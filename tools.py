"""
Tools for EcoHome Energy Advisor Agent
"""
import glob
import os
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models.energy import DatabaseManager

db_manager = DatabaseManager()

DOCUMENTS_DIR = "data/documents"
VECTORSTORE_DIR = "data/vectorstore"

CONDITION_MAP = {
    "Clear": "sunny",
    "Clouds": "partly_cloudy",
    "Rain": "rainy",
    "Drizzle": "rainy",
    "Thunderstorm": "rainy",
    "Snow": "cloudy",
    "Mist": "cloudy",
    "Fog": "cloudy",
    "Haze": "cloudy",
}


def _safe_round(value: Optional[float], digits: int = 2) -> Optional[float]:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _parse_date(date: Optional[str]) -> datetime:
    if not date:
        return datetime.now()
    return datetime.strptime(date, "%Y-%m-%d")


def _estimate_irradiance(hour: int, condition: str) -> int:
    """Estimate solar irradiance in W/m2 from hour of day and sky condition."""
    if hour < 6 or hour > 18:
        return 0
    hour_factor = 1 - abs(hour - 12) / 6
    multipliers = {
        "sunny": 1.0,
        "partly_cloudy": 0.65,
        "cloudy": 0.3,
        "rainy": 0.12,
    }
    return max(0, int(950 * hour_factor * multipliers.get(condition, 0.4)))


def _mock_weather_forecast(location: str, days: int) -> Dict[str, Any]:
    """Generate a realistic mock forecast when a live weather API is unavailable."""
    days = max(1, min(int(days), 7))
    seed = abs(hash((location.lower().strip(), datetime.now().strftime("%Y-%m-%d")))) % (2**32)
    rng = random.Random(seed)

    conditions = ["sunny", "partly_cloudy", "cloudy", "rainy"]
    weights = [0.4, 0.35, 0.18, 0.07]
    daily_conditions = [rng.choices(conditions, weights=weights, k=1)[0] for _ in range(days)]

    base_temp = rng.uniform(16, 24)
    current_hour = datetime.now().hour
    current_condition = daily_conditions[0]
    hourly: List[Dict[str, Any]] = []

    for day_offset in range(days):
        day_condition = daily_conditions[day_offset]
        day_start = (datetime.now() + timedelta(days=day_offset)).replace(
            minute=0, second=0, microsecond=0
        )
        for hour in range(24):
            timestamp = day_start.replace(hour=hour)
            diurnal = 4 * (1 - abs(hour - 15) / 15)
            temperature = base_temp + diurnal + rng.uniform(-1.5, 1.5)
            if hour < 6 or hour > 20:
                hour_condition = "cloudy" if day_condition == "sunny" else day_condition
            else:
                hour_condition = day_condition
            hourly.append(
                {
                    "hour": hour,
                    "timestamp": timestamp.isoformat(),
                    "temperature_c": round(temperature, 1),
                    "condition": hour_condition,
                    "solar_irradiance": _estimate_irradiance(hour, hour_condition),
                    "humidity": int(rng.uniform(40, 80)),
                    "wind_speed": round(rng.uniform(1.5, 8.0), 1),
                }
            )

    return {
        "location": location,
        "forecast_days": days,
        "data_source": "mock",
        "generated_at": datetime.now().isoformat(),
        "current": {
            "temperature_c": round(base_temp + 2, 1),
            "condition": current_condition,
            "humidity": int(rng.uniform(45, 75)),
            "wind_speed": round(rng.uniform(2.0, 6.0), 1),
            "hour": current_hour,
        },
        "hourly": hourly,
        "note": (
            "Live weather API was unavailable. This forecast is a local mock used so "
            "scheduling recommendations can still be produced."
        ),
    }


def _live_weather_forecast(location: str, days: int, api_key: str) -> Dict[str, Any]:
    """Fetch a forecast from OpenWeather. Raises on failure so callers can fall back."""
    days = max(1, min(int(days), 7))
    raw_location = location.strip()
    city_part = raw_location.split(",")[0].strip() if "," in raw_location else raw_location
    geo_params = {"q": f"{city_part},US", "limit": 1, "appid": api_key}
    geo_resp = requests.get(
        "http://api.openweathermap.org/geo/1.0/direct",
        params=geo_params,
        timeout=15,
    )
    geo_resp.raise_for_status()
    geo_data = geo_resp.json()
    if not geo_data:
        raise ValueError(f"Location not found: {location}")

    lat = geo_data[0]["lat"]
    lon = geo_data[0]["lon"]

    weather_resp = requests.get(
        "https://api.openweathermap.org/data/2.5/forecast",
        params={"lat": lat, "lon": lon, "units": "metric", "appid": api_key},
        timeout=20,
    )
    weather_resp.raise_for_status()
    weather_data = weather_resp.json()
    entries = weather_data.get("list") or []
    if not entries:
        raise ValueError("Weather API returned no forecast entries")

    hourly: List[Dict[str, Any]] = []
    end_time = datetime.now() + timedelta(days=days)
    for entry in entries:
        ts = datetime.fromtimestamp(entry["dt"])
        if ts > end_time:
            break
        main_condition = (entry.get("weather") or [{}])[0].get("main", "Clouds")
        condition = CONDITION_MAP.get(main_condition, "cloudy")
        hourly.append(
            {
                "hour": ts.hour,
                "timestamp": ts.isoformat(),
                "temperature_c": _safe_round(entry.get("main", {}).get("temp"), 1),
                "condition": condition,
                "solar_irradiance": _estimate_irradiance(ts.hour, condition),
                "humidity": entry.get("main", {}).get("humidity"),
                "wind_speed": _safe_round(entry.get("wind", {}).get("speed"), 1),
            }
        )

    current_entry = entries[0]
    current_main = (current_entry.get("weather") or [{}])[0].get("main", "Clouds")
    return {
        "location": location,
        "latitude": lat,
        "longitude": lon,
        "forecast_days": days,
        "data_source": "openweathermap",
        "generated_at": datetime.now().isoformat(),
        "current": {
            "temperature_c": _safe_round(current_entry.get("main", {}).get("temp"), 1),
            "condition": CONDITION_MAP.get(current_main, "cloudy"),
            "humidity": current_entry.get("main", {}).get("humidity"),
            "wind_speed": _safe_round(current_entry.get("wind", {}).get("speed"), 1),
        },
        "hourly": hourly,
    }


@tool
def get_weather_forecast(location: str, days: int = 3) -> Dict[str, Any]:
    """
    Get weather forecast for a specific location and number of days.

    Args:
        location (str): Location to get weather for (e.g., "San Francisco, CA")
        days (int): Number of days to forecast (1-7)

    Returns:
        Dict[str, Any]: Weather forecast data including temperature, conditions, and solar irradiance
    """
    try:
        if not location or not str(location).strip():
            return {"error": "location is required, for example 'San Francisco, CA'"}
        days = max(1, min(int(days or 3), 7))
        api_key = os.getenv("OPENWEATHER_API_KEY") or os.getenv("OPENWEATHERMAP_API_KEY")
        if api_key:
            try:
                return _live_weather_forecast(location, days, api_key)
            except Exception as live_error:
                mock = _mock_weather_forecast(location, days)
                mock["live_api_error"] = str(live_error)
                return mock
        return _mock_weather_forecast(location, days)
    except Exception as e:
        return {"error": f"Failed to fetch weather forecast: {str(e)}"}


@tool
def get_electricity_prices(date: str = None) -> Dict[str, Any]:
    """
    Get electricity prices for a specific date or current day.

    Args:
        date (str): Date in YYYY-MM-DD format (defaults to today)

    Returns:
        Dict[str, Any]: Electricity pricing data with hourly rates.
    """
    try:
        dt = _parse_date(date)
        date = dt.strftime("%Y-%m-%d")
        is_weekend = dt.weekday() >= 5
        base_rate = 0.22
        weekend_multiplier = 0.9 if is_weekend else 1.0

        hourly_rates = []
        for hour in range(24):
            if hour in [0, 1, 2, 3, 4, 5, 23]:
                period = "off_peak"
                rate = base_rate * 0.6
                demand_charge = 0.0
            elif 6 <= hour <= 15:
                period = "mid_peak"
                rate = base_rate * 1.0
                demand_charge = 0.0
            elif 16 <= hour <= 21:
                period = "on_peak"
                rate = base_rate * 1.5
                demand_charge = 0.10
            else:
                period = "mid_peak"
                rate = base_rate * 1.1
                demand_charge = 0.05

            rate *= weekend_multiplier
            demand_charge *= weekend_multiplier
            hourly_rates.append(
                {
                    "hour": hour,
                    "rate": round(rate, 4),
                    "period": period,
                    "demand_charge": round(demand_charge, 4),
                }
            )

        cheapest = min(hourly_rates, key=lambda row: row["rate"])
        expensive = max(hourly_rates, key=lambda row: row["rate"])
        return {
            "date": date,
            "weekday": dt.strftime("%A"),
            "pricing_type": "time_of_use",
            "currency": "USD",
            "unit": "per_kWh",
            "is_weekend": is_weekend,
            "base_rate_usd_per_kwh": base_rate,
            "cheapest_hour": cheapest["hour"],
            "cheapest_rate": cheapest["rate"],
            "most_expensive_hour": expensive["hour"],
            "most_expensive_rate": expensive["rate"],
            "hourly_rates": hourly_rates,
        }
    except ValueError:
        return {"error": "date must use YYYY-MM-DD format"}
    except Exception as e:
        return {"error": f"Failed to get electricity prices: {str(e)}"}


@tool
def query_energy_usage(start_date: str, end_date: str, device_type: str = None) -> Dict[str, Any]:
    """
    Query energy usage data from the database for a specific date range.

    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        device_type (str): Optional device type filter (e.g., "EV", "HVAC", "appliance")

    Returns:
        Dict[str, Any]: Energy usage data with consumption details
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        if end_dt <= start_dt:
            return {"error": "end_date must be on or after start_date"}

        records = db_manager.get_usage_by_date_range(start_dt, end_dt)
        if device_type:
            records = [r for r in records if (r.device_type or "").lower() == device_type.lower()]

        device_breakdown: Dict[str, Dict[str, Any]] = {}
        for record in records:
            device = record.device_type or "unknown"
            if device not in device_breakdown:
                device_breakdown[device] = {"consumption_kwh": 0.0, "cost_usd": 0.0, "records": 0}
            device_breakdown[device]["consumption_kwh"] += record.consumption_kwh
            device_breakdown[device]["cost_usd"] += record.cost_usd or 0
            device_breakdown[device]["records"] += 1

        for data in device_breakdown.values():
            data["consumption_kwh"] = round(data["consumption_kwh"], 2)
            data["cost_usd"] = round(data["cost_usd"], 2)

        sample_limit = 48
        usage_data = {
            "start_date": start_date,
            "end_date": end_date,
            "device_type": device_type,
            "total_records": len(records),
            "total_consumption_kwh": round(sum(r.consumption_kwh for r in records), 2),
            "total_cost_usd": round(sum(r.cost_usd or 0 for r in records), 2),
            "device_breakdown": device_breakdown,
            "records": [],
        }
        for record in records[:sample_limit]:
            usage_data["records"].append(
                {
                    "timestamp": record.timestamp.isoformat(),
                    "consumption_kwh": round(record.consumption_kwh, 3),
                    "device_type": record.device_type,
                    "device_name": record.device_name,
                    "cost_usd": round(record.cost_usd or 0, 4),
                }
            )
        if len(records) > sample_limit:
            usage_data["records_truncated"] = True
            usage_data["records_returned"] = sample_limit
        return usage_data
    except ValueError:
        return {"error": "start_date and end_date must use YYYY-MM-DD format"}
    except Exception as e:
        return {"error": f"Failed to query energy usage: {str(e)}"}


@tool
def query_solar_generation(start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Query solar generation data from the database for a specific date range.

    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format

    Returns:
        Dict[str, Any]: Solar generation data with production details
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        if end_dt <= start_dt:
            return {"error": "end_date must be on or after start_date"}

        records = db_manager.get_generation_by_date_range(start_dt, end_dt)
        span_days = max(1, (end_dt - start_dt).days)
        weather_breakdown: Dict[str, Dict[str, Any]] = {}
        for record in records:
            weather = record.weather_condition or "unknown"
            if weather not in weather_breakdown:
                weather_breakdown[weather] = {"generation_kwh": 0.0, "records": 0}
            weather_breakdown[weather]["generation_kwh"] += record.generation_kwh
            weather_breakdown[weather]["records"] += 1

        for data in weather_breakdown.values():
            data["generation_kwh"] = round(data["generation_kwh"], 2)

        sample_limit = 48
        generation_data = {
            "start_date": start_date,
            "end_date": end_date,
            "total_records": len(records),
            "total_generation_kwh": round(sum(r.generation_kwh for r in records), 2),
            "average_daily_generation": round(
                sum(r.generation_kwh for r in records) / span_days, 2
            ),
            "weather_breakdown": weather_breakdown,
            "records": [],
        }
        for record in records[:sample_limit]:
            generation_data["records"].append(
                {
                    "timestamp": record.timestamp.isoformat(),
                    "generation_kwh": round(record.generation_kwh, 3),
                    "weather_condition": record.weather_condition,
                    "temperature_c": record.temperature_c,
                    "solar_irradiance": record.solar_irradiance,
                }
            )
        if len(records) > sample_limit:
            generation_data["records_truncated"] = True
            generation_data["records_returned"] = sample_limit
        return generation_data
    except ValueError:
        return {"error": "start_date and end_date must use YYYY-MM-DD format"}
    except Exception as e:
        return {"error": f"Failed to query solar generation: {str(e)}"}


@tool
def get_recent_energy_summary(hours: int = 24) -> Dict[str, Any]:
    """
    Get a summary of recent energy usage and solar generation.

    Args:
        hours (int): Number of hours to look back (default 24)

    Returns:
        Dict[str, Any]: Summary of recent energy data
    """
    try:
        hours = max(1, min(int(hours or 24), 24 * 60))
        usage_records = db_manager.get_recent_usage(hours)
        generation_records = db_manager.get_recent_generation(hours)

        weather_counts: Dict[str, int] = {}
        for record in generation_records:
            weather = record.weather_condition or "unknown"
            weather_counts[weather] = weather_counts.get(weather, 0) + 1
        average_weather = (
            max(weather_counts, key=weather_counts.get) if weather_counts else "unknown"
        )

        summary = {
            "time_period_hours": hours,
            "usage": {
                "total_consumption_kwh": round(sum(r.consumption_kwh for r in usage_records), 2),
                "total_cost_usd": round(sum(r.cost_usd or 0 for r in usage_records), 2),
                "device_breakdown": {},
            },
            "generation": {
                "total_generation_kwh": round(
                    sum(r.generation_kwh for r in generation_records), 2
                ),
                "average_weather": average_weather,
            },
        }

        for record in usage_records:
            device = record.device_type or "unknown"
            if device not in summary["usage"]["device_breakdown"]:
                summary["usage"]["device_breakdown"][device] = {
                    "consumption_kwh": 0,
                    "cost_usd": 0,
                    "records": 0,
                }
            summary["usage"]["device_breakdown"][device]["consumption_kwh"] += record.consumption_kwh
            summary["usage"]["device_breakdown"][device]["cost_usd"] += record.cost_usd or 0
            summary["usage"]["device_breakdown"][device]["records"] += 1

        for device_data in summary["usage"]["device_breakdown"].values():
            device_data["consumption_kwh"] = round(device_data["consumption_kwh"], 2)
            device_data["cost_usd"] = round(device_data["cost_usd"], 2)

        net_kwh = (
            summary["generation"]["total_generation_kwh"]
            - summary["usage"]["total_consumption_kwh"]
        )
        summary["net_generation_minus_usage_kwh"] = round(net_kwh, 2)
        summary["grid_import_needed"] = net_kwh < 0
        return summary
    except Exception as e:
        return {"error": f"Failed to get recent energy summary: {str(e)}"}


def _load_tip_documents() -> List:
    documents = []
    for doc_path in sorted(glob.glob(os.path.join(DOCUMENTS_DIR, "*.txt"))):
        loader = TextLoader(doc_path)
        documents.extend(loader.load())
    return documents


def _embedding_client() -> OpenAIEmbeddings:
    kwargs = {"model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")}
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    if base_url:
        kwargs["base_url"] = base_url
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        kwargs["api_key"] = api_key
    return OpenAIEmbeddings(**kwargs)


def _get_vectorstore() -> Chroma:
    persist_directory = VECTORSTORE_DIR
    os.makedirs(persist_directory, exist_ok=True)
    embeddings = _embedding_client()
    chroma_path = os.path.join(persist_directory, "chroma.sqlite3")
    if os.path.exists(chroma_path):
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    documents = _load_tip_documents()
    if not documents:
        raise FileNotFoundError(
            f"No energy tip documents found in {DOCUMENTS_DIR}. "
            "Add .txt files and rerun 02_rag_setup.ipynb."
        )
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(documents)
    return Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory,
    )


@tool
def search_energy_tips(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search for energy-saving tips and best practices using RAG.

    Args:
        query (str): Search query for energy tips
        max_results (int): Maximum number of results to return

    Returns:
        Dict[str, Any]: Relevant energy tips and best practices
    """
    try:
        if not query or not str(query).strip():
            return {"error": "query is required"}
        max_results = max(1, min(int(max_results or 5), 10))
        vectorstore = _get_vectorstore()
        docs = vectorstore.similarity_search(query, k=max_results)
        results = {"query": query, "total_results": len(docs), "tips": []}
        for i, doc in enumerate(docs):
            results["tips"].append(
                {
                    "rank": i + 1,
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "relevance_score": "high" if i < 2 else "medium" if i < 4 else "low",
                }
            )
        return results
    except Exception as e:
        return {"error": f"Failed to search energy tips: {str(e)}"}


@tool
def calculate_energy_savings(
    device_type: str,
    current_usage_kwh: float,
    optimized_usage_kwh: float,
    price_per_kwh: float = 0.12,
) -> Dict[str, Any]:
    """
    Calculate potential energy savings from optimization.

    Args:
        device_type (str): Type of device being optimized
        current_usage_kwh (float): Current energy usage in kWh
        optimized_usage_kwh (float): Optimized energy usage in kWh
        price_per_kwh (float): Price per kWh (default 0.12)

    Returns:
        Dict[str, Any]: Savings calculation results
    """
    try:
        current_usage_kwh = float(current_usage_kwh)
        optimized_usage_kwh = float(optimized_usage_kwh)
        price_per_kwh = float(price_per_kwh)
        if current_usage_kwh < 0 or optimized_usage_kwh < 0 or price_per_kwh < 0:
            return {"error": "usage and price values must be zero or positive"}

        savings_kwh = current_usage_kwh - optimized_usage_kwh
        savings_usd = savings_kwh * price_per_kwh
        savings_percentage = (
            (savings_kwh / current_usage_kwh) * 100 if current_usage_kwh > 0 else 0
        )
        return {
            "device_type": device_type,
            "current_usage_kwh": round(current_usage_kwh, 3),
            "optimized_usage_kwh": round(optimized_usage_kwh, 3),
            "savings_kwh": round(savings_kwh, 2),
            "savings_usd": round(savings_usd, 2),
            "savings_percentage": round(savings_percentage, 1),
            "price_per_kwh": price_per_kwh,
            "monthly_savings_usd": round(savings_usd * 30, 2),
            "annual_savings_usd": round(savings_usd * 365, 2),
        }
    except (TypeError, ValueError):
        return {"error": "current_usage_kwh, optimized_usage_kwh, and price_per_kwh must be numbers"}
    except Exception as e:
        return {"error": f"Failed to calculate energy savings: {str(e)}"}


TOOL_KIT = [
    get_weather_forecast,
    get_electricity_prices,
    query_energy_usage,
    query_solar_generation,
    get_recent_energy_summary,
    search_energy_tips,
    calculate_energy_savings,
]
