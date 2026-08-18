import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from tools import TOOL_KIT

load_dotenv()

DEFAULT_SYSTEM_INSTRUCTIONS = """
You are the EcoHome Energy Advisor. You help homeowners cut electricity cost and
carbon impact for homes that may include rooftop solar, an EV, HVAC, appliances,
a pool pump, and optional battery storage.

Available tools:
- get_weather_forecast(location, days): hourly temperature, sky condition, and solar irradiance.
- get_electricity_prices(date): hourly time-of-use rates (YYYY-MM-DD).
- query_energy_usage(start_date, end_date, device_type): household consumption history.
- query_solar_generation(start_date, end_date): historical solar production.
- get_recent_energy_summary(hours): recent usage, cost, and generation snapshot.
- search_energy_tips(query): energy-saving tips from the knowledge base.
- calculate_energy_savings(device_type, current_usage_kwh, optimized_usage_kwh, price_per_kwh):
  kWh, dollar, monthly, and annual savings.

How to choose tools:
1. Scheduling questions ("when should I charge / run ..."):
   Call get_weather_forecast AND get_electricity_prices. Prefer hours that are both
   high-solar and low-price. If the user mentions history, also query usage.
2. Thermostat questions:
   Call get_electricity_prices and get_weather_forecast. Give a numeric setpoint
   in F and C, plus a pre-cool or pre-heat window before peak prices.
3. "Based on my usage / history / last month / last 48 hours":
   Call get_recent_energy_summary. If the user names a date range such as last
   month, also call query_energy_usage. Then call search_energy_tips. Name the
   highest-cost devices and give 3 concrete actions.
4. "How much can I save":
   Call get_electricity_prices, then calculate_energy_savings. State assumptions
   (kWh per cycle, hours of runtime, peak vs off-peak rate).
5. Tips / best practices:
   Call search_energy_tips and rewrite the retrieved tips in plain language.
6. Solar outlook:
   Call get_weather_forecast. Mention sunny hours and irradiance. If useful,
   compare with query_solar_generation.

Context rules:
- Use the current date provided below to resolve "today", "tomorrow", "this weekend",
  "Wednesday", and "this week" into YYYY-MM-DD before calling date tools.
- If a location is provided in extra context, pass that same location to weather tools.
- Default device assumptions when the user does not give sizes:
  EV charge: 10 kWh, dishwasher: 1.5 kWh/cycle, washer: 1.0 kWh/cycle,
  dryer: 3.0 kWh/cycle, pool pump: 1.5 kWh per hour for 6 hours,
  HVAC: 2.0 kWh per hour.
- Combine devices into one coordinated schedule when the user names more than one load.
- Answer the question first, then give 2-3 numbered actions with hours, rates, and
  estimated dollars. Keep units consistent (kWh, USD, C/F).
- Only call the tools needed for the question. Do not call every tool.
  Savings questions need prices plus calculate_energy_savings.
  Scheduling questions need weather plus prices.
  History questions need get_recent_energy_summary plus search_energy_tips.
- If a tool returns an error or empty data, say that live data was unavailable and
  continue with clear best-practice advice. Do not invent meter readings.
- Never claim you ran a tool if you did not. Prefer calling tools over guessing.
- Always produce a final written answer after tool results. Do not stop on an empty message.
"""


def _weekday_date(today: datetime, weekday: int, force_future: bool = True) -> str:
    days_ahead = (weekday - today.weekday()) % 7
    if force_future and days_ahead == 0:
        days_ahead = 7
    return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")


def build_system_instructions(base_instructions: Optional[str] = None) -> str:
    """Attach current date context so the agent can resolve relative days."""
    today = datetime.now()
    date_block = f"""
Current local datetime: {today.strftime("%Y-%m-%d %H:%M")}
Today is {today.strftime("%A")} ({today.strftime("%Y-%m-%d")}).
Tomorrow is {(today + timedelta(days=1)).strftime("%A")} ({(today + timedelta(days=1)).strftime("%Y-%m-%d")}).
Next Wednesday is {_weekday_date(today, 2, force_future=True)}.
This weekend Saturday is {_weekday_date(today, 5, force_future=False)}.
When calling get_electricity_prices or usage tools, convert relative days to YYYY-MM-DD.
"""
    return (base_instructions or DEFAULT_SYSTEM_INSTRUCTIONS).strip() + "\n" + date_block.strip()


def _message_text(message) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
        content = "\n".join(part for part in parts if part)
    return str(content or "").strip()


def _final_answer(result: Dict[str, Any]) -> str:
    messages = []
    if isinstance(result, dict):
        messages = result.get("messages") or []
    for message in reversed(messages):
        name = type(message).__name__
        if name in {"AIMessage", "AIMessageChunk"}:
            text = _message_text(message)
            if text:
                return text
    if messages:
        return _message_text(messages[-1])
    return str(result)


class Agent:
    def __init__(self, instructions: Optional[str] = None, model: str = "gpt-4o-mini") -> None:
        """
        EcoHome Energy Advisor Agent.

        Args:
            instructions: Optional custom system instructions. If None, the
                         default EcoHome advisor prompt is used.
            model: OpenAI model name.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to a .env file or export it "
                "in your environment before creating the agent."
            )

        self.system_instructions = build_system_instructions(instructions)
        llm_kwargs = {
            "model": os.getenv("OPENAI_MODEL", model),
            "temperature": 0.0,
            "api_key": api_key,
            "max_retries": 2,
        }
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
        if base_url:
            llm_kwargs["base_url"] = base_url
        self.llm = ChatOpenAI(**llm_kwargs)
        self.graph = create_react_agent(
            name="energy_advisor",
            prompt=SystemMessage(content=self.system_instructions),
            model=self.llm,
            tools=TOOL_KIT,
        )
        self.history: List[Tuple[str, str]] = []

    def invoke(
        self,
        question: str,
        context: Optional[str] = None,
        reset_history: bool = False,
    ) -> Dict[str, Any]:
        """
        Ask the Energy Advisor a question about energy optimization.

        Args:
            question: The user's question about energy optimization.
            context: Optional extra context (e.g., "Location: San Francisco, CA").
            reset_history: If True, clears prior conversation history.

        Returns:
            The raw LangGraph result (a dict with a `messages` list).
        """
        if not question or not str(question).strip():
            error_text = "Please ask a question about energy use, pricing, solar, or device scheduling."
            return {
                "messages": [
                    SystemMessage(content=self.system_instructions),
                    AIMessage(content=error_text),
                ]
            }

        try:
            if reset_history:
                self.history = []

            messages: List[Tuple[str, str]] = []
            if context:
                messages.append(
                    (
                        "system",
                        (
                            "Additional household context for this request: "
                            f"{context}. Use this location for weather tools and "
                            "personalize device advice when possible."
                        ),
                    )
                )

            for role, content in self.history[-8:]:
                messages.append((role, content))

            messages.append(("user", question.strip()))
            result = self.graph.invoke(
                {"messages": messages},
                config={"recursion_limit": 30},
            )
            answer_text = _final_answer(result)

            if not answer_text:
                retry_messages = messages + [
                    (
                        "user",
                        "Please give the final schedule and savings answer now in plain text.",
                    )
                ]
                result = self.graph.invoke(
                    {"messages": retry_messages},
                    config={"recursion_limit": 20},
                )
                answer_text = _final_answer(result)

            if not answer_text:
                answer_text = (
                    "I gathered energy data but did not finish the written answer. "
                    "Please ask again with the device name and a specific day."
                )
                if isinstance(result, dict):
                    result.setdefault("messages", []).append(AIMessage(content=answer_text))

            self.history.append(("user", question.strip()))
            self.history.append(("assistant", answer_text))
            return result
        except Exception as e:
            error_text = (
                "I could not finish the energy analysis because of an internal error. "
                "Try asking again with a specific date, location, and device. "
                f"({type(e).__name__}: {e})"
            )
            self.history.append(("user", question.strip()))
            self.history.append(("assistant", error_text))
            return {
                "messages": [
                    SystemMessage(content="EcoHome Energy Advisor encountered an internal error."),
                    HumanMessage(content=question),
                    AIMessage(content=error_text),
                ]
            }

    def get_agent_tools(self) -> List[str]:
        """Get list of available tools for the Energy Advisor."""
        return [t.name for t in TOOL_KIT]
