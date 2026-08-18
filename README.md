# EcoHome Energy Advisor

An AI energy advisor for homes with solar, EVs, HVAC, appliances, and optional storage. The agent uses weather, time-of-use prices, household history, and a RAG knowledge base to recommend when to run devices and how much that can save.

## What this project includes

- SQLite history for energy use and solar generation
- Weather forecasts with solar irradiance (live OpenWeather if a key is present, otherwise a local mock)
- Time-of-use electricity prices
- RAG search over energy-saving documents
- A LangGraph ReAct agent with conversation memory, date context, and error handling

## Project structure

```
ecohome_solution/
├── models/
│   ├── __init__.py
│   └── energy.py
├── data/
│   ├── documents/
│   ├── energy_data.db
│   └── vectorstore/
├── agent.py
├── tools.py
├── requirements.txt
├── 01_db_setup.ipynb
├── 02_rag_setup.ipynb
├── 03_run_and_evaluate.ipynb
└── README.md
```

## Local environment

- Python 3.12.0
- Packages from `requirements.txt`

Versions used in this local run:

| Package | Version |
| --- | --- |
| Python | 3.12.0 |
| langchain | 0.3.30 |
| langchain-core | 0.3.86 |
| langchain-community | 0.3.31 |
| langchain-openai | 0.3.35 |
| langchain-chroma | 0.2.6 |
| langchain-text-splitters | 0.3.11 |
| langgraph | 0.6.11 |
| chromadb | 1.5.9 |
| SQLAlchemy | 2.0.52 |
| openai | 2.54.0 |
| python-dotenv | 1.2.3 |
| requests | 2.34.2 |
| pandas | 3.0.5 |
| httpx | 0.28.1 |

Default chat model: `gpt-4o-mini`.

This machine could not call OpenAI chat models because of a regional restriction. Local evaluation used an OpenAI-compatible proxy with `OPENAI_MODEL=deepseek/deepseek-chat` and embeddings model `openai/text-embedding-3-small`. Set `OPENAI_API_KEY` for a normal OpenAI account and leave `OPENAI_MODEL` unset to use `gpt-4o-mini`.

## Setup

1. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
2. Optional: set `OPENWEATHER_API_KEY`. If it is missing, weather tools use a local mock forecast.
3. Create a virtual environment and install dependencies:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4. Run the notebooks in order:

- `01_db_setup.ipynb`: create tables and sample energy data
- `02_rag_setup.ipynb`: embed documents into ChromaDB
- `03_run_and_evaluate.ipynb`: run the agent on 10 scenarios and score tool use

## Knowledge base

Starter documents:

- `tip_device_best_practices.txt`
- `tip_energy_savings.txt`

Added documents:

- `tip_hvac_optimization_strategies.txt`
- `tip_smart_home_automation.txt`
- `tip_renewable_energy_integration.txt`
- `tip_seasonal_energy_management.txt`
- `tip_energy_storage_optimization.txt`

## Agent behavior

The Energy Advisor:

- Resolves today, tomorrow, Wednesday, and this weekend into dates
- Calls weather and pricing together for scheduling questions
- Uses usage history for personalized advice
- Retrieves RAG tips for best-practice questions
- Returns hours, setpoints, and dollar estimates
- Continues with general advice if a tool fails

## Evaluation (local run)

10 scenarios, 0 failed tests:

- Avg tool completeness: 1.00
- Avg tool appropriateness: 0.95
- Avg tool overall: 0.97
- Avg response overall: 0.79
- Avg usefulness: 1.00

The agent used the expected tools for EV charging, thermostat setpoints, dishwasher savings, solar outlook, usage history, multi-device scheduling, tips, recent summaries, and pool pump timing.

## Example questions

- When should I charge my electric car tomorrow to minimize cost and maximize solar power?
- What temperature should I set my thermostat on Wednesday afternoon if electricity prices spike?
- Suggest three ways I can reduce energy use based on my usage history.
- How much can I save by running my dishwasher during off-peak hours?
- What's the best time to run my pool pump this week based on the weather forecast?
