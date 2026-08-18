# EcoHome Energy Advisor

Home energy advisor: weather, time-of-use prices, usage history, and RAG tips.

## Setup

```
cp .env.example .env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set `OPENAI_API_KEY` in `.env`. `OPENWEATHER_API_KEY` is optional (mock weather if missing).

## Run

1. `01_db_setup.ipynb`
2. `02_rag_setup.ipynb`
3. `03_run_and_evaluate.ipynb`
