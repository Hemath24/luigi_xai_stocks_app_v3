
#!/usr/bin/env bash
set -e
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# Optional OpenAI (only if you want cloud LLM)
# export OPENAI_API_KEY="sk-proj-..."
# export OPENAI_BASE_URL="https://api.openai.com/v1"
# export OPENAI_MODEL="gpt-4o-mini"
uvicorn api.main:app --host 0.0.0.0 --port 8000
