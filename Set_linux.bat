#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

python - <<'PY'
import sys; print(sys.executable)
import faiss, streamlit
print("faiss & streamlit OK")
PY

python -m streamlit run app.py
