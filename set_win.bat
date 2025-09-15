@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  py -3.12 -m venv .venv || py -3 -m venv .venv
)

echo Upgrading pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip

echo Installing deps...
".venv\Scripts\python.exe" -m pip install -r requirements.txt

echo Sanity check...
".venv\Scripts\python.exe" -c "import sys; print(sys.executable); import faiss, streamlit; print('faiss & streamlit OK')"

echo Launching app...
".venv\Scripts\python.exe" -m streamlit run app.py
endlocal
