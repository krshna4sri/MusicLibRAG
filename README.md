musicrag/
├─ app.py
├─ requirements.txt
├─ README.md
├─ .gitignore
├─ setup_win.bat
├─ setup_unix.sh
└─ client_secrets.sample.json   # rename to client_secrets.json (and fill)

# MusicRAG — Ask your Music Library

Single-file Streamlit app that:
- Syncs PDFs from a Google Drive folder (optionally recursive)
- Indexes them with FAISS (embeddings via Ollama)
- Answers questions using only your library (citations included)

## Quick Start

### 0) Prereqs
- Python 3.10+
- [Ollama](https://ollama.com/download)
- (Optional) Google Cloud project with **Drive API** enabled

### 1) Clone & install
```bash
git clone <your_repo_url> musicrag
cd musicrag
# Windows
setup_win.bat
# Mac/Linux
./setup_unix.sh

