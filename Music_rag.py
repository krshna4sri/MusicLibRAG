# ------------------------------------------------------------
# Music Library RAG — Single File Streamlit App
# ------------------------------------------------------------
# Features:
# - Google Drive PDF sync (your folder ID is set below)
# - PDF text extraction (PyMuPDF)
# - Embeddings (Ollama: nomic-embed-text)
# - Vector search (FAISS, IP w/ normalized vectors)
# - Local LLM answer generation (Ollama: llama3.1)
# - Simple UI: one question box, one answer box (+ citations)
#
# First run:
# 1) Place client_secrets.json next to this file (OAuth Desktop Client).
# 2) Install requirements (see bottom of this file).
# 3) Start Ollama and pull models:
#       ollama pull nomic-embed-text
#       ollama pull llama3.1
# 4) Run:
#       streamlit run app.py
# ------------------------------------------------------------

import os
import io
import glob
import pickle
from typing import List, Dict, Tuple

import numpy as np
import requests
import faiss
import fitz  # PyMuPDF
import streamlit as st

# --------- CONFIG: EDIT IF NEEDED -------------------------------------------
DRIVE_FOLDER_ID = "11SNLb_ponkPGhbf9tKhdoQJFtPjfdnmH"  # <-- your folder ID
DATA_DIR   = "./data"
PDF_DIR    = os.path.join(DATA_DIR, "pdfs")
INDEX_DIR  = os.path.join(DATA_DIR, "index")

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL  = "llama3.1"

CHUNK_SIZE     = 1200
CHUNK_OVERLAP  = 200
TOP_K          = 6
BATCH_EMBED    = 64
OLLAMA_URL_EMB = "http://localhost:11434/api/embeddings"
OLLAMA_URL_GEN = "http://localhost:11434/api/generate"

# ----------------------------------------------------------------------------
# Google Drive (PyDrive2) auth + download (all inline for single-file app)
# ----------------------------------------------------------------------------
# We lazy-import pydrive2 so the app can still run Q&A without Drive syncing
def _ensure_dirs():
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)

def _gauth():
    from pydrive2.auth import GoogleAuth
    gauth = GoogleAuth()
    # Must have client_secrets.json in the same folder as this file
    if not os.path.exists("client_secrets.json"):
        raise FileNotFoundError(
            "Missing client_secrets.json. Create an OAuth Client ID (Desktop) "
            "in Google Cloud Console and download it here as client_secrets.json."
        )
    gauth.LoadClientConfigFile("client_secrets.json")
    # Reuse token if present
    gauth.LoadCredentialsFile("token.json")
    if gauth.credentials is None:
        # first run: opens a browser for OAuth
        gauth.LocalWebserverAuth()
        gauth.SaveCredentialsFile("token.json")
    elif gauth.access_token_expired:
        gauth.Refresh()
        gauth.SaveCredentialsFile("token.json")
    else:
        gauth.Authorize()
    return gauth

def list_pdfs_in_folder(folder_id: str):
    from pydrive2.drive import GoogleDrive
    gauth = _gauth()
    drive = GoogleDrive(gauth)
    q = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
    return drive.ListFile({'q': q}).GetList()

def download_pdfs(folder_id: str, out_dir: str) -> List[str]:
    from pydrive2.drive import GoogleDrive
    gauth = _gauth()
    drive = GoogleDrive(gauth)
    q = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
    file_list = drive.ListFile({'q': q}).GetList()

    paths = []
    for f in file_list:
        title = f['title']
        local_path = os.path.join(out_dir, title)
        # Skip if already exists
        if os.path.exists(local_path):
            paths.append(local_path)
            continue
        f.GetContentFile(local_path)
        paths.append(local_path)
    return paths

# ----------------------------------------------------------------------------
# PDF utils (text extraction + chunking)
# ----------------------------------------------------------------------------
def extract_text_per_page(pdf_path: str) -> List[Dict]:
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

# ----------------------------------------------------------------------------
# Embeddings + Index (Ollama + FAISS)
# ----------------------------------------------------------------------------
def ollama_embed(texts: List[str]) -> np.ndarray:
    vecs = []
    for t in texts:
        resp = requests.post(OLLAMA_URL_EMB, json={"model": EMBED_MODEL, "prompt": t})
        resp.raise_for_status()
        vecs.append(np.array(resp.json()["embedding"], dtype=np.float32))
    return np.vstack(vecs)

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    # Use Inner Product + normalized vectors => cosine
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def save_index(index: faiss.IndexFlatIP, meta: List[Dict]):
    _ensure_dirs()
    faiss.write_index(index, os.path.join(INDEX_DIR, "vectors.faiss"))
    with open(os.path.join(INDEX_DIR, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

def load_index() -> Tuple[faiss.IndexFlatIP, List[Dict]]:
    vec_path = os.path.join(INDEX_DIR, "vectors.faiss")
    meta_path = os.path.join(INDEX_DIR, "meta.pkl")
    if not (os.path.exists(vec_path) and os.path.exists(meta_path)):
        raise FileNotFoundError("Index not found. Build the index first.")
    index = faiss.read_index(vec_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return index, meta

def search(index: faiss.IndexFlatIP, meta: List[Dict], query: str, k: int = TOP_K) -> List[Dict]:
    qvec = ollama_embed([query])
    faiss.normalize_L2(qvec)
    D, I = index.search(qvec, k)
    hits = []
    for score, idx in zip(D[0], I[0]):
        item = meta[idx].copy()
        item["score"] = float(score)
        hits.append(item)
    return hits

# ----------------------------------------------------------------------------
# Generation (Ollama chat)
# ----------------------------------------------------------------------------
def ollama_chat(prompt: str) -> str:
    resp = requests.post(
        OLLAMA_URL_GEN,
        json={"model": CHAT_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "")

def format_prompt(query: str, passages: List[Dict]) -> str:
    ctx_lines = []
    for p in passages:
        src = f"{p['source']} (p.{p['page']})"
        ctx_lines.append(f"[{src}] {p['text']}")
    context = "\n\n".join(ctx_lines)
    system = (
        "You answer ONLY from the provided CONTEXT below. "
        "Cite sources using [filename (p.X)]. "
        "If the answer is not in the context, say you don't know."
    )
    return f"{system}\n\nCONTEXT:\n{context}\n\nUSER QUESTION: {query}\n\nASSISTANT ANSWER:"

# ----------------------------------------------------------------------------
# Index builder (from local PDFs)
# ----------------------------------------------------------------------------
def build_index_from_pdfs(pdf_dir: str):
    pdfs = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdfs:
        raise RuntimeError(f"No PDFs found in {pdf_dir}. Download or copy some PDFs first.")

    texts: List[str] = []
    meta: List[Dict] = []

    for pdf_path in pdfs:
        pages = extract_text_per_page(pdf_path)
        for page in pages:
            txt = (page["text"] or "").strip()
            if not txt:
                continue
            for ch in chunk_text(txt, CHUNK_SIZE, CHUNK_OVERLAP):
                texts.append(ch)
                meta.append({
                    "source": os.path.basename(pdf_path),
                    "page": page["page"],
                    "text": ch[:800]  # store a snippet
                })

    if not texts:
        raise RuntimeError("No extractable text found in your PDFs (they may be scans). Consider OCR later.")

    # Embed in batches to avoid long single requests
    embs = []
    for i in range(0, len(texts), BATCH_EMBED):
        embs.append(ollama_embed(texts[i:i + BATCH_EMBED]))
    embs = np.vstack(embs)

    index = build_faiss_index(embs)
    save_index(index, meta)

# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Music Library QA", page_icon="🎵", layout="centered")
st.title("🎵 Ask your Music Library")

with st.sidebar:
    st.header("Library Controls")
    st.caption("First run? Do these top-to-bottom.")

    colA, colB = st.columns(2)
    with colA:
        if st.button("1) Download/Refresh PDFs"):
            try:
                _ensure_dirs()
                paths = download_pdfs(DRIVE_FOLDER_ID, PDF_DIR)
                st.success(f"Downloaded/checked {len(paths)} PDFs into {PDF_DIR}")
            except Exception as e:
                st.error(f"Download error: {e}")

    with colB:
        if st.button("2) Build/Refresh Index"):
            try:
                _ensure_dirs()
                build_index_from_pdfs(PDF_DIR)
                st.success("Index built/refreshed.")
            except Exception as e:
                st.error(f"Index error: {e}")

    st.divider()
    st.subheader("Settings")
    top_k = st.slider("Top-K passages", 3, 12, TOP_K)
    st.caption("If answers seem shallow, increase Top-K.")

    st.divider()
    st.caption("Models (Ollama):")
    EMBED_MODEL = st.text_input("Embedding model", EMBED_MODEL)
    CHAT_MODEL  = st.text_input("Chat model", CHAT_MODEL)
    if st.button("Check Ollama"):
        try:
            r = requests.post(OLLAMA_URL_GEN, json={"model": CHAT_MODEL, "prompt": "Hello", "stream": False}, timeout=20)
            r.raise_for_status()
            st.success("Ollama reachable ✅")
        except Exception as e:
            st.error(f"Ollama not reachable: {e}")

# Main Q&A box
question = st.text_area(
    "Your question",
    height=130,
    placeholder="e.g., What is voice leading? How do I route sidechain compression in Ableton?"
)

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        # Load index (if present)
        try:
            index, meta = load_index()
        except Exception as e:
            st.error(f"Couldn't load index: {e}")
            st.stop()

        with st.spinner("Searching your library..."):
            try:
                hits = search(index, meta, question, k=top_k)
            except Exception as e:
                st.error(f"Search error: {e}")
                st.stop()

        if not hits:
            st.warning("No relevant passages found. Try rebuilding the index or broadening your question.")
        else:
            with st.spinner("Generating answer..."):
                prompt = format_prompt(question, hits)
                try:
                    answer = ollama_chat(prompt)
                except Exception as e:
                    st.error(f"Generation error: {e}")
                    st.stop()

            st.subheader("Answer")
            st.write(answer)

            st.caption("Sources")
            for h in hits:
                st.write(f"- {h['source']} (p.{h['page']}) • score={h['score']:.3f}")

st.markdown("---")
with st.expander("Notes & Tips"):
    st.markdown(
        """
- This indexes **text** inside your PDFs. If some manuals are scans (images), consider adding OCR later.
- If answers feel too shallow, increase **Top-K** in the sidebar.
- To re-sync Drive or re-build the index, use the sidebar buttons.
- For true music notation understanding (OMR → MusicXML), you can layer in **Audiveris** + **music21** later.
        """
    )

# ----------------------------- END OF FILE -----------------------------------
