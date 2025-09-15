# music_rag_app.py
# Single-file RAG app for a local Music PDF library.
# UI: one input for question, one box for answer—on the same screen.
#
# Run:
#   pip install streamlit pypdf sentence-transformers faiss-cpu transformers torch --extra-index-url https://download.pytorch.org/whl/cpu
#   streamlit run music_rag_app.py
#
# What it does:
# - Walks a folder, reads all PDFs
# - Extracts text per page (skips images-only pages)
# - Chunks text, builds FAISS vector index with SentenceTransformers
# - Answers questions using retrieval-augmented generation (FLAN-T5 on CPU)
# - Keeps index on disk to avoid re-ingesting every run

import os
import sys
import time
import pickle
import hashlib
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict

import streamlit as st
from pypdf import PdfReader

import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --------------------------
# Config (you can tweak)
# --------------------------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # fast & good enough
GEN_MODEL_NAME = "google/flan-t5-base"  # lightweight instruction model for CPU
CHUNK_SIZE = 900         # characters
CHUNK_OVERLAP = 200      # characters
TOP_K = 5                # retrieved chunks
MAX_INPUT_CHARS = 7000   # safety cap for generator prompt
INDEX_DIR = ".music_rag_index"  # created beside your data dir

# --------------------------
# Utilities
# --------------------------
def read_pdf_text(pdf_path: Path) -> List[Tuple[int, str]]:
    """Return list of (page_number, text) for a PDF."""
    out = []
    try:
        reader = PdfReader(str(pdf_path))
        for i, page in enumerate(reader.pages):
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            # Clean minimal
            txt = txt.replace("\x00", "").strip()
            if txt:
                out.append((i + 1, txt))
    except Exception as e:
        print(f"[WARN] Failed to read {pdf_path}: {e}", file=sys.stderr)
    return out

def chunk_text(doc_id: str, page_num: int, text: str, chunk_size: int, overlap: int) -> List[Dict]:
    """Split long page text into overlapping chunks with metadata."""
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        meta = {
            "doc_id": doc_id,
            "page": page_num,
            "char_start": start,
            "char_end": end
        }
        chunks.append({"text": chunk, "meta": meta})
        if end == n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def hash_dir(path: str) -> str:
    """Stable hash for a directory path (used to separate indexes)."""
    return hashlib.sha256(path.encode("utf-8")).hexdigest()[:10]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_pickle(obj, p: Path):
    with open(p, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)

# --------------------------
# Vector Index
# --------------------------
class VectorIndex:
    def __init__(self, embed_model: SentenceTransformer, dim: int, index_path: Path):
        self.embed_model = embed_model
        self.dim = dim
        self.index_path = index_path
        self.faiss_idx = None
        self.texts: List[str] = []
        self.metas: List[Dict] = []

    def build(self, chunks: List[Dict]):
        self.texts = [c["text"] for c in chunks]
        self.metas = [c["meta"] for c in chunks]
        # Compute embeddings in batches
        embs = self.embed_model.encode(self.texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        self.faiss_idx = faiss.IndexFlatIP(self.dim)
        self.faiss_idx.add(embs)
        self._persist()

    def _persist(self):
        ensure_dir(self.index_path)
        faiss.write_index(self.faiss_idx, str(self.index_path / "vectors.faiss"))
        save_pickle(self.texts, self.index_path / "texts.pkl")
        save_pickle(self.metas, self.index_path / "metas.pkl")

    def load(self):
        self.faiss_idx = faiss.read_index(str(self.index_path / "vectors.faiss"))
        self.texts = load_pickle(self.index_path / "texts.pkl")
        self.metas = load_pickle(self.index_path / "metas.pkl")

    def is_built(self) -> bool:
        return (self.index_path / "vectors.faiss").exists() and (self.index_path / "texts.pkl").exists() and (self.index_path / "metas.pkl").exists()

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        q_emb = self.embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.faiss_idx.search(q_emb, top_k)
        hits = []
        for idx, score in zip(I[0], D[0]):
            if idx < 0 or idx >= len(self.texts):
                continue
            hits.append({
                "text": self.texts[idx],
                "meta": self.metas[idx],
                "score": float(score)
            })
        return hits

# --------------------------
# Generator (answer composer)
# --------------------------
class Generator:
    def __init__(self, model_name=GEN_MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def answer(self, question: str, contexts: List[Dict]) -> str:
        # Build RAG-style prompt (keep within max length)
        context_blocks = []
        total = 0
        for c in contexts:
            block = f"[Doc: {c['meta']['doc_id']} p.{c['meta']['page']}] {c['text'].strip()}"
            if total + len(block) > MAX_INPUT_CHARS:
                break
            context_blocks.append(block)
            total += len(block)

        context_str = "\n\n".join(context_blocks) if context_blocks else "No context provided."
        prompt = (
            "You are a helpful assistant answering ONLY from the provided library excerpts. "
            "If the answer is not in the excerpts, say you don't know.\n\n"
            f"QUESTION:\n{question}\n\n"
            "EXCERPTS:\n"
            f"{context_str}\n\n"
            "Answer concisely in a few sentences. If relevant, cite doc_id and page numbers in parentheses."
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.2,
            do_sample=False
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# --------------------------
# Ingestion pipeline
# --------------------------
def ingest_folder(data_dir: str, embed_model: SentenceTransformer, index_dir_base: str = INDEX_DIR) -> Tuple[VectorIndex, int]:
    data_dir = os.path.abspath(data_dir)
    dir_hash = hash_dir(data_dir)
    idx_path = Path(index_dir_base) / f"idx_{dir_hash}"
    vi = VectorIndex(embed_model, dim=embed_model.get_sentence_embedding_dimension(), index_path=idx_path)

    if vi.is_built():
        vi.load()
        return vi, len(vi.texts)

    # Walk and read PDFs
    all_chunks = []
    pdf_count = 0
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdf_count += 1
                pdf_path = Path(root) / f
                doc_id = str(pdf_path.relative_to(data_dir))
                pages = read_pdf_text(pdf_path)
                for page_num, text in pages:
                    chunks = chunk_text(doc_id, page_num, text, CHUNK_SIZE, CHUNK_OVERLAP)
                    all_chunks.extend(chunks)

    if not all_chunks:
        raise RuntimeError("No extractable text found in PDFs. (Scanned PDFs may need OCR.)")

    vi.build(all_chunks)
    return vi, len(all_chunks)

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Music Library QA", page_icon="🎵", layout="wide")

st.title("🎵 Music Library Q&A (Local PDFs)")
st.caption("Ask questions and get answers grounded in your own music theory books, DAW manuals, and sound design PDFs.")

with st.sidebar:
    st.header("Library & Index")
    data_dir = st.text_input(
        "Folder path containing your PDFs",
        value=str(Path.home()),
        help="Example: C:/Users/you/MusicLibrary or /Users/you/Documents/MusicBooks"
    )

    rebuild = st.button("(Re)build index", type="primary")
    st.markdown("---")
    st.subheader("Status")
    status_box = st.empty()
    st.markdown(
        """
        **Notes**
        - Text-only extraction: scans/images won't index without OCR.
        - First build can take a while on large libraries.
        - Index is cached per folder so you don't rebuild every time.
        """
    )

# Lazy init models
@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)

@st.cache_resource(show_spinner=False)
def get_generator():
    return Generator(GEN_MODEL_NAME)

@st.cache_resource(show_spinner=True)
def build_index_cached(data_dir: str):
    embedder = get_embedder()
    return ingest_folder(data_dir, embedder)

# Build or load index
vector_index = None
num_chunks = 0
if data_dir:
    try:
        if rebuild:
            # force rebuild by clearing cache for this function call
            build_index_cached.clear()
        with st.spinner("Building/Loading index..."):
            vector_index, num_chunks = build_index_cached(data_dir)
        status_box.success(f"Index ready. Chunks: {num_chunks:,}")
    except Exception as e:
        status_box.error(f"Index error: {e}")

# Main UI: Question box and Answer box side-by-side
col_q, col_a = st.columns([1, 1])

with col_q:
    st.subheader("Ask a question")
    question = st.text_area("Your question about the library", height=160, placeholder="e.g., What is voice leading? How do I set up sidechain compression in Ableton Live?")

    ask_btn = st.button("Get Answer", type="primary", use_container_width=True, disabled=vector_index is None or not question.strip())

with col_a:
    st.subheader("Answer")
    answer_box = st.empty()
    sources_box = st.empty()

if ask_btn and vector_index is not None and question.strip():
    try:
        with st.spinner("Searching your library..."):
            hits = vector_index.search(question, top_k=TOP_K)

        # Show top sources
        if hits:
            with st.expander("Show retrieved excerpts (sources)"):
                for i, h in enumerate(hits, 1):
                    meta = h["meta"]
                    st.markdown(f"**{i}. {meta['doc_id']} — p.{meta['page']} (score {h['score']:.3f})**")
                    st.code(h["text"][:1200] + ("..." if len(h["text"]) > 1200 else ""))

        # Generate answer
        gen = get_generator()
        with st.spinner("Composing answer from sources..."):
            ans = gen.answer(question, hits)

        answer_box.markdown(ans if ans.strip() else "_No answer generated._")

    except Exception as e:
        answer_box.error(f"Error: {e}")

# Footer
st.markdown("---")
st.caption("Tip: If many pages are scans, consider adding OCR later. I can extend this file to run Tesseract for image-only pages.")
