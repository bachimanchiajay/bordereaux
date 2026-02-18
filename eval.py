"""
Streamlit UI for PDF Knowledge Bot
====================================
Simple 2-tab layout:  📄 Upload & Process  |  💬 Chat

PERFORMANCE:  The UI calls each backend service DIRECTLY —
  • PDFs → text_extraction_api (8001) — no middleman
  • Chunks → rag_api (8002) — no middleman
  • Questions → chat_api (8003) — lightweight call

This avoids the double-hop / double-serialisation that made the
old UI slow (UI → chat_api → text_extraction → back → rag → back).

Run:  streamlit run app.py --server.port 8501
Prerequisites:  All 3 API services must be running.
"""

import streamlit as st
import os
import pickle
import time
import base64
import requests
from typing import Dict, List, Optional
from pathlib import Path

import fitz  # PyMuPDF – for PDF rendering in the viewer

import config

# ──────────────────────────────────────────────
# Service URLs
# ──────────────────────────────────────────────
TEXT_EXTRACTION_URL = os.environ.get(
    "TEXT_EXTRACTION_URL",
    getattr(config, "TEXT_EXTRACTION_URL", "http://localhost:8001"),
)
RAG_SERVICE_URL = os.environ.get(
    "RAG_SERVICE_URL",
    getattr(config, "RAG_SERVICE_URL", "http://localhost:8002"),
)
CHAT_API_URL = os.environ.get(
    "CHAT_API_URL",
    getattr(config, "CHAT_API_URL", "http://localhost:8003"),
)

# Reusable session for HTTP connection pooling (keeps TCP connections alive)
_http = requests.Session()
_http.headers.update({"Connection": "keep-alive"})

REQUEST_TIMEOUT = 1800  # generous for large PDFs on CPU


# ──────────────────────────────────────────────
# Page config & CSS
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Knowledge Bot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {font-size:2.2rem;color:#1E88E5;text-align:center;margin-bottom:.5rem}
    .source-box {background:#f0f2f6;padding:.8rem;border-radius:.5rem;margin:.3rem 0}
    .confidence-high {color:#4CAF50;font-weight:bold}
    .confidence-medium {color:#FF9800;font-weight:bold}
    .confidence-low {color:#F44336;font-weight:bold}
    .pdf-viewer-container {border:2px solid #e0e0e0;border-radius:8px;padding:10px;
        background:#fafafa;max-height:80vh;overflow-y:auto}
    .answer-container {background:#f8f9fa;padding:1.2rem;border-radius:8px;
        border-left:4px solid #1E88E5;margin:.8rem 0}
    .svc-ok  {color:#4CAF50;font-weight:bold}
    .svc-down{color:#F44336;font-weight:bold}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Session state defaults
# ──────────────────────────────────────────────
_defaults = {
    "indexed": False,
    "chat_history": [],
    "current_pdf_path": None,
    "current_page": 1,
    "uploaded_pdf_paths": {},
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════
#  Service helpers  (DIRECT calls – no middleman)
# ══════════════════════════════════════════════

def _svc_ok(url: str) -> bool:
    try:
        r = _http.get(f"{url}/health", timeout=5)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except Exception:
        return False


def check_all_services() -> Dict:
    result = {}
    for name, url in [
        ("Text Extraction (8001)", TEXT_EXTRACTION_URL),
        ("RAG Service (8002)", RAG_SERVICE_URL),
        ("Chat Orchestrator (8003)", CHAT_API_URL),
    ]:
        try:
            r = _http.get(f"{url}/health", timeout=5)
            result[name] = r.json() if r.status_code == 200 else {"status": "error"}
        except Exception:
            result[name] = {"status": "unreachable"}
    return result


# ── Step 1: Extract (calls 8001 DIRECTLY) ────
def extract_single_pdf(uploaded_file, use_ocr: bool, progress_text) -> List[Dict]:
    """Send one PDF directly to text_extraction_api → get chunks back."""
    progress_text.write(f"  ⏳ Extracting **{uploaded_file.name}** …")
    file_bytes = uploaded_file.getbuffer()

    resp = _http.post(
        f"{TEXT_EXTRACTION_URL}/extract_and_chunk",
        files={"file": (uploaded_file.name, file_bytes, "application/pdf")},
        data={
            "use_ocr": str(use_ocr).lower(),
            "chunk_size": str(config.CHUNK_SIZE),
            "overlap": str(config.CHUNK_OVERLAP),
        },
        timeout=REQUEST_TIMEOUT,
    )
    if resp.status_code != 200:
        raise Exception(f"Extraction failed for {uploaded_file.name}: {resp.text}")

    result = resp.json()
    st.session_state.uploaded_pdf_paths[uploaded_file.name] = result.get("pdf_path", "")
    progress_text.write(
        f"  ✅ **{uploaded_file.name}**: {result['num_pages']} pages → "
        f"{result['num_chunks']} chunks  ({result['elapsed_seconds']:.1f}s)"
    )
    return result["chunks"]


# ── Step 2: Index (calls 8002 DIRECTLY) ──────
def index_chunks(chunks: List[Dict], use_kg: bool) -> Dict:
    """Send chunks directly to rag_api for FAISS+TF-IDF+KG indexing."""
    resp = _http.post(
        f"{RAG_SERVICE_URL}/index",
        json={"chunks": chunks, "use_kg": use_kg},
        timeout=REQUEST_TIMEOUT,
    )
    if resp.status_code != 200:
        raise Exception(f"Indexing failed: {resp.text}")
    index_result = resp.json()

    # Persist indices to disk
    _http.post(f"{RAG_SERVICE_URL}/save", timeout=120)

    # Save pdf_paths
    os.makedirs(config.INDICES_DIR, exist_ok=True)
    with open(os.path.join(config.INDICES_DIR, "pdf_paths.pkl"), "wb") as pf:
        pickle.dump(dict(st.session_state.uploaded_pdf_paths), pf)

    return index_result


# ── Step 3: Ask (calls 8003 – lightweight) ───
def ask_question(question: str, top_k: int, use_hybrid: bool, use_kg: bool) -> Dict:
    """Send question to chat_api /ask – only does retrieval + LLM (fast)."""
    resp = _http.post(
        f"{CHAT_API_URL}/ask",
        json={
            "question": question,
            "top_k": top_k,
            "use_hybrid": use_hybrid,
            "use_kg": use_kg,
        },
        timeout=REQUEST_TIMEOUT,
    )
    if resp.status_code != 200:
        raise Exception(f"Ask failed: {resp.text}")
    return resp.json()


# ── Load saved indices ───────────────────────
def load_saved_indices():
    """Tell RAG service to load persisted FAISS+TF-IDF+KG from disk."""
    resp = _http.post(f"{RAG_SERVICE_URL}/load", timeout=60)
    if resp.status_code != 200:
        raise Exception(f"Load failed: {resp.text}")
    # Also tell chat_api to load pdf_paths
    try:
        _http.post(f"{CHAT_API_URL}/load_indices", timeout=30)
    except Exception:
        pass  # chat_api may not be running yet, that's fine
    return resp.json()


# ══════════════════════════════════════════════
#  PDF Viewer helpers
# ══════════════════════════════════════════════

def render_pdf_page(pdf_path: str, page_number: int) -> Optional[str]:
    try:
        doc = fitz.open(pdf_path)
        if page_number < 1 or page_number > len(doc):
            return None
        pix = doc[page_number - 1].get_pixmap(matrix=fitz.Matrix(2, 2))
        b64 = base64.b64encode(pix.tobytes("png")).decode()
        doc.close()
        return f"data:image/png;base64,{b64}"
    except Exception as e:
        st.error(f"Error rendering PDF page: {e}")
        return None


def display_pdf_viewer(pdf_path: Optional[str], page_number: int):
    if not pdf_path or not os.path.exists(pdf_path):
        st.info("📄 No PDF selected. Click a source page number in chat to view it here.")
        return
    try:
        doc = fitz.open(pdf_path)
        total = len(doc)
        doc.close()
    except Exception as e:
        st.error(f"Error opening PDF: {e}")
        return

    st.markdown(f"**📄 {os.path.basename(pdf_path)}** — Page **{page_number}** / {total}")

    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if st.button("⬅️ Prev", disabled=(page_number <= 1), key="pdf_prev"):
            st.session_state.current_page = max(1, page_number - 1)
            st.rerun()
    with c2:
        new = st.number_input("Go to", 1, total, page_number, key="page_input")
        if new != page_number:
            st.session_state.current_page = new
            st.rerun()
    with c3:
        if st.button("Next ➡️", disabled=(page_number >= total), key="pdf_next"):
            st.session_state.current_page = min(total, page_number + 1)
            st.rerun()

    img = render_pdf_page(pdf_path, page_number)
    if img:
        st.markdown(
            f'<div class="pdf-viewer-container">'
            f'<img src="{img}" style="width:100%;height:auto;">'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.error("Failed to render PDF page")


def confidence_html(val: float) -> str:
    if val >= 0.7:
        return f'<span class="confidence-high">High ({val:.2%})</span>'
    if val >= 0.4:
        return f'<span class="confidence-medium">Medium ({val:.2%})</span>'
    return f'<span class="confidence-low">Low ({val:.2%})</span>'


# ══════════════════════════════════════════════
#  Main UI
# ══════════════════════════════════════════════
def main():
    st.markdown('<h1 class="main-header">📚 PDF Knowledge Bot</h1>', unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info(f"LLM: `{config.LLM_MODEL}`")
        st.info(f"Embeddings: `{config.EMBEDDING_MODEL}`")

        st.divider()

        # Service health
        st.header("🔌 Services")
        if st.button("🔄 Check Services"):
            svc = check_all_services()
            for name, info in svc.items():
                status = info.get("status", "unknown")
                if status == "ok":
                    st.markdown(f'<span class="svc-ok">✅ {name}</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="svc-down">❌ {name}</span>', unsafe_allow_html=True)

        st.divider()

        # Load saved indices shortcut
        if not st.session_state.indexed:
            if st.button("📂 Load Saved Indices"):
                with st.spinner("Loading indices …"):
                    try:
                        result = load_saved_indices()
                        st.session_state.indexed = True
                        st.success(
                            f"✅ Loaded {result.get('num_chunks', 0)} chunks, "
                            f"{result.get('faiss_vectors', 0)} vectors"
                        )
                    except Exception as e:
                        st.warning(f"No saved indices: {e}")
        else:
            st.success(f"✅ Documents indexed")

        st.divider()

        # Retrieval settings
        st.header("🎛️ Retrieval")
        use_hybrid = st.checkbox("Hybrid Search", True)
        use_kg = st.checkbox("Knowledge Graph", True)
        top_k = st.slider("Results", 1, 10, config.DEFAULT_TOP_K)

    # ═════════════════════════════════════════
    #  TABS
    # ═════════════════════════════════════════
    tab_upload, tab_chat = st.tabs(["📄 Upload & Process", "💬 Chat"])

    # ══════════════════════════════════════════
    #  TAB 1 — Upload & Process
    #  Calls 8001 + 8002 DIRECTLY (no chat_api middleman)
    # ══════════════════════════════════════════
    with tab_upload:
        st.header("Upload & Process PDFs")
        st.caption(
            "PDFs are sent **directly** to Text Extraction (8001) and "
            "chunks are sent **directly** to RAG Service (8002) — no middleman."
        )

        c1, c2 = st.columns([2, 1])
        with c1:
            files = st.file_uploader(
                "Choose PDF files",
                type=["pdf"],
                accept_multiple_files=True,
            )
        with c2:
            st.write("**Options:**")
            enable_ocr = st.checkbox("Enable OCR (scanned pages)", True)
            enable_kg = st.checkbox("Build Knowledge Graph", True)

        if files:
            st.write(f"**Selected:** {len(files)} PDF(s)")
            for f in files:
                st.write(f"  - {f.name} ({f.size / 1024 / 1024:.2f} MB)")

            if st.button("🚀 Process PDFs", type="primary"):
                # Check required services
                if not _svc_ok(TEXT_EXTRACTION_URL):
                    st.error("❌ Text Extraction (8001) is not running. Start: `python text_extraction_api.py`")
                    return
                if not _svc_ok(RAG_SERVICE_URL):
                    st.error("❌ RAG Service (8002) is not running. Start: `python rag_api.py`")
                    return

                all_chunks = []
                progress = st.empty()
                t0 = time.time()

                # ── Phase 1: Extract each PDF (direct → 8001) ──
                st.subheader("Phase 1: Extracting text …")
                phase1_status = st.container()
                for f in files:
                    try:
                        chunks = extract_single_pdf(f, enable_ocr, phase1_status)
                        all_chunks.extend(chunks)
                    except Exception as e:
                        st.error(f"❌ {f.name}: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                        return

                extraction_time = time.time() - t0
                st.success(
                    f"✅ Extraction done: **{len(all_chunks)}** chunks "
                    f"from **{len(files)}** PDF(s) in **{extraction_time:.1f}s**"
                )

                # ── Phase 2: Index chunks (direct → 8002) ──
                st.subheader("Phase 2: Indexing (embeddings + FAISS + KG) …")
                with st.spinner("Building FAISS index, TF-IDF, and Knowledge Graph …"):
                    t1 = time.time()
                    try:
                        result = index_chunks(all_chunks, enable_kg)
                        index_time = time.time() - t1
                    except Exception as e:
                        st.error(f"❌ Indexing failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                        return

                st.session_state.indexed = True
                total_time = time.time() - t0

                st.success(
                    f"✅ All done!  \n"
                    f"**{result['num_chunks']}** chunks indexed, "
                    f"**{result['faiss_vectors']}** FAISS vectors, "
                    f"**{result.get('kg_nodes', 0)}** KG nodes  \n"
                    f"⏱️ Extraction: {extraction_time:.1f}s | "
                    f"Indexing: {index_time:.1f}s | "
                    f"Total: {total_time:.1f}s"
                )
                st.balloons()

    # ══════════════════════════════════════════
    #  TAB 2 — Chat   (LEFT = Q&A | RIGHT = PDF Viewer)
    # ══════════════════════════════════════════
    with tab_chat:
        if not st.session_state.indexed:
            st.warning(
                "⚠️ No documents indexed yet.  \n"
                "Upload & process PDFs in the first tab, "
                "or click **Load Saved Indices** in the sidebar."
            )
            return

        col_chat, col_pdf = st.columns([1, 1], gap="medium")

        # ─── LEFT: Chat Q&A ─────────────────
        with col_chat:
            st.markdown("### 💬 Ask Questions")

            with st.expander("💡 Example Questions", expanded=False):
                for eq in [
                    "What is the BATCH ID?",
                    "What are the default values for action codes?",
                    "How do I delete FAN dealer group information?",
                    "What does pressing PF4 do?",
                    "What is the routing number field used for?",
                ]:
                    if st.button(eq, key=f"ex_{eq}"):
                        st.session_state.current_question = eq

            question = st.text_area(
                "Your question:",
                value=st.session_state.get("current_question", ""),
                height=80,
                placeholder="Type your question here…",
            )

            b1, b2 = st.columns(2)
            with b1:
                ask_btn = st.button("🔍 Ask", type="primary", use_container_width=True)
            with b2:
                if st.button("🗑️ Clear History", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()

            if ask_btn and question:
                # Check chat service
                if not _svc_ok(CHAT_API_URL):
                    st.error("❌ Chat Orchestrator (8003) is not running. Start: `python chat_api.py`")
                else:
                    with st.spinner("Searching …"):
                        try:
                            t0 = time.time()
                            result = ask_question(question, top_k, use_hybrid, use_kg)
                            elapsed = time.time() - t0

                            if result.get("pdf_path") and result.get("page_number"):
                                st.session_state.current_pdf_path = result["pdf_path"]
                                st.session_state.current_page = result["page_number"]

                            st.session_state.chat_history.append({
                                "question": question,
                                "answer": result["answer"],
                                "sources": result.get("sources", []),
                                "images": result.get("images", []),
                                "confidence": result.get("confidence", 0),
                                "time": round(elapsed, 2),
                            })
                            st.rerun()

                        except Exception as e:
                            st.error(f"Error: {e}")
                            import traceback
                            st.code(traceback.format_exc())

            # ── Chat history ──────────────────
            st.divider()
            st.markdown("### 📜 Chat History")

            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                num = len(st.session_state.chat_history) - i
                with st.container():
                    st.markdown(f"**❓ Question {num}**")
                    st.write(chat["question"])

                    st.markdown('<div class="answer-container">', unsafe_allow_html=True)
                    st.markdown("**💡 Answer:**")
                    st.write(chat["answer"])
                    conf = confidence_html(chat["confidence"])
                    st.markdown(
                        f"**Confidence:** {conf} &nbsp;|&nbsp; ⏱️ {chat.get('time', '?')}s",
                        unsafe_allow_html=True,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Reference images
                    images = chat.get("images", [])
                    if images:
                        st.markdown("**🖼️ Reference Images:**")
                        img_cols = st.columns(min(len(images), 3))
                        for idx, img_info in enumerate(images):
                            col_idx = idx % min(len(images), 3)
                            with img_cols[col_idx]:
                                try:
                                    st.image(
                                        img_info["path"],
                                        caption=f'{img_info["source"]} — Page {img_info["page"]}',
                                        use_container_width=True,
                                    )
                                except Exception:
                                    st.write(f"⚠️ Image not found")

                    # Sources with PDF viewer links
                    st.markdown("**📚 Sources** (click page → view on right):")
                    for j, src in enumerate(chat.get("sources", [])):
                        sc1, sc2 = st.columns([4, 1])
                        with sc1:
                            st.write(f"{src['source']} — Relevance: {src['relevance_score']:.3f}")
                        with sc2:
                            ppath = src.get("pdf_path")
                            pnum = src["page"]
                            if ppath and os.path.exists(ppath):
                                if st.button(f"📄 Page {pnum}", key=f"pg_{i}_{j}"):
                                    st.session_state.current_pdf_path = ppath
                                    st.session_state.current_page = pnum
                                    st.rerun()
                            else:
                                st.write(f"Page {pnum}")
                    st.divider()

        # ─── RIGHT: PDF Viewer ───────────────
        with col_pdf:
            st.markdown("### 📄 PDF Viewer")
            display_pdf_viewer(
                st.session_state.current_pdf_path,
                st.session_state.current_page,
            )


if __name__ == "__main__":
    main()
