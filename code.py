"""
Streamlit UI – PDF Knowledge Bot
==================================
Clean split-screen:  LEFT = Chat  |  RIGHT = PDF Viewer

Run:  streamlit run app.py
"""

# ── Fix: PyTorch + Streamlit file-watcher clash ──────────
import torch
torch.classes.__path__ = []

import streamlit as st
import os
import time
import base64
from pathlib import Path

import fitz  # PyMuPDF

import config
from chat_orchestrator import ChatOrchestrator

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Knowledge Bot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# Professional CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ─────────────────────────────── */
    .block-container {padding-top:1.5rem;padding-bottom:1rem}
    [data-testid="stSidebar"] {background:#f7f8fa}

    /* ── Header ─────────────────────────────── */
    .app-header {
        display:flex;align-items:center;gap:.6rem;
        padding:.6rem 0 .4rem;border-bottom:2px solid #e3e7ee;margin-bottom:1rem;
    }
    .app-header h1 {font-size:1.5rem;margin:0;font-weight:700;color:#1a1a2e}
    .app-header .tag {
        font-size:.7rem;padding:2px 8px;border-radius:10px;
        background:#e8f0fe;color:#1967d2;font-weight:600;
    }

    /* ── Status pill (top-right) ────────────── */
    .status-pill {
        display:inline-flex;align-items:center;gap:4px;
        font-size:.75rem;padding:3px 10px;border-radius:12px;font-weight:600;
    }
    .status-ready {background:#e6f4ea;color:#1e7e34}
    .status-idle  {background:#fce8e6;color:#c5221f}

    /* ── Answer card ────────────────────────── */
    .answer-card {
        background:#fff;padding:1.2rem 1.4rem;border-radius:10px;
        border:1px solid #e3e7ee;margin:.6rem 0;
        box-shadow:0 1px 3px rgba(0,0,0,.06);
    }

    /* ── Confidence badge ───────────────────── */
    .badge {
        display:inline-block;font-size:.7rem;padding:2px 8px;
        border-radius:10px;font-weight:600;
    }
    .badge-high   {background:#e6f4ea;color:#1e7e34}
    .badge-medium {background:#fef7e0;color:#e37400}
    .badge-low    {background:#fce8e6;color:#c5221f}

    /* ── Source chip ─────────────────────────── */
    .src-chip {
        display:inline-flex;align-items:center;gap:4px;
        font-size:.78rem;padding:3px 10px;border-radius:6px;
        background:#f0f2f6;margin:2px 0;
    }

    /* ── PDF viewer frame ───────────────────── */
    .pdf-frame {
        border:1px solid #e3e7ee;border-radius:10px;padding:8px;
        background:#fafbfc;max-height:82vh;overflow-y:auto;
    }

    /* ── Chat Q bubble ──────────────────────── */
    .q-bubble {
        background:#e8f0fe;padding:.6rem 1rem;border-radius:14px 14px 14px 2px;
        margin:.4rem 0;font-size:.92rem;color:#1a1a2e;
    }

    /* ── Hide Streamlit branding ────────────── */
    #MainMenu, footer, header {visibility:hidden}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────
_defaults = {
    "bot": None,
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
#  Helpers
# ══════════════════════════════════════════════
def _init_bot():
    api_key = config.LLM_API_KEY if config.LLM_API_KEY != "your-api-key-here" else None
    api_key = api_key or st.session_state.get("_api_key")
    if not api_key:
        st.error("Set **LLM_API_KEY** in `config.py` to continue.")
        return None
    try:
        return ChatOrchestrator(llm_api_key=api_key, llm_base_url=config.LLM_BASE_URL)
    except Exception as e:
        st.error(f"Initialization failed: {e}")
        return None


def _render_page(pdf_path: str, page_num: int) -> str | None:
    try:
        doc = fitz.open(pdf_path)
        if page_num < 1 or page_num > len(doc):
            return None
        pix = doc[page_num - 1].get_pixmap(matrix=fitz.Matrix(2, 2))
        b64 = base64.b64encode(pix.tobytes("png")).decode()
        doc.close()
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None


def _pdf_viewer(pdf_path: str | None, page_num: int):
    if not pdf_path or not os.path.exists(pdf_path):
        st.markdown(
            '<div class="pdf-frame" style="display:flex;align-items:center;'
            'justify-content:center;height:60vh;color:#999">'
            '<div style="text-align:center">'
            '<div style="font-size:3rem">📄</div>'
            '<p style="margin-top:.5rem">Click a <b>page number</b> in the chat '
            'to preview the source document here.</p></div></div>',
            unsafe_allow_html=True,
        )
        return

    try:
        doc = fitz.open(pdf_path)
        total = len(doc)
        doc.close()
    except Exception as e:
        st.error(f"Cannot open PDF: {e}")
        return

    # Navigation bar
    nav1, nav2, nav3 = st.columns([1, 3, 1])
    with nav1:
        if st.button("◀", disabled=(page_num <= 1), key="pv_prev", use_container_width=True):
            st.session_state.current_page = max(1, page_num - 1)
            st.rerun()
    with nav2:
        st.markdown(
            f'<div style="text-align:center;font-size:.85rem;padding:6px 0">'
            f'<b>{os.path.basename(pdf_path)}</b> &nbsp;·&nbsp; '
            f'Page {page_num} / {total}</div>',
            unsafe_allow_html=True,
        )
    with nav3:
        if st.button("▶", disabled=(page_num >= total), key="pv_next", use_container_width=True):
            st.session_state.current_page = min(total, page_num + 1)
            st.rerun()

    img = _render_page(pdf_path, page_num)
    if img:
        st.markdown(
            f'<div class="pdf-frame"><img src="{img}" style="width:100%;height:auto"></div>',
            unsafe_allow_html=True,
        )
    else:
        st.error("Failed to render page")


def _process_pdfs(pdf_files, use_kg: bool, use_ocr: bool):
    pdf_dir = Path(config.UPLOADED_PDFS_DIR)
    pdf_dir.mkdir(exist_ok=True)
    paths: list[str] = []
    for f in pdf_files:
        p = pdf_dir / f.name
        p.write_bytes(f.getbuffer())
        paths.append(str(p))
        st.session_state.uploaded_pdf_paths[f.name] = str(p)
    st.session_state.bot.pdf_processor.use_ocr = use_ocr
    t0 = time.time()
    n = st.session_state.bot.process_pdfs(paths, use_kg=use_kg)
    elapsed = time.time() - t0
    st.session_state.bot.save_indices()
    return n, elapsed


def _confidence_badge(val: float) -> str:
    if val >= 0.7:
        cls, label = "badge-high", "High"
    elif val >= 0.4:
        cls, label = "badge-medium", "Medium"
    else:
        cls, label = "badge-low", "Low"
    return f'<span class="badge {cls}">{label} · {val:.0%}</span>'


# ══════════════════════════════════════════════
#  Minimal sidebar — settings only
# ══════════════════════════════════════════════
def _sidebar():
    with st.sidebar:
        st.markdown("#### ⚙️ Settings")

        # API key prompt — only when not configured
        if config.LLM_API_KEY == "your-api-key-here":
            st.text_input("API Key", type="password", key="_api_key",
                          placeholder="Paste your LLM API key")

        # Search tuning
        use_hybrid = st.toggle("Hybrid search", value=True)
        use_kg = st.toggle("Knowledge graph", value=True)
        top_k = st.slider("Sources to retrieve", 1, 10, config.DEFAULT_TOP_K)

        # Compact status
        if st.session_state.indexed and st.session_state.bot:
            chunks = len(st.session_state.bot.retriever.chunks)
            pdfs = len(st.session_state.bot.processed_pdfs)
            st.caption(f"📊 {chunks:,} chunks · {pdfs} PDFs indexed")

    return use_hybrid, use_kg, top_k


# ══════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════
def main():
    use_hybrid, use_kg, top_k = _sidebar()

    # ── Header row ───────────────────────────
    hdr_l, hdr_r = st.columns([4, 1])
    with hdr_l:
        st.markdown(
            '<div class="app-header">'
            '<h1>📚 PDF Knowledge Bot</h1>'
            f'<span class="tag">{config.LLM_MODEL}</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    with hdr_r:
        if st.session_state.bot and st.session_state.indexed:
            st.markdown('<span class="status-pill status-ready">● Ready</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-pill status-idle">● Not ready</span>',
                        unsafe_allow_html=True)

    # ── Auto-initialize bot on first load ────
    if st.session_state.bot is None:
        bot = _init_bot()
        if bot:
            st.session_state.bot = bot
            # Auto-load saved indices if available
            idx_file = os.path.join(config.INDICES_DIR, "faiss.index")
            if os.path.exists(idx_file):
                st.session_state.bot.load_indices()
                st.session_state.indexed = True
            st.rerun()
        else:
            return

    # ── Tabs ─────────────────────────────────
    tab_upload, tab_qa = st.tabs(["📄 Upload & Process", "💬 Ask Questions"])

    # ── Upload tab ───────────────────────────
    with tab_upload:
        files = st.file_uploader(
            "Drop PDF files here",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if files:
            cols = st.columns(min(len(files), 4))
            for i, f in enumerate(files):
                with cols[i % len(cols)]:
                    st.markdown(
                        f'<div style="background:#f7f8fa;padding:.5rem .8rem;'
                        f'border-radius:8px;font-size:.82rem;margin:3px 0">'
                        f'📄 **{f.name}**<br>'
                        f'<span style="color:#666">{f.size/1024/1024:.1f} MB</span></div>',
                        unsafe_allow_html=True,
                    )

            opt1, opt2, opt3 = st.columns([1, 1, 2])
            with opt1:
                enable_ocr = st.checkbox("OCR", True, help="Enable for scanned pages")
            with opt2:
                enable_kg = st.checkbox("Knowledge Graph", True)
            with opt3:
                if st.button("🚀 Process PDFs", type="primary", use_container_width=True):
                    with st.spinner(f"Processing {len(files)} PDF(s)…"):
                        try:
                            n, t = _process_pdfs(files, enable_kg, enable_ocr)
                            st.session_state.indexed = True
                            st.success(f"✅ **{n:,}** chunks indexed in **{t:.1f}s**")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
                            import traceback
                            st.code(traceback.format_exc())
        else:
            st.info("Drag and drop PDF files above to get started.")

    # ── Q&A tab ──────────────────────────────
    with tab_qa:
        if not st.session_state.indexed:
            st.info("Upload and process PDFs first, or place existing indices in `./indices/`.")
            return

        col_chat, col_pdf = st.columns([1, 1], gap="medium")

        # ── LEFT: Chat ───────────────────────
        with col_chat:
            question = st.chat_input("Ask a question about your documents…")

            if question:
                with st.spinner("Thinking…"):
                    try:
                        result = st.session_state.bot.ask(
                            question, top_k=top_k,
                            use_hybrid=use_hybrid, use_kg=use_kg,
                        )
                        if result.get("pdf_path") and result.get("page_number"):
                            st.session_state.current_pdf_path = result["pdf_path"]
                            st.session_state.current_page = result["page_number"]

                        st.session_state.chat_history.append({
                            "question": question,
                            "answer": result["answer"],
                            "sources": result["sources"],
                            "images": result.get("images", []),
                            "confidence": result["confidence"],
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

            # Render history (latest first)
            if not st.session_state.chat_history:
                st.markdown(
                    '<div style="text-align:center;color:#999;margin-top:3rem">'
                    '<div style="font-size:2.5rem">💬</div>'
                    '<p>Ask a question to get started</p></div>',
                    unsafe_allow_html=True,
                )

            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                # Question bubble
                st.markdown(f'<div class="q-bubble">{chat["question"]}</div>',
                            unsafe_allow_html=True)

                # Answer card
                st.markdown(
                    f'<div class="answer-card">'
                    f'{chat["answer"]}<br><br>'
                    f'{_confidence_badge(chat["confidence"])}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Reference images (compact grid)
                images = chat.get("images", [])
                if images:
                    img_cols = st.columns(min(len(images), 3))
                    for idx, img_info in enumerate(images):
                        with img_cols[idx % min(len(images), 3)]:
                            try:
                                st.image(
                                    img_info["path"],
                                    caption=f'p.{img_info["page"]}',
                                    use_container_width=True,
                                )
                            except Exception:
                                pass

                # Source chips with page buttons
                src_cols = st.columns(min(len(chat["sources"]), 3))
                for j, src in enumerate(chat["sources"]):
                    with src_cols[j % min(len(chat["sources"]), 3)]:
                        ppath = src.get("pdf_path")
                        pnum = src["page"]
                        label = f"📄 {src['source']}  ·  p.{pnum}"
                        if ppath and os.path.exists(ppath):
                            if st.button(label, key=f"s_{i}_{j}",
                                         use_container_width=True):
                                st.session_state.current_pdf_path = ppath
                                st.session_state.current_page = pnum
                                st.rerun()

                st.markdown("---")

            # Clear button at the bottom
            if st.session_state.chat_history:
                if st.button("🗑️ Clear conversation", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()

        # ── RIGHT: PDF Viewer ────────────────
        with col_pdf:
            _pdf_viewer(
                st.session_state.current_pdf_path,
                st.session_state.current_page,
            )


if __name__ == "__main__":
    main()
