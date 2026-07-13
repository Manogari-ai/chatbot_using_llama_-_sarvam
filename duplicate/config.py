"""
config.py — Central configuration for PDF Extract Chatbot
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directories
UPLOAD_DIR   = os.path.join(BASE_DIR, "uploads")
CHUNKS_DIR   = os.path.join(BASE_DIR, "chunks")
INDEX_DIR    = os.path.join(BASE_DIR, "index")
LOG_DIR      = os.path.join(BASE_DIR, "logs")
IMAGES_DIR   = os.path.join(BASE_DIR, "images")

for d in [UPLOAD_DIR, CHUNKS_DIR, INDEX_DIR, LOG_DIR, IMAGES_DIR]:
    os.makedirs(d, exist_ok=True)

# Files
FAISS_INDEX_PATH  = os.path.join(INDEX_DIR, "faiss.index")
CHUNKS_JSON_PATH  = os.path.join(CHUNKS_DIR, "chunks.json")
CHAT_HISTORY_PATH = os.path.join(LOG_DIR, "chat_history.txt")

# Chunking
CHUNK_SIZE    = 512          # characters
CHUNK_OVERLAP = 80

# Embedding model
# Swapped from BAAI/bge-m3 (Beijing Academy of AI / China) to mxbai-embed
# -large-v1 (mixedbread ai / Germany). Same 1024-dim output, so nothing
# downstream (FAISS index construction, EMBED_DIM) needs to change.
# ⚠️ VECTOR SPACE CHANGE: embeddings from the old and new models are NOT
# comparable. After this swap you MUST delete index/faiss.index and
# chunks/chunks.json (or call ingest.delete_pdf_from_index() per PDF) and
# re-ingest every PDF — otherwise old bge-m3 vectors and new mxbai vectors
# sit in the same FAISS index and cosine similarity between them is
# meaningless, which will silently degrade retrieval quality.
# Loaded via sentence-transformers, not FlagEmbedding (BAAI-specific lib).
EMBED_MODEL   = "mixedbread-ai/mxbai-embed-large-v1"
EMBED_DIM     = 1024

# Re-ranker model
# Swapped from BAAI/bge-reranker-v2-m3 (China) to mxbai-rerank-large-v1
# (mixedbread ai / Germany). Loaded via sentence-transformers' CrossEncoder,
# not FlagEmbedding's FlagReranker — different library, see rag.py.
# mxbai-rerank returns raw logits (not the 0-1 normalized scores
# FlagReranker gave with normalize=True), so rag.py applies a sigmoid
# before comparing against MIN_RELEVANCE_SCORE / _RERANK_SKIP_THRESHOLD.
# Recalibrate those two against your own chat_history.txt after the
# swap — the score distribution won't be identical to bge-reranker's.
RERANKER_MODEL = "mixedbread-ai/mxbai-rerank-large-v1"

# Retrieval
# TOP_K_RERANK=1 was the cause of a prior quality drop ("Coimbatore"
# answer) — the correct chunk was ranked #2, not #1, so at top_k=1 it
# never reached the LLM at all. 3 is the minimum that reliably avoids
# this for directory-style/cross-referenced content. Don't drop below 3.
TOP_K_RETRIEVE = 5          # initial FAISS hits
TOP_K_RERANK   = 3          # after re-ranking — DO NOT reduce, causes missed answers

# LLM
# Swapped from qwen2.5:3b (Alibaba / China) to llama3.2:3b (Meta / US) —
# same 3B parameter class, so speed is comparable and the OLLAMA_NUM_CTX /
# OLLAMA_NUM_PREDICT tuning below (already dialed in for a 3B model on
# this hardware) stays valid without retuning.
# Pull it once before running: `ollama pull llama3.2:3b`
OLLAMA_MODEL = "llama3.2:3b"       # non-Chinese, same size/speed class as qwen2.5:3b
OLLAMA_URL   = "http://localhost:11434"

# 3 chunks (≤512 chars each) + system prompt + question needs headroom.
# 1536 was cutting it close enough to risk silent context truncation on
# longer chunks — bumped back up just enough to be safe without paying
# the full cost of the original 2048+.
OLLAMA_NUM_CTX = 2048            # 1800

# 200 tokens is fine for one-line facts but truncates multi-field answers
# (address + phone + in-charge name in one response). Raised modestly.
OLLAMA_NUM_PREDICT = 450            # 300

# NOTE: this was previously defined but silently unused — rag.py hardcoded
# 0.1 in the Ollama options dict regardless of this value. rag.py now
# reads this directly. 0.0 = fully deterministic, which matters more on a
# small model (3B) than a large one: lower temperature meaningfully
# reduces hallucination/drift on smaller models.
OLLAMA_TEMPERATURE = 0.0
# MUST be an int, not the string "-1" — Ollama's keep_alive expects either
# a number of seconds or a Go duration string with a unit (e.g. "30m").
# A bare "-1" has no unit and fails Go's duration parser with exactly
# 'missing unit in duration "-1"'. -1 (int) means "keep the model loaded
# indefinitely, never unload".
OLLAMA_KEEP_ALIVE  = -1

# ─── Document-level meta-query budget ──────────────────────────────────────
# Used only by rag.handle_doc_intent() for "summarize this document" /
# "main topics" / "what does page N say" answers. These need more context
# than a normal 3-chunk answer (a full page, or a whole-document sample,
# is bigger than 3×512 chars) and a longer output budget (a summary is
# longer than a one-line fact). Kept separate from OLLAMA_NUM_CTX /
# OLLAMA_NUM_PREDICT above so normal Q&A stays fast and doesn't pay this
# larger cost on every query — only the four meta-questions do.
DOC_META_NUM_CTX     = 4096
DOC_META_NUM_PREDICT = 500

# ─── Relevance floor ────────────────────────────────────────────────────────
# If the best chunk's rerank_score after reranking is below this, the
# retrieved context is considered too weak to answer from — rag.answer()
# and rag.answer_stream() return "I couldn't find that information in the
# uploaded PDF." instead of calling the LLM at all. This is what stops a
# smaller model (Llama 3.2 3B) from confidently generating an answer out
# of irrelevant context instead of admitting it doesn't know.
#
# bge-reranker-v2-m3 normalized scores for a genuinely relevant top chunk
# are typically well above 0.5; below ~0.3 the match is usually noise.
# Start at 0.35 and tune against your own logs/chat_history.txt — if
# legitimate answers start getting rejected, lower it; if garbage answers
# still slip through, raise it.
MIN_RELEVANCE_SCORE = 0.35

# Flask
ALLOWED_EXTENSIONS = {"pdf"}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024   # 100 MB

# OCR  (PaddleOCR)
OCR_LANG = "en"
OCR_USE_GPU = False

TABLE_PRECHECK_MIN_ROWS = 2

# ─── Ingestion-time cost gates (used by ingest.py) ─────────────────────────
ENABLE_TABLE_EXTRACTION = True
EXTRACT_EMBEDDED_IMAGES = True
MIN_IMAGE_OCR_AREA = 900

# ─── Query-time cost gates (used by rag.py) ────────────────────────────────
FASTPATH_THRESHOLD = 85

# ==========================
# Cache Settings
# ==========================
ANSWER_CACHE_SIZE = 100
QUERY_CACHE_SIZE  = 500

# rm -f index/faiss.index chunks/chunks.json             vetor table and chunking removed completely command