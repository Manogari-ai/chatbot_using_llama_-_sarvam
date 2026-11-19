"""
config.py — Central configuration for PDF Extract Chatbot (fast + accurate, multilingual)
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directories
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
CHUNKS_DIR = os.path.join(BASE_DIR, "chunks")
INDEX_DIR  = os.path.join(BASE_DIR, "index")
LOG_DIR    = os.path.join(BASE_DIR, "logs")
IMAGES_DIR = os.path.join(BASE_DIR, "images")

for d in [UPLOAD_DIR, CHUNKS_DIR, INDEX_DIR, LOG_DIR, IMAGES_DIR]:
    os.makedirs(d, exist_ok=True)

# Files
FAISS_INDEX_PATH  = os.path.join(INDEX_DIR, "faiss.index")
CHUNKS_JSON_PATH  = os.path.join(CHUNKS_DIR, "chunks.json")
CHAT_HISTORY_PATH = os.path.join(LOG_DIR, "chat_history.txt")

# ─── Device ────────────────────────────────────────────────────────────────
try:
    import torch
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
except ImportError:
    DEVICE = "cpu"

# Chunking
CHUNK_SIZE    = 700
CHUNK_OVERLAP = 100
TABLE_ROWS_PER_CHUNK = 12

# ─── Models (multilingual) ──────────────────────────────────────────────────
# After changing EMBED_MODEL run:  python ingest.py --rebuild
EMBED_MODEL        = "BAAI/bge-m3"
EMBED_DIM          = 1024
EMBED_MAX_SEQ      = 512          # max tokens per chunk when embedding (Tamil needs more tokens than English)
EMBED_QUERY_PROMPT = ""           # bge-m3 needs no instruction prefix

RERANKER_MODEL    = "BAAI/bge-reranker-v2-m3"
RERANK_MAX_LENGTH = 512
USE_RERANKER      = False         # False = fast (bge-m3 cosine score decides). True = slower but more precise on CPU.

# Score thresholds. With USE_RERANKER=False these are bge-m3 cosine scores.
MIN_RELEVANCE_SCORE     = 0.30    # below this -> "not found"
DIRECT_ANSWER_MIN_SCORE = 0.55    # a clearly-best FAQ / directory entry is answered verbatim (no LLM)
DIRECT_ANSWER_GAP       = 0.03

# Retrieval: hybrid (dense + BM25) -> fuse -> rerank short list -> top K
TOP_K_RETRIEVE    = 20
RERANK_CANDIDATES = 8
TOP_K_RERANK      = 4

# LLM (Ollama)
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_URL   = "http://localhost:11434"
OLLAMA_NUM_CTX          = 3584
OLLAMA_NUM_PREDICT      = 350
OLLAMA_NUM_PREDICT_LONG = 700
OLLAMA_TEMPERATURE = 0.0
OLLAMA_KEEP_ALIVE  = -1
WARMUP_OLLAMA      = True         # False: don't pre-load the Ollama model at start-up (e.g. you only use Sarvam)

# Document-level meta questions
DOC_META_NUM_CTX     = 4096
DOC_META_NUM_PREDICT = 700

# Flask
ALLOWED_EXTENSIONS = {"pdf"}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024

# OCR (only used for scanned / image PDFs)
OCR_LANG    = "en"
OCR_USE_GPU = False
OCR_SCALE   = 1.5
OCR_WORKERS = 4
MIN_IMAGE_OCR_AREA = 40000
MAX_IMAGE_OCR      = 30
VISION_MODEL       = None      # e.g. "moondream" (ollama pull moondream) to caption text-less images

# Ingestion gates
ENABLE_TABLE_EXTRACTION = True
EXTRACT_EMBEDDED_IMAGES = True

# Fast path (structured directory lookup)
FASTPATH_THRESHOLD = 88
FASTPATH_MAX_WORDS = 12
LLM_ON_FAST_PATH   = True        # False: instant lookups (directory / FAQ) answer on their own, verbatim + fast.
ENABLE_AGENT       = True        # True : send the instant card first, THEN let the LLM add its own text under it.

# FAQ lookup: how closely a question must match a question that the PDF already contains
QA_MIN_MATCH = 0.60            # word-overlap (Jaccard) of the meaningful words
QA_MIN_GAP   = 0.15            # best match must beat the runner-up by this much

# Caches
ANSWER_CACHE_SIZE = 100
QUERY_CACHE_SIZE  = 500

OLLAMA_TRANSLATE_TO_ENGLISH = True

QA_DIRECT_MIN_COVER = 0.6

# To wipe the index:  delete index/faiss.index and chunks/chunks.json

# ─── Sarvam AI ──────────────────────────────────────────────────────────────
# Keep the key OUT of the code:  setx SARVAM_API_KEY "your-key"   (then open a new terminal)
SARVAM_API_KEY          = os.environ.get("SARVAM_API_KEY", "sk_y7if6q0z_gPAwGEGIqyLGsjsM0Noml6An")
SARVAM_BASE_URL         = "https://api.sarvam.ai"
SARVAM_CHAT_MODEL       = "sarvam-105b-conversations"
SARVAM_TRANSLATE_FALLBACK_MODEL = "sarvam-translate:v1"
SARVAM_FALLBACK_MODEL   = "sarvam-105b"                 # tried if the first model fails
SARVAM_MAX_TOKENS       = 4000                          # 105b models may think before answering
SARVAM_REASONING_EFFORT = None
SARVAM_TRANSLATE_ANSWERS = True      # True: answer in English with Sarvam LLM, then translate
                                     # False: ask the LLM to write directly in the chosen language
                                     # (answers from a document already in the user's language are never re-translated)
SARVAM_STT_MODEL        = "saarika:v2.5"
SARVAM_TRANSLATE_MODEL  = "mayura:v1"
SARVAM_TTS_MODEL        = "bulbul:v3"
SARVAM_TTS_SPEAKER      = "shubh"
TTS_MAX_CHARS           = 1500

LANGUAGES = {
    "en-IN": "English", "ta-IN": "Tamil", "te-IN": "Telugu", "hi-IN": "Hindi",
    "kn-IN": "Kannada", "ml-IN": "Malayalam", "bn-IN": "Bengali", "mr-IN": "Marathi",
    "gu-IN": "Gujarati", "pa-IN": "Punjabi", "od-IN": "Odia",
}

# Only needed for a NEW language with a script that sarvam.py does not know yet, e.g. {"ur": "ARABIC"}
LANG_SCRIPT = {}









