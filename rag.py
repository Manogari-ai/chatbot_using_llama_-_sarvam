# rag


import os
import re
import json
import time
import logging
import datetime
import threading
from functools import lru_cache
from typing import List, Dict, Optional, Tuple

import numpy as np

import config
import ingest

logger = logging.getLogger(__name__)

# ─── Lazy singletons ──────────────────────────────────────────────────────────

_embed_model  = None
_reranker     = None
_faiss_index  = None
_chunks: List[Dict] = []
_model_lock   = threading.Lock()

# ─── Answer cache ─────────────────────────────────────────────────────────────
# Plain bounded dict, not a strict LRU — good enough for "don't recompute
# an answer we just gave a minute ago" without adding a dependency. Cleared
# wholesale on load_index() so it can never serve a stale answer after the
# corpus changes. Defined here (before load_index() is ever called) since
# load_index() references it directly on module import.

_answer_cache: Dict[str, Dict] = {}
_answer_cache_lock = threading.Lock()

def _cache_key(query: str) -> str:
    return query.strip().lower()

def get_cached_answer(query: str) -> Optional[Dict]:
    with _answer_cache_lock:
        cached = _answer_cache.get(_cache_key(query))
        return dict(cached) if cached else None

def set_cached_answer(query: str, result: Dict):
    with _answer_cache_lock:
        if len(_answer_cache) >= config.ANSWER_CACHE_SIZE:
            _answer_cache.pop(next(iter(_answer_cache)))   # evict oldest entry
        _answer_cache[_cache_key(query)] = {
            "answer":  result["answer"],
            "sources": result["sources"],
        }

def get_embed_model():
    """
    mxbai-embed-large-v1 loads via sentence-transformers, not FlagEmbedding
    (that library is BAAI-specific — it doesn't know this model's config).
    No query_instruction_for_retrieval kwarg exists on SentenceTransformer,
    so the query prompt is applied by hand in _embed_query() below instead
    of at construction time.
    """
    global _embed_model
    if _embed_model is None:
        with _model_lock:
            if _embed_model is None:
                from sentence_transformers import SentenceTransformer
                _embed_model = SentenceTransformer(config.EMBED_MODEL)
                logger.info(f"Embedding model ready: {config.EMBED_MODEL}")
    return _embed_model

def get_reranker():
    """
    mxbai-rerank-large-v1 loads via sentence-transformers' CrossEncoder,
    not FlagEmbedding's FlagReranker (BAAI-specific). Deliberately v1, not
    v2 — mxbai-rerank-v2 is built on a Qwen (Alibaba) base model
    internally, which would defeat the point of moving off BAAI's
    reranker. v1 is a plain BERT-based cross-encoder.
    """
    global _reranker
    if _reranker is None:
        with _model_lock:
            if _reranker is None:
                from sentence_transformers import CrossEncoder
                model_name = getattr(config, "RERANKER_MODEL",
                                     "mixedbread-ai/mxbai-rerank-large-v1")
                _reranker = CrossEncoder(model_name)
                logger.info(f"Re-ranker ready: {model_name}")
    return _reranker

def load_index():
    """Load (or reload) FAISS index + chunks into module globals."""
    global _faiss_index, _chunks
    if not os.path.exists(config.FAISS_INDEX_PATH):
        logger.warning("FAISS index not found. Ingest PDFs first.")
        _faiss_index = None
        _chunks = []
        return
    import faiss
    _faiss_index = faiss.read_index(config.FAISS_INDEX_PATH)
    with open(config.CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        _chunks = json.load(f)
    # Bust query embedding cache when index changes
    _embed_query.cache_clear()
    # Stale cached answers reference content that may no longer be current
    # (a PDF could have just been deleted/replaced) — clear them too.
    _answer_cache.clear()
    logger.info(f"Index loaded: {_faiss_index.ntotal} vectors, {len(_chunks)} chunks")

# ─── Query embedding with cache ───────────────────────────────────────────────
# Bumped cache size 256 → 512: cheap (a few KB per entry) and raises the hit
# rate for repeated/near-repeated questions, skipping embedding entirely.

# mxbai-embed-large-v1's recommended retrieval prompt — same wording BGE-M3
# used, applied by hand here since SentenceTransformer has no constructor-
# level instruction kwarg (FlagModel did). Document-side embedding
# (ingest.py's embed_texts) intentionally does NOT get this prefix — only
# queries do, same asymmetry as before.
_QUERY_PROMPT = "Represent this sentence for searching relevant passages: "

@lru_cache(maxsize=512)
def _embed_query(query: str) -> bytes:
    """
    Embed a query string and return the vector as raw bytes for caching.
    lru_cache requires hashable args; numpy arrays aren't, so we serialise.
    """
    model = get_embed_model()
    q_raw = model.encode([_QUERY_PROMPT + query], convert_to_numpy=True).astype(np.float32)
    norm  = np.linalg.norm(q_raw, axis=1, keepdims=True)
    q_vec = q_raw / np.where(norm == 0, 1.0, norm)
    return q_vec.tobytes()          # serialise → hashable

def embed_query(query: str) -> np.ndarray:
    return np.frombuffer(_embed_query(query), dtype=np.float32).reshape(1, -1)

# Load on import — must be after _embed_query is defined (cache_clear reference)
load_index()

# ─── Directory entry formatting (fast-path output) ────────────────────────────
# The fast path returns raw chunk text by design (zero LLM cost), but raw
# chunk text still has PyMuPDF artifacts in it: a leading "6. Chennai"
# header, "o" bullet markers PyMuPDF leaves behind, inconsistent spacing.
# This turns that into clean, bold-headed bullets WITHOUT touching the
# underlying data or going anywhere near the LLM — same field values,
# same accuracy, just legible formatting.

# A light emoji per recognised field — purely cosmetic, never affects
# matching/parsing logic, which still lives in ingest._parse_entry_fields.
_FIELD_EMOJI = {
    "in-charge": "👤",
    "officer in charge": "👤",
    "designation": "🧾",
    "contact person": "👤",
    "address": "📍",
    "phone": "📞",
    "telephone": "📞",
    "fax": "📠",
    "email": "✉️",
    "e-mail": "✉️",
    "contact": "📞",
}

# Canonical display label per parsed field key (ingest.py lower-cases and
# normalises labels like "in-charge" / "officer-in-charge" / "e-mail").
_FIELD_DISPLAY = {
    "in-charge": "In-Charge",
    "officer in charge": "Officer-in-Charge",
    "designation": "Designation",
    "contact person": "Contact Person",
    "address": "Address",
    "phone": "Phone",
    "telephone": "Telephone",
    "fax": "Fax",
    "email": "Email",
    "e-mail": "Email",
    "contact": "Contact",
}

def format_directory_entry(chunk: Dict) -> str:
    """
    Re-parses a directory_entry chunk's raw text with the SAME field
    parser used at ingest/query time (ingest._parse_entry_fields), then
    renders it as clean markdown bullets:

        📍 **Address:** ...
        📞 **Phone:** ...
        ✉️ **Email:** ...

    Falls back to the raw chunk text untouched if parsing finds nothing
    (e.g. a non-directory chunk slipped through) — this is a pure
    presentation layer, never a source of fabricated content.
    """
    fields = ingest._parse_entry_fields(chunk.get("text", ""))
    if not fields:
        return chunk.get("text", "").strip()

    location = chunk.get("location_name", "").strip()
    lines = []
    if location:
        lines.append(f"📍 **{location}**")

    for key, value in fields.items():
        if not value:
            continue
        emoji = _FIELD_EMOJI.get(key, "•")
        label = _FIELD_DISPLAY.get(key, key.title())
        lines.append(f"{emoji} **{label}:** {value}")

    # "  \n" = markdown hard line-break (two trailing spaces + newline).
    # A bare "\n" gets collapsed into a single space by most markdown
    # renderers (and plain terminals), which is why fields were showing
    # up run-together on one line instead of stacked.
    return "  \n".join(lines) if lines else chunk.get("text", "").strip()


def format_single_field(field: str, value: str) -> str:
    """
    Formats a single-field fast-path answer (e.g. user only asked for the
    phone number) with the same emoji/bold convention as the full entry,
    so a one-line answer still looks consistent with a full lookup.
    """
    emoji = _FIELD_EMOJI.get(field, "•")
    label = _FIELD_DISPLAY.get(field, field.title())
    return f"{emoji} **{label}:** {value}"

# ─── Fast path: structured directory lookup ──────────────────────────────────
# Reuses ingest.py's query_location() — the SAME fuzzy matcher and field
# parser used everywhere else — so "FRRO Chennai phone number" or "Cochin
# POE in-charge" gets answered directly from the parsed entry, with zero
# embedding/FAISS/reranking/LLM cost. This is the single biggest lever for
# response time on directory-style content (FRRO/POE/etc.), since it skips
# every expensive step entirely instead of just speeding one of them up.
#
# Returns None (falls through to the normal pipeline) whenever:
#   - there are no directory_entry chunks in the index at all, or
#   - the query doesn't fuzzy-match a location with enough confidence
# so non-directory PDFs and genuinely free-text questions are unaffected.

def try_fast_path(query: str) -> Optional[Dict]:
    if not _chunks:
        return None
    try:
        result = ingest.query_location(query, chunks=_chunks,
                                        threshold=config.FASTPATH_THRESHOLD)
    except Exception as e:
        logger.warning(f"Fast path lookup failed, falling back to full RAG: {e}")
        return None
    if not result:
        return None

    if result.get("field"):
        source_chunk = result["chunk"]
        raw_answer   = result["answer"]
        if not raw_answer or not raw_answer.strip():
            return None
        answer_text = format_single_field(result["field"], raw_answer.strip())
    else:
        # A location matched but no specific field was asked for — hand
        # back the whole entry, cleanly formatted, rather than guessing
        # which field they want.
        source_chunk = result["chunks"][0]
        answer_text  = format_directory_entry(source_chunk)
        if not answer_text or not answer_text.strip():
            return None

    return {
        "answer": answer_text,
        "sources": [{
            "pdf":     source_chunk.get("pdf", ""),
            "page":    source_chunk.get("page", ""),
            "source":  source_chunk.get("source", ""),
            "score":   1.0,
            "snippet": source_chunk["text"][:200],
        }],
        "fast_path": True,
    }

# ─── Document-level meta-query routing ───────────────────────────────────────
# "Summarize this document" / "main topics" / "list tables" / "page N" are
# document-level questions. Semantic top-K retrieval against the literal
# query text returns near-random chunks for these (there's no passage in a
# PDF that's semantically "about" the word "summarize"), which is why a
# capable model like Qwen3 could sometimes bluff a plausible answer from
# garbage context, but a small model like Qwen2.5:3B just fails openly
# (echoes the question, or answers something unrelated). This layer
# detects the intent and builds context directly from chunk metadata
# instead of going anywhere near FAISS — more accurate AND faster, since
# it skips embedding + retrieval + reranking entirely for these four
# always-available questions.

_PAGE_RE    = re.compile(r"\bpage\s+(\d+)\b", re.I)
_TABLES_RE  = re.compile(r"\b(list|show|what).{0,15}\btables?\b", re.I)
_TOPICS_RE  = re.compile(r"\b(main|key)\s+topics?\b|\bwhat.{0,15}\b(cover|about)\b", re.I)
_SUMMARY_RE = re.compile(r"\bsummar(y|ize|ise)\b|\btl;?dr\b|\boverview\b", re.I)

def detect_doc_intent(query: str) -> Tuple[Optional[str], Optional[int]]:
    m = _PAGE_RE.search(query)
    if m:
        return "page", int(m.group(1))
    if _TABLES_RE.search(query):
        return "tables", None
    if _TOPICS_RE.search(query):
        return "topics", None
    if _SUMMARY_RE.search(query):
        return "summary", None
    return None, None

def _active_pdf() -> Optional[str]:
    """
    Most recently ingested PDF — used when a doc-level query doesn't name
    a specific file. Generalises to any number of ingested PDFs without
    hardcoding a filename anywhere.
    """
    return _chunks[-1]["pdf"] if _chunks else None

def _page_chunks(pdf: str, page: int) -> List[Dict]:
    return [c for c in _chunks if c.get("pdf") == pdf and c.get("page") == page]

def _table_chunks(pdf: str) -> List[Dict]:
    return [c for c in _chunks if c.get("pdf") == pdf and c.get("source") == "docling_table"]

def _page_sort_key(page):
    """
    Sort key that never blows up on mixed types. Chunk "page" values are
    supposed to always be ints, but this is defensive: if any chunk in
    chunks.json ever has a non-numeric page (e.g. an old index written
    before the ingest.py page_no fix, or any future source that forgets
    to set an int), sorted() would otherwise raise:
        TypeError: '<' not supported between instances of 'str' and 'int'
    Non-numeric values are coerced to a large sentinel int so they sort
    last instead of crashing the whole summary/topics request.
    """
    try:
        return int(page)
    except (TypeError, ValueError):
        return 10**9

def _overview_sample(pdf: str, max_chars: int = 3500) -> str:
    """
    Builds a representative cross-section of the WHOLE document — one
    chunk sampled per page, in page order, truncated to fit the context
    budget — instead of a semantic top-K slice. This is what makes
    "summarize" and "main topics" actually cover the document instead of
    3 arbitrary paragraphs that happened to score highest against the
    literal word "summarize".
    """
    by_page: Dict = {}
    for c in _chunks:
        if c.get("pdf") != pdf:
            continue
        by_page.setdefault(c.get("page", 0), c)   # first chunk seen per page
    pieces, total = [], 0
    for page in sorted(by_page, key=_page_sort_key):
        text = by_page[page]["text"].strip()
        if not text:
            continue
        piece = f"[Page {page}] {text[:400]}"
        if total + len(piece) > max_chars:
            break
        pieces.append(piece)
        total += len(piece)
    return "\n\n".join(pieces)

def handle_doc_intent(query: str) -> Optional[Dict]:
    """
    Returns a complete result dict (same shape as try_fast_path) if this
    query is a document-level meta-query AND we have chunks to answer it
    from. Returns None otherwise, so the caller falls through to normal
    retrieval for everything else.
    """
    intent, arg = detect_doc_intent(query)
    if intent is None:
        return None
    pdf = _active_pdf()
    if pdf is None:
        return None

    if intent == "page":
        chunks = _page_chunks(pdf, arg)
        if not chunks:
            return {
                "answer": f"I couldn't find page {arg} in the uploaded PDF.",
                "sources": [],
                "fast_path": True,
            }
        context = "\n\n".join(c["text"] for c in chunks)
        prompt = (
            f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}"
            f"\n\nQUESTION:\nWhat does this page say? Summarize it clearly.\n\nANSWER:"
        )
        source_chunks = chunks

    elif intent == "tables":
        chunks = _table_chunks(pdf)
        if not chunks:
            return {
                "answer": "I couldn't find any tables in the uploaded PDF.",
                "sources": [],
                "fast_path": True,
            }
        # Tables are already clean markdown from Docling — no LLM call
        # needed at all, so this branch is effectively instant.
        answer_text = "\n\n".join(c["text"] for c in chunks)
        return {
            "answer": answer_text,
            "sources": [{
                "pdf": c.get("pdf", ""), "page": c.get("page", ""),
                "source": c.get("source", ""), "score": 1.0,
                "snippet": c["text"][:200],
            } for c in chunks],
            "fast_path": True,
        }

    else:  # "summary" or "topics"
        context = _overview_sample(pdf)
        if not context:
            return None
        question = (
            "Summarize this document in a few clear paragraphs."
            if intent == "summary" else
            "What are the main topics covered in this document? List them clearly."
        )
        prompt = (
            f"{SYSTEM_PROMPT}\n\nCONTEXT (excerpts spanning the full document):\n{context}"
            f"\n\nQUESTION:\n{question}\n\nANSWER:"
        )
        source_chunks = [c for c in _chunks if c.get("pdf") == pdf][:5]

    # Meta-queries need more room than a normal 3-chunk answer (a full-page
    # or full-document context is bigger than 3×512 chars), so they get a
    # larger context/output budget for this one call only.
    response = call_ollama(
        prompt,
        num_ctx=config.DOC_META_NUM_CTX,
        num_predict=config.DOC_META_NUM_PREDICT,
    )
    return {
        "answer": response,
        "sources": [{
            "pdf": c.get("pdf", ""), "page": c.get("page", ""),
            "source": c.get("source", ""), "score": 1.0,
            "snippet": c["text"][:200],
        } for c in source_chunks],
        "fast_path": True,
    }

# ─── Retrieval ────────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = config.TOP_K_RETRIEVE) -> List[Dict]:
    """Embed query (cached) and fetch top-K chunks from FAISS."""
    if _faiss_index is None or not _chunks:
        return []

    q_vec = embed_query(query)
    scores, indices = _faiss_index.search(q_vec, min(top_k, len(_chunks)))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = dict(_chunks[idx])
        chunk["retrieval_score"] = float(score)
        results.append(chunk)
    return results

# ─── Adaptive re-ranking ─────────────────────────────────────────────────────

# Skip reranker when the top FAISS score is already above this threshold
# (the semantic match is already very strong — reranking adds little)
_RERANK_SKIP_THRESHOLD = float(os.environ.get("RERANK_SKIP_THRESHOLD", "0.88"))

def rerank(query: str, candidates: List[Dict],
           top_k: int = config.TOP_K_RERANK) -> List[Dict]:
    """
    Re-rank candidates with the reranker model.
    Skips reranking entirely when the best retrieval score exceeds the threshold
    (saves 200–800 ms on confident matches).
    """
    if not candidates:
        return []

    best_score = candidates[0].get("retrieval_score", 0.0)
    if best_score >= _RERANK_SKIP_THRESHOLD:
        logger.info(f"Rerank skipped (top score={best_score:.3f} ≥ {_RERANK_SKIP_THRESHOLD})")
        for c in candidates:
            c["rerank_score"] = c["retrieval_score"]
        return candidates[:top_k]

    try:
        reranker   = get_reranker()
        pairs      = [[query, c["text"]] for c in candidates]
        # CrossEncoder.predict() returns raw logits, unlike FlagReranker's
        # compute_score(normalize=True) which gave 0-1 scores directly.
        # Apply sigmoid by hand so rerank_score stays in the same 0-1
        # range MIN_RELEVANCE_SCORE / _RERANK_SKIP_THRESHOLD expect —
        # otherwise those thresholds would silently stop meaning anything.
        raw_scores = reranker.predict(pairs)
        scores     = 1.0 / (1.0 + np.exp(-np.asarray(raw_scores, dtype=np.float64)))
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k]
    except Exception as e:
        logger.warning(f"Re-ranking failed ({e}); using retrieval order.")
        for c in candidates:
            c.setdefault("rerank_score", c.get("retrieval_score", 0.0))
        return candidates[:top_k]

# ─── Prompt Builder ───────────────────────────────────────────────────────────
# Conversational tone, no citation instructions, key terms highlighted in bold,
# light emoji use for scannability, bullet formatting for multi-field answers
# (mirrors the fast-path formatter above so directory-style answers look
# consistent whether they came from the fast path or the LLM). Still accurate
# and professional — friendliness never overrides correctness.
# NOTE: source/page/pdf metadata is intentionally NOT included in the prompt
# text below — the model can only cite what it can see, so keeping that
# metadata out of its context is the most reliable way to stop it from
# mentioning pages/sources, and it also shortens the prompt (faster prefill).

SYSTEM_PROMPT = (
    "You are a friendly, knowledgeable assistant having a natural conversation "
    "with the user about their documents.\n\n"
    "Rules:\n"
    "- Answer using only the information in the CONTEXT below.\n"
    "- If the answer isn't in the context, say so naturally and warmly, e.g. "
    "\"Hmm, I couldn't find that in the documents you've shared 🤔 — want to "
    "try rephrasing, or ask about something else?\"\n"
    "- Write like you're explaining things to a colleague: clear, warm, and "
    "conversational — not robotic, and never mention \"the context\" itself.\n"
    "- Never mention page numbers, source numbers, or which file a fact came "
    "from. Just answer the question directly, as if you simply know it.\n"
    "- Bold the key terms, names, numbers, and important keywords using "
    "**markdown bold** so the most important parts of your answer stand out "
    "at a glance.\n"
    "- Use 1-3 relevant emojis where they naturally fit (e.g. 📍 for "
    "locations/addresses, 📞 for phone numbers, ✉️ for email, 👤 for a "
    "person/role, 📠 for fax, ✅ for confirmations) — never force them in, "
    "and never use more than one emoji per line.\n"
    "- If an answer has MULTIPLE fields or details (e.g. a contact entry "
    "with an in-charge, address, phone, and email), NEVER write it as one "
    "paragraph or a numbered narrative. Instead, put each field on its own "
    "bullet line with a bold label, like:\n"
    "  📍 **Address:** ...\n"
    "  📞 **Phone:** ...\n"
    "  ✉️ **Email:** ...\n"
    "- If the answer involves steps or a process, use a numbered list so "
    "it's easy to follow.\n"
    "- Keep it concise but complete — get to the point without padding, "
    "filler intros, or restating the question.\n"
    "- If the answer has multiple points, items, or steps — even 5 or "
    "more — include ALL of them completely. Never skip, hide, or "
    "shorten any point unless the user explicitly asks for a shorter "
    "version.\n"
    "- Display each point separately and clearly, with proper spacing "
    "and line breaks between them so the answer is easy to scan.\n"
    "- Every point must be meaningful and complete on its own — no "
    "vague or half-finished items.\n"
    "- For tables, format them clearly with markdown tables.\n"
    "- Stay professional and accurate at all times — friendliness, emojis, "
    "and formatting should never come at the cost of correctness. Never "
    "make anything up or add information that isn't in the context."
)

def build_prompt(query: str, context_chunks: List[Dict]) -> str:
    context_text = "\n\n---\n\n".join(c["text"] for c in context_chunks)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT:\n{context_text}"
        f"\n\nQUESTION:\n{query}\n\nANSWER:"
    )

# ─── Citation / page-reference cleanup ───────────────────────────────────────
# Defense-in-depth: the prompt above already tells the model not to cite
# pages/sources, but models occasionally slip up. This strips common
# citation-style artifacts from the final text before it reaches the user.

_CITATION_PATTERNS = [
    re.compile(r"\(?\s*according to\s+page\s+\d+\s*\)?", re.IGNORECASE),
    re.compile(r"\(?\s*as (?:stated|mentioned|noted) (?:in|on)\s+page\s+\d+\s*\)?", re.IGNORECASE),
    re.compile(r"\(?\s*page\s+\d+\s*\)?", re.IGNORECASE),
    re.compile(r"\[?\s*source\s*\d+\s*\]?\s*:?", re.IGNORECASE),
    re.compile(r"\(?\s*p\.\s?\d+\s*\)?", re.IGNORECASE),
]

def strip_citations(text: str) -> str:
    """Remove any stray page/source references the model may have added."""
    cleaned = text
    for pattern in _CITATION_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    # collapse whitespace / stray punctuation left behind by the removals
    cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()

# ─── LLM (Qwen via Ollama) — streaming ───────────────────────────────────────
# keep_alive keeps the model loaded in memory between requests so it doesn't
# have to be reloaded from disk on every query — this is usually the single
# biggest win for perceived response time on a local Ollama setup.
#
# num_ctx / num_predict are optional per-call overrides: normal retrieval
# answers use config.OLLAMA_NUM_CTX / config.OLLAMA_NUM_PREDICT, but
# document-meta answers (summarize / topics / whole-page) legitimately
# need a bigger context and output budget, so handle_doc_intent() passes
# config.DOC_META_NUM_CTX / config.DOC_META_NUM_PREDICT explicitly for
# that one call instead of raising the defaults for every query.

def _resolve_keep_alive():
    """
    os.environ.get() always returns a string even for a non-string
    default, and Ollama's keep_alive rejects a bare numeric string like
    "-1" (needs a unit, e.g. "-1s", or to be an actual int/float) — that
    mismatch is what caused 'missing unit in duration "-1"'. This coerces
    any plain integer-looking string (from the env var or config) back to
    a real int before it ever reaches the ollama client, and only falls
    back to passing the raw string through for genuine duration strings
    like "30m".
    """
    raw = os.environ.get("OLLAMA_KEEP_ALIVE", config.OLLAMA_KEEP_ALIVE)
    if isinstance(raw, (int, float)):
        return raw
    raw = str(raw).strip()
    try:
        return int(raw)
    except ValueError:
        return raw   # e.g. "30m" — a real Go duration string, pass through as-is

_OLLAMA_KEEP_ALIVE = _resolve_keep_alive()

def stream_ollama(prompt: str, num_ctx: Optional[int] = None,
                   num_predict: Optional[int] = None):
    """
    Generator that yields raw text tokens from Ollama as they arrive.
    """
    try:
        import ollama
        for chunk in ollama.generate(
            model=config.OLLAMA_MODEL,
            prompt=prompt,
            stream=True,
            keep_alive=_OLLAMA_KEEP_ALIVE,
            options={
                # Was hardcoded to 0.1 regardless of config — config.py's
                # OLLAMA_TEMPERATURE was defined but silently never read.
                # Lower temperature matters more on a 3B model: it hallucinates
                # and drifts off-context more readily than a larger model at
                # the same setting.
                "temperature": config.OLLAMA_TEMPERATURE,
                "top_p":       0.9,
                # Smaller context window than the previous 4096 default —
                # less attention compute per token = faster prefill on
                # every request. Overridable per-call for doc-meta answers.
                "num_ctx":     num_ctx or config.OLLAMA_NUM_CTX,
                # Hard cap on generated tokens so a single verbose answer
                # can't dominate response time. Overridable per-call.
                "num_predict": num_predict or config.OLLAMA_NUM_PREDICT,
            }
        ):
            token = chunk.get("response", "")
            if token:
                yield token
    except Exception as e:
        logger.error(f"Ollama stream failed: {e}")
        yield f"[LLM Error] Could not reach Ollama ({config.OLLAMA_MODEL}). Error: {e}"


_SENTENCE_END = re.compile(r"([.!?])(\s+)")

def stream_ollama_clean(prompt: str, num_ctx: Optional[int] = None,
                         num_predict: Optional[int] = None):
    """
    Wraps stream_ollama: buffers tokens into sentence-sized chunks so that
    strip_citations() can see a whole phrase before it's sent to the client.
    Preserves original whitespace (including newlines before bullets/lists)
    instead of collapsing everything to single spaces.
    """
    buffer = ""
    while True:
        match = _SENTENCE_END.search(buffer)
        if not match:
            break
        end = match.end()
        sentence  = buffer[:match.start() + 1]   # text up to and including . ! ?
        separator = match.group(2)               # the actual whitespace that followed
        cleaned = strip_citations(sentence)
        if cleaned:
            yield cleaned + separator
        buffer = buffer[end:]
        continue

    for token in stream_ollama(prompt, num_ctx=num_ctx, num_predict=num_predict):
        buffer += token
        while True:
            match = _SENTENCE_END.search(buffer)
            if not match:
                break
            sentence  = buffer[:match.start() + 1]
            separator = match.group(2)
            cleaned = strip_citations(sentence)
            if cleaned:
                yield cleaned + separator
            buffer = buffer[match.end():]

    if buffer:
        cleaned = strip_citations(buffer)
        if cleaned:
            yield cleaned

def call_ollama(prompt: str, num_ctx: Optional[int] = None,
                 num_predict: Optional[int] = None) -> str:
    """Blocking call — collects the full streamed response, then sanitizes it."""
    raw = "".join(stream_ollama(prompt, num_ctx=num_ctx, num_predict=num_predict))
    return strip_citations(raw)

# ─── Warmup ───────────────────────────────────────────────────────────────────
# Every model here is normally lazy-loaded on first use, which means
# whoever sends the FIRST real query pays for loading the embedder, the
# reranker, AND a cold Ollama model load all at once — often several
# seconds. Calling this once at app startup moves that entire cost off
# the request path so it never shows up as "response time" to a user.

def warmup():
    def _warm():
        try:
            t0 = time.time()
            get_embed_model()
            get_reranker()
            import ollama
            ollama.generate(
                model=config.OLLAMA_MODEL,
                prompt="hi",
                stream=False,
                keep_alive=_OLLAMA_KEEP_ALIVE,
                options={"num_predict": 1},
            )
            logger.info(f"Warmup complete in {time.time() - t0:.2f}s "
                        f"(embed model, reranker, Ollama all loaded)")
        except Exception as e:
            logger.warning(f"Warmup failed (non-fatal — models will lazy-load instead): {e}")
    threading.Thread(target=_warm, daemon=True).start()

# ─── Async chat logging ───────────────────────────────────────────────────────

def _format_duration(elapsed: float) -> str:
    """e.g. 136.03 -> '136.03 seconds = 2 minutes 16.03 seconds'"""
    minutes, seconds = divmod(elapsed, 60)
    return f"{elapsed:.2f} seconds = {int(minutes)} minutes {seconds:.2f} seconds"

def log_chat(query: str, answer: str, sources: List[Dict], elapsed: float):
    """Fire-and-forget: write Q&A log (with response time) in a background thread."""
    def _write():
        try:
            os.makedirs(config.LOG_DIR, exist_ok=True)
            ts       = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            duration = _format_duration(elapsed)
            with open(config.CHAT_HISTORY_PATH, "a", encoding="utf-8") as f:
                f.write(
                    f"\n{'='*70}\n"
                    f"[{ts}] (response time: {duration})\n"
                    f"Q: {query}\n\nA: {answer}\n\nSources:\n"
                )
                for s in sources:
                    score = s.get("rerank_score", s.get("retrieval_score", "?"))
                    score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
                    f.write(f"  - PDF: {s.get('pdf')} | Page: {s.get('page')} | "
                            f"Type: {s.get('source')} | Score: {score_str}\n")
        except Exception as e:
            logger.warning(f"Chat logging failed: {e}")
    threading.Thread(target=_write, daemon=True).start()

# ─── Main RAG Entry ───────────────────────────────────────────────────────────

def answer(query: str) -> Dict:
    """
    Full pipeline: fast-path → doc-meta → cache → retrieve → rerank →
    relevance floor → generate → clean → log.
    Returns dict with 'answer', 'sources', 'elapsed'.
    """
    t0 = time.time()

    if _faiss_index is None:
        return {
            "answer":  "No documents have been ingested yet. Please upload and ingest a PDF first.",
            "sources": [],
            "elapsed": 0.0
        }

    # 0. Fast path — structured directory lookup, no LLM involved at all.
    fast = try_fast_path(query)
    if fast:
        elapsed = round(time.time() - t0, 4)
        log_chat(query, fast["answer"], [], elapsed)
        fast["elapsed"] = elapsed
        return fast

    # 0a. Document-level meta-query — "summarize", "main topics", "list
    # tables", "page N". Bypasses FAISS; builds context from metadata.
    doc_meta = handle_doc_intent(query)
    if doc_meta:
        elapsed = round(time.time() - t0, 4)
        log_chat(query, doc_meta["answer"], [], elapsed)
        doc_meta["elapsed"] = elapsed
        return doc_meta

    # 0b. Exact-repeat question already answered before.
    cached = get_cached_answer(query)
    if cached:
        cached["elapsed"] = round(time.time() - t0, 4)
        cached["cached"]  = True
        return cached

    candidates = retrieve(query, top_k=config.TOP_K_RETRIEVE)
    top_chunks = rerank(query, candidates, top_k=config.TOP_K_RERANK)

    # 3b. Relevance floor — if even the best chunk after reranking is a
    # weak match, don't hand it to the LLM and hope. This is what stops a
    # 3B model from confidently answering off irrelevant context instead
    # of admitting it doesn't know.
    if not top_chunks or top_chunks[0].get("rerank_score", 0.0) < config.MIN_RELEVANCE_SCORE:
        elapsed = round(time.time() - t0, 2)
        result = {
            "answer":  "I couldn't find that information in the uploaded PDF.",
            "sources": [],
            "elapsed": elapsed
        }
        log_chat(query, result["answer"], [], elapsed)
        return result

    prompt   = build_prompt(query, top_chunks)
    response = call_ollama(prompt)   # already cleaned of citation artifacts

    sources = [{
        "pdf":     c.get("pdf", ""),
        "page":    c.get("page", ""),
        "source":  c.get("source", ""),
        "score":   round(c.get("rerank_score", c.get("retrieval_score", 0.0)), 4),
        "snippet": c["text"][:200] + ("…" if len(c["text"]) > 200 else "")
    } for c in top_chunks]

    elapsed = round(time.time() - t0, 2)
    log_chat(query, response, top_chunks, elapsed)   # async — does not block return

    result = {
        "answer":  response,
        "sources": sources,
        "elapsed": elapsed
    }
    set_cached_answer(query, result)
    return result


# ─── Streaming answer (for /api/chat/stream) ─────────────────────────────────

def answer_stream(query: str):
    """
    Generator for SSE streaming:
      Yields JSON lines:  {"token": "..."}
      Final line:         {"done": true, "sources": [...], "elapsed": X}
    Tokens are pre-cleaned of citation artifacts via stream_ollama_clean().
    Fast-path, doc-meta, and cache hits skip straight to a single token +
    done event.
    """
    t0 = time.time()

    if _faiss_index is None:
        yield json.dumps({"token": "No documents have been ingested yet."})
        yield json.dumps({"done": True, "sources": [], "elapsed": 0.0})
        return

    fast = try_fast_path(query)
    if fast:
        elapsed = round(time.time() - t0, 4)
        log_chat(query, fast["answer"], [], elapsed)
        yield json.dumps({"token": fast["answer"]})
        yield json.dumps({"done": True, "sources": fast["sources"],
                           "elapsed": elapsed, "fast_path": True})
        return

    doc_meta = handle_doc_intent(query)
    if doc_meta:
        elapsed = round(time.time() - t0, 4)
        log_chat(query, doc_meta["answer"], [], elapsed)
        yield json.dumps({"token": doc_meta["answer"]})
        yield json.dumps({"done": True, "sources": doc_meta["sources"],
                           "elapsed": elapsed, "fast_path": True})
        return

    cached = get_cached_answer(query)
    if cached:
        elapsed = round(time.time() - t0, 4)
        yield json.dumps({"token": cached["answer"]})
        yield json.dumps({"done": True, "sources": cached["sources"],
                           "elapsed": elapsed, "cached": True})
        return

    candidates = retrieve(query, top_k=config.TOP_K_RETRIEVE)
    top_chunks = rerank(query, candidates, top_k=config.TOP_K_RERANK)

    if not top_chunks or top_chunks[0].get("rerank_score", 0.0) < config.MIN_RELEVANCE_SCORE:
        elapsed = round(time.time() - t0, 2)
        msg = "I couldn't find that information in the uploaded PDF."
        log_chat(query, msg, [], elapsed)
        yield json.dumps({"token": msg})
        yield json.dumps({"done": True, "sources": [], "elapsed": elapsed})
        return

    prompt      = build_prompt(query, top_chunks)
    full_answer = []

    for token in stream_ollama_clean(prompt):
        full_answer.append(token)
        yield json.dumps({"token": token})

    sources = [{
        "pdf":     c.get("pdf", ""),
        "page":    c.get("page", ""),
        "source":  c.get("source", ""),
        "score":   round(c.get("rerank_score", c.get("retrieval_score", 0.0)), 4),
        "snippet": c["text"][:200] + ("…" if len(c["text"]) > 200 else "")
    } for c in top_chunks]

    elapsed     = round(time.time() - t0, 2)
    final_text  = "".join(full_answer)
    log_chat(query, final_text, top_chunks, elapsed)
    set_cached_answer(query, {"answer": final_text, "sources": sources})

    yield json.dumps({"done": True, "sources": sources, "elapsed": elapsed})


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is this document about?"
    result = answer(q)
    print("\nAnswer:", result["answer"])
    print(f"\nSources ({len(result['sources'])}):")
    for s in result["sources"]:
        print(f"  [{s['pdf']} | p.{s['page']}] {s['snippet']}")
    print(f"\nElapsed: {result['elapsed']}s")