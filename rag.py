"""
rag.py — retrieval + answer pipeline (fast + accurate, multilingual)

fast path (directory lookup) -> doc-level meta questions -> FAQ lookup -> answer cache
-> HYBRID retrieve (dense FAISS + BM25 in parallel, RRF fusion) -> rerank short list
-> relevance floor -> table/image-aware selection -> LLM (streams) -> grounding guard

Languages: tokenizing / normalising is script-independent (see sarvam.py), so every language in
config.LANGUAGES - and any language you add later - works. Retrieval needs a MULTILINGUAL embedding
model (BAAI/bge-m3) and reranker (BAAI/bge-reranker-v2-m3).

The answer is ALWAYS shown in the language selected in the UI (English, Hindi, Tamil, ...), whatever
language the PDF text is written in (see _localize + sarvam.translate_to).

Needs:  pip install rank_bm25 rapidfuzz
"""

import os
import re
import json
import hashlib
import time
import logging
import datetime
import threading
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Tuple
import sarvam
import numpy as np

import config
import ingest

logger = logging.getLogger(__name__)

# ─── Index state: ONE tuple, swapped atomically ──────────────────────────────

_state: Tuple = (None, [], None)       # (faiss_index, chunks, bm25)
_faiss_index = None                    # app.py reads this
_chunks: List[Dict] = []
_state_lock = threading.Lock()
_index_problem = ""                    # message shown while the index is unavailable / being rebuilt
_rebuilding    = False                 # automatic rebuild running
_rebuild_tried = False                 # never loop: try the automatic rebuild only once per start
_model_lock = threading.Lock()
_reranker   = None
_pool       = ThreadPoolExecutor(max_workers=4)

# one embedding model for the whole app (shared with ingestion)
get_embed_model = ingest.get_embed_model

_answer_cache: Dict[str, Dict] = {}
_answer_cache_lock = threading.Lock()

_STOP = {"the", "a", "an", "of", "is", "are", "what", "whats", "please", "tell", "me", "give",
         "for", "in", "at", "to", "about", "who"}


def _cache_key(q: str) -> str:
    return " ".join(w for w in sarvam.tokens(q) if w not in _STOP)


def get_cached_answer(q: str) -> Optional[Dict]:
    with _answer_cache_lock:
        c = _answer_cache.get(_cache_key(q))
        return dict(c) if c else None


def set_cached_answer(q: str, result: Dict):
    if looks_negative(result["answer"]):          # never remember "not found" / errors — they poisoned later queries
        return
    with _answer_cache_lock:
        if len(_answer_cache) >= config.ANSWER_CACHE_SIZE:
            _answer_cache.pop(next(iter(_answer_cache)))
        _answer_cache[_cache_key(q)] = {"answer": result["answer"], "sources": result["sources"]}

# ─── Models ───────────────────────────────────────────────────────────────────

def get_reranker():
    global _reranker
    if _reranker is None:
        with _model_lock:
            if _reranker is None:
                from sentence_transformers import CrossEncoder
                _reranker = CrossEncoder(config.RERANKER_MODEL, device=config.DEVICE,
                                         max_length=config.RERANK_MAX_LENGTH)
                logger.info(f"Re-ranker ready: {config.RERANKER_MODEL} on {config.DEVICE}")
    return _reranker

# ─── BM25 ─────────────────────────────────────────────────────────────────────

def _tok(text: str) -> List[str]:
    return sarvam.tokens(text)             # script-independent (keeps Indic vowel signs / pulli)


def _bm25_text(c: Dict) -> str:
    src = c.get("source")
    if src == "directory_entry":
        extra = f"{c.get('directory_key', '')} {c.get('directory_title', '')} {c.get('location_name', '')} "
    elif src == "qa_pair":
        q = c.get("question", "")
        extra = f"{c.get('section', '')} {q} {q} "          # the question is what users search for
    else:
        extra = ""
    return extra + c["text"]


def _build_bm25(chunks: List[Dict]):
    try:
        from rank_bm25 import BM25Okapi
        return BM25Okapi([_tok(_bm25_text(c)) for c in chunks]) if chunks else None
    except ImportError:
        logger.warning("rank_bm25 not installed — dense-only retrieval (pip install rank_bm25)")
        return None


def load_index():
    """Build everything locally, then swap in ONE assignment."""
    global _state, _faiss_index, _chunks, _index_problem
    if not os.path.exists(config.FAISS_INDEX_PATH) or not os.path.exists(config.CHUNKS_JSON_PATH):
        logger.warning("FAISS index not found. Ingest PDFs first.")
        with _state_lock:
            _state, _faiss_index, _chunks = (None, [], None), None, []
        return
    import faiss
    index = faiss.read_index(config.FAISS_INDEX_PATH)
    with open(config.CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    dim = getattr(config, "EMBED_DIM", None)
    if dim and index.d != dim:                       # index was built with another embedding model
        bm25 = _build_bm25(chunks)
        with _state_lock:                            # no vectors, but chunks stay -> FAQ / directory answers work
            _state, _faiss_index, _chunks = (None, chunks, bm25), None, chunks
        if not _rebuild_tried:
            _index_problem = ("⏳ The search index is being rebuilt for the new multilingual model "
                              "(one-time, a few minutes). Please ask again shortly.")
            logger.warning(f"Index has {index.d}-dim vectors but {config.EMBED_MODEL} makes {dim}: "
                           f"rebuilding automatically ...")
            _auto_rebuild()
        else:
            _index_problem = (f"The saved index uses {index.d}-dimension vectors but {config.EMBED_MODEL} "
                              f"makes {dim}. Check EMBED_DIM in config.py, or run:  python ingest.py --rebuild")
            logger.error(_index_problem)
        return
    _index_problem = ""
    bm25 = _build_bm25(chunks)
    with _state_lock:
        _state = (index, chunks, bm25)
        _faiss_index, _chunks = index, chunks
    _embed_query.cache_clear()
    with _answer_cache_lock:
        _answer_cache.clear()
    logger.info(f"Index loaded: {index.ntotal} vectors, {len(chunks)} chunks")

def _auto_rebuild():
    """Re-embed chunks.json with the current model in the background, then load the new index."""
    global _rebuilding, _rebuild_tried
    if _rebuilding:
        return
    _rebuilding = _rebuild_tried = True

    def _job():
        global _rebuilding, _index_problem
        try:
            ingest.rebuild_index()
            _rebuilding = False
            load_index()                             # loads the fresh index; clears _index_problem
            logger.info("Automatic index rebuild finished - chat is ready.")
        except Exception as e:
            logger.exception("Automatic index rebuild failed")
            _index_problem = f"Automatic rebuild failed ({e}). Run:  python ingest.py --rebuild"
        finally:
            _rebuilding = False

    threading.Thread(target=_job, daemon=True, name="index-rebuild").start()


def _instant_only(query: str, chunks: List[Dict]) -> Optional[Dict]:
    """While the vector index is unavailable: FAQ / directory answers need no vectors."""
    if not chunks:
        return None
    return try_fast_path(query, chunks) or try_qa_fast(query, chunks)

# ─── Query embedding (cached) ─────────────────────────────────────────────────

# BGE-M3 needs NO instruction prefix. (English-only BGE models use
# "Represent this sentence for searching relevant passages: " -> set config.EMBED_QUERY_PROMPT for those.)
_QUERY_PROMPT = getattr(config, "EMBED_QUERY_PROMPT", "")


@lru_cache(maxsize=config.QUERY_CACHE_SIZE)
def _embed_query(query: str) -> bytes:
    q = get_embed_model().encode([_QUERY_PROMPT + query], convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=False)
    return q.astype(np.float32).tobytes()


def embed_query(query: str) -> np.ndarray:
    return np.frombuffer(_embed_query(query), dtype=np.float32).reshape(1, -1)


load_index()

# ─── Friendly formatting (icons / labels come from the data, nothing fixed) ───

def format_single_field(label: str, value: str) -> str:
    return f"{ingest.field_icon(label, value)} **{label.title()}:** {value}"


def _dir_label(c: Dict) -> str:
    return (c.get("directory_key") or c.get("directory_title") or "").strip()


def format_directory_entry(chunk: Dict, with_title: bool = False) -> str:
    fields = ingest._parse_entry_fields(chunk.get("text", ""))
    if not fields:
        return chunk.get("text", "").strip()
    loc   = chunk.get("location_name", "").strip()
    title = _dir_label(chunk)
    head  = f"{title} — {loc}" if (with_title and title) else loc
    lines = [f"🏢 **{head}**"] if head else []
    lines += [format_single_field(k, v) for k, v in fields.items() if v]
    return "  \n".join(lines)


def render_directory(items: List[Tuple[Dict, Optional[Dict]]]) -> str:
    """items = [(chunk, fields-or-None)]. Friendly lead-in + cards + closing line."""
    multi, parts = len(items) > 1, []
    asked = any(f for _, f in items)
    heads = [f"{_dir_label(c)} — {c.get('location_name', '')}".strip(" —") for c, _ in items]
    for (c, fields), head in zip(items, heads):
        if fields:
            body = "  \n".join(format_single_field(k, v) for k, v in fields.items())
            if multi:
                body = f"🏢 **{head}**  \n{body}"
        else:
            body = format_directory_entry(c, with_title=True)
        parts.append(body)
    if not multi and asked:
        lead = f"Sure! 😊 Here's what you asked for — **{heads[0]}**:"
    elif multi:
        lead = "Here's what I found 👇"
    else:
        lead = "Here are the details I found 👇"
    return lead + "\n\n" + "\n\n---\n\n".join(parts) + "\n\n💬 Need anything else? Just ask!"


def format_qa_answer(question: str, answer: str) -> str:
    """Answer only - the user's question is NOT repeated back."""
    a, ql = answer.strip(), question.lower()
    if re.match(r"(?i)yes\b", a):
        icon = "✅"
    elif re.match(r"(?i)no\b", a):
        icon = "❌"
    elif re.match(r"(what (is|are|does)|define|meaning)", ql):
        icon = "📘"
    else:
        icon = "💡"
    return f"{icon} {a}\n\n😊 Anything else you'd like to know?"


def _src(c: Dict, score=1.0) -> Dict:
    d = {"pdf": c.get("pdf", ""), "page": c.get("page", ""), "source": c.get("source", ""),
         "score": round(score, 4),
         "snippet": c["text"][:200] + ("…" if len(c["text"]) > 200 else "")}
    if c.get("directory_key"):
        d["directory"] = c["directory_key"]
    if c.get("question"):
        d["question"] = c["question"]
    if c.get("image_path"):
        d["image"] = "/images/" + os.path.relpath(c["image_path"], config.IMAGES_DIR).replace(os.sep, "/")
    return d

# ─── Fast path 1: directory lookup ───────────────────────────────────────────

def try_fast_path(query: str, chunks: List[Dict]) -> Optional[Dict]:
    if not chunks:
        return None
    try:
        result = ingest.query_location(query, chunks=chunks)
    except Exception:
        logger.exception("Directory fast path crashed — falling back to full RAG")   # full traceback, never silent
        return None
    if not result:
        return None
    items = [(m["chunk"], m.get("fields")) for m in result["matches"]]
    return {"answer": render_directory(items), "sources": [_src(c) for c, _ in items],
            "fast_path": True, "route": "directory", "chunks": [c for c, _ in items]}

# ─── Fast path 2: FAQ question lookup (instant, verbatim from the document) ──

_QA_STOPWORDS = {"what", "whats", "is", "are", "was", "were", "an", "a", "the", "of", "to", "for", "how", "do",
                 "does", "did", "can", "could", "i", "my", "me", "we", "you", "your", "in", "on", "at", "it",
                 "be", "and", "or", "with", "about", "tell", "explain", "define", "meaning", "please", "there",
                 "this", "that", "if", "will", "would", "should", "may", "have", "has", "any", "need", "want",
                 "know", "give", "from"}
_qa_cache = (None, [])


def _qa_tokens(text: str) -> frozenset:
    out = []
    for w in sarvam.tokens(text):
        if w in _QA_STOPWORDS:
            continue
        if w.isascii() and len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]                                   # plural stemming for English words only
        out.append(w)
    return frozenset(out)


def _qa_index(chunks: List[Dict]):
    """[(chunk_index, token_set, normalised_question)] — one entry per distinct question."""
    global _qa_cache
    if _qa_cache[0] is chunks:
        return _qa_cache[1]
    seen, idx = set(), []
    for i, c in enumerate(chunks):
        if c.get("source") == "qa_pair" and c.get("question") and c["question"] not in seen:
            seen.add(c["question"])
            idx.append((i, _qa_tokens(c["question"]), " ".join(sarvam.tokens(c["question"]))))
    _qa_cache = (chunks, idx)
    return idx


def try_qa_fast(query: str, chunks: List[Dict]) -> Optional[Dict]:
    """The question is (almost) one that the PDF already asks -> answer straight from the PDF.
    Score = word overlap of the meaningful words (Jaccard), so 'eta' / 'what is eta ?' -> 'What is an ETA?'
    but 'visa' alone (matches dozens of questions) is NOT answered blindly. Works in every script."""
    idx = _qa_index(chunks)
    qt = _qa_tokens(query)
    if not idx or not qt or len(query.split()) > 25:
        return None
    scored = sorted(((len(qt & t) / len(qt | t), i) for i, t, _ in idx if t and (qt & t)), reverse=True)
    best, bi = (scored[0] if scored else (0.0, -1))
    second = scored[1][0] if len(scored) > 1 else 0.0
    if not (best >= config.QA_MIN_MATCH and best - second >= config.QA_MIN_GAP):
        # typo-tolerant fallback on the whole question text
        try:
            from rapidfuzz import fuzz
            qn = " ".join(sarvam.tokens(query))
            fz = sorted(((fuzz.token_sort_ratio(qn, norm), i) for i, _, norm in idx), reverse=True)
            if not fz or fz[0][0] < 92 or (len(fz) > 1 and fz[0][0] - fz[1][0] < 5):
                return None
            best, bi = fz[0][0] / 100.0, fz[0][1]
        except Exception:
            return None
    c = chunks[bi]
    return {"answer": format_qa_answer(c["question"], c["answer"]), "sources": [_src(c, best)],
            "fast_path": True, "route": "qa", "chunks": [c]}


# ─── Lexical check: does the FAQ really answer THIS question? ────────────────
# Embeddings say "similar topic"; they cannot tell "which language got classical status in 2004?" from
# "in which year did Tamil get classical status?". So a FAQ pair is only used when most of the meaningful
# words of the user's question appear in it (prefix match, so Tamil suffixes like மொழி / மொழிக்குச் agree).

def _stem_match(a: str, b: str) -> bool:
    if a == b:
        return True
    n = min(len(a), len(b))
    if n < 3:
        return False
    p = 0
    while p < n and a[p] == b[p]:
        p += 1
    return p >= max(3, int(n * 0.75))


def _coverage(query: str, text: str) -> float:
    """Share (0..1) of the question's meaningful words that also occur in `text`."""
    qt = [t for t in _qa_tokens(query) if len(t) > 2]
    if not qt:
        return 1.0
    tt = _qa_tokens(text)
    return sum(1 for q in qt if any(_stem_match(q, t) for t in tt)) / len(qt)


def _same_script(query: str, text: str) -> bool:
    """Different scripts (e.g. English question, Tamil PDF) cannot be compared word by word."""
    qs = sarvam.dominant_script(query)
    return qs is not None and qs == sarvam.dominant_script(text)


def _faq_unsupported(query: str, top: List[Dict]) -> bool:
    """Best hit is a FAQ pair, but none of the (same-script) hits contains enough of the question's words."""
    if not top or top[0].get("source") != "qa_pair":
        return False
    thr  = getattr(config, "QA_DIRECT_MIN_COVER", 0.6)
    same = [c for c in top if _same_script(query, c["text"])]
    if not same:
        return False
    return not any(_coverage(query, _bm25_text(c)) >= thr for c in same)


def try_direct_answer(query: str, top: List[Dict], all_chunks: List[Dict]) -> Optional[Dict]:
    """Retrieval found ONE clearly-best directory entry / FAQ pair -> answer verbatim (no LLM needed)."""
    best = top[0]
    if best.get("source") not in ("directory_entry", "qa_pair") or len(query.split()) > 15:
        return None
    s0 = best.get("rerank_score", 0.0)
    if s0 < config.DIRECT_ANSWER_MIN_SCORE:
        return None
    if len(top) > 1 and s0 - top[1].get("rerank_score", 0.0) < config.DIRECT_ANSWER_GAP:
        return None                       # ambiguous -> let the LLM see several candidates
    if best["source"] == "qa_pair":
        if (_same_script(query, best["text"])
                and _coverage(query, best["text"]) < getattr(config, "QA_DIRECT_MIN_COVER", 0.6)):
            return None                   # similar topic, different question -> never answer it verbatim
        return {"answer": format_qa_answer(best["question"], best["answer"]), "sources": [_src(best, s0)],
                "fast_path": True, "route": "qa", "chunks": [best]}
    named = ingest.named_directories(query, all_chunks)
    if named and best.get("directory_key") not in named:
        return None
    flds = ingest._parse_entry_fields(best["text"])
    # words of THIS directory's title ('contact', 'directory', ...) name the directory, not a field
    ignore = ingest.title_words(all_chunks, {best.get("directory_key")})
    want = ingest.requested_fields(query, flds, ignore)
    return {"answer": render_directory([(best, {k: flds[k] for k in want} if want else None)]),
            "sources": [_src(best, s0)], "fast_path": True, "route": "directory", "chunks": [best]}

# ─── Document-level meta questions ───────────────────────────────────────────

_PAGE_RE    = re.compile(r"\bpage\s+(\d+)\b", re.I)
_TABLES_RE = re.compile(
    r"\b(?:list|show|extract|display|give|get|find)\b(?:\s+\w+){0,3}?\s+tables\b"
    r"|\ball\s+tables\b|\btables\s+(?:found|list)\b|\bhow\s+many\s+tables\b", re.I)
_TABLE_NUM_RE = re.compile(
    r"\b(?:show|extract|display|give|get|open)\b.{0,15}\btable\s*#?(\d{1,3})\b", re.I)
_TOPICS_RE  = re.compile(r"\b(main|key)\s+topics?\b|\bwhat\s+is\s+this\s+(document|pdf|file)\s+about\b", re.I)
_SUMMARY_RE = re.compile(r"\bsummar(y|ize|ise)\b|\btl;?dr\b|\boverview\b", re.I)
_DOC_WORDS  = re.compile(r"\b(document|pdf|file|doc)\b", re.I)
_TABLE_SRC  = ("table", "docling_table")


def detect_doc_intent(query: str) -> Tuple[Optional[str], Optional[int]]:
    words = len(query.split())
    m = _PAGE_RE.search(query)
    if m and words <= 10:
        return "page", int(m.group(1))
    m = _TABLE_NUM_RE.search(query)
    if m and words <= 10:
        return "tables", int(m.group(1))          # "show table 3"
    if _TABLES_RE.search(query) and words <= 10:
        return "tables", None                     # "list all tables"
    whole_doc = bool(_DOC_WORDS.search(query)) or words <= 5
    if _TOPICS_RE.search(query) and whole_doc:
        return "topics", None
    if _SUMMARY_RE.search(query) and whole_doc:
        return "summary", None
    return None, None


def _all_pdfs(chunks: List[Dict]) -> List[str]:
    seen = []
    for c in chunks:
        p = c.get("pdf")
        if p and p not in seen:
            seen.append(p)
    return sorted(seen, key=str.lower)


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _target_pdfs(query: str, chunks: List[Dict]) -> List[str]:
    """ALL documents, unless the question names one of them."""
    pdfs = _all_pdfs(chunks)
    qn = _norm_name(query)
    named = [p for p in pdfs if _norm_name(p) and _norm_name(p) in qn]
    return named or pdfs


def _page_sort_key(page):
    try:
        return int(page)
    except (TypeError, ValueError):
        return 10**9


def _overview_sample(pdf: str, chunks: List[Dict], max_chars: int = 3500) -> str:
    """Excerpts spread evenly over the whole document (not just the first pages)."""
    by_page: Dict = {}
    for c in chunks:
        if c.get("pdf") == pdf:
            by_page.setdefault(c.get("page", 0), c)
    pages = sorted(by_page, key=_page_sort_key)
    budget = max(1, max_chars // 420)
    if len(pages) > budget:
        step = len(pages) / budget
        pages = [pages[int(i * step)] for i in range(budget)]
    pieces = []
    for page in pages:
        text = by_page[page]["text"].strip()
        if text:
            pieces.append(f"[Page {page}] {text[:400]}")
    return "\n\n".join(pieces)


DOC_SYSTEM_PROMPT = (
    "You are a warm, friendly assistant chatting with the user about THEIR uploaded documents.\n"
    "Rules:\n"
    "- The CONTEXT has excerpts of one or more documents. Each starts with '=== DOCUMENT: name ==='.\n"
    "- Cover EVERY document, each under its own bold heading with the document name.\n"
    "- Write like a person talking: a short friendly opening line, then 2-4 simple sentences or bullets "
    "per document. No copy-paste from the text.\n"
    "- Use ONLY the CONTEXT. Never invent facts, names or numbers.\n"
    "- Use 1-3 fitting emojis (📄 📊 💡 ✅) and finish with a short friendly offer such as "
    "'Want me to go deeper into any of these? 😊'"
)


def _tnum(label: str) -> int:
    m = re.search(r"\d+", label or "")
    return int(m.group()) if m else 0


def _collect_tables(pdfs, chunks: List[Dict]) -> Dict:
    """(pdf, page, '[TABLE n]') -> header + ALL rows (chunk_table splits big tables by rows)."""
    found: Dict = {}
    for c in chunks:
        if c.get("pdf") not in pdfs or c.get("source") not in _TABLE_SRC:
            continue
        lines = [l for l in c["text"].split("\n") if l.strip()]
        if not lines:
            continue
        rec = found.setdefault((c["pdf"], c.get("page"), lines[0]),
                               {"chunk": c, "head": lines[:3], "rows": []})
        rec["rows"] += lines[3:]
    return found


def _render_tables(pdfs: List[str], chunks: List[Dict], only: Optional[int] = None):
    """-> (markdown text, [first chunk of each table shown])"""
    found = _collect_tables(set(pdfs), chunks)
    blocks, used = [], []
    for pdf in pdfs:
        mine = [(k, v) for k, v in found.items() if k[0] == pdf]
        if only is not None:
            mine = [(k, v) for k, v in mine if _tnum(k[2]) == only]
        if not mine:
            if only is None:
                blocks.append(f"📄 **{pdf}** — no tables found")
            continue
        mine.sort(key=lambda kv: _tnum(kv[0][2]))
        blocks.append(f"📄 **{pdf}** — {len(mine)} table(s)")
        for (_, page, label), rec in mine:
            body = "\n".join(rec["head"][1:3] + rec["rows"])
            blocks.append(f"**Table {_tnum(label)} · page {page}** ({len(rec['rows'])} rows)\n\n{body}")
            used.append(rec["chunk"])
    return "\n\n".join(blocks), used


def plan_doc_intent(query: str, chunks: List[Dict]) -> Optional[Dict]:
    intent, arg = detect_doc_intent(query)
    if intent is None or not chunks:
        return None
    pdfs = _target_pdfs(query, chunks)
    if not pdfs:
        return None
    multi = len(pdfs) > 1

    # ---- page N (of every document) ----
    if intent == "page":
        blocks, srcs = [], []
        for pdf in pdfs:
            pc = [c for c in chunks if c.get("pdf") == pdf and c.get("page") == arg]
            if pc:
                blocks.append(f"=== DOCUMENT: {pdf} (page {arg}) ===\n"
                              + "\n".join(c["text"] for c in pc)[:1500])
                srcs += [_src(c) for c in pc]
        if not blocks:
            return {"answer": f"😕 I couldn't find page {arg} in " +
                              ("any of your documents." if multi else f"**{pdfs[0]}**."), "sources": []}
        user = ("CONTEXT:\n" + "\n\n".join(blocks) +
                f"\n\nQUESTION:\nWhat does page {arg} say? Explain it in a friendly way, "
                f"one section per document.")
        return {"llm": (DOC_SYSTEM_PROMPT, user), "sources": srcs}

    # ---- tables: full content of every table (or one table: "show table 5") ----
    if intent == "tables":
        text, used = _render_tables(pdfs, chunks, only=arg)
        if not used:
            what = f"table {arg}" if arg else "any tables"
            return {"answer": f"😕 I couldn't find {what} in " +
                              ("your documents." if multi else f"**{pdfs[0]}**."),
                    "sources": []}
        n = len(used)
        head = (f"📊 Sure! Here is table {arg} 👇" if arg else
                f"📊 Sure! I found **{n} table{'s' if n != 1 else ''}**"
                + (f" across your {len(pdfs)} documents" if multi else "")
                + " — here they all are 👇")
        return {"answer": head + "\n\n" + text +
                          "\n\n💬 Want me to explain any table? Just ask! 😊",
                "sources": [_src(c) for c in used], "route": "tables"}

    # ---- summary / main topics (every document) ----
    per = max(900, 3000 // len(pdfs))
    blocks = []
    for pdf in pdfs:
        s = _overview_sample(pdf, chunks, max_chars=per)
        if s:
            blocks.append(f"=== DOCUMENT: {pdf} ===\n{s}")
    if not blocks:
        return None
    what = "Summarize" if intent == "summary" else "List the main topics of"
    question = (f"{what} EACH document separately, in a friendly conversational way."
                if multi else f"{what} this document in a friendly conversational way.")
    user = "CONTEXT (excerpts from the documents):\n" + "\n\n".join(blocks) + f"\n\nQUESTION:\n{question}"
    srcs = [_src(next(c for c in chunks if c.get("pdf") == p)) for p in pdfs]
    return {"llm": (DOC_SYSTEM_PROMPT, user), "sources": srcs}

# ─── Hybrid retrieval (dense + BM25 in parallel -> RRF) ──────────────────────

_TABLE_Q = re.compile(r"\b(table|row|column|rate|rates|fees?|price|cost|charges?|amount|total|"
                      r"how much|schedule|compare|versus|vs)\b", re.I)
_IMAGE_Q = re.compile(r"\b(image|figure|chart|diagram|picture|photo|graph|map|logo|scanned?|stamp)\b", re.I)


def retrieve(query: str, state: Tuple, top_k: int = config.TOP_K_RETRIEVE) -> List[Dict]:
    index, chunks, bm25 = state
    if index is None or not chunks:
        return []
    k = min(top_k, len(chunks))

    def dense():
        _, idx = index.search(embed_query(query), k)
        return [int(i) for i in idx[0] if i >= 0]

    def sparse():
        if bm25 is None:
            return []
        s = bm25.get_scores(_tok(query))
        return [int(i) for i in np.argsort(s)[::-1][:k] if s[i] > 0] if s.max() > 0 else []

    fd, fs = _pool.submit(dense), _pool.submit(sparse)
    fused: Dict[int, float] = {}
    for ids in (fd.result(), fs.result()):
        for r, i in enumerate(ids):
            fused[i] = fused.get(i, 0.0) + 1.0 / (60 + r)

    want_t, want_i = bool(_TABLE_Q.search(query)), bool(_IMAGE_Q.search(query))
    for i in fused:                                   # nudge the right content type up
        src = chunks[i].get("source")
        if (want_t and src in _TABLE_SRC) or (want_i and src == "image_ocr"):
            fused[i] += 0.01

    q_vec = embed_query(query)[0]
    out = []
    for i in sorted(fused, key=fused.get, reverse=True)[:config.RERANK_CANDIDATES]:
        c = dict(chunks[i])
        c["retrieval_score"] = float(index.reconstruct(i) @ q_vec)
        c["fusion_score"]    = fused[i]
        out.append(c)
    return out


def rerank(query: str, candidates: List[Dict], top_k: int = config.RERANK_CANDIDATES) -> List[Dict]:
    if not candidates:
        return []
    if not getattr(config, "USE_RERANKER", True):    # fast mode: bge-m3 cosine score decides
        for c in candidates:
            c["rerank_score"] = c.get("retrieval_score", 0.0)
        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
    try:
        raw    = get_reranker().predict([[query, c["text"]] for c in candidates],
                                        batch_size=len(candidates), show_progress_bar=False)
        scores = 1.0 / (1.0 + np.exp(-np.asarray(raw, dtype=np.float64)))
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
    except Exception as e:
        logger.warning(f"Re-ranking failed ({e}); using fusion order.")
        for c in candidates:
            c["rerank_score"] = c.get("retrieval_score", 0.0)
        return candidates[:top_k]


def _select(ranked: List[Dict], k: int, query: str) -> List[Dict]:
    """Top-k, but a table / image chunk is guaranteed a slot when the question asks for one."""
    top = ranked[:k]
    for kinds, pat in ((_TABLE_SRC, _TABLE_Q), (("image_ocr",), _IMAGE_Q)):
        if pat.search(query) and not any(c.get("source") in kinds for c in top):
            extra = next((c for c in ranked[k:] if c.get("source") in kinds
                          and c.get("rerank_score", 0) >= config.MIN_RELEVANCE_SCORE), None)
            if extra:
                top = top[:-1] + [extra]
    return top

# ─── Prompt ───────────────────────────────────────────────────────────────────
# NOTE: no example answer line in here — a 3B model copies it into every answer
# (that is where "Address: Not specified / Phone: Not specified" came from).

SYSTEM_PROMPT = (
    "You are a warm, friendly assistant that answers questions about the user's documents.\n"
    "Rules:\n"
    "- Use ONLY the CONTEXT. If it really does not contain the answer, reply with exactly: NOT_FOUND\n"
    "- Talk like a friendly person, not like a document: explain the details in your own words, "
    "keeping every fact, number, phone and e-mail EXACT.\n"
    "- Start DIRECTLY with the answer. Do NOT write a greeting line (like 'Sure!' or 'Here is what you "
    "need to know') and do NOT write a closing line (like 'Hope that helps'). The app adds those itself.\n"
    "- Put every list item on its OWN line starting with '- '. Never put several items on one line.\n"
    "- Never use country-flag emojis.\n"
    "- If the CONTEXT has a 'Q:' / 'A:' pair that matches the question, give that answer completely, "
    "but rephrase it naturally instead of copying it word for word.\n"
    "- Never reply with just 'Yes.' or 'No.'. Start with a short friendly direct answer, then add the key "
    "details, conditions and exceptions from the CONTEXT.\n"
    "- Never invent names, numbers, phone numbers, emails or addresses.\n"
    "- Never mention 'the context', page numbers, or file names. Just answer.\n"
    "- Answer exactly what was asked, ONCE. Do not repeat the answer or add extra sections.\n"
    "- Give address / phone / email / in-charge ONLY if the question asks for them AND the CONTEXT contains "
    "them. If a detail is missing, leave it out. Never write 'Not specified', 'Not available' or 'N/A'.\n"
    "- Use 1-3 fitting emojis as icons: ✅ allowed / yes, ❌ not allowed / no, ⚠️ condition or warning, "
    "📄 documents, ✈️ travel, 📞 phone, 📍 place, 💡 tip. Start key lines or bullets with an emoji.\n"
    "- Use bullets for several points, a numbered list for steps, a markdown table for tabular data. "
    "Bold key terms with **markdown**. Include EVERY relevant item of a list.\n"
    "- Tables in the CONTEXT are markdown: read the correct row AND column, and copy numbers exactly.\n"
    "- Text marked [IMAGE ...] came from an image (OCR or a description); quote digits exactly.\n"
    "- Write complete sentences only; never stop in the middle of a sentence.\n"
)


def build_messages(query: str, chunks: List[Dict]) -> Tuple[str, str]:
    ctx = "\n\n---\n\n".join(c["text"] for c in chunks)
    return SYSTEM_PROMPT, f"CONTEXT:\n{ctx}\n\nQUESTION:\n{query}"


_LONG_Q = re.compile(r"\b(list|all|every|steps?|procedure|process|explain|compare)\b", re.I)


def _predict_budget(query: str) -> int:
    return config.OLLAMA_NUM_PREDICT_LONG if _LONG_Q.search(query) else config.OLLAMA_NUM_PREDICT

# ─── Answer cleanup ───────────────────────────────────────────────────────────

_CITATION_PATTERNS = [
    re.compile(r"\(?\s*according to\s+(?:the\s+)?(?:page|source)\s+\d+\s*\)?", re.I),
    re.compile(r"\(?\s*as (?:stated|mentioned|noted) (?:in|on)\s+(?:page|source)\s+\d+\s*\)?", re.I),
    re.compile(r"\[\s*source\s*\d+\s*\]", re.I),
    re.compile(r"\(\s*(?:page|p\.)\s*\d+\s*\)", re.I),
]

# "<anything>: Not specified / N/A / None" -> removed (works for ANY label, e.g. "📍 **Address:** Not specified")
_EMO = r'[\U0001F300-\U0001FAFF\u2600-\u27BF]\ufe0f?'
_EMPTY_FIELD_RE = re.compile(
    r'(?i)(?:' + _EMO + r'\s*)?\*{0,2}[\w\-/ ]{2,25}?\*{0,2}\s*:\s*\*{0,2}\s*'
    r'(?:not\s+(?:specified|available|mentioned|provided|found)|n/?a)\b\.?')


def strip_empty_fields(text: str) -> str:
    text = _EMPTY_FIELD_RE.sub("", text)
    text = re.sub(r'(?m)^[ \t]*[-•*][ \t]*$\n?', '', text)           # bullet-only lines (blank lines are kept!)
    return re.sub(r'[ \t]{2,}', ' ', text).strip()


_FLAG_RE   = re.compile(r"[\U0001F1E6-\U0001F1FF]+")          # flag emojis show up as 'KE', 'IN' ... on Windows
_BULLET_RE = re.compile(r"[ \t]*[•▪●][ \t]*")


def strip_citations(text: str) -> str:
    for p in _CITATION_PATTERNS:
        text = p.sub("", text)
    text = _FLAG_RE.sub("", text)
    if text.count("•") >= 2:                                         # "• a • b • c" on ONE line -> real list
        text = _BULLET_RE.sub("\n- ", text).strip()
        text = re.sub(r"([^\n])\n- ", r"\1\n\n- ", text, count=1)
    text = strip_empty_fields(text)
    text = re.sub(r"[ \t]+([.,;:])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ─── Greeting / closing: written by the APP (varied, meaningful), never by the 3B model ───

_LEAD_FILLER = re.compile(r"(?i)^\s*(?:sure|certainly|of course|absolutely|great|okay|ok|alright|happy to help)"
                          r"\s*[!,.:]\s*(?:😊|🙂)?\s*")
_LEAD_LINE   = re.compile(r"(?i)^\s*here(?:'s| is| are)\b[^\n]{0,70}(?:know|details|information|found|following|answer)"
                          r"[^\n]{0,20}\n+")
_TAIL_INLINE = re.compile(r"(?i)[ \t]*(?:i )?hope (?:that|this|it) helps[^\n]{0,30}$")
_TAIL_LINE   = re.compile(r"(?i)(hope (?:that|this|it) helps|let me know if|feel free to ask|anything else|"
                          r"is there anything|don'?t hesitate|happy to help)")


def strip_frame(text: str) -> str:
    """Remove a greeting ('Sure! Here is what you need to know') and a closing ('Hope that helps!')."""
    t = _LEAD_FILLER.sub("", text.strip(), count=1)
    t = _LEAD_LINE.sub("", t, count=1)
    t = _TAIL_INLINE.sub("", t).rstrip()
    lines = t.split("\n")
    while len(lines) > 1 and lines[-1].strip() and len(lines[-1]) < 140 and _TAIL_LINE.search(lines[-1]):
        lines.pop()
    t = "\n".join(lines).strip()
    return (t[:1].upper() + t[1:]) if t and t[0].islower() else t


_FRAMES = {
    "en": {
        "open": ["Good question! Here's what I found in your documents 😊",
                 "Happy to help! Here are the details you asked for 👇",
                 "Here you go! I've pulled the information together for you 👇",
                 "Sure thing! This is what your documents say about it 😊"],
        "close": ["Would you like more detail on any of this? Just ask 😊",
                  "If anything is unclear, tell me and I'll explain it in more depth 🙂",
                  "Let me know if you'd like me to look up anything else for you 😊"],
    },
    "ta": {
        "open": ["நல்ல கேள்வி! உங்கள் ஆவணங்களில் எனக்குக் கிடைத்த தகவல் இதோ 😊",
                 "கண்டிப்பாக! நீங்கள் கேட்ட விவரங்களை எளிமையாகக் கீழே தருகிறேன் 👇",
                 "வணக்கம்! உங்கள் கேள்விக்கான தெளிவான பதில் இதோ 👇",
                 "சரிங்க! இதைப் பற்றி ஆவணங்கள் சொல்வதை இங்கே பார்க்கலாம் 😊",
                 "உங்களுக்காக இந்த விவரங்களைத் தேடி எடுத்து வந்திருக்கிறேன் 👇"],
        "close": ["இன்னும் ஏதாவது தெரிந்துகொள்ள வேண்டுமா? தயங்காமல் கேளுங்கள் 😊",
                  "இந்தத் தகவல் உங்களுக்குப் பயனுள்ளதாக இருக்கும் என்று நம்புகிறேன் 😊",
                  "வேறு ஏதேனும் சந்தேகம் இருந்தால் எப்போது வேண்டுமானாலும் என்னிடம் கேளுங்கள் 🙂",
                  "இதில் புரியாத பகுதி இருந்தால் சொல்லுங்கள், இன்னும் விளக்கமாகச் சொல்கிறேன் 😊"],
    },
    "hi": {
        "open": ["बहुत अच्छा सवाल! आपके दस्तावेज़ों से मिली जानकारी यह रही 😊",
                 "ज़रूर! आपने जो पूछा, उसका जवाब नीचे दिया है 👇",
                 "नमस्ते! आपके सवाल का साफ़ जवाब यह रहा 👇",
                 "आपके लिए यह जानकारी ढूँढकर लाया हूँ 👇"],
        "close": ["अगर आप और कुछ जानना चाहते हैं, तो बेझिझक पूछिए 😊",
                  "उम्मीद है कि यह जानकारी आपके काम आएगी 😊",
                  "कुछ समझ न आए तो मुझे बताइए, मैं और विस्तार से समझा दूँगा 🙂"],
    },
}


@lru_cache(maxsize=64)
def _translated_frame(text: str, lang: str) -> str:
    try:
        return sarvam.translate(text, "en-IN", lang)
    except Exception:
        return text


def _frame_line(kind: str, lang: str, n: int, translate_on: bool) -> str:
    pool = _FRAMES.get(lang.split("-")[0])
    if pool:
        return pool[kind][n % len(pool[kind])]
    eng = _FRAMES["en"][kind][n % len(_FRAMES["en"][kind])]
    return _translated_frame(eng, lang) if translate_on else eng


def add_frame(text: str, query: str, lang: str, translate_on: bool = False) -> str:
    """Friendly, complete opening + closing sentence in the selected language (same question -> same wording)."""
    h = int(hashlib.md5(query.encode("utf-8")).hexdigest(), 16)
    return (f"{_frame_line('open', lang, h, translate_on)}\n\n{text}\n\n"
            f"{_frame_line('close', lang, h // 7, translate_on)}")

# ─── Refusal detection / final clean-up ───────────────────────────────────────

_NEG_RE = re.compile(r"(?i)(\bNOT_FOUND\b|no information (?:about|on|regarding)|not (?:specified|mentioned|stated|"
                     r"provided|available|found) in|(?:couldn'?t|could not|cannot|can'?t|unable to) find|"
                     r"\[LLM Error\]|\b(?:the|provided|given) context\b)")


def looks_negative(text: str) -> bool:
    return len(text) < 400 and bool(_NEG_RE.search(text))


def finalize_answer(raw: str, frame: bool = False) -> Tuple[str, bool]:
    """-> (clean friendly text, cacheable). Refusals become one friendly message and are never cached."""
    if raw.lstrip().startswith("[LLM Error]"):
        logger.error(raw.strip()[:500])
        return (f"⚠️ {raw.strip()[:300]}", False)
    t = strip_citations(raw)
    if frame:
        t = strip_frame(t)
    if not t or looks_negative(t):
        logger.warning(f"Answer treated as NOT_FOUND. Raw LLM output: {raw[:300]!r}")
        return NOT_FOUND, False
    return t, True

# ─── Grounding guard ──────────────────────────────────────────────────────────

def _grounded(answer: str, context: str) -> bool:
    ctx_low, ctx_digits = context.lower(), re.sub(r"\D", "", context)
    for e in re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", answer):
        if e.lower() not in ctx_low:
            return False
    for n in re.findall(r"\d[\d\s\-]{4,}\d", answer):
        d = re.sub(r"\D", "", n)
        if len(d) >= 5 and d not in ctx_digits:
            return False
    return True


def _keeps_facts(src: str, out: str) -> bool:
    """True if every number (2+ digits) of the source text is still in the rewritten answer."""
    out_digits = sarvam._ascii_digits(out)
    return all(n in out_digits for n in re.findall(r"\d{2,}", src))


def _safe_fallback(top_chunks: List[Dict], response: str) -> str:
    if top_chunks and top_chunks[0].get("source") == "directory_entry":
        return format_directory_entry(top_chunks[0], with_title=True)
    return response + "\n\n⚠️ Please double-check the numbers/emails above against the document."

# ─── LLM (Ollama chat API) ────────────────────────────────────────────────────

def _resolve_keep_alive():
    raw = os.environ.get("OLLAMA_KEEP_ALIVE", config.OLLAMA_KEEP_ALIVE)
    if isinstance(raw, (int, float)):
        return raw
    try:
        return int(str(raw).strip())
    except ValueError:
        return str(raw).strip()


_OLLAMA_KEEP_ALIVE = _resolve_keep_alive()


def stream_ollama(system: str, user: str, num_ctx: Optional[int] = None,
                  num_predict: Optional[int] = None):
    try:
        import ollama
        for part in ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
            stream=True,
            keep_alive=_OLLAMA_KEEP_ALIVE,
            options={"temperature": config.OLLAMA_TEMPERATURE, "top_p": 0.9,
                     "num_ctx": num_ctx or config.OLLAMA_NUM_CTX,
                     "num_predict": num_predict or config.OLLAMA_NUM_PREDICT},
        ):
            token = part["message"]["content"]
            if token:
                yield token
    except Exception as e:
        logger.error(f"Ollama stream failed: {e}")
        yield f"[LLM Error] Could not reach Ollama ({config.OLLAMA_MODEL}). Error: {e}"


def call_ollama(system: str, user: str, num_ctx=None, num_predict=None) -> str:
    return strip_citations("".join(stream_ollama(system, user, num_ctx, num_predict)))

# ─── Warmup ───────────────────────────────────────────────────────────────────

def warmup():
    def _warm():
        t0 = time.time()
        steps = [
            lambda: ingest.get_embed_model().encode(["warmup"]),
            lambda: embed_query("warmup"),
            lambda: get_reranker().predict([["warmup", "warmup text"]])
                    if getattr(config, "USE_RERANKER", True) else None,
            lambda: __import__("ollama").chat(
                model=config.OLLAMA_MODEL, messages=[{"role": "user", "content": "hi"}],
                stream=False, keep_alive=_OLLAMA_KEEP_ALIVE, options={"num_predict": 1})
                    if getattr(config, "WARMUP_OLLAMA", True) else None,
        ]
        for s in steps:
            try:
                s()
            except Exception as e:
                logger.warning(f"Warmup step failed (non-fatal): {e}")
        logger.info(f"Warmup complete in {time.time() - t0:.2f}s")
    threading.Thread(target=_warm, daemon=True).start()

# ─── Logging ──────────────────────────────────────────────────────────────────

def _format_duration(elapsed: float) -> str:
    m, s = divmod(elapsed, 60)
    return f"{elapsed:.2f} seconds = {int(m)} minutes {s:.2f} seconds"


def log_chat(query: str, answer: str, sources: List[Dict], elapsed: float):
    def _write():
        try:
            os.makedirs(config.LOG_DIR, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(config.CHAT_HISTORY_PATH, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*70}\n[{ts}] (response time: {_format_duration(elapsed)})\n"
                        f"Q: {query}\n\nA: {answer}\n\nSources:\n")
                for s in sources:
                    sc = s.get("rerank_score", s.get("retrieval_score", s.get("score", "?")))
                    sc_txt = f"{sc:.4f}" if isinstance(sc, (int, float)) else str(sc)
                    f.write(f"  - PDF: {s.get('pdf')} | Page: {s.get('page')} | "
                            f"Type: {s.get('source')} | Score: {sc_txt}\n")
        except Exception as e:
            logger.warning(f"Chat logging failed: {e}")
    threading.Thread(target=_write, daemon=True).start()

# ─── Shared pre-LLM routing ───────────────────────────────────────────────────

NOT_FOUND = "😕 No answer is available for this question in the uploaded PDFs."


def _ocr_notice() -> str:
    """A scanned PDF that is still being read (or whose OCR failed) must not look like 'no answer'."""
    for name, j in list(ingest.JOBS.items()):
        if j.get("stage") == "ocr":
            prog = f" ({j['done']}/{j['total']} pages)" if j.get("total") else ""
            return f"⏳ **{name}** is still being read with OCR{prog}. Please ask again in a minute."
    for name, j in list(ingest.JOBS.items()):
        if j.get("stage") == "error":
            return f"⚠️ OCR failed for **{name}**: {j.get('error', 'unknown error')}"
    return ""


def _route(query: str, state: Tuple, use_cache: bool = True):
    index, chunks, _ = state
    fast = try_fast_path(query, chunks)
    if fast:
        return "done", fast
    plan = plan_doc_intent(query, chunks)
    if plan:
        return ("llm", plan) if "llm" in plan else ("done", plan)
    qa = try_qa_fast(query, chunks)
    if qa:
        return "done", qa
    cached = get_cached_answer(query) if use_cache else None
    if cached:
        cached["cached"] = True
        return "done", cached

    ranked = rerank(query, retrieve(query, state))
    if not ranked or ranked[0].get("rerank_score", 0.0) < config.MIN_RELEVANCE_SCORE:
        return "none", {"answer": NOT_FOUND, "sources": []}
    top = _select(ranked, config.TOP_K_RERANK, query)
    if _faq_unsupported(query, top):
        return "none", {"answer": NOT_FOUND, "sources": []}
    direct = try_direct_answer(query, top, chunks)
    if direct:
        if use_cache:
            set_cached_answer(query, direct)
        return "done", direct
    return "rag", top

# ─── Diagnostics: "why did the bot answer like that?" (no LLM call) ──────────

def explain(query: str) -> Dict:
    import traceback
    from collections import Counter
    query = sarvam.clean(query)
    index, chunks, _ = _state
    out: Dict = {"query": query, "tokens": sarvam.tokens(query), "script": sarvam.dominant_script(query),
                 "chunks": len(chunks),
                 "by_source": dict(Counter(c.get("source") for c in chunks)),
                 "directories": dict(Counter(c.get("directory_key") for c in chunks
                                             if c.get("source") == "directory_entry")),
                 "qa_pairs": sum(1 for c in chunks if c.get("source") == "qa_pair")}
    try:
        r = ingest.query_location(query, chunks=chunks)
        out["directory_lookup"] = None if not r else [
            {"directory": m["chunk"].get("directory_key"), "location": m["chunk"].get("location_name"),
             "fields": list((m.get("fields") or {}).keys()) or "full entry",
             "entry_text_tail": m["chunk"]["text"][-120:]} for m in r["matches"]]
    except Exception:
        out["directory_lookup_ERROR"] = traceback.format_exc()
    try:
        q = try_qa_fast(query, chunks)
        out["faq_lookup"] = None if not q else q["sources"][0].get("question")
    except Exception:
        out["faq_lookup_ERROR"] = traceback.format_exc()
    try:
        ranked = rerank(query, retrieve(query, _state))
        out["retrieval_top"] = [{"source": c.get("source"), "page": c.get("page"),
                                 "rerank": round(c.get("rerank_score", 0), 3),
                                 "snippet": c["text"][:90].replace("\n", " ")} for c in ranked[:5]]
    except Exception:
        out["retrieval_ERROR"] = traceback.format_exc()
    return out

# ─── Main RAG entry (blocking) ───────────────────────────────────────────────

def answer(query: str) -> Dict:
    t0 = time.time()
    query = sarvam.clean(query)
    state = _state
    if state[0] is None:
        hit = _instant_only(query, state[1])
        if hit:
            hit = {k: v for k, v in hit.items() if k != "chunks"}
            hit["elapsed"] = round(time.time() - t0, 4)
            return hit
        return {"answer": _index_problem or "No documents have been ingested yet. Please upload and ingest a PDF first.",
                "sources": [], "elapsed": 0.0}

    kind, payload = _route(query, state)

    if kind in ("done", "none"):
        payload = {k: v for k, v in payload.items() if k != "chunks"}
        payload["elapsed"] = round(time.time() - t0, 4)
        if not payload.get("cached"):
            log_chat(query, payload["answer"], payload["sources"], payload["elapsed"])
        return payload

    if kind == "llm":
        text, _ = finalize_answer(call_ollama(*payload["llm"], num_ctx=config.DOC_META_NUM_CTX,
                                              num_predict=config.DOC_META_NUM_PREDICT))
        result = {"answer": text, "sources": payload["sources"], "fast_path": True, "route": "doc"}
        result["elapsed"] = round(time.time() - t0, 4)
        log_chat(query, text, result["sources"], result["elapsed"])
        return result

    top = payload
    system, user = build_messages(query, top)
    response, ok = finalize_answer(call_ollama(system, user, num_predict=_predict_budget(query)))
    if ok and not _grounded(response, "\n".join(c["text"] for c in top)):
        response = _safe_fallback(top, response)

    result = {"answer": response, "route": "rag", "sources": [_src(c, c.get("rerank_score", 0.0)) for c in top]}
    result["elapsed"] = round(time.time() - t0, 2)
    log_chat(query, response, result["sources"], result["elapsed"])
    set_cached_answer(query, result)
    return result

# ─── Streaming entry (/api/chat/stream) ──────────────────────────────────────
# Events: {"instant": ...}  {"token": ...} …  optional {"replace": "<clean text>"}  {"done": true, ...}


def _llm_tokens(system, user, provider="ollama", lang="en-IN", num_ctx=None, num_predict=None):
    if provider == "sarvam":
        yield from sarvam.chat_stream(system + sarvam.language_rule(lang), user)
    else:
        yield from stream_ollama(system, user, num_ctx=num_ctx, num_predict=num_predict)


_LETTER_NAMES = {
    "TAMIL": dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                      "ஏ பி சி டி இ எஃப் ஜி எச் ஐ ஜே கே எல் எம் என் ஓ பி கியூ ஆர் எஸ் டி யூ வி டபிள்யூ எக்ஸ் வை இசட்".split())),
    "DEVANAGARI": dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                           "ए बी सी डी ई एफ जी एच आई जे के एल एम एन ओ पी क्यू आर एस टी यू वी डब्ल्यू एक्स वाई ज़ेड".split())),
}
_CODE_RE = re.compile(r"\b([A-Z])-(\d{1,3})\b")


def _fix_codes(text: str, lang: str) -> str:
    """Visa-style codes (S-1, G-20 ...) are written in the answer's own script, so Tamil / Hindi answers
    never mix 'எஸ்-1' and 'S-2'."""
    want = sarvam.lang_script(lang)
    names = _LETTER_NAMES.get(want)
    if not names or sarvam.script_share(text, want) < 0.3:
        return text
    return _CODE_RE.sub(lambda m: f"{names[m.group(1)]}-{m.group(2)}", text)


def _localize(text: str, enabled: bool, lang: str) -> str:
    """Show `text` in the language the user selected (any source language -> lang).
    `enabled` = Sarvam is the provider (translation needs the Sarvam API).
    The source language of EVERY line is detected from its own script, so a Tamil FAQ answer is
    translated to English / Hindi, English lines are translated to Tamil, and lines that are already
    in the target script are never touched. Phone numbers / e-mails are verified after translation."""
    if (not enabled or not text.strip() or text.lstrip().startswith("[TABLE")
            or "| ---" in text):
        return text
    try:
        out = sarvam.translate_to(text, lang)
    except Exception:
        logger.exception("Translation failed; returning original text")
        return text
    if out != text and not sarvam.facts_preserved(text, out):
        logger.warning("Translation altered a number/e-mail; showing the original text instead")
        out = text
    return _fix_codes(out, lang)


def answer_stream(query: str, provider: str = "ollama", lang: str = "en-IN"):
    t0        = time.time()
    state     = _state
    query     = sarvam.clean(query)               # NFC + invisible chars: same text -> same bytes
    sarvam_on = provider == "sarvam"
    if not sarvam_on:
        lang = "en-IN"                            # Ollama = English only, whatever the UI sends
    non_en    = sarvam_on and lang != "en-IN"
    # translation (Tamil PDF text -> English / Hindi / ...) uses the Sarvam translate API.
    # For Ollama it is only used to turn non-English PDF text into English.
    translate_on = sarvam_on or getattr(config, "OLLAMA_TRANSLATE_TO_ENGLISH", True)
    use_cache = not sarvam_on
    two_step  = non_en and getattr(config, "SARVAM_TRANSLATE_ANSWERS", True)   # English answer -> translate
    want      = sarvam.lang_script(lang)

    if state[0] is None:
        hit = _instant_only(query, state[1])         # FAQ / directory answers work even without vectors
        if hit:
            yield json.dumps({"token": _localize(hit["answer"], translate_on, lang)})
            yield json.dumps({"done": True, "sources": hit["sources"], "elapsed": round(time.time() - t0, 4),
                              "fast_path": True, "route": hit.get("route", ""),
                              "provider": provider, "language": lang})
            return
        yield json.dumps({"token": _index_problem or _localize(_ocr_notice() or "No documents have been ingested yet.", translate_on, lang)})
        yield json.dumps({"done": True, "sources": [], "elapsed": 0.0})
        return

    # multi-part / cross-document questions -> small tool-using agent (see agent.py)
    if getattr(config, "ENABLE_AGENT", True) and detect_doc_intent(query)[0] is None:
        import agent                                   # lazy import avoids a circular import
        if agent.is_complex(query):
            yield from agent.run(query, state, provider, lang)
            return

    _q = {}

    def get_q_en() -> str:
        """English version of the question - translated only when really needed (one API call, cached).
        The question's real language is detected from its own script (not assumed from the dropdown)."""
        if "v" not in _q:
            v = query
            sc  = sarvam.dominant_script(query)
            src = sarvam.script_lang(sc) if sc else None
            if translate_on and src and src != "en-IN":
                try:
                    v = sarvam.translate(query, src, "en-IN")
                    logger.info(f"Query translated: {query!r} -> {v!r}")
                except Exception:
                    logger.exception("Query translation failed; using the original text")
            _q["v"] = v
        return _q["v"]

    # 1) the user's own wording first (matches documents written in the user's language; no network call)
    kind, payload = _route(query, state, use_cache)
    # 2) English translation only as a fallback (matches English directories / FAQs)
    if kind == "none" and translate_on:
        q_en = get_q_en()
        if q_en != query:
            kind, payload = _route(q_en, state, use_cache)

    if kind == "none":
        note = _ocr_notice()
        if note:
            payload = {"answer": note, "sources": []}

    fast_answer, fast_src = None, ""
    if (kind == "done" and payload.get("fast_path") and payload.get("chunks")
            and config.LLM_ON_FAST_PATH and payload.get("route") == "qa"):
        # FAQ pair -> the LLM may rephrase it. Directory cards (phone / e-mail / address) are NEVER sent to the
        # LLM: a 3B model drops fields and the answer changed with the capital letters of the question.
        fast_answer = payload["answer"]
        fast_src    = "\n".join(c["text"] for c in payload["chunks"])
        if not non_en:                        # raw text shown instantly, then replaced by the friendly answer
            yield json.dumps({"instant": payload["answer"], "sources": payload["sources"]})
        kind, payload = "rag", payload["chunks"][:3]

    # Instant answers (FAQ / directory / cache / not-found): same content, shown in the selected language
    if kind in ("done", "none"):
        elapsed = round(time.time() - t0, 4)
        text = _localize(payload["answer"], translate_on, lang)
        if not payload.get("cached"):
            log_chat(query, text, payload["sources"], elapsed)
        yield json.dumps({"token": text})
        yield json.dumps({"done": True, "sources": payload["sources"], "elapsed": elapsed,
                          "fast_path": payload.get("fast_path", False),
                          "cached": payload.get("cached", False), "route": payload.get("route", ""),
                          "provider": provider, "language": lang})
        return

    # ── LLM answer (Ollama, or Sarvam) ──
    def run(system, user, step2, llang, **kw):
        parts = []
        for tok in _llm_tokens(system, user, provider=provider, lang=llang, **kw):
            parts.append(tok)
            if not step2:                          # two-step: show nothing until translated
                yield json.dumps({"token": tok})
        return "".join(parts)

    top = []
    if kind == "llm":
        system, user = payload["llm"]
        sources, route, step2 = payload["sources"], "doc", two_step
        raw = yield from run(system, user, step2, "en-IN" if step2 else lang,
                             num_ctx=config.DOC_META_NUM_CTX, num_predict=config.DOC_META_NUM_PREDICT)
    else:
        top = payload
        ctx = "\n".join(c["text"] for c in top)
        # context already in the user's language -> answer directly in it (no English detour)
        native = bool(want) and want != "LATIN" and sarvam.script_share(ctx, want) > 0.5
        step2  = two_step and not native
        qtxt   = get_q_en() if step2 else query
        system, user = build_messages(qtxt, top)
        sources = [_src(c, c.get("rerank_score", 0.0)) for c in top]
        route = "rag"
        raw = yield from run(system, user, step2, "en-IN" if step2 else lang,
                             num_predict=_predict_budget(qtxt))

    framed = kind != "llm"                                   # summary / page answers keep their own opening
    final, ok = finalize_answer(raw, frame=framed)
    if final == NOT_FOUND:
        final = _localize(_ocr_notice() or final, translate_on, lang)
    else:
        if ok and fast_answer is not None and not _keeps_facts(fast_src, final):
            logger.warning("LLM rewrite lost a number of the FAQ answer -> showing the FAQ answer as it is")
            final, framed = fast_answer, False                # verbatim answer already has its own friendly lines
        elif ok and top and not _grounded(final, "\n".join(c["text"] for c in top)):
            final = _safe_fallback(top, final)
        if ok:
            final = _localize(final, translate_on, lang)     # force the selected language (no-op if already right)
        if ok and framed:
            final = add_frame(final, query, lang, translate_on)

    if step2:
        yield json.dumps({"token": final})
    elif final != raw:
        yield json.dumps({"replace": final})

    elapsed = round(time.time() - t0, 2)
    log_chat(query, final, sources, elapsed)
    if ok and use_cache and route == "rag":
        set_cached_answer(query, {"answer": final, "sources": sources})
    yield json.dumps({"done": True, "sources": sources, "elapsed": elapsed, "route": route,
                      "provider": provider, "language": lang})


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is this document about?"
    r = answer(q)
    print("\nAnswer:", r["answer"])
    print(f"\nSources ({len(r['sources'])}):")
    for s in r["sources"]:
        print(f"  [{s['pdf']} | p.{s['page']}] {s['snippet']}")
    print(f"\nElapsed: {r['elapsed']}s")