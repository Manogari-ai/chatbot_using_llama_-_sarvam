"""agent.py — small tool-using agent for multi-part / cross-document questions.
Plan (JSON) -> run a tool -> repeat (max 3 steps) -> friendly final answer."""
import re
import json
import time
import logging

import config
import rag
import sarvam

logger = logging.getLogger(__name__)

MAX_STEPS  = 3
NOTE_CHARS = 2500

_COMPLEX = re.compile(r"\b(compare|comparison|difference|differences|differ|versus|vs|both|each|"
                      r"every document|all (?:the )?(?:documents|pdfs|files)|which (?:document|pdf|file)|"
                      r"as well as|and also)\b", re.I)


def is_complex(q: str) -> bool:
    return len(q.split()) >= 5 and bool(_COMPLEX.search(q))


PLANNER = (
    "You plan the NEXT step for a document assistant. Reply with ONE JSON object and nothing else.\n"
    'Tools:\n'
    '  {"tool": "search",    "input": "<short search query>"}\n'
    '  {"tool": "tables",    "input": "<document name or empty for all>"}\n'
    '  {"tool": "summary",   "input": "<document name or empty for all>"}\n'
    '  {"tool": "page",      "input": "<document name>|<page number>"}\n'
    '  {"tool": "documents", "input": ""}\n'
    '  {"tool": "final",     "input": ""}   <- use when the NOTES already answer the question\n'
    "Rules: split a multi-part question into one search per part. Never repeat a tool call. "
    "Use document names exactly as listed."
)


def _match_docs(name: str, pdfs):
    n = rag._norm_name(name)
    if not n:
        return pdfs
    hit = [p for p in pdfs if rag._norm_name(p) and (rag._norm_name(p) in n or n in rag._norm_name(p))]
    return hit or pdfs


def _t_search(arg, state):
    ranked = rag.rerank(arg, rag.retrieve(arg, state))
    ranked = [c for c in ranked if c.get("rerank_score", 0.0) >= config.MIN_RELEVANCE_SCORE]
    top = rag._select(ranked, 3, arg)
    return "\n---\n".join(f"[{c['pdf']}] {c['text']}" for c in top) or "(nothing relevant found)", top


def _t_tables(arg, state):
    chunks = state[1]
    text, used = rag._render_tables(_match_docs(arg, rag._all_pdfs(chunks)), chunks)
    return text, used


def _t_summary(arg, state):
    chunks = state[1]
    out = [f"=== {p} ===\n{rag._overview_sample(p, chunks, max_chars=900)}"
           for p in _match_docs(arg, rag._all_pdfs(chunks))]
    return "\n\n".join(out), []


def _t_page(arg, state):
    chunks = state[1]
    doc, _, num = arg.rpartition("|")
    m = re.search(r"\d+", num)
    if not m:
        return "(no page number given)", []
    pg = int(m.group())
    pc = [c for c in chunks if c.get("pdf") in _match_docs(doc, rag._all_pdfs(chunks)) and c.get("page") == pg]
    return ("\n".join(f"[{c['pdf']} p{pg}] {c['text']}" for c in pc) or f"(page {pg} not found)"), pc


def _t_documents(arg, state):
    return "\n".join(rag._all_pdfs(state[1])), []


TOOLS = {"search": _t_search, "tables": _t_tables, "summary": _t_summary,
         "page": _t_page, "documents": _t_documents}


def _ask(system: str, user: str, provider: str) -> str:
    if provider == "sarvam":
        return "".join(sarvam.chat_stream(system, user))
    return "".join(rag.stream_ollama(system, user, num_predict=150))


def _parse(raw: str):
    m = re.search(r"\{.*?\}", raw, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return str(d.get("tool", "")).strip().lower(), str(d.get("input", "")).strip()
    except Exception:
        return None


def run(query: str, state, provider: str = "ollama", lang: str = "en-IN"):
    t0 = time.time()
    pdfs = rag._all_pdfs(state[1])
    notes, sources, seen = [], [], set()
    translate_on = provider == "sarvam"

    for _ in range(MAX_STEPS):
        user = (f"DOCUMENTS: {', '.join(pdfs)}\nQUESTION: {query}\n"
                f"NOTES SO FAR:\n{''.join(notes) or '(none)'}\nNext step?")
        try:
            tool, arg = _parse(_ask(PLANNER, user, provider)) or ("search", query)
        except Exception:
            logger.exception("Planner failed; falling back to plain search")
            tool, arg = "search", query
        if tool == "final":
            break
        if tool not in TOOLS:
            tool, arg = "search", query
        if (tool, arg.lower()) in seen:
            break
        seen.add((tool, arg.lower()))
        logger.info(f"Agent step: {tool}({arg!r})")
        text, used = TOOLS[tool](arg, state)
        notes.append(f"### {tool}({arg})\n{text[:NOTE_CHARS]}\n\n")
        sources += [rag._src(c, c.get("rerank_score", 0.0)) for c in used]

    if not notes:                                   # planner said "final" at once -> still search
        text, used = _t_search(query, state)
        notes.append(f"### search({query})\n{text[:NOTE_CHARS]}\n\n")
        sources += [rag._src(c, c.get("rerank_score", 0.0)) for c in used]

    ctx  = "".join(notes)
    user = f"CONTEXT:\n{ctx}\n\nQUESTION:\n{query}"
    parts = []
    for tok in rag._llm_tokens(rag.SYSTEM_PROMPT, user, provider=provider, lang=lang,
                               num_ctx=config.DOC_META_NUM_CTX,
                               num_predict=config.OLLAMA_NUM_PREDICT_LONG):
        parts.append(tok)
        yield json.dumps({"token": tok})
    raw = "".join(parts)

    final, ok = rag.finalize_answer(raw, frame=True)
    if final == rag.NOT_FOUND:
        final = rag._localize(final, translate_on, lang)
    elif ok:
        if not rag._grounded(final, ctx):
            final = rag._safe_fallback([], final)
        final = rag._localize(final, translate_on, lang)
        final = rag.add_frame(final, query, lang, translate_on)
    if final != raw:
        yield json.dumps({"replace": final})

    elapsed = round(time.time() - t0, 2)
    rag.log_chat(query, final, sources, elapsed)
    yield json.dumps({"done": True, "sources": sources, "elapsed": elapsed, "route": "agent",
                      "provider": provider, "language": lang})