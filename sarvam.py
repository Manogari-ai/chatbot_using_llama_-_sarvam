"""sarvam.py — thin client for Sarvam AI: chat (streaming), translate, speech-to-text, text-to-speech."""
import re
import json
import logging
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import requests
import config

logger = logging.getLogger(__name__)


class SarvamError(RuntimeError):
    pass


def _headers(json_body: bool = True) -> dict:
    if not config.SARVAM_API_KEY:
        raise SarvamError("SARVAM_API_KEY is not set on the server")
    h = {"api-subscription-key": config.SARVAM_API_KEY}
    if json_body:
        h["Content-Type"] = "application/json"
    return h


def _check(r: requests.Response):
    if not r.ok:
        raise SarvamError(f"Sarvam API {r.status_code}: {r.text[:300]}")


# ─── Text helpers (any language / script) ─────────────────────────────────────
# Python's \w+ does not treat Indic vowel signs / pulli as letters, so "தமிழ்நாட்டின்" is cut into
# fragments. A token here = a run of Letters + Marks + Numbers, which is right for EVERY script.

_ZERO = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None)

# language code (before "-") -> Unicode script. Add a line to config.LANG_SCRIPT = {"ur": "ARABIC"} for a
# new language; if a language is missing everywhere the script check is simply switched off.
_DEFAULT_LANG_SCRIPT = {
    "en": "LATIN", "ta": "TAMIL", "te": "TELUGU", "kn": "KANNADA", "ml": "MALAYALAM",
    "bn": "BENGALI", "gu": "GUJARATI", "pa": "GURMUKHI", "od": "ORIYA",
    "hi": "DEVANAGARI", "mr": "DEVANAGARI",
}


def lang_script(lang: str):
    m = {**_DEFAULT_LANG_SCRIPT, **getattr(config, "LANG_SCRIPT", {})}
    return m.get(lang.split("-")[0])


def clean(t: str) -> str:
    """NFC-normalise + drop invisible characters, so equal text always has equal code points."""
    return unicodedata.normalize("NFC", t).translate(_ZERO).replace("\u00a0", " ").strip()


def tokens(text: str):
    """Lower-cased word tokens for ANY script."""
    out, cur = [], []
    for ch in clean(text).lower():
        if unicodedata.category(ch)[0] in "LMN":          # Letter / Mark / Number
            cur.append(ch)
        elif cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out


def _script(ch: str):
    try:
        return unicodedata.name(ch).split()[0]            # "TAMIL LETTER A" -> "TAMIL"
    except ValueError:
        return None


def dominant_script(text: str):
    """'TAMIL', 'LATIN', 'DEVANAGARI' ... or None when the text has no letters."""
    c = Counter(_script(ch) for ch in text if ch.isalpha())
    c.pop(None, None)
    return c.most_common(1)[0][0] if c else None


def script_share(text: str, script: str) -> float:
    """Fraction (0..1) of the letters of `text` that belong to `script`."""
    letters = [ch for ch in text if ch.isalpha()]
    return sum(_script(ch) == script for ch in letters) / len(letters) if letters else 0.0


def wrong_script(text: str, lang: str) -> bool:
    """True if `text` has letters that are neither the target language's script nor Latin."""
    want = lang_script(lang)
    if not want:
        return False                                      # unknown language: never block
    return any(ch.isalpha() and _script(ch) not in (want, "LATIN", None) for ch in text)


def language_rule(lang: str) -> str:
    name = config.LANGUAGES.get(lang, "English")
    return (f"\n- Write your ENTIRE answer in {name}. Keep phone numbers, e-mail addresses, URLs, "
            f"acronyms and proper names exactly as they appear in the CONTEXT."
            f"\n- Exception: if the CONTEXT does not contain the answer, reply with exactly NOT_FOUND "
            f"(English letters, nothing else - do not translate it).")

# ─── Chat (OpenAI-compatible, streaming) ──────────────────────────────────────

_OPEN, _CLOSE = "<think>", "</think>"


def _strip_think(buf: str, in_think: bool):
    """Remove <think>…</think> blocks from a token stream (tags may be split across tokens)."""
    out = ""
    while True:
        if in_think:
            j = buf.find(_CLOSE)
            if j == -1:
                return out, buf[-len(_CLOSE):], True
            buf, in_think = buf[j + len(_CLOSE):], False
        else:
            i = buf.find(_OPEN)
            if i != -1:
                out += buf[:i]
                buf, in_think = buf[i + len(_OPEN):], True
                continue
            keep = 0
            for n in range(min(len(_OPEN) - 1, len(buf)), 0, -1):
                if _OPEN.startswith(buf[-n:]):
                    keep = n
                    break
            return out + buf[:len(buf) - keep], buf[len(buf) - keep:], False


def _attempt(model, system, user, max_tokens, effort, state):
    payload = {"model": model, "stream": True, "temperature": 0.2, "max_tokens": max_tokens,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}]}
    if effort:
        payload["reasoning_effort"] = effort
    with requests.post(f"{config.SARVAM_BASE_URL}/v1/chat/completions", headers=_headers(),
                       json=payload, stream=True, timeout=(10, 180)) as r:
        _check(r)
        r.encoding = "utf-8"
        buf, in_think = "", False
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                choice = json.loads(data)["choices"][0]
            except (ValueError, KeyError, IndexError):
                continue
            state["finish"] = choice.get("finish_reason") or state["finish"]
            buf += (choice.get("delta") or {}).get("content") or ""
            out, buf, in_think = _strip_think(buf, in_think)
            if out:
                state["emitted"] = state["emitted"] or bool(out.strip())
                yield out
        if buf and not in_think:
            state["emitted"] = state["emitted"] or bool(buf.strip())
            yield buf
        state["thinking_unclosed"] = in_think


def chat_stream(system: str, user: str, max_tokens: int = None):
    max_tokens = max_tokens or config.SARVAM_MAX_TOKENS
    models = [(config.SARVAM_CHAT_MODEL, config.SARVAM_REASONING_EFFORT)]
    fb = getattr(config, "SARVAM_FALLBACK_MODEL", None)
    if fb and fb != config.SARVAM_CHAT_MODEL:
        models.append((fb, None))

    problems = []
    for model, effort in models:
        state = {"emitted": False, "finish": None, "thinking_unclosed": False}
        try:
            yield from _attempt(model, system, user, max_tokens, effort, state)
        except Exception as e:
            problems.append(f"{model}: {e}")
            logger.error(f"Sarvam {problems[-1]}")
            if state["emitted"]:
                return                      # part of an answer is already on screen
            continue
        if state["emitted"]:
            return
        problems.append(f"{model}: no answer text (finish_reason={state['finish']})")
        logger.warning(f"Sarvam {problems[-1]} - trying next model")

    yield "[LLM Error] Sarvam gave no answer. " + " | ".join(problems)

# ─── Translate ────────────────────────────────────────────────────────────────

def _translate_once(text: str, src: str, tgt: str, model: str) -> str:
    body = {"input": text, "source_language_code": src,
            "target_language_code": tgt, "model": model}
    if model.startswith("mayura"):
        body["enable_preprocessing"] = False
    r = requests.post(f"{config.SARVAM_BASE_URL}/translate", headers=_headers(),
                      timeout=30, json=body)
    _check(r)
    return r.json()["translated_text"]


def translate(text: str, src: str, tgt: str) -> str:
    if not text.strip() or src == tgt:
        return text
    models = [config.SARVAM_TRANSLATE_MODEL]
    alt = getattr(config, "SARVAM_TRANSLATE_FALLBACK_MODEL", None)
    if alt and alt not in models:
        models.append(alt)
    for model in models:
        for _ in range(2):                                  # same model twice, then next model
            try:
                out = _translate_once(text, src, tgt, model)
            except Exception as e:
                logger.warning(f"translate {model} failed: {e}")
                continue
            if not wrong_script(out, tgt):
                return out
            logger.warning(f"{model} returned wrong script for {tgt}: {out[:60]!r}")
    return text            # better clean English than a mix of two scripts


def translate_long(text: str, src: str, tgt: str, limit: int = 900) -> str:
    """Translate in pieces below the API's per-request limit."""
    pieces, cur = [], ""
    for line in text.split("\n"):
        if cur and len(cur) + len(line) + 1 > limit:
            pieces.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        pieces.append(cur)
    return "\n".join(translate(p, src, tgt) for p in pieces)


_PREFIX_RE = re.compile(
    r"^(\s*(?:[-*•]\s+|\d{1,2}[.)]\s+|#{1,6}\s+)?(?:[\U0001F300-\U0001FAFF\u2600-\u27BF]\ufe0f?\s*)*)(.*)$")


def translate_markdown(text: str, src: str, tgt: str) -> str:
    """Translate line by line so bullets, emojis, line breaks and paragraphs survive."""
    parts = []
    for ln in text.split("\n"):
        if not ln.strip() or re.fullmatch(r"[\s\-|:]{3,}", ln):
            parts.append((ln, None, ""))
            continue
        m = _PREFIX_RE.match(ln)
        prefix = m.group(1)
        body = m.group(2).replace("**", "").replace("__", "").strip()
        trail = "  " if ln.endswith("  ") else ""
        parts.append((prefix, body, trail) if re.search(r"[A-Za-z]", body) else (ln, None, ""))

    with ThreadPoolExecutor(max_workers=6) as pool:
        done = list(pool.map(lambda p: translate_long(p[1], src, tgt) if p[1] else None, parts))
    return "\n".join((p[0] + t + p[2]) if t is not None else p[0] for p, t in zip(parts, done))


# ─── Any-language -> selected language (used by rag._localize) ────────────────

def script_lang(script: str):
    """'TAMIL' -> 'ta-IN', 'LATIN' -> 'en-IN', 'DEVANAGARI' -> 'hi-IN' ..."""
    for code in config.LANGUAGES:
        if lang_script(code) == script:
            return code
    return None


def translate_any(text: str, src: str, tgt: str) -> str:
    """Any language -> any language. Indic -> Indic goes through English."""
    if not text.strip() or src == tgt:
        return text
    if src != "en-IN" and tgt != "en-IN":
        mid = translate_long(text, src, "en-IN")
        if mid == text:                       # first hop failed -> keep original
            return text
        return translate_long(mid, "en-IN", tgt)
    return translate_long(text, src, tgt)


def translate_to(text: str, tgt: str) -> str:
    """Make `text` appear in language `tgt`. The source language of EVERY line is detected from
    its own script, so Tamil lines and English lines in the same answer are both handled."""
    want = lang_script(tgt)
    parts = []                                 # (prefix, body, trail, src)  body=None -> keep line as is
    for ln in text.split("\n"):
        if not ln.strip() or re.fullmatch(r"[\s\-|:]{3,}", ln):
            parts.append((ln, None, "", None))
            continue
        m = _PREFIX_RE.match(ln)
        prefix = m.group(1)
        body = m.group(2).replace("**", "").replace("__", "").strip()
        trail = "  " if ln.endswith("  ") else ""
        sc = dominant_script(body)
        src = script_lang(sc) if sc else None
        if not src or sc == want or src == tgt:      # no letters / already in the right language
            parts.append((ln, None, "", None))
        else:
            parts.append((prefix, body, trail, src))

    with ThreadPoolExecutor(max_workers=6) as pool:
        done = list(pool.map(lambda p: translate_any(p[1], p[3], tgt) if p[1] else None, parts))
    return "\n".join((p[0] + t + p[2]) if t is not None else p[0] for p, t in zip(parts, done))


def _ascii_digits(s: str) -> str:
    return "".join(str(unicodedata.decimal(c)) for c in s if c.isdigit())


def facts_preserved(src: str, out: str) -> bool:
    """True if every e-mail and every 5+ digit number of `src` still appears in `out`."""
    for e in re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", src):
        if e.lower() not in out.lower():
            return False
    out_digits = _ascii_digits(out)
    for n in re.findall(r"\d[\d\s\-]{4,}\d", src):
        d = _ascii_digits(n)
        if len(d) >= 5 and d not in out_digits:
            return False
    return True

# ─── Speech to text ───────────────────────────────────────────────────────────

def speech_to_text(audio: bytes, filename: str, mimetype: str, lang: str):
    r = requests.post(f"{config.SARVAM_BASE_URL}/speech-to-text", headers=_headers(False), timeout=60,
                      files={"file": (filename, audio, mimetype)},
                      data={"model": config.SARVAM_STT_MODEL, "language_code": lang or "unknown"})
    _check(r)
    j = r.json()
    return clean(j.get("transcript", "")), j.get("language_code")

# ─── Text to speech ───────────────────────────────────────────────────────────

def speakable(text: str) -> str:
    t = re.sub(r"```.*?```", " ", text, flags=re.S)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u27BF\ufe0f]", "", t)
    t = re.sub(r"[*_`#>|]+|-{3,}", " ", t)
    lines = []
    for ln in t.split("\n"):
        ln = re.sub(r"^\s*[-•]\s+", "", ln).strip()
        if ln:
            lines.append(ln if re.search(r"[.!?।॥:]$", ln) else ln + ".")
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def split_for_tts(text: str, limit: int = 400):
    chunks, cur = [], ""
    for s in re.split(r"(?<=[.!?।॥])\s+", text):
        if cur and len(cur) + len(s) + 1 > limit:
            chunks.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur)
    out = []
    for c in chunks:
        while len(c) > limit:
            out.append(c[:limit])
            c = c[limit:]
        if c:
            out.append(c)
    return out


def text_to_speech(text: str, lang: str, speaker: str = None):
    """-> list of base64 WAV strings, in playback order."""
    chunks = split_for_tts(speakable(text)[:config.TTS_MAX_CHARS])

    def one(c):
        r = requests.post(f"{config.SARVAM_BASE_URL}/text-to-speech", headers=_headers(), timeout=60,
                          json={"text": c, "target_language_code": lang,
                                "speaker": speaker or config.SARVAM_TTS_SPEAKER,
                                "model": config.SARVAM_TTS_MODEL, "pace": 1.0,
                                "speech_sample_rate": 22050, "enable_preprocessing": True})
        _check(r)
        return r.json()["audios"][0]

    with ThreadPoolExecutor(max_workers=3) as pool:
        return list(pool.map(one, chunks))