"""
ingest.py — PDF ingestion pipeline (fast + accurate)

* Stage 1 (seconds): PyMuPDF text + tables -> chunks -> embeddings -> FAISS. Searchable at once.
* Stage 2 (background): OCR of SCANNED pages + embedded images (parallel), then re-commit.
  - works with RapidOCR (English), EasyOCR / PaddleOCR / Tesseract (English + Tamil + Hindi ...)
  - progress is visible in JOBS[name] ({"stage": "ocr", "done": 12, "total": 57}); errors are NEVER silent
* Every directory in a PDF (FRRO, POE, ...) is parsed as its OWN section with its own key,
  so "poe jaipur" and "frro jaipur" can never return each other's entry.

Everything about directories is learned from the PDF text itself (no fixed names / aliases).
"""

import os
import re
import io
import json
import time
import bisect
import hashlib
import logging
import threading
import warnings
import importlib.util
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Tuple
from collections import Counter

import fitz                          # PyMuPDF
import numpy as np
from PIL import Image
import sarvam
import config

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Singletons / shared state ────────────────────────────────────────────────

_ocr_engine     = None
_embed_model    = None
_singleton_lock = threading.Lock()
_ocr_lock       = threading.Lock()
_index_lock     = threading.RLock()
on_commit       = None                 # app.py sets this to rag.load_index
JOBS: Dict[str, Dict] = {}             # upload progress, read by /api/ingest-status/<name>

# ─── OCR engines ──────────────────────────────────────────────────────────────
# config.OCR_ENGINE = "auto" (default) | "rapidocr" | "easyocr" | "paddle" | "tesseract"
# config.OCR_LANGS  = ["en"]  or ["ta", "en"]  or ["hi", "en"]   (falls back to [config.OCR_LANG])
#   rapidocr  : English / Chinese only, fastest        pip install rapidocr onnxruntime
#   easyocr   : English + Tamil / Hindi / Telugu ...   pip install easyocr
#   paddle    : many languages                         pip install paddleocr paddlepaddle
#   tesseract : many languages (needs tesseract.exe)   pip install pytesseract
# "auto" picks the first INSTALLED engine that supports the languages.

_ENGINE_MODULES = {"rapidocr": ("rapidocr", "rapidocr_onnxruntime"), "easyocr": ("easyocr",),
                   "paddle": ("paddleocr",), "tesseract": ("pytesseract",)}
_PADDLE_LANG = {"kn": "ka"}
_TESS_LANG   = {"en": "eng", "ta": "tam", "hi": "hin", "te": "tel", "kn": "kan", "ml": "mal",
                "bn": "ben", "mr": "mar", "gu": "guj", "pa": "pan", "ur": "urd"}

OCR_INSTALL_HELP = ("A scanned PDF needs an OCR engine, and none is installed. Install ONE of these:\n"
                    "  English only :  pip install rapidocr onnxruntime\n"
                    "  Tamil / Hindi / English :  pip install easyocr\n"
                    "then restart the app and upload the PDF again.")


def _ocr_langs() -> List[str]:
    langs = getattr(config, "OCR_LANGS", None) or [getattr(config, "OCR_LANG", None) or "en"]
    return [str(l).lower() for l in langs]


def _engine_order() -> List[str]:
    pref = str(getattr(config, "OCR_ENGINE", "auto")).lower()
    if pref != "auto":
        return [pref]
    english_only = all(l in ("en", "latin", "ch") for l in _ocr_langs())
    return (["rapidocr"] if english_only else []) + ["easyocr", "paddle", "tesseract"]


def _installed(kind: str) -> bool:
    return any(importlib.util.find_spec(m) is not None for m in _ENGINE_MODULES.get(kind, ()))


def ocr_available() -> Optional[str]:
    """Name of the first installed OCR engine (cheap check - nothing is loaded), or None."""
    return next((k for k in _engine_order() if _installed(k)), None)


def _init_engine(kind: str):
    langs = _ocr_langs()
    if kind == "rapidocr":
        try:
            from rapidocr import RapidOCR                       # new package (works on Python 3.13)
        except ImportError:
            from rapidocr_onnxruntime import RapidOCR           # old package
        return RapidOCR()
    if kind == "easyocr":
        import easyocr
        return easyocr.Reader(langs, gpu=bool(getattr(config, "OCR_USE_GPU", False)))
    if kind == "paddle":
        from paddleocr import PaddleOCR
        lang = _PADDLE_LANG.get(langs[0], langs[0])
        try:                                                    # PaddleOCR 2.x
            return PaddleOCR(use_angle_cls=True, lang=lang,
                             use_gpu=bool(getattr(config, "OCR_USE_GPU", False)), show_log=False)
        except (TypeError, ValueError):                         # PaddleOCR 3.x
            return PaddleOCR(lang=lang, use_doc_orientation_classify=False,
                             use_doc_unwarping=False, use_textline_orientation=False)
    if kind == "tesseract":
        import pytesseract
        pytesseract.get_tesseract_version()                     # raises if tesseract.exe is missing
        return pytesseract
    raise ValueError(f"Unknown OCR engine '{kind}'")


def get_ocr():
    """-> (kind, engine). Loads the first engine that really works; clear error if none does."""
    global _ocr_engine
    if _ocr_engine is None:
        with _singleton_lock:
            if _ocr_engine is None:
                errors = []
                for kind in _engine_order():
                    if not _installed(kind):
                        continue
                    try:
                        _ocr_engine = (kind, _init_engine(kind))
                        logger.info(f"OCR engine ready: {kind}  languages={_ocr_langs()}")
                        break
                    except Exception as e:
                        errors.append(f"{kind}: {e}")
                        logger.warning(f"OCR engine '{kind}' failed to start: {e}")
                if _ocr_engine is None:
                    raise RuntimeError(("OCR engine could not start (" + " | ".join(errors) + "). "
                                        if errors else "") + OCR_INSTALL_HELP)
    return _ocr_engine


def _bbox(points) -> Tuple[float, float, float, float]:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _items_to_text(items: List[Tuple[Tuple[float, float, float, float], str]]) -> str:
    """OCR words / boxes -> text with REAL LINES (directory + FAQ detection depends on line breaks)."""
    items = [(b, t.strip()) for b, t in items if t and t.strip()]
    if not items:
        return ""
    heights = sorted(max(b[3] - b[1], 1.0) for b, _ in items)
    med = heights[len(heights) // 2]
    items.sort(key=lambda it: (it[0][1] + it[0][3]) / 2)
    lines, cur, cy = [], [], 0.0
    for b, t in items:
        c = (b[1] + b[3]) / 2
        if cur and abs(c - cy) > 0.6 * med:
            lines.append(cur)
            cur = []
        if not cur:
            cy = c
        cur.append((b[0], t))
    if cur:
        lines.append(cur)
    return "\n".join(" ".join(t for _, t in sorted(line)) for line in lines)


def _run_ocr(kind: str, eng, arr: np.ndarray) -> str:
    if kind == "rapidocr":
        out = eng(arr)
        if hasattr(out, "txts"):                                # new rapidocr
            if out.txts is None or out.boxes is None:
                return ""
            return _items_to_text([(_bbox(b), t) for b, t in zip(out.boxes, out.txts)])
        res = out[0] if isinstance(out, tuple) else out         # old rapidocr_onnxruntime
        return _items_to_text([(_bbox(r[0]), r[1]) for r in (res or [])])
    if kind == "easyocr":
        return _items_to_text([(_bbox(b), t) for b, t, _ in eng.readtext(arr)])
    if kind == "paddle":
        if hasattr(eng, "predict"):                             # PaddleOCR 3.x
            items = []
            for r in eng.predict(arr):
                texts = r.get("rec_texts") or []
                polys = r.get("rec_polys")
                if polys is None:
                    polys = r.get("rec_boxes")
                if polys is None:
                    continue
                items += [(_bbox(np.array(p).reshape(-1, 2)), t) for p, t in zip(polys, texts)]
            return _items_to_text(items)
        res = eng.ocr(arr, cls=True)                            # PaddleOCR 2.x
        lines = res[0] if res and res[0] else []
        return _items_to_text([(_bbox(l[0]), l[1][0]) for l in lines if l and l[1]])
    if kind == "tesseract":
        lang = "+".join(_TESS_LANG.get(l, l) for l in _ocr_langs())
        return eng.image_to_string(Image.fromarray(arr), lang=lang)
    return ""


def ocr_array(arr: np.ndarray) -> str:
    kind, eng = get_ocr()
    if kind == "rapidocr":                                      # onnx sessions are safe to share
        return _run_ocr(kind, eng, arr)
    with _ocr_lock:                                             # torch / paddle / tesseract: one at a time
        return _run_ocr(kind, eng, arr)


def get_embed_model():
    """THE single embedding model for the whole app (rag.py reuses this one)."""
    global _embed_model
    if _embed_model is None:
        with _singleton_lock:
            if _embed_model is None:
                from sentence_transformers import SentenceTransformer
                m = SentenceTransformer(config.EMBED_MODEL, device=config.DEVICE)
                m.max_seq_length = config.EMBED_MAX_SEQ
                _embed_model = m
                logger.info(f"Embedding model loaded: {config.EMBED_MODEL} on {config.DEVICE}")
    return _embed_model


def embed_texts(texts: List[str]) -> np.ndarray:
    emb = get_embed_model().encode(texts, batch_size=64, convert_to_numpy=True,
                                   normalize_embeddings=True, show_progress_bar=False)
    return emb.astype(np.float32)

# ─── Stage 1: text + tables (fast, no OCR) ────────────────────────────────────

_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None)


def clean_text(t: str) -> str:
    t = sarvam.clean(t)                    # NFC + zero-width chars + nbsp
    return re.sub(r"[\uf000-\uf8ff]", "•", t)


def _rows_to_markdown(rows) -> str:
    clean = [[(c or "").replace("\n", " ").replace("|", "/").strip() for c in r] for r in rows]
    width = max(len(r) for r in clean)
    clean = [r + [""] * (width - len(r)) for r in clean]
    out = ["| " + " | ".join(clean[0]) + " |",
           "| " + " | ".join(["---"] * width) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in clean[1:]]
    return "\n".join(out)


def _table_rows(page, t) -> List[List[str]]:
    """Cell text via clip= (keeps the spaces that t.extract() loses), then tidy merged-cell layouts."""
    rows = []
    for row in t.rows:
        rows.append(["" if cb is None else " ".join(page.get_text("text", clip=fitz.Rect(cb)).split())
                     for cb in row.cells])
    rows = [r for r in rows if any(r)]
    if not rows:
        return rows
    w    = max(len(r) for r in rows)
    keep = [j for j in range(w) if any(j < len(r) and r[j] for r in rows)]      # drop empty columns
    rows = [[r[j] if j < len(r) else "" for j in keep] for r in rows]
    modal = Counter(sum(1 for c in r if c) for r in rows).most_common(1)[0][0]
    if modal and len(keep) > modal:               # merged cells -> compact each row to its filled cells
        compact = [[c for c in r if c] for r in rows]
        rows = []
        for r in compact:
            if rows and len(r) < modal:           # continuation line -> glue to the row above
                rows[-1][-1] = (rows[-1][-1] + " " + " ".join(r)).strip()
            else:
                rows.append(r)
    return rows


def _is_scanned(page, text: str) -> bool:
    """No text layer -> scanned. A page with only a few words but ONE big picture is a scan too
    (stamp / header text on top of a scanned image)."""
    n = len(text)
    if n < 30:
        return True
    if n < 200:
        try:
            area = page.rect.get_area() or 1.0
            for info in page.get_image_info():
                if (fitz.Rect(info["bbox"]) & page.rect).get_area() / area > 0.6:
                    return True
        except Exception:
            pass
    return False


def extract_fast(pdf_path: str):
    """One pass: text + tables. Returns (pages, tables, scanned_page_numbers, has_images)."""
    doc = fitz.open(pdf_path)
    pages, tables, scanned, n_t, has_images = [], [], [], 0, False
    for pn, page in enumerate(doc, 1):
        text = clean_text(page.get_text("text")).strip()
        pages.append({"page": pn, "text": text, "source": "pymupdf"})
        if _is_scanned(page, text):
            scanned.append(pn)
        elif config.ENABLE_TABLE_EXTRACTION and len(page.get_drawings()) >= 4:
            # find_tables() is the slow call; ruled tables always have >=4 line/rect drawings,
            # so pages without them are skipped (0.1 s scan instead of ~15 s on 80 pages)
            try:
                for t in page.find_tables().tables:
                    rows = _table_rows(page, t)
                    if len(rows) < 2 or max(len(r) for r in rows) < 2:
                        continue
                    n_t += 1
                    tables.append({"page": pn, "source": "table",
                                   "text": f"[TABLE {n_t}]\n{_rows_to_markdown(rows)}"})
            except Exception as e:
                logger.warning(f"Table scan p{pn}: {e}")
        if not has_images and page.get_images():
            has_images = True
    doc.close()
    return pages, tables, scanned, has_images

# ─── Stage 2: OCR scanned pages + embedded images (parallel) ──────────────────

def _caption(path: str) -> str:
    import ollama
    r = ollama.chat(model=config.VISION_MODEL, options={"num_predict": 120, "temperature": 0},
                    messages=[{"role": "user", "images": [path],
                               "content": "Describe this image in two sentences. Transcribe any numbers or labels."}])
    return r["message"]["content"].strip()


def _render_page(doc, pn: int, scale: float) -> np.ndarray:
    pix = doc[pn - 1].get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n).copy()


def extract_ocr(pdf_path: str, img_out: str, scanned_pages: List[int], progress=None):
    """OCR every scanned page (rendered a few at a time, so 500 pages never fill the RAM) and the big
    embedded images of the other pages. progress(done, total) is called after every batch.
    Raises if the OCR engine cannot read ANY page, so a broken engine is never silent."""
    os.makedirs(img_out, exist_ok=True)
    stem, doc = Path(pdf_path).stem, fitz.open(pdf_path)
    scale   = float(getattr(config, "OCR_SCALE", 2.0))
    workers = max(1, int(getattr(config, "OCR_WORKERS", 2)))
    total   = len(scanned_pages)
    skip    = set(scanned_pages)
    ocr_pages, errors, image_records = {}, [], []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # --- scanned pages -------------------------------------------------
        batch = max(2, workers * 2)
        for b in range(0, total, batch):
            part = scanned_pages[b:b + batch]
            futs = [(pn, pool.submit(ocr_array, _render_page(doc, pn, scale))) for pn in part]
            for pn, fut in futs:
                try:
                    t = fut.result()
                    if len(t.strip()) >= 3:
                        ocr_pages[pn] = t
                except Exception as e:
                    errors.append((pn, str(e)))
                    logger.warning(f"OCR page {pn}: {e}")
            if progress:
                progress(min(b + batch, total), total)
            logger.info(f"OCR pages {min(b + batch, total)}/{total}")
        if total and not ocr_pages and errors:
            doc.close()
            raise RuntimeError(f"OCR failed on all {total} pages. First error: {errors[0][1]}")

        # --- images embedded in text pages --------------------------------
        img_jobs, seen = [], set()
        if config.EXTRACT_EMBEDDED_IMAGES:
            for pn, page in enumerate(doc, 1):
                if pn in skip or len(img_jobs) >= config.MAX_IMAGE_OCR:
                    continue                   # a scanned page's big image is already OCR'd
                for k, info in enumerate(page.get_images(full=True)):
                    xref = info[0]
                    if xref in seen or info[2] * info[3] < config.MIN_IMAGE_OCR_AREA:
                        continue
                    seen.add(xref)
                    try:
                        base = doc.extract_image(xref)
                        path = os.path.join(img_out, f"{stem}_p{pn}_img{k}.{base['ext']}")
                        with open(path, "wb") as f:
                            f.write(base["image"])
                        img_jobs.append((pn, path, base["image"]))
                    except Exception as e:
                        logger.warning(f"Image p{pn}: {e}")
                    if len(img_jobs) >= config.MAX_IMAGE_OCR:
                        break
        doc.close()

        def _img(path, data):
            txt = ocr_array(np.array(Image.open(io.BytesIO(data)).convert("RGB")))
            if len(txt.strip()) < 15 and config.VISION_MODEL:
                txt = _caption(path)
            return txt

        imf = [(pn, path, pool.submit(_img, path, data)) for pn, path, data in img_jobs]
        for pn, path, fut in imf:
            try:
                t = fut.result()
                if t.strip():
                    image_records.append({"page": pn, "source": "image_ocr", "image_path": path,
                                          "text": f"[IMAGE p{pn}] {t}"})
            except Exception as e:
                logger.warning(f"Image OCR p{pn}: {e}")
    return ocr_pages, image_records

# ─── Directory detection — SCHEMA-FREE ────────────────────────────────────────
# Nothing here knows about FRRO, POE, "Location", "Jurisdiction" ... Everything is learned from
# the PDF text itself:
#   entry   = a header line ("1. Delhi Office", or a plain "Mumbai" line) followed by >=2 "Label: value" lines
#   section = consecutive entries; it ends at the next ALL-CAPS heading / Q:A: marker / numbering reset
#   title   = the heading line just above the first entry (any wording: "...DIRECTORY", "HELPLINE NUMBERS", ...)
#   key     = acronym in the title "(POE)" / first ALL-CAPS word / first meaningful word
#   labels  = whatever "Label:" words the entries really use (Phone, Mobile, Tel, Mail, Head, Zone, ...)
#   end     = the last entry stops where it leaves the pattern the other entries follow
#             (unknown label, heading, question, extra wrapped lines, text after the final e-mail)

_STOP    = {"of", "the", "and", "for", "to", "in", "at", "a", "an", "on"}
_GENERIC = _STOP | {"directory", "directories", "list", "lists", "contact", "contacts", "office", "offices",
                    "details", "detail", "information", "info", "address", "addresses", "branch",
                    "branches", "department"}
_LABEL_LINE_RE  = re.compile(r'^\s*(?:[o•·▪●◦\-*–]\s+)?([A-Za-z][A-Za-z0-9 .\-/&()]{1,28}?)\s*:\s*(.*?)\s*$')
_BULLET_ONLY_RE = re.compile(r'^[o•·▪●◦\-*–]$')
_NUM_HEAD_RE    = re.compile(r'^\s*(\d{1,3})\s*[.)]\s*(.*?)\s*$')
_NAME_OK_RE     = re.compile(r"^[A-Za-z][A-Za-z .\-/()&']{1,58}$")
_QA_RE          = re.compile(r'^\s*(?:[•·▪●◦\-*–]\s*)?(?:Q|A|Ans|Question|Answer)\s*[:.]')
_EMAIL_RE       = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')

_EM            = r'[\w.+-]+@[\w-]+\.[\w.-]+'
_EMAIL_LIST_RE = re.compile(rf'(?:{_EM}(?:\s*\([^)]{{0,40}}\))?\s*(?:[;,/]|\bor\b|\band\b)?\s*)+', re.I)
_TAIL_SEP_RE   = re.compile(r'(?:\s*(?:[;,/|]|\bor\b|\band\b))+\s*$', re.I)


def _lines(text: str):
    out, pos = [], 0
    for raw in text.split("\n"):
        out.append((pos, pos + len(raw), raw.strip()))
        pos += len(raw) + 1
    return out


def _label_of(s: str):
    m = _LABEL_LINE_RE.match(s)
    if not m:
        return None
    label, val = m.group(1).strip(), m.group(2)
    if len(label.split()) > 4 or val.startswith("//"):          # sentences / URLs are not labels
        return None
    return label, val


def _is_caps_heading(s: str) -> bool:
    if not s or ":" in s or len(s) > 90 or _BULLET_ONLY_RE.match(s) or _NUM_HEAD_RE.match(s):
        return False
    letters = [c for c in s if c.isalpha()]
    return (len(letters) >= 6 and len(s.split()) <= 12
            and sum(c.isupper() for c in letters) / len(letters) >= 0.8)


def _is_title_heading(s: str) -> bool:
    ws = s.split()
    if (not s or ":" in s or "," in s or s[-1] in ".;" or not (2 <= len(ws) <= 8)
            or _NUM_HEAD_RE.match(s) or _BULLET_ONLY_RE.match(s)):
        return False
    core = [w for w in ws if w.lower() not in _GENERIC and any(c.isalpha() for c in w)]
    return bool(core) and all(w.lstrip('("\'“‘[')[:1].isupper() for w in core)


def _norm_label(label: str) -> str:
    """Light normalisation so 'In charge' / 'Incharge' / 'In-Charge' and 'E-mail' / 'Email' agree."""
    l = re.sub(r"\s+", " ", label.strip().lower())
    squashed = re.sub(r"[^a-z]", "", l)
    if squashed in ("incharge", "officerincharge"):
        return "in-charge"
    if squashed in ("email", "emailid", "mailid", "mail"):
        return "email"
    if squashed in ("tel", "telephone", "phone", "phoneno", "phonenumber"):
        return "phone"
    return l


def _phoneish(v: str) -> bool:
    digits = sum(c.isdigit() for c in v)
    body   = len(re.sub(r"\s", "", v))
    return digits >= 7 and body > 0 and digits / body >= 0.5


def _clip_value(v: str) -> str:
    """A value that STARTS with e-mail(s) ends with the last e-mail (+ an optional '(note)').
    Anything after it ('FAQ (Frequently Asked Questions) 1.What is...') is not part of the value."""
    v = " ".join(v.split())
    if _EMAIL_RE.match(v):
        m = _EMAIL_LIST_RE.match(v)
        if m:
            return _TAIL_SEP_RE.sub("", m.group(0)).strip().rstrip(".")
    return v


def _cut(entry: Dict, pos: int, text: str):
    entry["end"], entry["raw"] = pos, text[entry["start"]:pos].strip()


def _parse_entry_fields(block: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    current: Optional[str] = None
    for raw in block.split("\n"):
        line = raw.strip()
        if not line or _BULLET_ONLY_RE.match(line):
            continue
        lv = _label_of(line)
        if lv:
            current = _norm_label(lv[0])
            fields[current] = lv[1].strip()
        elif current:
            fields[current] = (fields[current] + " " + line).strip()
    out: Dict[str, str] = {}
    for k, v in fields.items():                          # "011-123 | a@b.in" -> phone + email
        v = _clip_value(re.sub(r"\s+", " ", v).strip())
        if not v:
            continue
        emails = _EMAIL_RE.findall(v)
        rest   = re.sub(r'\s*\|\s*', ' ', _EMAIL_RE.sub("", v)).strip(" ;,|")
        if emails and _phoneish(rest):
            out.setdefault("phone", rest)
            out.setdefault("email", "; ".join(emails))
        else:
            out[k] = v
    return out


def _find_headers(L):
    heads, n = [], len(L)
    for i, (s, e, t) in enumerate(L):
        m = _NUM_HEAD_RE.match(t)
        if not m:
            continue
        name, name_line = m.group(2), i
        if not name:                                     # "2." then the name on the next line
            j = i + 1
            while j < n and not L[j][2]:
                j += 1
            if j >= n:
                continue
            name, name_line = L[j][2], j
        if _NAME_OK_RE.match(name) and not _label_of(name) and len(name.split()) <= 7:
            heads.append({"line": i, "name_line": name_line, "number": int(m.group(1)),
                          "name": name.strip(), "start": s})
    return heads


def _find_plain_headers(L, skip_ranges):
    """Un-numbered entries: a short name line directly followed by a 'Label: value' line."""
    heads, n = [], len(L)
    for i, (s, e, t) in enumerate(L):
        if (not t or ":" in t or _BULLET_ONLY_RE.match(t) or not _NAME_OK_RE.match(t)
                or len(t.split()) > 6 or any(a <= s < b for a, b in skip_ranges)):
            continue
        j = i + 1
        while j < n and (not L[j][2] or _BULLET_ONLY_RE.match(L[j][2])):
            j += 1
        if j < n and j - i <= 3 and _label_of(L[j][2]):
            heads.append({"line": i, "name_line": i, "number": None, "name": t, "start": s})
    return heads


def _block_end(L, h, next_start):
    for k in range(h["name_line"] + 1, len(L)):
        s, _, t = L[k]
        if s >= next_start:
            return next_start
        if _QA_RE.match(t) or _is_caps_heading(t) or _NUM_HEAD_RE.match(t) or t.endswith("?"):
            return s                                     # FAQ / next section starts here
    return next_start


def _section_title(L, first_head):
    fallback = None
    k, seen = first_head["line"] - 1, 0
    while k >= 0 and seen < 4:
        t = L[k][2]
        if t and not _BULLET_ONLY_RE.match(t):
            seen += 1
            if _is_caps_heading(t):
                return t, L[k][0]
            if fallback is None and _is_title_heading(t):
                fallback = (t, L[k][0])
        k -= 1
    return fallback or ("", first_head["start"])


def _dir_key(title: str, idx: int) -> str:
    m = re.search(r'\(([A-Za-z]{2,8})\)', title)
    if m:
        return m.group(1).upper()
    for w in re.findall(r'\b[A-Z]{2,6}\b', title):
        if w.lower() not in _GENERIC:
            return w
    for w in re.findall(r"[A-Za-z]{3,}", title):
        if w.lower() not in _GENERIC:
            return w.upper()
    return f"DIR{idx}"


def _strip_common_suffix(names: List[str]) -> List[str]:
    """If >=80% of the names end with the same word ('Office', 'Branch', ...), drop it."""
    last = [n.split()[-1].lower() for n in names if len(n.split()) > 1]
    if len(names) >= 3 and last:
        word, cnt = Counter(last).most_common(1)[0]
        if cnt / len(names) >= 0.8:
            return [" ".join(n.split()[:-1]) if len(n.split()) > 1 and n.split()[-1].lower() == word else n
                    for n in names]
    return names


def _field_runs(block: str):
    """[(label, continuation_line_count)] for one entry block."""
    runs, cur = [], None
    for raw in block.split("\n"):
        t = raw.strip()
        if not t or _BULLET_ONLY_RE.match(t):
            continue
        lv = _label_of(t)
        if lv:
            cur = [_norm_label(lv[0]), 0]
            runs.append(cur)
        elif cur:
            cur[1] += 1
    return [(a, b) for a, b in runs]


def _trim_last(L, sec, text):
    """The last entry of a directory has no next header to stop it, so it can swallow whatever follows
    (a FAQ, a paragraph...). Learn where an entry ends from the OTHER entries: which labels the directory
    uses, how many wrapped lines each label really has, which label comes last. Cut the last entry as soon
    as it leaves that pattern (unknown label, heading, question, extra lines, text after the final e-mail)."""
    if len(sec) < 2:
        return
    others = sec[:-1]
    stats: Dict[str, int] = {}
    seen, finals = Counter(), Counter()
    for e in others:
        runs = _field_runs(e["raw"])
        for lab, c in runs:
            stats[lab] = max(stats.get(lab, 0), c)
        seen.update({lab for lab, _ in runs})
        if runs:
            finals[runs[-1][0]] += 1
    known = {l for l, c in seen.items() if c >= max(1, len(others) // 2)}
    final = finals.most_common(1)[0][0] if finals else None

    last, cur, cont, val = sec[-1], None, 0, ""
    for k in range(last["name_line"] + 1, len(L)):
        s, _, t = L[k]
        if s >= last["end"]:
            break
        if not t or _BULLET_ONLY_RE.match(t):
            continue
        lv = _label_of(t)
        if lv:
            lab = _norm_label(lv[0])
            if lab not in known:                       # 'Ans', 'Id', 'Helpline Number'... not part of this directory
                _cut(last, s, text)
                return
            raw_val = " ".join(lv[1].split())
            clip    = _clip_value(raw_val)
            if clip and len(clip) < len(raw_val) and (lab == final or field_type(lab, clip) == "email"):
                tail = clip.split()[-1]                # junk on the SAME line after the e-mail
                pos  = text.find(tail, s)
                if pos != -1:
                    _cut(last, pos + len(tail), text)
                    return
            cur, cont, val = lab, 0, raw_val
            continue
        if cur is None:
            continue
        cont += 1
        wrapped_paren = val.count("(") > val.count(")")           # "(OCI/PIO" + "Enquiries)" is a real wrap
        headingish = (_is_title_heading(t) or _is_caps_heading(t) or t.endswith("?")
                      or bool(_NUM_HEAD_RE.match(t)) or bool(_QA_RE.match(t)))
        contact    = cur == final or field_type(cur, val) in ("email", "phone", "fax")
        slack      = 0 if contact else 1                          # address-like text may wrap one line more
        if not wrapped_paren and (cont > stats.get(cur, 0) + slack or (headingish and contact)):
            _cut(last, s, text)
            return
        val += " " + t


def _group(L, heads, text, numbered: bool):
    """-> list of sections; each = list of valid entry dicts (with 'end')."""
    n, valid = len(text), []
    for i, h in enumerate(heads):
        nxt   = heads[i + 1]["start"] if i + 1 < len(heads) else n
        end   = _block_end(L, h, nxt)
        block = text[h["start"]:end]
        fields = _parse_entry_fields(block)
        if len(fields) >= 2:
            valid.append({**h, "end": end, "raw": block.strip(), "first_label": next(iter(fields))})
    if not numbered and valid:                           # plain mode: entries must start alike
        modal = Counter(v["first_label"] for v in valid).most_common(1)[0][0]
        valid = [v for v in valid if v["first_label"] == modal]
    sections, cur = [], []
    for v in valid:
        if cur and ((numbered and v["number"] <= cur[-1]["number"]) or cur[-1]["end"] < v["start"]):
            sections.append(cur)
            cur = []
        cur.append(v)
    if cur:
        sections.append(cur)
    for sec in sections:
        _trim_last(L, sec, text)
    return sections


def detect_directory_entries(full_text: str, min_entries: int = 3
                             ) -> Tuple[List[Dict], List[Tuple[int, int]]]:
    """Returns (entries, spans). spans = char ranges the directories occupy (removed from plain text chunks)."""
    L = _lines(full_text)
    raw_secs = [(s, L) for s in _group(L, _find_headers(L), full_text, True)]
    covered  = [(s[0]["start"], s[-1]["end"]) for s, _ in raw_secs if len(s) >= min_entries]
    raw_secs += [(s, L) for s in _group(L, _find_plain_headers(L, covered), full_text, False)]

    entries, spans, used = [], [], Counter()
    for sec, _ in sorted(raw_secs, key=lambda x: x[0][0]["start"] if x[0] else 0):
        if len(sec) < min_entries:
            continue
        title, tstart = _section_title(L, sec[0])
        title = re.sub(r"\s+", " ", title).strip()
        key   = _dir_key(title, len(spans) + 1)
        used[key] += 1
        if used[key] > 1:
            key = f"{key}{used[key]}"
        names = _strip_common_suffix([e["name"] for e in sec])
        for e, short in zip(sec, names):
            entries.append({"number": "" if e["number"] is None else str(e["number"]),
                            "location_name": short.strip(), "location_full": e["name"],
                            "raw_text": e["raw"], "char_start": e["start"],
                            "directory_title": title, "directory_key": key})
        spans.append((tstart, sec[-1]["end"]))
    return entries, sorted(spans)


def entries_to_chunks(entries: List[Dict], pdf_name: str,
                      page_offsets: List[Tuple[int, int]]) -> List[Dict]:
    offsets = [o for o, _ in page_offsets]
    out = []
    for i, e in enumerate(entries):
        pos  = bisect.bisect_right(offsets, e["char_start"]) - 1
        page = page_offsets[max(pos, 0)][1]
        out.append({
            "chunk_id": hashlib.md5(
                f"{pdf_name}_dir_{e['directory_key']}_{e['number'] or i}_{e['location_full']}".encode()
            ).hexdigest()[:12],
            "pdf": pdf_name, "page": page, "source": "directory_entry",
            "directory_title": e["directory_title"], "directory_key": e["directory_key"],
            "entry_number": e["number"], "location_name": e["location_name"],
            "location_full": e["location_full"], "text": e["raw_text"]})
    return out


def _subtract(s: int, e: int, spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """[s,e) minus the spans -> remaining (start,end) pieces."""
    out, cur = [], s
    for a, b in spans:
        if b <= cur or a >= e:
            continue
        if a > cur:
            out.append((cur, a))
        cur = max(cur, b)
    if cur < e:
        out.append((cur, e))
    return out

# ─── Chunking ─────────────────────────────────────────────────────────────────

def _cid(*parts) -> str:
    return hashlib.md5("_".join(str(p) for p in parts).encode()).hexdigest()[:12]


def chunk_text(text: str, page: int, source: str, pdf_name: str,
               chunk_size: int = config.CHUNK_SIZE,
               overlap: int    = config.CHUNK_OVERLAP,
               extra: Optional[Dict] = None) -> List[Dict]:
    """Overlapping chunks that prefer to end at a paragraph / sentence / space."""
    text = text.strip()
    if not text:
        return []
    chunks, start, idx, n = [], 0, 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            floor = start + int(chunk_size * 0.6)
            dot   = text.rfind(". ", floor, end)
            cut   = max(text.rfind("\n\n", floor, end), text.rfind("\n", floor, end),
                        dot + 1 if dot != -1 else -1, text.rfind(" ", floor, end))
            if cut > start:
                end = cut
        piece = text[start:end].strip()
        if piece:
            d = {"chunk_id": _cid(pdf_name, page, source, idx),
                 "pdf": pdf_name, "page": page, "source": source, "text": piece}
            if extra:
                d.update(extra)
            chunks.append(d)
            idx += 1
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_table(text: str, page: int, source: str, pdf_name: str,
                rows_per_chunk: int = config.TABLE_ROWS_PER_CHUNK) -> List[Dict]:
    """Split a markdown table by ROWS; repeat label + header in every chunk."""
    lines = [l for l in text.strip().split("\n") if l.strip()]
    if len(lines) < 4:
        return chunk_text(text, page, source, pdf_name)
    label, header, rows = lines[0], lines[1:3], lines[3:]
    chunks = []
    for idx, i in enumerate(range(0, len(rows), rows_per_chunk)):
        body = "\n".join([label] + header + rows[i:i + rows_per_chunk])
        chunks.append({"chunk_id": _cid(pdf_name, page, source, label, idx),
                       "pdf": pdf_name, "page": page, "source": source, "text": body})
    return chunks

# ─── Q&A detection — schema-free (FAQ pages become one chunk per question) ────
# Recognises, without any per-document setting:
#   "Q: question ...  A: answer"          "1.What is X?  Ans: answer"        "2. Question?  plain answer"
# An answer runs until the next question, an ALL-CAPS heading, or a directory block.

_Q_MARK_RE = re.compile(r'^\s*(?:Q|Question)\s*[:.]\s*(.*)$')
_A_MARK_RE = re.compile(r'^\s*(?:[•·▪●◦\-*–]\s*)?(?:A\s*:|Ans\s*[:.]|Answer\s*[:.])\s*(.*)$')
_NUMQ_RE   = re.compile(r'^\s*\d{1,3}\s*[.)]\s*(.*)$')
_BULLET_CH = "•·▪●◦"
MAX_QA_CHARS = 1800


def _try_question(L, k, barrier):
    """If line k starts a question -> (question_text, next_line_index), else None."""
    t = L[k][2]
    if not t or barrier(k):
        return None
    n = len(L)
    m = _Q_MARK_RE.match(t)
    if m:                                                    # "Q: ..." (may wrap over a few lines)
        q, j = [m.group(1)], k + 1
        while (j < n and len(q) < 4 and not q[-1].rstrip().endswith("?") and L[j][2]
               and not _A_MARK_RE.match(L[j][2]) and not barrier(j)):
            q.append(L[j][2])
            j += 1
        text = " ".join(x for x in q if x).strip()
        return (text, j) if len(text.split()) >= 2 else None
    m = _NUMQ_RE.match(t)
    body = m.group(1) if m else t
    if not body or (not m and not body.rstrip().endswith("?")):
        return None
    q, j = [body], k + 1
    while (not q[-1].rstrip().endswith("?") and len(q) < 3 and j < n and L[j][2]
           and not _A_MARK_RE.match(L[j][2]) and not barrier(j)):
        q.append(L[j][2])
        j += 1
    text = " ".join(q).strip()
    if not text.endswith("?") or not (2 <= len(text.split()) <= 40):
        return None
    return text, j


def _tidy_answer(lines: List[str]) -> str:
    """Join wrapped lines into sentences, keep bullets as '- item', drop a trailing sub-heading."""
    out, pending, first = [], False, True
    for raw in lines:
        t = raw.strip()
        if not t:
            continue
        if first:
            first = False
            m = _A_MARK_RE.match(t)
            if m:
                t = m.group(1).strip()
                if not t:
                    continue
        if _BULLET_ONLY_RE.match(t):
            pending = True
            continue
        is_bullet = pending or t[0] in _BULLET_CH
        t = t.lstrip(_BULLET_CH + " ").strip()
        if not t:
            continue
        if is_bullet:
            out.append("- " + t)
            pending = False
        elif out and not re.match(r'^\d{1,2}[.)]\s', t):
            out[-1] += " " + t
        else:
            out.append(t)
    if len(out) >= 2 and _is_title_heading(out[-1]) and re.search(r'[.!?:;)]$', out[-2].rstrip()):
        out.pop()
    return "\n".join(out).strip()


def _sub_heading(L, k, barrier, prev_text: str) -> bool:
    """A short Title-Case line that follows a finished sentence and is followed by a question
    ('Important Restrictions & Extensions', 'FAQ (Frequently Asked Questions)') -> a heading, not answer text."""
    t = L[k][2]
    if not t or barrier(k) or _label_of(t) or not _is_title_heading(t):
        return False
    if not prev_text or not re.search(r'[.!?)]$', prev_text.rstrip()):
        return False
    j = k + 1
    while j < len(L) and not L[j][2]:
        j += 1
    return j < len(L) and _try_question(L, j, barrier) is not None


def detect_qa_pairs(full_text: str, skip_spans: List[Tuple[int, int]]):
    """-> (items, spans). item = {question, answer, section, char_start}; spans = char ranges used."""
    L = _lines(full_text)
    n = len(L)

    def barrier(k):
        s = L[k][0]
        return any(a <= s < b for a, b in skip_spans)

    items, spans, section, sub, i = [], [], "", "", 0
    while i < n:
        t = L[i][2]
        if barrier(i):
            section, sub = "", ""                                # a directory block ends the previous topic
            i += 1
            continue
        if _is_caps_heading(t):
            section, sub = t, ""
            i += 1
            continue
        q = _try_question(L, i, barrier)
        if not q and _sub_heading(L, i, barrier, "?"):          # heading right above a question
            sub = t
            i += 1
            continue
        if not q:
            i += 1
            continue
        question, j = q
        k, a_lines = j, []
        while k < n:
            if (barrier(k) or _is_caps_heading(L[k][2]) or _try_question(L, k, barrier)
                    or _sub_heading(L, k, barrier, a_lines[-1] if a_lines else "?")):
                break
            a_lines.append(L[k][2])
            k += 1
        answer = _tidy_answer(a_lines)
        if answer:
            end = L[max(k - 1, j - 1)][1]
            items.append({"question": re.sub(r"\s+", " ", question), "answer": answer,
                          "section": f"{section} — {sub}" if (section and sub) else (sub or section),
                          "char_start": L[i][0]})
            spans.append((L[i][0], end))
        i = max(k, i + 1)
    return items, spans


def qa_to_chunks(items: List[Dict], pdf_name: str, page_offsets: List[Tuple[int, int]]) -> List[Dict]:
    offsets, out = [o for o, _ in page_offsets], []
    for i, it in enumerate(items):
        pos  = bisect.bisect_right(offsets, it["char_start"]) - 1
        page = page_offsets[max(pos, 0)][1]
        base = {"pdf": pdf_name, "page": page, "source": "qa_pair", "question": it["question"],
                "answer": it["answer"], "section": it["section"]}
        text = f"Q: {it['question']}\nA: {it['answer']}"
        if len(text) <= MAX_QA_CHARS:
            out.append({**base, "chunk_id": _cid(pdf_name, "qa", i), "text": text})
        else:                                           # long answer: repeat the question on every part
            for p, part in enumerate(chunk_text(it["answer"], page, "qa_pair", pdf_name,
                                                chunk_size=MAX_QA_CHARS - 200, overlap=100)):
                out.append({**base, "chunk_id": _cid(pdf_name, "qa", i, p),
                            "text": f"Q: {it['question']}\nA: {part['text']}"})
    return out


def _merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    out = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out

# ─── Embedding + FAISS ────────────────────────────────────────────────────────

def chunk_embed_text(c: Dict) -> str:
    """Text used for EMBEDDING only (stored chunk text stays clean)."""
    src = c.get("source")
    if src == "directory_entry":
        head = f"{c.get('directory_key', '')} {c.get('directory_title', '')} - {c.get('location_name', '')}"
    elif src == "qa_pair":
        head = f"{c.get('section', '')}\n{c.get('question', '')}"          # the question carries the meaning
    else:
        head = c.get("pdf", "")
    return f"{head}\n{c['text']}"


def _load_existing() -> tuple:
    import faiss
    if os.path.exists(config.FAISS_INDEX_PATH) and os.path.exists(config.CHUNKS_JSON_PATH):
        index = faiss.read_index(config.FAISS_INDEX_PATH)
        with open(config.CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        return index, chunks
    return None, []


def save_index(index, chunks: List[Dict]):
    import faiss
    faiss.write_index(index, config.FAISS_INDEX_PATH)
    with open(config.CHUNKS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    logger.info(f"Saved {index.ntotal} vectors, {len(chunks)} chunks")


def load_index():
    import faiss
    index = faiss.read_index(config.FAISS_INDEX_PATH)
    with open(config.CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks


def _drop_pdf(index, chunks: List[Dict], pdf_name: str):
    """Remove one PDF's chunks WITHOUT re-embedding (reconstruct vectors from the flat index)."""
    import faiss
    target = pdf_name.replace(".pdf", "")
    keep   = [i for i, c in enumerate(chunks) if c.get("pdf", "").replace(".pdf", "") != target]
    if len(keep) == len(chunks):
        return index, chunks
    kept = [chunks[i] for i in keep]
    if not kept:
        return None, []
    if index is not None and index.ntotal == len(chunks):
        vecs = index.reconstruct_n(0, index.ntotal)[keep]
        new  = faiss.IndexFlatIP(index.d)
        new.add(np.ascontiguousarray(vecs, dtype=np.float32))
        return new, kept
    new = faiss.IndexFlatIP(config.EMBED_DIM)
    new.add(embed_texts([chunk_embed_text(c) for c in kept]))
    return new, kept

# ─── Location lookup (fast path) — also schema-free ───────────────────────────

_loc_cache = (None, {})
_aux_cache = (None, {})            # per-chunks-object: directory aliases + intent words + title words


def build_location_index(chunks: List[Dict]) -> Dict[str, List[int]]:
    global _loc_cache
    if _loc_cache[0] is chunks:
        return _loc_cache[1]
    index: Dict[str, List[int]] = {}
    for i, c in enumerate(chunks):
        if c.get("source") == "directory_entry":
            for nm in {c["location_name"], c.get("location_full", c["location_name"])}:
                index.setdefault(nm.strip().lower(), []).append(i)
    _loc_cache = (chunks, index)
    return index


def _aux(chunks: List[Dict]) -> Dict:
    """Directory aliases, 'this sounds like a directory question' words and the words that make up the
    directory TITLES (they name the directory, they never ask for a field) — all learned from the chunks."""
    global _aux_cache
    if _aux_cache[0] is chunks:
        return _aux_cache[1]
    titles: Dict[str, str] = {}
    labels = set()
    for c in chunks:
        if c.get("source") == "directory_entry":
            titles[c["directory_key"]] = c.get("directory_title", "")
            labels.update(_parse_entry_fields(c["text"]).keys())
    weak = {k: {w for w in re.findall(r"[a-z]{3,}", t.lower()) if w not in _GENERIC}
            for k, t in titles.items()}
    freq = Counter(w for ws in weak.values() for w in ws)
    weak = {k: {w for w in ws if freq[w] == 1 and w != k.lower()} for k, ws in weak.items()}  # words shared by 2+ dirs are useless
    strong = {k: {k.lower()} for k in titles}                       # the key itself (acronym / first word)
    aliases = {"strong": strong, "weak": weak}
    intent = (_GENERIC - _STOP) | _CONCEPT_WORDS | {w for l in labels for w in re.findall(r"[a-z]{3,}", l)} \
             | {"where", "located", "head", "official", "jurisdiction"}
    title_words = {k: set(re.findall(r"[a-z]+", t.lower())) | {k.lower()} for k, t in titles.items()}
    out = {"aliases": aliases, "intent": intent, "title_words": title_words}
    _aux_cache = (chunks, out)
    return out


def title_words(chunks: List[Dict], keys=None) -> set:
    """Words of the title(s) of the given directories (all directories if keys is None)."""
    tw = _aux(chunks)["title_words"]
    return set().union(*[w for k, w in tw.items() if keys is None or k in keys]) if tw else set()


def _named(q: str, chunks: List[Dict]):
    words = set(re.findall(r"[a-z]+", q.lower()))
    al = _aux(chunks)["aliases"]
    strong = {k for k, a in al["strong"].items() if words & a}
    weak   = {k for k, a in al["weak"].items() if words & a}
    return strong, weak


def named_directories(q: str, chunks: List[Dict]) -> set:
    strong, weak = _named(q, chunks)
    return strong | weak


def select_candidates(cands: List[Dict], chunks: List[Dict], q: str) -> List[Dict]:
    """Keep only the directory the user named. If it has no such location: [] when the user typed the
    directory's key (never answer from the wrong directory), but ignore a mere title word like 'numbers'."""
    strong, weak = _named(q, chunks)
    named = strong | weak
    if not named:
        return cands
    kept = [c for c in cands if c.get("directory_key") in named]
    if kept:
        return kept
    return [] if strong else cands


# ---- fields: type is inferred from the VALUE, not from a fixed label list ----

def field_type(label: str, value: str) -> str:
    if _EMAIL_RE.search(value) and not _phoneish(value):
        return "email"
    if _phoneish(value):
        return "fax" if "fax" in label else "phone"
    return "text"


def field_icon(label: str, value: str) -> str:
    t, l = field_type(label, value), label.lower()
    if t == "email":
        return "✉️"
    if t == "fax":
        return "📠"
    if t == "phone":
        return "📞"
    if re.search(r"addr|locat|situat|place", l):
        return "📍"
    if re.search(r"charge|officer|head|person|name|design|manager|director|incharge", l):
        return "👤"
    if re.search(r"jurisd|area|region|zone|cover|district", l):
        return "🗺️"
    return "•"


_CONCEPTS = {
    "phone":   ("phone", "telephone", "mobile", "number", "call", "tel", "contact"),
    "email":   ("email", "mail", "contact"),
    "address": ("address", "location", "located", "where", "situated"),
}
_CONCEPT_WORDS = {w for ws in _CONCEPTS.values() for w in ws}


def requested_fields(query: str, fields: Dict[str, str], ignore=frozenset()) -> List[str]:
    """Which of this entry's fields does the question ask for? Label words first, then phone/email/address
    concepts. Words that are part of a directory TITLE ('contact', 'directory', 'frro' ...) name the
    directory, not a field, so they are ignored: 'FRRO CONTACT DIRECTORY Mumbai' == 'frro mumbai'."""
    words  = [w for w in re.findall(r"[a-z]+", query.lower()) if w not in ignore]
    qwords = set(words)
    qnorm  = "".join(words)
    hits: List[str] = []
    for k in fields:                                     # 1) the question names the label itself
        ws = [w for w in re.findall(r"[a-z]+", k) if len(w) >= 4]
        kn = re.sub(r"[^a-z]", "", k)
        if any(w in qwords for w in ws) or (len(kn) >= 4 and kn in qnorm):
            hits.append(k)
    for concept, cw in _CONCEPTS.items():                # 2) generic concepts matched on value type
        if qwords & set(cw):
            for k, v in fields.items():
                t = field_type(k, v)
                ok = (concept == "phone" and t == "phone") or (concept == "email" and t == "email") or \
                     (concept == "address" and re.search(r"addr|locat", k))
                if ok and k not in hits:
                    hits.append(k)
    return [k for k in fields if k in hits]


def query_location(query: str, chunks: Optional[List[Dict]] = None, threshold: int = 0) -> Optional[Dict]:
    """
    Returns {"location", "matches": [{"chunk", "fields"?}]} or None.
    * short, directory-sounding queries only (everything else goes to full RAG)
    * if the query names a directory (learned from the PDF titles) only that directory is used
    * same location in several directories + none named -> ALL are returned
    * words of the directory title never count as a field request
    """
    q = query.lower().strip()
    if len(q.split()) > config.FASTPATH_MAX_WORDS:
        return None
    if chunks is None:
        _, chunks = load_index()
    loc_index = build_location_index(chunks)
    if not loc_index:
        return None
    if not (set(re.findall(r"[a-z]+", q)) & _aux(chunks)["intent"]
            or len(q.split()) <= 3 or named_directories(q, chunks)):
        return None

    keys = _find_location_keys(q, list(loc_index.keys()))
    if not keys:
        return None
    key   = max(keys, key=len)
    cands = select_candidates([chunks[i] for i in loc_index[key]], chunks, q)
    if not cands:
        return None

    ignore  = title_words(chunks, {c.get("directory_key") for c in cands})   # only the directory being answered
    matches = []
    for c in cands:
        flds = _parse_entry_fields(c["text"])
        want = requested_fields(query, flds, ignore)
        if want:
            matches.append({"chunk": c, "fields": {k: flds[k] for k in want}})
    if matches:
        return {"location": key, "matches": matches}
    return {"location": key, "matches": [{"chunk": c} for c in cands]}


def _find_location_keys(q: str, keys: List[str]) -> List[str]:
    """Whole-word match first ('goa' won't match 'goal'); typo-tolerant fallback for longer names."""
    hits = [k for k in keys if re.search(rf"(?<!\w){re.escape(k)}(?!\w)", q)]
    if hits:
        return hits
    from rapidfuzz import fuzz
    words, out = re.findall(r"[a-z]+", q), []
    for k in keys:
        if len(k) < 5:
            continue
        n = len(k.split())
        for i in range(len(words) - n + 1):
            if fuzz.ratio(" ".join(words[i:i + n]), k) >= config.FASTPATH_THRESHOLD:
                out.append(k)
                break
    return out

# ─── Delete PDF from index ────────────────────────────────────────────────────

def delete_pdf_from_index(pdf_name: str) -> Dict:
    with _index_lock:
        if not os.path.exists(config.CHUNKS_JSON_PATH):
            return {"status": "error", "message": "No index found"}
        index, chunks = _load_existing()
        total = len(chunks)
        new_index, kept = _drop_pdf(index, chunks, pdf_name)
        removed = total - len(kept)
        if removed == 0:
            return {"status": "not_found", "message": f"No chunks for '{pdf_name}'"}
        if not kept:
            for path in (config.FAISS_INDEX_PATH, config.CHUNKS_JSON_PATH):
                if os.path.exists(path):
                    os.remove(path)
            return {"status": "ok", "pdf": pdf_name, "removed_chunks": removed, "remaining_chunks": 0}
        save_index(new_index, kept)
        return {"status": "ok", "pdf": pdf_name, "removed_chunks": removed, "remaining_chunks": len(kept)}

# ─── Build chunks (directories + text + tables + images) ──────────────────────

def _build_chunks(pdf_name, page_records, image_records, table_records):
    full_text, page_offsets, ranges = "", [], []
    for rec in page_records:
        s = len(full_text)
        page_offsets.append((s, rec["page"]))
        full_text += rec["text"] + "\n"
        ranges.append((s, len(full_text), rec))

    entries, dir_spans = detect_directory_entries(full_text)
    chunks = entries_to_chunks(entries, pdf_name, page_offsets) if entries else []
    n_dir  = len(chunks)
    keys   = ", ".join(sorted({e["directory_key"] for e in entries}))

    qa_items, qa_spans = detect_qa_pairs(full_text, dir_spans)
    qa_chunks = qa_to_chunks(qa_items, pdf_name, page_offsets)
    chunks.extend(qa_chunks)
    spans = _merge_spans(dir_spans + qa_spans)
    logger.info(f"Structure found: directories=[{keys or '-'}] entries={n_dir}  qa_pairs={len(qa_items)}")

    for s, e, rec in ranges:          # everything that is NOT a directory entry / Q&A pair stays plain text
        pieces = [full_text[a:b] for a, b in _subtract(s, e, spans)] if spans else [rec["text"]]
        text = "\n".join(p.strip() for p in pieces if p.strip())
        if text:
            chunks.extend(chunk_text(text, rec["page"], rec["source"], pdf_name))
    for rec in image_records:
        chunks.extend(chunk_text(rec["text"], rec["page"], rec["source"], pdf_name,
                                 extra={"image_path": rec["image_path"]}))
    for rec in table_records:
        chunks.extend(chunk_table(rec["text"], rec["page"], rec["source"], pdf_name))
    return chunks, keys, n_dir, len(qa_items)


def _commit(pdf_name: str, new_chunks: List[Dict]) -> int:
    """Embed OUTSIDE the lock, swap into the index INSIDE it."""
    import faiss
    new_emb = embed_texts([chunk_embed_text(c) for c in new_chunks])
    with _index_lock:
        index, chunks = _load_existing()
        if index is not None and (index.ntotal != len(chunks) or index.d != new_emb.shape[1]):
            index = None
            if chunks:
                index = faiss.IndexFlatIP(new_emb.shape[1])
                index.add(embed_texts([chunk_embed_text(c) for c in chunks]))
        index, chunks = _drop_pdf(index, chunks, pdf_name)
        if index is None:
            index = faiss.IndexFlatIP(new_emb.shape[1])
        index.add(new_emb)
        final = chunks + new_chunks
        save_index(index, final)
    if on_commit:
        on_commit()
    return len(final)


def rebuild_index() -> None:
    """Re-embed chunks.json with the CURRENT embedding model:  python ingest.py --rebuild"""
    import faiss
    with _index_lock:
        with open(config.CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        if not chunks:
            logger.warning("chunks.json is empty - nothing to rebuild")
            return
        total, t0 = len(chunks), time.time()
        texts = [chunk_embed_text(c) for c in chunks]
        order = sorted(range(total), key=lambda i: len(texts[i]))   # similar lengths together = much less padding
        logger.info(f"Re-embedding {total} chunks with {config.EMBED_MODEL} on {config.DEVICE} "
                    f"(one-time - please do not press Ctrl+C) ...")
        emb = None
        for b in range(0, total, 16):
            ids = order[b:b + 16]
            v = embed_texts([texts[i] for i in ids])
            if emb is None:
                emb = np.zeros((total, v.shape[1]), dtype=np.float32)
            emb[ids] = v                                           # put every vector back in its ORIGINAL position
            done = min(b + 16, total)
            left = (total - done) * (time.time() - t0) / done
            logger.info(f"  rebuild {done}/{total} ({done * 100 // total}%)  ~{left:.0f}s left")
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        save_index(index, chunks)
    logger.info(f"Done: {index.ntotal} vectors, dim {emb.shape[1]}, {time.time() - t0:.0f}s")

# ─── Main entry ───────────────────────────────────────────────────────────────

def _ocr_stage(pdf_path, name, img_out, pages, tables, scanned):
    t0 = time.time()
    try:
        JOBS[name] = {"stage": "ocr", "done": 0, "total": len(scanned)}

        def progress(done, total):
            JOBS[name] = {"stage": "ocr", "done": done, "total": total}

        ocr_pages, image_records = extract_ocr(pdf_path, img_out, scanned, progress)
        if not ocr_pages and not image_records:
            msg = ("OCR ran but could not read any text from this PDF. Check the scan quality and "
                   "set config.OCR_LANGS (e.g. ['ta','en'] for Tamil) and config.OCR_SCALE = 2.5.")
            logger.error(f"{name}: {msg}")
            JOBS[name] = {"stage": "error", "error": msg}
            return
        pages = [dict(r) for r in pages]                     # never mutate the stage-1 records
        for rec in pages:
            if rec["page"] in ocr_pages:
                rec["text"], rec["source"] = clean_text(ocr_pages[rec["page"]]), "ocr"
        JOBS[name] = {"stage": "ocr", "done": len(scanned), "total": len(scanned), "note": "indexing"}
        chunks, _, _, _ = _build_chunks(name, pages, image_records, tables)
        total = _commit(name, chunks)
        JOBS[name] = {"stage": "done", "images": len(image_records), "ocr_pages": len(ocr_pages),
                      "total_indexed": total, "elapsed_sec": round(time.time() - t0, 1)}
        logger.info(f"OCR stage done for {name}: {JOBS[name]}")
    except Exception as e:
        logger.exception("OCR stage failed")
        JOBS[name] = {"stage": "error", "error": str(e)}


def ingest_pdf(pdf_path: str) -> Dict:
    t0, name = time.time(), Path(pdf_path).stem
    img_out  = os.path.join(config.IMAGES_DIR, name)

    pages, tables, scanned, has_images = extract_fast(pdf_path)
    t_ext = time.time() - t0
    chunks, keys, n_dir, n_qa = _build_chunks(name, pages, [], tables)
    need_ocr = bool(scanned) or (config.EXTRACT_EMBEDDED_IMAGES and has_images)

    # a scanned PDF with no OCR engine installed: say so NOW (never a silent "0 chunks")
    ocr_engine = ocr_available() if need_ocr else None
    if need_ocr and not ocr_engine:
        logger.error(f"{name}: no OCR engine installed")
        if scanned and not chunks:
            JOBS[name] = {"stage": "error", "error": OCR_INSTALL_HELP}
            return {"status": "ocr_unavailable", "pdf": name, "pages": len(pages),
                    "scanned_pages": len(scanned), "message": OCR_INSTALL_HELP}
        need_ocr = False                                   # text pages are fine; only images are skipped

    total, t1 = None, time.time()
    if chunks:
        total = _commit(name, chunks)
    logger.info(f"Stage 1: extract={t_ext:.1f}s embed+save={time.time() - t1:.1f}s chunks={len(chunks)} "
                f"scanned_pages={len(scanned)}")

    JOBS[name] = {"stage": "ocr" if need_ocr else "done", "done": 0, "total": len(scanned)}
    if need_ocr:
        threading.Thread(target=_ocr_stage, daemon=True, name=f"ocr-{name}",
                         args=(pdf_path, name, img_out, pages, tables, scanned)).start()
    if not chunks and not need_ocr:
        return {"status": "empty", "chunks": 0}

    message = ""
    if scanned:
        message = (f"{len(scanned)} scanned page(s) are being read with OCR ({ocr_engine}) in the background. "
                   f"Ask your questions in a few minutes (progress: /api/ingest-status/{name}).")
    return {"status": "ok", "pdf": name, "pages": len(pages), "tables": len(tables),
            "directory_title": keys, "directory_entries": n_dir, "qa_pairs": n_qa,
            "total_chunks": len(chunks), "total_indexed": total,
            "background_ocr": need_ocr, "scanned_pages": len(scanned), "ocr_engine": ocr_engine,
            "message": message, "elapsed_sec": round(time.time() - t0, 2)}


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path/to/file.pdf>   |   python ingest.py --rebuild")
        sys.exit(1)
    if sys.argv[1] == "--rebuild":
        rebuild_index()
        sys.exit(0)
    result = ingest_pdf(sys.argv[1])
    print(json.dumps(result, indent=2))
    stem = Path(sys.argv[1]).stem
    while JOBS.get(stem, {}).get("stage") == "ocr":          # command line: wait for the OCR thread
        j = JOBS[stem]
        print(f"OCR {j.get('done', 0)}/{j.get('total', 0)} ...")
        time.sleep(5)
    print(json.dumps(JOBS.get(stem, {}), indent=2))