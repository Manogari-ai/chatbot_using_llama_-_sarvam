"""
local_ai.py - FREE, offline AI4Bharat fallback for the Sarvam API.  No API key, no credits.

    chat       -> Ollama, config.LOCAL_CHAT_MODEL = Gemma (the Ollama provider keeps its own llama model)
    translate  -> AI4Bharat IndicTrans2 (Indian, open source, 22 languages)
    speech->txt-> faster-whisper (Tamil, Hindi, Telugu ... supported)
    text->speech-> Meta MMS-TTS (Tamil, Hindi, Telugu, Kannada, Malayalam ...)

sarvam.py calls these automatically when the Sarvam limit is reached (or config.AI_BACKEND == "local"),
so rag.py / agent.py / app.py do not change.

pip install transformers sentencepiece sacremoses IndicTransToolkit faster-whisper scipy
"""
import base64
import io
import logging
import re
import threading

import config

logger = logging.getLogger(__name__)

_lock = threading.RLock()          # models are loaded / run one request at a time
_models = {}

# language code (before "-") -> FLORES code (IndicTrans2), MMS code (TTS), Whisper code (STT)
FLORES = {"en": "eng_Latn", "ta": "tam_Taml", "te": "tel_Telu", "hi": "hin_Deva", "kn": "kan_Knda",
          "ml": "mal_Mlym", "bn": "ben_Beng", "mr": "mar_Deva", "gu": "guj_Gujr", "pa": "pan_Guru",
          "od": "ory_Orya"}
MMS = {"en": "eng", "ta": "tam", "te": "tel", "hi": "hin", "kn": "kan", "ml": "mal", "bn": "ben",
       "mr": "mar", "gu": "guj", "pa": "pan", "od": "ory"}
WHISPER = {"od": "or"}             # everything else: same 2-letter code


def _pre(code: str) -> str:
    return (code or "en").split("-")[0].lower()


def _device() -> str:
    return "cuda" if getattr(config, "DEVICE", "cpu") == "cuda" else "cpu"


# ─── Chat (Ollama) ────────────────────────────────────────────────────────────

def chat_stream(system: str, user: str, max_tokens: int = None):
    model = getattr(config, "LOCAL_CHAT_MODEL", config.OLLAMA_MODEL)
    try:
        import ollama
        for part in ollama.chat(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            stream=True,
            keep_alive=config.OLLAMA_KEEP_ALIVE,
            options={"temperature": 0.1, "top_p": 0.9,
                     "num_ctx": getattr(config, "LOCAL_NUM_CTX", 4096),
                     "num_predict": max_tokens or getattr(config, "LOCAL_NUM_PREDICT", 600)},
        ):
            tok = part["message"]["content"]
            if tok:
                yield tok
    except Exception as e:
        logger.error(f"Local chat failed: {e}")
        yield f"[LLM Error] Could not run Ollama model '{model}'. Run:  ollama pull {model}   Error: {e}"


# ─── Translation (IndicTrans2) ────────────────────────────────────────────────

def _load_mt(direction: str):
    with _lock:
        if direction not in _models:
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            try:
                from IndicTransToolkit.processor import IndicProcessor
            except ImportError:
                from IndicTransToolkit import IndicProcessor
            name = f"ai4bharat/indictrans2-{direction}-dist-200M"        # en-indic | indic-en
            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(name, trust_remote_code=True).to(_device()).eval()
            _models[direction] = (tok, mdl, IndicProcessor(inference=True), torch)
            logger.info(f"IndicTrans2 loaded: {name}")
        return _models[direction]


def preload():
    """Load IndicTrans2 (and Gemma) in the background so the first fallback call is not a cold start."""
    def _job():
        try:
            _load_mt("en-indic")
            _load_mt("indic-en")
        except Exception as e:
            if isinstance(e, ImportError):
                _models["mt_failed"] = True
            logger.warning(f"IndicTrans2 preload failed: {e}")
        if getattr(config, "WARMUP_LOCAL_CHAT", True):
            try:
                import ollama
                ollama.chat(model=getattr(config, "LOCAL_CHAT_MODEL", config.OLLAMA_MODEL),
                            messages=[{"role": "user", "content": "hi"}], stream=False,
                            keep_alive=config.OLLAMA_KEEP_ALIVE, options={"num_predict": 1})
            except Exception as e:
                logger.warning(f"Gemma preload failed: {e}")
    threading.Thread(target=_job, daemon=True, name="mt-preload").start()


def translate(text: str, src: str, tgt: str) -> str:
    """Same signature as sarvam.translate. Indic -> Indic goes through English."""
    if not text.strip() or _pre(src) == _pre(tgt):
        return text
    if _models.get("mt_failed"):                       # IndicTrans2 cannot load: do not retry on every line
        return text
    try:
        if _pre(src) != "en" and _pre(tgt) != "en":
            text, src = translate(text, src, "en-IN"), "en-IN"
        direction = "en-indic" if _pre(src) == "en" else "indic-en"
        s_code, t_code = FLORES[_pre(src)], FLORES[_pre(tgt)]
        sents = [s for s in re.split(r"(?<=[.!?।])\s+", text.strip()) if s]
        out = []
        with _lock:
            tok, mdl, ip, torch = _load_mt(direction)
            for i in range(0, len(sents), 8):
                batch = ip.preprocess_batch(sents[i:i + 8], src_lang=s_code, tgt_lang=t_code)
                enc = tok(batch, truncation=True, padding="longest", max_length=256,
                          return_tensors="pt").to(_device())
                with torch.no_grad():
                    gen = mdl.generate(**enc, use_cache=True, min_length=0, max_length=256,
                                       num_beams=4, num_return_sequences=1)
                dec = tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                out += ip.postprocess_batch(dec, lang=t_code)
        return " ".join(out)
    except ImportError:
        _models["mt_failed"] = True
        logger.error("IndicTrans2 cannot load (transformers / IndicTransToolkit version mismatch). "
                     "Run:  pip install \"transformers==4.46.3\"  and restart. Local translation is off until then.")
        return text
    except Exception:
        logger.exception("Local translation failed; returning original text")
        return text


# ─── Speech to text (faster-whisper) ──────────────────────────────────────────

def speech_to_text(audio: bytes, filename: str, mimetype: str, lang: str):
    from faster_whisper import WhisperModel
    import sarvam
    with _lock:
        if "stt" not in _models:
            _models["stt"] = WhisperModel(getattr(config, "WHISPER_MODEL", "small"),
                                          device=_device(), compute_type="int8")
        model = _models["stt"]
        code = _pre(lang) if lang and lang != "unknown" else None
        code = WHISPER.get(code, code)
        segs, info = model.transcribe(io.BytesIO(audio), language=code, vad_filter=True, beam_size=5)
        text = " ".join(s.text.strip() for s in segs)
    det = info.language
    det = "od" if det == "or" else det
    return sarvam.clean(text), f"{det}-IN"


# ─── Text to speech (Meta MMS-TTS) ────────────────────────────────────────────

def _load_tts(code: str):
    with _lock:
        if code not in _models:
            from transformers import VitsModel, AutoTokenizer
            name = f"facebook/mms-tts-{code}"
            _models[code] = (AutoTokenizer.from_pretrained(name), VitsModel.from_pretrained(name).eval())
        return _models[code]


def tts_stream(chunks, lang: str):
    """Yield one base64 WAV per text chunk, as soon as each one is synthesised (voice starts after chunk 1)."""
    import numpy as np
    import torch
    from scipy.io import wavfile
    import sarvam
    code = MMS.get(_pre(lang), "eng")
    tok, mdl = _load_tts(code)
    for c in chunks:
        c = sarvam._tts_prepare(c, lang, native=True)      # acronyms -> letter names in the target script
        enc = tok(c, return_tensors="pt")
        if enc["input_ids"].shape[1] == 0:
            continue
        with _lock, torch.no_grad():
            wav = mdl(**enc).waveform[0].cpu().numpy()
        pcm = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
        buf = io.BytesIO()
        wavfile.write(buf, mdl.config.sampling_rate, pcm)
        yield base64.b64encode(buf.getvalue()).decode()


def text_to_speech(text: str, lang: str, speaker: str = None):
    """-> list of base64 WAV strings, in playback order (same as sarvam.text_to_speech)."""
    import sarvam
    return list(tts_stream(sarvam.tts_chunks(text, lang), lang))