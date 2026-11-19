"""
app.py — Flask backend for PDF Extract Chatbot
Routes:
  GET  /                          → chat.html
  GET  /upload                    → upload.html
  POST /api/upload                → ingest PDF(s)  (text+tables now, OCR of scanned pages in background)
  GET  /api/ingest-status/<name>  → background OCR progress  ({"stage": "ocr", "done": 12, "total": 57})
  POST /api/chat                  → RAG answer
  POST /api/chat/stream           → RAG answer (SSE)
  GET  /api/status                → index stats (+ running OCR jobs)
  GET  /api/history, DELETE /api/clear, GET /api/pdfs, DELETE /api/delete-pdf
"""

import os
import json
import logging
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import (Flask, request, jsonify, render_template,
                   send_from_directory, Response, stream_with_context)

import config
import ingest
import rag
import sarvam
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

# After EVERY index commit (stage 1 and the background OCR stage) reload the search index.
# This also clears the answer cache, so nothing stale is served.
ingest.on_commit = rag.load_index

# ─── Helpers ──────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS

def list_ingested_pdfs():
    if not os.path.exists(config.CHUNKS_JSON_PATH):
        return []
    try:
        with open(config.CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        return sorted(set(c.get("pdf", "") for c in chunks if c.get("pdf")))
    except Exception:
        return []

def ocr_jobs():
    """PDFs whose scanned pages are still being read, or whose OCR failed."""
    return {n: j for n, j in list(ingest.JOBS.items()) if j.get("stage") in ("ocr", "error")}

# ─── Pages ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/upload")
def upload_page():
    return render_template("upload.html")

# ─── API: Upload & Ingest ─────────────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    files = request.files.getlist("file")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No files selected"}), 400

    results = []
    for file in files:
        if not allowed_file(file.filename):
            results.append({"file": file.filename, "error": "Only PDF files are allowed"})
            continue
        filename  = secure_filename(file.filename)
        save_path = os.path.join(config.UPLOAD_DIR, filename)
        file.save(save_path)
        logger.info(f"Saved upload: {save_path}")
        try:
            summary = ingest.ingest_pdf(save_path)      # locks internally; reloads index via on_commit
            if summary.get("status") == "ocr_unavailable":   # scanned PDF but no OCR engine -> show a real error
                results.append({"file": filename, "error": summary["message"]})
            else:
                results.append({"file": filename, "result": summary})
        except Exception as e:
            logger.exception(f"Ingestion error for {filename}")
            results.append({"file": filename, "error": str(e)})
    return jsonify({"uploads": results})

@app.route("/api/ingest-status/<name>")
def api_ingest_status(name):
    stem = Path(name).stem if name.lower().endswith(".pdf") else name
    return jsonify(ingest.JOBS.get(stem, {"stage": "unknown"}))

# ─── API: Chat ────────────────────────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data  = request.get_json(force=True)
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400
    try:
        return jsonify(rag.answer(query))
    except Exception as e:
        logger.exception("RAG error")
        return jsonify({"error": str(e)}), 500

def _lang(data) -> str:
    l = data.get("language") or "en-IN"
    return l if l in config.LANGUAGES else "en-IN"

@app.route("/api/config")
def api_config():
    return jsonify({
        "sarvam_enabled": bool(config.SARVAM_API_KEY),
        "languages": config.LANGUAGES,
        "providers": [{"id": "ollama", "label": f"Ollama · {config.OLLAMA_MODEL}"},
                      {"id": "sarvam", "label": f"Sarvam AI · {config.SARVAM_CHAT_MODEL}"}],
    })

@app.route("/api/chat/stream", methods=["POST"])
def api_chat_stream():
    data     = request.get_json(force=True)
    query    = (data.get("query") or "").strip()
    provider = data.get("provider", "ollama")
    if not query:
        return jsonify({"error": "Empty query"}), 400
    if provider == "sarvam" and not config.SARVAM_API_KEY:
        return jsonify({"error": "SARVAM_API_KEY is not set on the server"}), 400
    lang = _lang(data) if provider == "sarvam" else "en-IN"

    def generate():
        try:
            for line in rag.answer_stream(query, provider=provider, lang=lang):
                yield f"data: {line}\n\n"
        except Exception as e:
            logger.exception("Stream RAG error")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.route("/api/stt", methods=["POST"])
def api_stt():
    f = request.files.get("audio")
    if not f:
        return jsonify({"error": "No audio uploaded"}), 400
    try:
        text, detected = sarvam.speech_to_text(f.read(), f.filename or "speech.wav",
                                               f.mimetype or "audio/wav",
                                               request.form.get("language") or "unknown")
        return jsonify({"transcript": text, "language": detected})
    except Exception as e:
        logger.exception("STT error")
        return jsonify({"error": str(e)}), 500

@app.route("/api/tts", methods=["POST"])
def api_tts():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400
    try:
        return jsonify({"audios": sarvam.text_to_speech(text, _lang(data)), "format": "wav"})
    except Exception as e:
        logger.exception("TTS error")
        return jsonify({"error": str(e)}), 500

# ─── API: Status / PDFs / Delete / History ───────────────────────────────────

@app.route("/api/status", methods=["GET"])
def api_status():
    pdfs = list_ingested_pdfs()
    index_exists = os.path.exists(config.FAISS_INDEX_PATH)
    vector_count = rag._faiss_index.ntotal if (index_exists and rag._faiss_index is not None) else 0
    return jsonify({
        "index_ready": index_exists, "vector_count": vector_count,
        "pdf_count": len(pdfs), "pdfs": pdfs,
        "ocr_jobs": ocr_jobs(),
        "ocr_engine": ingest.ocr_available(),
        "embed_model": config.EMBED_MODEL, "llm_model": config.OLLAMA_MODEL,
        "reranker": config.RERANKER_MODEL,
        "top_k_retrieve": config.TOP_K_RETRIEVE, "top_k_rerank": config.TOP_K_RERANK,
        "table_extraction_enabled": config.ENABLE_TABLE_EXTRACTION,
        "embedded_image_ocr_enabled": config.EXTRACT_EMBEDDED_IMAGES,
        "fastpath_threshold": config.FASTPATH_THRESHOLD,
        "answer_cache_entries": len(rag._answer_cache),
        "answer_cache_capacity": config.ANSWER_CACHE_SIZE,
    })

@app.route("/api/debug", methods=["GET"])
def api_debug():
    """/api/debug?q=POE Jaipur  ->  which path would answer it, what was found, and any hidden error."""
    return jsonify(rag.explain((request.args.get("q") or "").strip() or "hello"))

@app.route("/api/pdfs", methods=["GET"])
def api_pdfs():
    return jsonify({"pdfs": list_ingested_pdfs(), "processing": ocr_jobs()})

@app.route("/api/delete-pdf", methods=["DELETE"])
def api_delete_pdf():
    data     = request.get_json(force=True)
    pdf_name = (data.get("pdf") or "").strip()
    if not pdf_name:
        return jsonify({"error": "Missing 'pdf' field"}), 400
    try:
        summary = ingest.delete_pdf_from_index(pdf_name)
        ingest.JOBS.pop(Path(pdf_name).stem if pdf_name.lower().endswith(".pdf") else pdf_name, None)
        rag.load_index()
        return jsonify(summary)
    except Exception as e:
        logger.exception(f"Delete error for {pdf_name}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/history", methods=["GET"])
def api_history():
    if not os.path.exists(config.CHAT_HISTORY_PATH):
        return jsonify({"history": ""})
    with open(config.CHAT_HISTORY_PATH, "r", encoding="utf-8") as f:
        return jsonify({"history": f.read()})

@app.route("/api/clear", methods=["DELETE"])
def api_clear_history():
    if os.path.exists(config.CHAT_HISTORY_PATH):
        open(config.CHAT_HISTORY_PATH, "w").close()
    return jsonify({"status": "cleared"})


@app.route("/api/tts/stream", methods=["POST"])
def tts_stream():
    d = request.get_json(force=True)
    def gen():
        for b64 in sarvam.text_to_speech_stream(d.get("text", ""), d.get("language", "en-IN")):
            yield json.dumps({"audio": b64}) + "\n"
    return Response(stream_with_context(gen()), mimetype="application/x-ndjson")

@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(config.IMAGES_DIR, filename)

if __name__ == "__main__":
    rag.warmup()
    app.run(host="0.0.0.0", port=8003, debug=False, use_reloader=False)