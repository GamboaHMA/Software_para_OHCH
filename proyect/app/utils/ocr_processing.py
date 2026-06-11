"""
OCR processing pipeline for the web OCR tool.

Steps
-----
1. PDF → text via PyMuPDF + Tesseract (pytesseract fallback if needed).
   Each page emits an SSE progress event (0 – 50 %).
2. Text → preprocessed text via one of:
   - 'salamandra' : Ollama hdnh2006/salamandra-7b-instruct:latest
   - 'qwen'       : Ollama qwen2.5:3b
   - 'artesanal'  : local StringChecker (spacy + hunspell + kenlm)
   - 'ninguno'    : no preprocessing
   Preprocessing phase emits SSE events (50 – 100 %).
3. Final 'done' event carries the processed text.

All events are plain dicts pushed into a task-specific queue.Queue; the Flask
SSE endpoint drains the queue and forwards events to the browser.
"""
from __future__ import annotations

import builtins
import os
import queue
import sys
import tempfile
import threading
import traceback
import uuid

# ---------------------------------------------------------------------------
# Workspace root (parent of proyect/)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))          # proyect/app/utils/
_WORKSPACE_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))

# ---------------------------------------------------------------------------
# Task store
# Each entry: {"queue": Queue, "pdf_path": str, "pdf_name": str,
#              "raw_text": str|None, "processed_text": str|None}
# The temporary PDF is kept until the user explicitly saves (or discards).
# ---------------------------------------------------------------------------
_tasks: dict[str, dict] = {}
_tasks_lock = threading.Lock()


def create_task(pdf_path: str, pdf_name: str) -> str:
    task_id = str(uuid.uuid4())
    with _tasks_lock:
        _tasks[task_id] = {
            "queue": queue.Queue(maxsize=1000),
            "pdf_path": pdf_path,
            "pdf_name": pdf_name,
            "raw_text": None,
            "processed_text": None,
        }
    return task_id


def get_task_queue(task_id: str) -> queue.Queue | None:
    entry = _tasks.get(task_id)
    return entry["queue"] if entry else None


def get_task_result(task_id: str) -> dict | None:
    """Return the stored result dict (pdf_path, pdf_name, raw_text, processed_text)
    or None if the task does not exist."""
    entry = _tasks.get(task_id)
    if entry is None:
        return None
    return {
        "pdf_path": entry["pdf_path"],
        "pdf_name": entry["pdf_name"],
        "raw_text": entry["raw_text"],
        "processed_text": entry["processed_text"],
    }


def cleanup_task(task_id: str) -> None:
    """Delete the temp PDF (if it still exists) and remove the task entry."""
    with _tasks_lock:
        entry = _tasks.pop(task_id, None)
    if entry and entry["pdf_path"]:
        try:
            os.unlink(entry["pdf_path"])
        except OSError:
            pass


def remove_task(task_id: str) -> None:
    """Remove task entry WITHOUT deleting the PDF (SSE stream close)."""
    with _tasks_lock:
        _tasks.pop(task_id, None)


# ---------------------------------------------------------------------------
# Model configs (mirrors procesar_todos_salamander_local.py)
# ---------------------------------------------------------------------------
_SYSTEM_MSG = (
    "Eres un experto en transcripción y corrección de documentos históricos en español. "
    "Tu única tarea es corregir errores de OCR sin alterar el contenido ni el estilo. "
    "Devuelve ÚNICAMENTE el texto reconstruido, sin explicaciones, sin comentarios, "
    "sin encabezados, sin delimitadores de bloque de código."
)

_USER_TEMPLATE = (
    "El siguiente texto fue extraído mediante OCR de un documento histórico impreso y "
    "contiene errores típicos: caracteres aislados sin sentido (letras o símbolos sueltos "
    "en líneas propias), palabras partidas con guión al final de línea que deben unirse, "
    "espacios incorrectos dentro de palabras, confusión entre caracteres similares "
    "(l/1, o/0, rn/m, cl/d, etc.) y signos de puntuación mal reconocidos.\n"
    "Reconstruye el texto corrigiendo únicamente los errores de OCR. Conserva el español "
    "histórico, la ortografía de la época (tildes arcaicas, formas antiguas) y el "
    "contenido exacto: NO añadas, inventes ni elimines información.\n\n"
    "TEXTO OCR:\n{text}"
)

_MODEL_CONFIGS: dict[str, dict] = {
    "salamandra": {
        "model": "hdnh2006/salamandra-7b-instruct:latest",
        "max_chars": 2_000,
        "options": {
            "num_ctx": 4096,
            "temperature": 0.1,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_gpu": 22,
        },
    },
    "qwen": {
        "model": "qwen2.5:3b",
        "max_chars": 6_000,
        "options": {
            "num_ctx": 8192,
            "temperature": 0.1,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_gpu": 99,
        },
    },
}

# ---------------------------------------------------------------------------
# Text chunking (adapted from procesar_todos_salamander_local.py)
# ---------------------------------------------------------------------------

def _chunk_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        needed = len(para) + (2 if current else 0)
        if current_len + needed > max_chars and current:
            chunks.append("\n\n".join(current))
            current, current_len = [], 0
        if len(para) > max_chars:
            sentences = para.replace(". ", ".\n").split("\n")
            for sent in sentences:
                needed_s = len(sent) + (2 if current else 0)
                if current_len + needed_s > max_chars and current:
                    chunks.append("\n\n".join(current))
                    current, current_len = [], 0
                current.append(sent)
                current_len += needed_s
        else:
            current.append(para)
            current_len += needed

    if current:
        chunks.append("\n\n".join(current))

    return chunks


# ---------------------------------------------------------------------------
# Phase 1: OCR  (pdf2image → pytesseract, same approach as probando_tesseract.py)
# ---------------------------------------------------------------------------

def _ocr_pdf(pdf_path: str, q: queue.Queue) -> str:
    """Convert each PDF page to an image via pdf2image (poppler) and run
    Tesseract OCR on it via pytesseract.  Page count is known upfront so the
    progress bar can be accurate from the start."""
    from pdf2image import convert_from_path  # type: ignore
    import pytesseract  # type: ignore

    q.put({
        "type": "progress",
        "step": "ocr",
        "current": 0,
        "total": 1,
        "pct": 2,
        "msg": "Convirtiendo PDF a imágenes…",
    })

    images = convert_from_path(pdf_path, dpi=300)
    total_pages = len(images)
    texts: list[str] = []

    for i, img in enumerate(images):
        pct = 2 + int(((i) / max(total_pages, 1)) * 48)  # 2 → 50 %
        q.put({
            "type": "progress",
            "step": "ocr",
            "current": i + 1,
            "total": total_pages,
            "pct": pct,
            "msg": f"OCR: página {i + 1} / {total_pages}",
        })
        text = pytesseract.image_to_string(img, lang="spa")
        texts.append(text)

    q.put({
        "type": "progress",
        "step": "ocr",
        "current": total_pages,
        "total": total_pages,
        "pct": 50,
        "msg": "OCR completado",
    })

    return "\n\n".join(texts)


# ---------------------------------------------------------------------------
# Phase 2: Preprocessing
# ---------------------------------------------------------------------------

def _preprocess_ollama(text: str, model_key: str, q: queue.Queue) -> str:
    from ollama import chat  # type: ignore

    cfg = _MODEL_CONFIGS[model_key]
    model_name: str = cfg["model"]
    max_chars: int = cfg["max_chars"]
    options: dict = cfg["options"]

    chunks = _chunk_text(text, max_chars)
    total_chunks = len(chunks)
    results: list[str] = []

    for i, chunk in enumerate(chunks):
        pct = 50 + int((i / max(total_chunks, 1)) * 48)
        q.put({
            "type": "progress",
            "step": "preprocess",
            "current": i + 1,
            "total": total_chunks,
            "pct": pct,
            "msg": f"Preprocesando con {model_key} — fragmento {i + 1} / {total_chunks}",
        })

        messages = [
            {"role": "system", "content": _SYSTEM_MSG},
            {"role": "user", "content": _USER_TEMPLATE.format(text=chunk)},
        ]
        resp = chat(model=model_name, messages=messages, options=options)
        results.append(resp.message.content.strip())

    return "\n\n".join(results)


def _preprocess_artisanal(text: str, q: queue.Queue) -> str:
    # Ensure workspace root is importable
    if _WORKSPACE_ROOT not in sys.path:
        sys.path.insert(0, _WORKSPACE_ROOT)

    _real_print = builtins.print
    builtins.print = lambda *a, **kw: None  # silence internal debug output

    try:
        from anomalies_checker import StringChecker, limpiar_espacios_basicos  # type: ignore

        q.put({
            "type": "progress",
            "step": "preprocess",
            "current": 1,
            "total": 4,
            "pct": 55,
            "msg": "Cargando modelo artesanal (spacy + hunspell + kenlm)…",
        })

        lm_path = os.path.join(_WORKSPACE_ROOT, "anagrama_models", "modelo_combinado_5.bin")
        checker = StringChecker(lm_model=lm_path)
        cleaned = limpiar_espacios_basicos(text)

        q.put({
            "type": "progress",
            "step": "preprocess",
            "current": 2,
            "total": 4,
            "pct": 68,
            "msg": "Analizando texto con spaCy…",
        })

        doc = checker.nlp(cleaned)
        anomaly_chain = []
        for token in doc:
            is_valid = checker._spell_cache.get(token.text)
            if is_valid is None:
                is_valid = checker.hobj.spell(token.text)
                checker._spell_cache[token.text] = is_valid
            anomaly_chain.append((token, is_valid))

        q.put({
            "type": "progress",
            "step": "preprocess",
            "current": 3,
            "total": 4,
            "pct": 82,
            "msg": "Precalentando sugerencias de hunspell…",
        })

        for token, valid in anomaly_chain:
            if not valid and token.text not in checker._suggest_cache:
                checker._suggest_cache[token.text] = checker.hobj.suggest(token.text)[:5]

        q.put({
            "type": "progress",
            "step": "preprocess",
            "current": 4,
            "total": 4,
            "pct": 94,
            "msg": "Corrigiendo con programación dinámica…",
        })

        checker.correjir_cadena(anomaly_chain)
        return checker.last_corrected_text

    finally:
        builtins.print = _real_print


# ---------------------------------------------------------------------------
# Background task runner
# ---------------------------------------------------------------------------

def _run_task(task_id: str, pdf_path: str, model: str) -> None:
    q = get_task_queue(task_id)
    if q is None:
        return

    try:
        # ── Phase 1: OCR ──────────────────────────────────────────────────────
        raw_text = _ocr_pdf(pdf_path, q)

        if not raw_text.strip():
            q.put({
                "type": "error",
                "msg": "El OCR no produjo texto. Verifique que el PDF sea legible.",
            })
            return

        # ── Phase 2: Preprocessing ────────────────────────────────────────────
        if model == "ninguno":
            final_text = raw_text
            q.put({
                "type": "progress",
                "step": "preprocess",
                "current": 1,
                "total": 1,
                "pct": 100,
                "msg": "Sin preprocesamiento — texto OCR directo",
            })
        elif model in ("salamandra", "qwen"):
            final_text = _preprocess_ollama(raw_text, model, q)
        elif model == "artesanal":
            final_text = _preprocess_artisanal(raw_text, q)
        else:
            final_text = raw_text

        # Store results in the task entry so the save endpoint can retrieve them
        with _tasks_lock:
            if task_id in _tasks:
                _tasks[task_id]["raw_text"] = raw_text
                _tasks[task_id]["processed_text"] = final_text

        q.put({
            "type": "done",
            "raw_text": raw_text,
            "processed_text": final_text,
            "pct": 100,
        })

    except Exception as exc:  # noqa: BLE001
        q.put({
            "type": "error",
            "msg": str(exc),
            "trace": traceback.format_exc(),
        })
        # On error still clean up the temp PDF
        try:
            os.unlink(pdf_path)
        except OSError:
            pass
        with _tasks_lock:
            _tasks.pop(task_id, None)


def start_ocr_task(pdf_bytes: bytes, pdf_name: str, model: str) -> str:
    """Save *pdf_bytes* to a temp file, register a task entry and launch a
    background thread.  Returns the task_id for the SSE stream endpoint."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    task_id = create_task(pdf_path=tmp_path, pdf_name=pdf_name)

    t = threading.Thread(
        target=_run_task,
        args=(task_id, tmp_path, model),
        daemon=True,
    )
    t.start()

    return task_id
