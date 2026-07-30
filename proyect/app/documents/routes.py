import json
import os
import queue as _queue_mod

from flask import (
    Blueprint, render_template, request, jsonify,
    abort, send_file, current_app, Response, stream_with_context,
)
from app.documents import services
from app.utils import ocr_processing

bp = Blueprint("main", __name__)


@bp.route("/")
def index():
    page = request.args.get("page", 1, type=int)
    search = request.args.get("q", "").strip()
    classification = request.args.get("cls", "").strip()
    date_from = request.args.get("df", None, type=int)
    date_to = request.args.get("dt", None, type=int)
    century = request.args.get("c", None, type=int)

    pagination = services.get_documents(
        page=page,
        per_page=current_app.config.get("DOCUMENTS_PER_PAGE", 24),
        search=search or None,
        classification=classification or None,
        date_from=date_from,
        date_to=date_to,
        century=century,
    )

    return render_template(
        "index.html",
        pagination=pagination,
        classifications=services.get_classifications(),
        centuries=services.get_centuries(),
        search=search,
        current_cls=classification,
        current_century=century,
        date_from=date_from,
        date_to=date_to,
    )


@bp.route("/documento/<int:doc_id>")
def document_detail(doc_id):
    doc = services.get_document(doc_id)
    return render_template("document_detail.html", document=doc)


@bp.route("/api/documento/<int:doc_id>/ocr", methods=["POST"])
def update_ocr(doc_id):
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Datos inválidos"}), 400
    ok = services.update_ocr_text(doc_id, data["text"])
    if not ok:
        return jsonify({"error": "Documento no encontrado"}), 404
    return jsonify({"ok": True, "message": "OCR guardado correctamente"})


@bp.route("/api/documento/<int:doc_id>/metadata", methods=["POST"])
def update_metadata(doc_id):
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Datos inválidos"}), 400
    ok = services.update_metadata(doc_id, data)
    if not ok:
        return jsonify({"error": "Documento no encontrado"}), 404
    return jsonify({"ok": True})


@bp.route("/api/pdf/<int:doc_id>")
def serve_pdf(doc_id):
    doc = services.get_document(doc_id)
    if not doc.pdf_filename:
        abort(404)
    pdf_path = os.path.join(current_app.config["PDF_DIR"], doc.pdf_filename)
    if not os.path.isfile(pdf_path):
        abort(404)
    return send_file(pdf_path, mimetype="application/pdf")


@bp.route("/importar", methods=["GET", "POST"])
def import_view():
    result = None
    if request.method == "POST":
        from app.utils.imports_docs import import_single_document
        result = import_single_document(request)
    return render_template("import.html", result=result)


# ── OCR Tool ──────────────────────────────────────────────────────────────────

@bp.route("/ocr-herramienta")
def ocr_tool():
    return render_template("ocr_tool.html")


@bp.route("/api/ocr/upload", methods=["POST"])
def ocr_upload():
    if "pdf" not in request.files:
        return jsonify({"error": "No se recibió ningún archivo PDF"}), 400

    pdf_file = request.files["pdf"]
    if not pdf_file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "El archivo debe ser un PDF"}), 400

    model = request.form.get("model", "artesanal")
    if model not in ("salamandra", "qwen", "artesanal", "ninguno"):
        return jsonify({"error": "Modelo no válido"}), 400

    pdf_name = pdf_file.filename
    pdf_bytes = pdf_file.read()
    task_id = ocr_processing.start_ocr_task(pdf_bytes, pdf_name, model)
    return jsonify({"task_id": task_id, "pdf_name": pdf_name})


@bp.route("/api/ocr/save", methods=["POST"])
def ocr_save():
    from datetime import datetime, timezone
    from app.models import Document
    from app import db
    import shutil

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Datos inválidos"}), 400

    task_id = data.get("task_id", "").strip()
    doc_id  = data.get("doc_id", "").strip()
    title   = data.get("title", "").strip()

    if not task_id:
        return jsonify({"error": "task_id requerido"}), 400
    if not doc_id or not title:
        return jsonify({"error": "ID de documento y título son obligatorios"}), 400

    if Document.query.filter_by(doc_id=doc_id).first():
        return jsonify({"error": f"Ya existe un documento con ID '{doc_id}'"}), 409

    result = ocr_processing.get_task_result(task_id)
    if result is None:
        return jsonify({"error": "Tarea no encontrada o ya expirada"}), 404

    # ── Save PDF to data/pdfs/ ─────────────────────────────────────────────
    pdf_dest_name = None
    if result["pdf_path"] and os.path.isfile(result["pdf_path"]):
        safe_name = os.path.basename(result["pdf_name"] or f"{doc_id}.pdf")
        pdf_dest = os.path.join(current_app.config["PDF_DIR"], safe_name)
        # Avoid collisions
        if os.path.exists(pdf_dest):
            base, ext = os.path.splitext(safe_name)
            safe_name = f"{base}_{doc_id}{ext}"
            pdf_dest = os.path.join(current_app.config["PDF_DIR"], safe_name)
        shutil.copy2(result["pdf_path"], pdf_dest)
        pdf_dest_name = safe_name

    # ── Compute century from year ──────────────────────────────────────────
    date_year = data.get("date_year", None)
    try:
        date_year = int(date_year) if date_year not in (None, "") else None
    except (ValueError, TypeError):
        date_year = None
    century = ((date_year - 1) // 100 + 1) if date_year else None

    # ── Create DB record ───────────────────────────────────────────────────
    doc = Document(
        doc_id=doc_id,
        title=title,
        archive=data.get("archive") or None,
        date_raw=data.get("date_raw") or None,
        date_year=date_year,
        century=century,
        location_country=data.get("location_country") or "España",
        classification=data.get("classification") or None,
        author=data.get("author") or None,
        language=data.get("language") or "español",
        ocr_text=result["raw_text"],
        ocr_text_edited=result["processed_text"] if result["processed_text"] != result["raw_text"] else None,
        pdf_filename=pdf_dest_name,
        notes=data.get("notes") or None,
    )
    db.session.add(doc)
    db.session.commit()

    print(
        f"[OCR] Guardado en base de datos: "
        f"doc_id='{doc.doc_id}'  título='{doc.title}'  "
        f"id={doc.id}  pdf='{pdf_dest_name}'",
        flush=True,
    )

    # ── Clean up temp PDF and task ─────────────────────────────────────────
    ocr_processing.cleanup_task(task_id)

    return jsonify({"ok": True, "db_id": doc.id, "doc_id": doc.doc_id})


@bp.route("/api/ocr/stream/<task_id>")
def ocr_stream(task_id):
    def generate():
        q = ocr_processing.get_task_queue(task_id)
        if q is None:
            yield f"data: {json.dumps({'type': 'error', 'msg': 'Tarea no encontrada'})}\n\n"
            return
        try:
            while True:
                try:
                    msg = q.get(timeout=60)
                    yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
                    if msg.get("type") in ("done", "error"):
                        break
                except _queue_mod.Empty:
                    # Keepalive so the browser doesn't time out
                    yield 'data: {"type":"keepalive"}\n\n'
        finally:
            ocr_processing.remove_task(task_id)

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Chat Histórico ─────────────────────────────────────────────────────────────

@bp.route("/chat")
def chat_view():
    """Página principal del chat histórico."""
    from app.utils import chat_service
    models = chat_service.list_available_models()
    ollama_models = list(chat_service.OLLAMA_MODEL_CONFIGS.keys())
    default_ollama = chat_service.DEFAULT_OLLAMA_MODEL
    return render_template(
        "chat.html",
        topic_models=models,
        ollama_models=ollama_models,
        default_ollama=default_ollama,
    )


@bp.route("/api/chat/search", methods=["POST"])
def chat_search():
    """
    Endpoint de búsqueda semántica de documentos.

    Recibe JSON con:
        query       (str)  – consulta histórica del usuario
        model_path  (str)  – ruta relativa al .pkl del modelo de temas
        threshold   (float)– umbral de similitud [0.0, 1.0], default 0.30
        max_results (int)  – máximo de documentos a devolver, default 50
        ollama_model(str)  – alias Ollama para extraer tema clave (opcional)

    Devuelve JSON con:
        documents   – lista de {doc_id, similarity, topic_idx}
        query_topics– distribución de temas de la consulta (para depuración)
        model_used  – ruta del pkl usado
    """
    from app.utils import chat_service
    from app.models import Document

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Datos inválidos"}), 400

    query      = (data.get("query") or "").strip()
    model_path = (data.get("model_path") or "").strip()
    threshold  = float(data.get("threshold", 0.30))
    max_results = int(data.get("max_results", 50))

    # Validaciones básicas de entrada
    if not query:
        return jsonify({"error": "La consulta no puede estar vacía"}), 400
    if not model_path:
        return jsonify({"error": "Se debe especificar un modelo de temas"}), 400
    if not (0.0 <= threshold <= 1.0):
        return jsonify({"error": "El umbral debe estar entre 0 y 1"}), 400

    try:
        state = chat_service.load_topic_model(model_path)
    except (FileNotFoundError, ValueError) as e:
        return jsonify({"error": str(e)}), 404

    # Buscar documentos similares
    results = chat_service.find_relevant_documents(
        query=query,
        state=state,
        threshold=threshold,
        max_results=max_results,
    )

    # Enriquecer resultados con datos de la base de datos
    enriched = []
    for item in results:
        db_doc = Document.query.filter_by(doc_id=item["doc_id"]).first()
        enriched.append({
            "doc_id":     item["doc_id"],
            "similarity": item["similarity"],
            "topic_idx":  item["topic_idx"],
            "db_id":      db_doc.id    if db_doc else None,
            "title":      db_doc.title if db_doc else None,
            "date_raw":   db_doc.date_raw if db_doc else None,
            "in_db":      db_doc is not None,
        })

    # Obtener vector de temas de la consulta (para mostrarlo en UI como debug)
    query_vector = chat_service.query_to_topic_vector(query, state)

    return jsonify({
        "documents":    enriched,
        "query_topics": query_vector.tolist(),
        "model_used":   model_path,
        "total_found":  len(enriched),
    })


@bp.route("/api/chat/summary/stream", methods=["POST"])
def chat_summary_stream():
    """
    Endpoint de generación de resumen histórico con streaming SSE.

    Recibe JSON con:
        query        (str)  – consulta histórica del usuario
        model_path   (str)  – ruta relativa al .pkl del modelo de temas
        ollama_model (str)  – alias del modelo Ollama ('salamandra' o 'qwen')
        threshold    (float)– umbral de similitud [0.0, 1.0], default 0.30
        max_docs     (int)  – máximo de documentos usados para el contexto, default 10

    Devuelve un stream SSE con eventos:
        {"type": "chunk",  "text": "..."}   – fragmento de texto generado
        {"type": "done"}                    – generación completada
        {"type": "error",  "msg":  "..."}   – error durante la generación
    """
    import json as _json
    from app.utils import chat_service

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Datos inválidos"}), 400

    query        = (data.get("query") or "").strip()
    model_path   = (data.get("model_path") or "").strip()
    ollama_alias = (data.get("ollama_model") or chat_service.DEFAULT_OLLAMA_MODEL).strip()
    threshold    = float(data.get("threshold", 0.30))
    max_docs     = int(data.get("max_docs", 10))

    # Validaciones básicas
    if not query:
        return jsonify({"error": "La consulta no puede estar vacía"}), 400
    if not model_path:
        return jsonify({"error": "Se debe especificar un modelo de temas"}), 400
    if ollama_alias not in chat_service.OLLAMA_MODEL_CONFIGS:
        return jsonify({"error": f"Modelo Ollama no reconocido: {ollama_alias}"}), 400

    try:
        state = chat_service.load_topic_model(model_path)
    except (FileNotFoundError, ValueError) as e:
        return jsonify({"error": str(e)}), 404

    # Recuperar documentos relevantes para construir el contexto
    results = chat_service.find_relevant_documents(
        query=query,
        state=state,
        threshold=threshold,
        max_results=max_docs,
    )

    def generate():
        """Genera el stream SSE del resumen de Ollama."""
        if not results:
            yield (
                "data: "
                + _json.dumps(
                    {"type": "error", "msg": "No se encontraron documentos relevantes para el umbral indicado."},
                    ensure_ascii=False,
                )
                + "\n\n"
            )
            return

        try:
            for chunk_text in chat_service.stream_summary(
                query=query,
                results=results,
                state=state,
                ollama_alias=ollama_alias,
            ):
                yield (
                    "data: "
                    + _json.dumps({"type": "chunk", "text": chunk_text}, ensure_ascii=False)
                    + "\n\n"
                )
        except Exception as exc:
            yield (
                "data: "
                + _json.dumps({"type": "error", "msg": str(exc)}, ensure_ascii=False)
                + "\n\n"
            )
            return

        yield "data: " + _json.dumps({"type": "done"}, ensure_ascii=False) + "\n\n"

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
