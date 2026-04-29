import os
from flask import (
    Blueprint, render_template, request, jsonify,
    abort, send_file, current_app,
)
from app.documents import services

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
