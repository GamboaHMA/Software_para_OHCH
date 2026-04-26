from flask import Blueprint, render_template, url_for

bp = Blueprint("main", __name__)

DOCUMENTS = [
    {
        "id": 1,
        "title": "Acta notarial de 1821",
        "classification": "documento legal",
        "date": "1821-09-15",
        "author": "Anónimo",
        "ocr_quality": "0.91",
        "pdf_file": "acta_1821.pdf",
    },
    {
        "id": 2,
        "title": "Carta histórica de 1798",
        "classification": "correspondencia",
        "date": "1798-03-02",
        "author": "Autor desconocido",
        "ocr_quality": "0.87",
        "pdf_file": "carta_1798.pdf",
    },
]

@bp.route("/", methods=['GET'])
def index() -> str:
    return render_template("index.html", documents=DOCUMENTS)

@bp.route("/documents/<int:doc_id>", methods=['GET'])
def document_detail(doc_id):
    doc = next((d for d in DOCUMENTS if d["id"] == doc_id), None)
    if doc is None:
        return "Documento no encontrado", 404
    
    pdf_url = url_for("static", filename=f"pdfs/{doc['pdf_file']}")
    return render_template("document_detail.html", document=doc, pdf_url=pdf_url)