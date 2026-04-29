import os
import csv
from flask import current_app
from app import db
from app.models import Document
from app.utils.text_cleaning import clean_ocr_text
from app.utils.filesystem import save_uploaded_file


def _year_to_century(year) -> int | None:
    if year is None:
        return None
    return ((int(year) - 1) // 100) + 1


def import_single_document(request_obj) -> dict:
    """Import one document from a multipart/form-data POST request."""
    form = request_obj.form
    files = request_obj.files

    doc_id = form.get("doc_id", "").strip()
    title = form.get("title", "").strip()

    if not doc_id or not title:
        return {"ok": False, "error": "doc_id y título son obligatorios."}

    if Document.query.filter_by(doc_id=doc_id).first():
        return {"ok": False, "error": f"Ya existe un documento con ID '{doc_id}'."}

    # Save PDF if provided
    pdf_filename = None
    if "pdf_file" in files and files["pdf_file"].filename:
        pdf_filename = save_uploaded_file(files["pdf_file"], current_app.config["PDF_DIR"])

    # OCR text: file upload takes priority over textarea
    ocr_text = None
    if "ocr_file" in files and files["ocr_file"].filename:
        raw = files["ocr_file"].read().decode("utf-8", errors="replace")
        ocr_text = clean_ocr_text(raw)
    elif form.get("ocr_text"):
        ocr_text = clean_ocr_text(form.get("ocr_text"))

    date_year = form.get("date_year", type=int)
    century = _year_to_century(date_year)

    doc = Document(
        doc_id=doc_id,
        title=title,
        archive=form.get("archive") or None,
        signature=form.get("signature") or None,
        date_raw=form.get("date_raw") or None,
        date_year=date_year,
        century=century,
        location_country=form.get("location_country") or "España",
        location_region=form.get("location_region") or None,
        location_city=form.get("location_city") or None,
        location_place=form.get("location_place") or None,
        classification=form.get("classification") or None,
        author=form.get("author") or None,
        language=form.get("language") or "español",
        ocr_text=ocr_text,
        ocr_quality=form.get("ocr_quality", type=float),
        pdf_filename=pdf_filename,
        notes=form.get("notes") or None,
    )
    db.session.add(doc)
    db.session.commit()

    return {
        "ok": True,
        "doc_id": doc.id,
        "message": f"Documento '{doc_id}' importado correctamente.",
    }


def import_from_csv(csv_path: str) -> dict:
    """
    Bulk-import documents from a CSV file.

    Expected columns: doc_id, title, archive, signature, date_raw, date_year,
    location_country, location_region, location_city, location_place,
    classification, author, language, pdf_filename, ocr_quality, notes

    PDFs must already be in data/pdfs/; OCR text files in data/ocr/ with
    the same base name as the PDF but with .txt extension.
    """
    imported = 0
    skipped = 0
    errors = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = row.get("doc_id", "").strip()
            if not doc_id:
                errors.append("Fila sin doc_id omitida.")
                skipped += 1
                continue

            if Document.query.filter_by(doc_id=doc_id).first():
                skipped += 1
                continue

            date_year = None
            raw_year = row.get("date_year", "").strip()
            if raw_year:
                try:
                    date_year = int(raw_year)
                except ValueError:
                    pass

            pdf_filename = row.get("pdf_filename", "").strip() or None
            ocr_text = None
            if pdf_filename:
                ocr_path = os.path.join(
                    current_app.config["OCR_DIR"],
                    os.path.splitext(pdf_filename)[0] + ".txt",
                )
                if os.path.isfile(ocr_path):
                    with open(ocr_path, encoding="utf-8", errors="replace") as ocr_f:
                        ocr_text = clean_ocr_text(ocr_f.read())

            ocr_quality = None
            if row.get("ocr_quality"):
                try:
                    ocr_quality = float(row["ocr_quality"])
                except ValueError:
                    pass

            doc = Document(
                doc_id=doc_id,
                title=row.get("title") or doc_id,
                archive=row.get("archive") or None,
                signature=row.get("signature") or None,
                date_raw=row.get("date_raw") or None,
                date_year=date_year,
                century=_year_to_century(date_year),
                location_country=row.get("location_country") or "España",
                location_region=row.get("location_region") or None,
                location_city=row.get("location_city") or None,
                location_place=row.get("location_place") or None,
                classification=row.get("classification") or None,
                author=row.get("author") or None,
                language=row.get("language") or "español",
                ocr_text=ocr_text,
                ocr_quality=ocr_quality,
                pdf_filename=pdf_filename,
                notes=row.get("notes") or None,
            )
            db.session.add(doc)
            imported += 1

    db.session.commit()
    return {"imported": imported, "skipped": skipped, "errors": errors}
