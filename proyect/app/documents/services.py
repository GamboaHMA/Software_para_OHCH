from datetime import datetime, timezone
from sqlalchemy import or_
from app import db
from app.models import Document


def get_documents(
    page: int = 1,
    per_page: int = 24,
    search: str | None = None,
    classification: str | None = None,
    date_from: int | None = None,
    date_to: int | None = None,
    century: int | None = None,
):
    q = Document.query

    if search:
        term = f"%{search}%"
        q = q.filter(
            or_(
                Document.title.ilike(term),
                Document.author.ilike(term),
                Document.archive.ilike(term),
                Document.doc_id.ilike(term),
                Document.ocr_text.ilike(term),
                Document.ocr_text_edited.ilike(term),
            )
        )

    if classification:
        q = q.filter(Document.classification == classification)

    if date_from is not None:
        q = q.filter(Document.date_year >= date_from)

    if date_to is not None:
        q = q.filter(Document.date_year <= date_to)

    if century is not None:
        q = q.filter(Document.century == century)

    return q.order_by(
        Document.date_year.asc().nullslast(), Document.doc_id.asc()
    ).paginate(page=page, per_page=per_page, error_out=False)


def get_document(doc_id: int):
    return db.get_or_404(Document, doc_id)


def get_classifications() -> list[str]:
    rows = (
        db.session.query(Document.classification)
        .filter(Document.classification.isnot(None), Document.classification != "")
        .distinct()
        .all()
    )
    return sorted(r[0] for r in rows)


def get_centuries() -> list[int]:
    rows = (
        db.session.query(Document.century)
        .filter(Document.century.isnot(None))
        .distinct()
        .all()
    )
    return sorted(r[0] for r in rows)


def update_ocr_text(doc_id: int, text: str) -> bool:
    doc = db.session.get(Document, doc_id)
    if not doc:
        return False
    doc.ocr_text_edited = text
    doc.updated_at = datetime.now(timezone.utc)
    db.session.commit()
    return True


def update_metadata(doc_id: int, data: dict) -> bool:
    doc = db.session.get(Document, doc_id)
    if not doc:
        return False
    allowed = {
        "title", "archive", "signature", "date_raw", "date_year",
        "location_country", "location_region", "location_city",
        "location_place", "classification", "author", "language", "notes",
    }
    for key, value in data.items():
        if key in allowed:
            setattr(doc, key, value)
    doc.updated_at = datetime.now(timezone.utc)
    db.session.commit()
    return True


def count_documents() -> int:
    return Document.query.count()
