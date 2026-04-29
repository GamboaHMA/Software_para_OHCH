from datetime import datetime, timezone
from app import db


class Document(db.Model):
    __tablename__ = "documents"

    id = db.Column(db.Integer, primary_key=True)
    doc_id = db.Column(db.String(50), unique=True, nullable=False, index=True)
    title = db.Column(db.String(500), nullable=False)
    archive = db.Column(db.String(400))
    signature = db.Column(db.String(300))
    date_raw = db.Column(db.String(150))
    date_year = db.Column(db.Integer, index=True)
    century = db.Column(db.Integer, index=True)
    location_country = db.Column(db.String(100), default="España")
    location_region = db.Column(db.String(150))
    location_city = db.Column(db.String(150))
    location_place = db.Column(db.String(300))
    classification = db.Column(db.String(100), index=True)
    author = db.Column(db.String(300))
    language = db.Column(db.String(50), default="español")
    ocr_text = db.Column(db.Text)
    ocr_text_edited = db.Column(db.Text)
    ocr_quality = db.Column(db.Float)
    pdf_filename = db.Column(db.String(400))
    notes = db.Column(db.Text)
    imported_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    @property
    def display_text(self):
        """Return edited OCR if available, otherwise original."""
        return self.ocr_text_edited or self.ocr_text or ""

    @property
    def has_pdf(self):
        return bool(self.pdf_filename)

    @property
    def is_edited(self):
        return bool(self.ocr_text_edited)

    def __repr__(self):
        return f"<Document {self.doc_id}>"
