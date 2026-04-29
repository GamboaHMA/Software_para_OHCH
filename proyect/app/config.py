import os

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-prod")
    BASE_DIR = BASE_DIR
    DATA_DIR = os.path.join(BASE_DIR, "data")
    PDF_DIR = os.path.join(BASE_DIR, "data", "pdfs")
    OCR_DIR = os.path.join(BASE_DIR, "data", "ocr")
    METADATA_DIR = os.path.join(BASE_DIR, "data", "metadata")
    DB_DIR = os.path.join(BASE_DIR, "data", "db")
    SQLALCHEMY_DATABASE_URI = (
        f"sqlite:///{os.path.join(BASE_DIR, 'data', 'db', 'corpus.db')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MAX_CONTENT_LENGTH = 200 * 1024 * 1024
    DOCUMENTS_PER_PAGE = 24