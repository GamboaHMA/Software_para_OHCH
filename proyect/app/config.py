import os

class Config():
    SECRET_KEY = "dev-secret-key"
    BAS_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    DATA_DIR = os.path.join(BAS_DIR, "data")
    PDF_DIR = os.path.join(DATA_DIR, "pdfs")
    OCR_DIR = os.path.join(DATA_DIR, "ocr")
    DB_PATH = os.path.join(DATA_DIR, "db", "corpus.db")
    MAX_CONTENT_LENGTH = 200 * 1024 * 1024