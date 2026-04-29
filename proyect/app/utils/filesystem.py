import os
from flask import current_app


def get_pdf_path(filename: str) -> str:
    return os.path.join(current_app.config["PDF_DIR"], filename)


def pdf_exists(filename: str | None) -> bool:
    if not filename:
        return False
    return os.path.isfile(get_pdf_path(filename))


def save_uploaded_file(file_obj, destination_dir: str, filename: str | None = None) -> str:
    """Save a werkzeug FileStorage object to destination_dir. Returns final filename."""
    filename = filename or file_obj.filename
    dest = os.path.join(destination_dir, filename)
    file_obj.save(dest)
    return filename
