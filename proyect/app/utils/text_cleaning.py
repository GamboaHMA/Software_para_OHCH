import re


def clean_ocr_text(text: str) -> str:
    """Basic normalization for OCR output."""
    if not text:
        return text
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
