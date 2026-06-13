#!/usr/bin/env python3
"""
load_from_reconstrucciones.py
─────────────────────────────
Empareja los PDFs en proyect/data/pdfs/ con los textos reconstruidos en
Reconstrucciones_qwen_deepseek/ e inserta / actualiza los registros en la
base de datos de la aplicación.

Uso básico:
    cd proyect/
    python load_from_reconstrucciones.py               # inserta los nuevos
    python load_from_reconstrucciones.py --dry-run     # solo muestra el match, sin tocar la DB
    python load_from_reconstrucciones.py --force       # sobreescribe registros existentes
    python load_from_reconstrucciones.py --solo-pdf    # incluye PDFs que todavía no tienen TXT

Estructura esperada:
  proyect/data/pdfs/
      ohcbh_cf_erlXXX/
          PDF_ohcbh_cf_erlXXX/
              ohcbh_cf_erlXXX_bn.pdf

  ../Reconstrucciones_qwen_deepseek/
      Articulos_ohcbh_cf_erlXXX/
          ohcbh_cf_erlXXX_001.pdf.txt
          ohcbh_cf_erlXXX_002.pdf.txt
          ...
"""

import os
import sys
import re
import csv
import argparse
from pathlib import Path

# ─── Ruta al paquete app (un nivel arriba de este script = proyect/) ──────────
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from app import create_app, db
from app.models import Document
from app.utils.text_cleaning import clean_ocr_text

# ─── Directorios ──────────────────────────────────────────────────────────────
ROOT_DIR = SCRIPT_DIR.parent                          # workspace root
RECONSTRUCCIONES_DIR = ROOT_DIR / "Reconstrucciones_qwen_deepseek"
CSV_PATH = ROOT_DIR / "Listado completo Facticia.csv"
PDF_DIR = SCRIPT_DIR / "data" / "pdfs"

# ─── Patrón de identificador de documento ─────────────────────────────────────
DOC_ID_RE = re.compile(r"^ohcbh_cf_erl(\d+[A-Za-z]?)$")


# ──────────────────────────────────────────────────────────────────────────────
# Fase 1: Escaneo de PDFs
# ──────────────────────────────────────────────────────────────────────────────

def _find_pdf_in_subdir(pdf_subdir: Path, doc_id: str) -> Path | None:
    """
    Busca el PDF del documento dentro de pdf_subdir con tolerancia a:
      - sufijo _bn opcional (ohcbh_cf_erlXXX_bn.pdf o ohcbh_cf_erlXXX.pdf)
      - diferencias de mayúsculas/minúsculas en la letra al final del ID
    """
    # Candidatos en orden de preferencia
    candidates = [
        f"{doc_id}_bn.pdf",
        f"{doc_id}.pdf",
        f"{doc_id.lower()}_bn.pdf",
        f"{doc_id.lower()}.pdf",
    ]
    for name in candidates:
        p = pdf_subdir / name
        if p.is_file():
            return p
    # Fallback: cualquier PDF en el directorio
    pdf_files = [f for f in pdf_subdir.iterdir() if f.suffix.lower() == ".pdf" and f.name != "index.html"]
    if len(pdf_files) == 1:
        return pdf_files[0]
    return None


def scan_pdfs() -> dict[str, Path]:
    """
    Recorre PDF_DIR buscando archivos PDF (preferiblemente *_bn.pdf).

    Retorna {doc_id: ruta_relativa_desde_PDF_DIR}
    Ejemplo: {'ohcbh_cf_erl1': PosixPath('ohcbh_cf_erl1/PDF_ohcbh_cf_erl1/ohcbh_cf_erl1_bn.pdf')}
    """
    pdfs: dict[str, Path] = {}
    if not PDF_DIR.is_dir():
        print(f"[ERROR] PDF_DIR no encontrado: {PDF_DIR}")
        return pdfs

    for doc_dir in sorted(PDF_DIR.iterdir()):
        if not doc_dir.is_dir():
            continue
        doc_id = doc_dir.name
        if not DOC_ID_RE.match(doc_id):
            continue

        pdf_subdir = doc_dir / f"PDF_{doc_id}"
        if not pdf_subdir.is_dir():
            continue

        pdf_file = _find_pdf_in_subdir(pdf_subdir, doc_id)
        if pdf_file:
            pdfs[doc_id] = pdf_file.relative_to(PDF_DIR)

    return pdfs


# ──────────────────────────────────────────────────────────────────────────────
# Fase 2: Escaneo de textos reconstruidos
# ──────────────────────────────────────────────────────────────────────────────

def _sort_key(path: Path) -> tuple:
    """Ordena txt por número de página embedded en el nombre."""
    m = re.search(r"_(\d+)\.pdf\.txt$", path.name)
    return (int(m.group(1)) if m else 0, path.name)


def scan_reconstrucciones() -> dict[str, list[Path]]:
    """
    Recorre Reconstrucciones_qwen_deepseek/ buscando subcarpetas Articulos_*.

    Retorna {doc_id: lista_de_txts_ordenada_por_página}
    """
    txts: dict[str, list[Path]] = {}
    if not RECONSTRUCCIONES_DIR.is_dir():
        print(f"[ERROR] Reconstrucciones no encontradas: {RECONSTRUCCIONES_DIR}")
        return txts

    for art_dir in sorted(RECONSTRUCCIONES_DIR.iterdir()):
        if not art_dir.is_dir():
            continue
        if not art_dir.name.startswith("Articulos_"):
            continue

        doc_id = art_dir.name[len("Articulos_"):]
        if not DOC_ID_RE.match(doc_id):
            continue

        txt_files = sorted(
            [f for f in art_dir.iterdir() if f.suffix == ".txt"],
            key=_sort_key,
        )
        if txt_files:
            txts[doc_id] = txt_files

    return txts


# ──────────────────────────────────────────────────────────────────────────────
# Fase 3: Carga de metadatos del CSV
# ──────────────────────────────────────────────────────────────────────────────

def load_csv_metadata() -> dict[str, dict]:
    """
    Lee el CSV 'Listado completo Facticia.csv' y construye un diccionario
    {doc_id: fila_de_metadatos}.

    Solo retiene la primera fila por documento (la del _bn.pdf).
    """
    meta: dict[str, dict] = {}
    if not CSV_PATH.is_file():
        print(f"[WARN] CSV no encontrado: {CSV_PATH}. Se omitirán metadatos.")
        return meta

    with open(CSV_PATH, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            nv = row.get("Nombre viejo", "").strip()
            # Solo queremos las filas del PDF _bn (representan el documento completo)
            m = re.match(r"(ohcbh_cf_erl\d+[A-Za-z]?)_bn\.pdf$", nv)
            if m:
                doc_id = m.group(1)
                if doc_id not in meta:
                    meta[doc_id] = row

    return meta


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def concat_texts(txt_files: list[Path]) -> str:
    """Lee y concatena los archivos de texto en orden de página."""
    parts = []
    for f in txt_files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                parts.append(text)
        except OSError as exc:
            print(f"  [WARN] No se pudo leer {f.name}: {exc}")
    return "\n\n".join(parts)


def _parse_year(raw: str) -> int | None:
    """Intenta extraer un año de 4 dígitos de un string de fecha."""
    if not raw:
        return None
    m = re.search(r"\b(1[3-9]\d{2}|20[01]\d)\b", raw)
    return int(m.group(1)) if m else None


def _year_to_century(year: int | None) -> int | None:
    if year is None:
        return None
    return ((year - 1) // 100) + 1


def _metadata_fields(row: dict) -> dict:
    """Extrae los campos relevantes de una fila del CSV."""
    title = (row.get("Título") or "").strip()
    author = (row.get("Autor") or "").strip() or None
    # Localización = signatura archivística (p.ej. F ERL 001)
    signature = (row.get("Localización") or "").strip() or None
    doc_type = (row.get("Tipo de documento") or "").strip() or None
    notes = (row.get("Nota General/Observaciones") or "").strip() or None
    location_city = (row.get("Lugar Fuente") or "").strip() or None
    date_raw = (row.get("Fecha Fuente") or "").strip() or None
    date_year = _parse_year(date_raw)
    return {
        "title": title,
        "author": author,
        "archive": "OHCBH - Fondo Emilio Roig de Leuchsenring",
        "signature": signature,
        "classification": doc_type,
        "notes": notes,
        "location_country": "Cuba",
        "location_city": location_city or "La Habana",
        "date_raw": date_raw,
        "date_year": date_year,
        "century": _year_to_century(date_year),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Fase 4: Carga en la base de datos
# ──────────────────────────────────────────────────────────────────────────────

def run(dry_run: bool = False, force: bool = False, solo_pdf: bool = False) -> None:
    app = create_app()
    with app.app_context():

        print("=" * 60)
        print("Cargando metadatos del CSV...")
        metadata = load_csv_metadata()
        print(f"  {len(metadata)} documentos encontrados en el CSV.")

        print("\nEscaneando PDFs...")
        pdfs = scan_pdfs()
        print(f"  {len(pdfs)} PDFs encontrados en data/pdfs/.")

        print("\nEscaneando textos reconstruidos...")
        txts = scan_reconstrucciones()
        print(f"  {len(txts)} carpetas de texto encontradas en Reconstrucciones_qwen_deepseek/.")

        # ── Match ──────────────────────────────────────────────────────────────
        matched: dict[str, tuple] = {
            doc_id: (pdfs[doc_id], txts[doc_id])
            for doc_id in pdfs
            if doc_id in txts
        }
        pdf_only = sorted(set(pdfs) - set(txts))   # PDF sin TXT aún
        txt_only = sorted(set(txts) - set(pdfs))   # TXT sin PDF (inesperado)

        print("\n" + "─" * 60)
        print(f"  ✓ Emparejados (PDF + TXT): {len(matched)}")
        print(f"  ⏳ Solo PDF (TXT pendiente): {len(pdf_only)}")
        if txt_only:
            print(f"  ⚠  Solo TXT (sin PDF): {len(txt_only)} — {txt_only}")
        print("─" * 60)

        if dry_run:
            print("\n[DRY-RUN] No se modificará la base de datos.")
            print("\nDocumentos que se insertarían/actualizarían:")
            for doc_id in sorted(matched):
                pdf_rel, txts_list = matched[doc_id]
                row = metadata.get(doc_id, {})
                title = (row.get("Título") or "").strip() or doc_id
                print(f"  {doc_id:30s}  PDF={str(pdf_rel):<60s}  TXTs={len(txts_list)}  Título={title[:50]!r}")
            if solo_pdf:
                print(f"\nDocumentos solo-PDF que se insertarían (sin texto aún):")
                for doc_id in pdf_only:
                    pdf_rel = pdfs[doc_id]
                    row = metadata.get(doc_id, {})
                    title = (row.get("Título") or "").strip() or doc_id
                    print(f"  {doc_id:30s}  PDF={str(pdf_rel)}")
            return

        inserted = 0
        updated = 0
        skipped = 0
        errors = 0

        def _process(doc_id: str, pdf_rel: Path, txt_files: list[Path] | None) -> None:
            nonlocal inserted, updated, skipped, errors
            try:
                existing = Document.query.filter_by(doc_id=doc_id).first()

                if existing and not force:
                    skipped += 1
                    return

                # Texto OCR concatenado
                ocr_text = None
                if txt_files:
                    raw = concat_texts(txt_files)
                    ocr_text = clean_ocr_text(raw) if raw else None

                # Metadatos del CSV
                row = metadata.get(doc_id, {})
                fields = _metadata_fields(row)
                if not fields["title"]:
                    fields["title"] = doc_id

                if existing:
                    # Actualiza campos — con --force sobreescribe todo
                    if force or not existing.ocr_text:
                        existing.ocr_text = ocr_text
                    if force or not existing.pdf_filename:
                        existing.pdf_filename = str(pdf_rel)
                    if force or not existing.title or existing.title == existing.doc_id:
                        existing.title = fields["title"]
                    if force or not existing.author:
                        existing.author = fields["author"]
                    if force or not existing.archive:
                        existing.archive = fields["archive"]
                    if force or not existing.signature:
                        existing.signature = fields["signature"]
                    if force or not existing.classification:
                        existing.classification = fields["classification"]
                    if force or not existing.notes:
                        existing.notes = fields["notes"]
                    if force or not existing.location_city:
                        existing.location_city = fields["location_city"]
                    if force or not existing.location_country:
                        existing.location_country = fields["location_country"]
                    if force or not existing.date_raw:
                        existing.date_raw = fields["date_raw"]
                    if force or not existing.date_year:
                        existing.date_year = fields["date_year"]
                        existing.century = fields["century"]
                    db.session.commit()
                    updated += 1
                    print(f"  [UPD] {doc_id}")
                else:
                    doc = Document(
                        doc_id=doc_id,
                        pdf_filename=str(pdf_rel),
                        ocr_text=ocr_text,
                        language="español",
                        **fields,
                    )
                    db.session.add(doc)
                    db.session.commit()
                    inserted += 1
                    print(f"  [NEW] {doc_id}")

            except Exception as exc:
                db.session.rollback()
                errors += 1
                print(f"  [ERR] {doc_id}: {exc}")

        # ── Procesar documentos emparejados (PDF + TXT) ────────────────────────
        print("\nProcesando documentos emparejados...")
        for doc_id in sorted(matched):
            pdf_rel, txt_files = matched[doc_id]
            _process(doc_id, pdf_rel, txt_files)

        # ── Procesar solo-PDF si se solicitó ──────────────────────────────────
        if solo_pdf and pdf_only:
            print(f"\nProcesando {len(pdf_only)} documentos solo-PDF (sin texto aún)...")
            for doc_id in pdf_only:
                _process(doc_id, pdfs[doc_id], None)

        # ── Resumen ───────────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print(f"RESUMEN")
        print(f"  Insertados:  {inserted}")
        print(f"  Actualizados: {updated}")
        print(f"  Omitidos (ya existían): {skipped}")
        print(f"  Errores: {errors}")
        if pdf_only and not solo_pdf:
            print(f"\n  {len(pdf_only)} PDFs sin texto todavía (usa --solo-pdf para cargarlos sin TXT).")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Empareja PDFs y textos reconstruidos, y los carga en la base de datos.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo muestra el emparejamiento sin modificar la base de datos.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Sobreescribe registros existentes con los nuevos datos.",
    )
    parser.add_argument(
        "--solo-pdf",
        action="store_true",
        help="También inserta documentos que solo tienen PDF (sin texto reconstruido aún).",
    )
    args = parser.parse_args()

    run(dry_run=args.dry_run, force=args.force, solo_pdf=args.solo_pdf)


if __name__ == "__main__":
    main()
