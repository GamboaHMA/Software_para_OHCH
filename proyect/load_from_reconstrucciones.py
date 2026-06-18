#!/usr/bin/env python3
"""
load_from_reconstrucciones.py
─────────────────────────────
Empareja los PDFs página a página con los textos reconstruidos e inserta /
actualiza los registros en la base de datos de la aplicación.

Uso básico:
    cd proyect/
    python load_from_reconstrucciones.py               # inserta los nuevos
    python load_from_reconstrucciones.py --dry-run     # solo muestra el match, sin tocar la DB
    python load_from_reconstrucciones.py --force       # sobreescribe registros existentes
    python load_from_reconstrucciones.py --solo-pdf    # incluye páginas sin TXT todavía

Estructura esperada:
  proyect/data/pdfs/
      ohcbh_cf_erlXXX/
          Articulos_ohcbh_cf_erlXXX/
              ohcbh_cf_erlXXX_001.pdf   ← PDF de la página 1
              ohcbh_cf_erlXXX_002.pdf
              ...

  ../Reconstrucciones_qwen_deepseek/
      Articulos_ohcbh_cf_erlXXX/
          ohcbh_cf_erlXXX_001.pdf.txt   ← texto reconstruido de la página 1
          ohcbh_cf_erlXXX_002.pdf.txt
          ...

El emparejamiento es página a página: el stem del PDF (sin .pdf) coincide con
el prefijo del TXT (sin .pdf.txt).
  ohcbh_cf_erlXXX_001.pdf  ↔  ohcbh_cf_erlXXX_001.pdf.txt
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

# ─── Patrón de identificador de documento padre ───────────────────────────────
DOC_ID_RE = re.compile(r"^ohcbh_cf_erl(\d+[A-Za-z]?)$")


# ──────────────────────────────────────────────────────────────────────────────
# Fase 1: Escaneo de PDFs página a página
# ──────────────────────────────────────────────────────────────────────────────

def scan_pdfs() -> dict[str, tuple[Path, str]]:
    """
    Recorre PDF_DIR buscando PDFs individuales dentro de
    ohcbh_cf_erlXXX/Articulos_ohcbh_cf_erlXXX/.

    Retorna {page_stem: (ruta_relativa_desde_PDF_DIR, parent_doc_id)}
    Ejemplo:
      'ohcbh_cf_erl1_001' → (PosixPath('ohcbh_cf_erl1/Articulos_.../ohcbh_cf_erl1_001.pdf'), 'ohcbh_cf_erl1')
    """
    pdfs: dict[str, tuple[Path, str]] = {}
    if not PDF_DIR.is_dir():
        print(f"[ERROR] PDF_DIR no encontrado: {PDF_DIR}")
        return pdfs

    for doc_dir in sorted(PDF_DIR.iterdir()):
        if not doc_dir.is_dir():
            continue
        parent_doc_id = doc_dir.name
        if not DOC_ID_RE.match(parent_doc_id):
            continue

        articulos_dir = doc_dir / f"Articulos_{parent_doc_id}"
        if not articulos_dir.is_dir():
            continue

        for pdf_file in sorted(articulos_dir.iterdir()):
            if pdf_file.suffix.lower() != ".pdf" or pdf_file.name == "index.html":
                continue
            page_stem = pdf_file.stem          # ohcbh_cf_erl1_001
            rel_path = pdf_file.relative_to(PDF_DIR)
            pdfs[page_stem] = (rel_path, parent_doc_id)

    return pdfs


# ──────────────────────────────────────────────────────────────────────────────
# Fase 2: Escaneo de textos reconstruidos página a página
# ──────────────────────────────────────────────────────────────────────────────

def scan_reconstrucciones() -> dict[str, tuple[Path, str]]:
    """
    Recorre Reconstrucciones_qwen_deepseek/ buscando archivos *.pdf.txt dentro
    de cada carpeta Articulos_ohcbh_cf_erlXXX/.

    Retorna {page_stem: (ruta_absoluta_al_txt, parent_doc_id)}
    Ejemplo:
      'ohcbh_cf_erl1_001' → (PosixPath('.../ohcbh_cf_erl1_001.pdf.txt'), 'ohcbh_cf_erl1')
    """
    txts: dict[str, tuple[Path, str]] = {}
    if not RECONSTRUCCIONES_DIR.is_dir():
        print(f"[ERROR] Reconstrucciones no encontradas: {RECONSTRUCCIONES_DIR}")
        return txts

    for art_dir in sorted(RECONSTRUCCIONES_DIR.iterdir()):
        if not art_dir.is_dir() or not art_dir.name.startswith("Articulos_"):
            continue

        parent_doc_id = art_dir.name[len("Articulos_"):]
        if not DOC_ID_RE.match(parent_doc_id):
            continue

        for txt_file in sorted(art_dir.iterdir()):
            if not txt_file.name.endswith(".pdf.txt"):
                continue
            # ohcbh_cf_erl1_001.pdf.txt → page_stem = ohcbh_cf_erl1_001
            page_stem = txt_file.name[: -len(".pdf.txt")]
            txts[page_stem] = (txt_file, parent_doc_id)

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

def concat_texts(txt_file: Path) -> str:
    """Lee el archivo de texto de una página."""
    try:
        return txt_file.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as exc:
        print(f"  [WARN] No se pudo leer {txt_file.name}: {exc}")
        return ""


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

        print("\nEscaneando PDFs página a página...")
        pdfs = scan_pdfs()
        print(f"  {len(pdfs)} páginas PDF encontradas en data/pdfs/.../Articulos_.../")

        print("\nEscaneando textos reconstruidos...")
        txts = scan_reconstrucciones()
        print(f"  {len(txts)} páginas TXT encontradas en Reconstrucciones_qwen_deepseek/.")

        # ── Match página a página ──────────────────────────────────────────────
        # pdfs:  {page_stem: (rel_path, parent_doc_id)}
        # txts:  {page_stem: (txt_path, parent_doc_id)}
        matched: dict[str, tuple] = {
            stem: (pdfs[stem][0], pdfs[stem][1], txts[stem][0])
            for stem in pdfs
            if stem in txts
        }
        pdf_only = sorted(set(pdfs) - set(txts))   # PDF sin TXT aún
        txt_only = sorted(set(txts) - set(pdfs))   # TXT sin PDF (inesperado)

        print("\n" + "─" * 60)
        print(f"  ✓ Emparejados (PDF + TXT): {len(matched)}")
        print(f"  ⏳ Solo PDF (TXT pendiente): {len(pdf_only)}")
        if txt_only:
            print(f"  ⚠  Solo TXT (sin PDF): {len(txt_only)} — {txt_only[:5]}")
        print("─" * 60)

        if dry_run:
            print("\n[DRY-RUN] No se modificará la base de datos.")
            print("\nEjemplo de páginas que se insertarían/actualizarían (primeras 10):")
            for stem in sorted(matched)[:10]:
                rel_path, parent_id, txt_path = matched[stem]
                row = metadata.get(parent_id, {})
                title = (row.get("Título") or "").strip() or parent_id
                print(f"  {stem:40s}  padre={parent_id}  título={title[:40]!r}")
            if solo_pdf:
                print(f"\nPáginas solo-PDF que se insertarían (primeras 5):")
                for stem in pdf_only[:5]:
                    rel_path, parent_id = pdfs[stem]
                    print(f"  {stem:40s}  PDF={str(rel_path)}")
            return

        inserted = 0
        updated = 0
        skipped = 0
        errors = 0

        def _process(page_stem: str, pdf_rel: Path, parent_doc_id: str,
                     txt_path: Path | None) -> None:
            nonlocal inserted, updated, skipped, errors
            try:
                existing = Document.query.filter_by(doc_id=page_stem).first()

                if existing and not force:
                    skipped += 1
                    return

                # Texto OCR de la página
                ocr_text = None
                if txt_path:
                    raw = concat_texts(txt_path)
                    ocr_text = clean_ocr_text(raw) if raw else None

                # Metadatos del CSV (heredados del documento padre)
                row = metadata.get(parent_doc_id, {})
                fields = _metadata_fields(row)
                if not fields["title"]:
                    fields["title"] = parent_doc_id

                if existing:
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
                    print(f"  [UPD] {page_stem}")
                else:
                    doc = Document(
                        doc_id=page_stem,
                        pdf_filename=str(pdf_rel),
                        ocr_text=ocr_text,
                        language="español",
                        **fields,
                    )
                    db.session.add(doc)
                    db.session.commit()
                    inserted += 1
                    print(f"  [NEW] {page_stem}")

            except Exception as exc:
                db.session.rollback()
                errors += 1
                print(f"  [ERR] {page_stem}: {exc}")

        # ── Procesar páginas emparejadas (PDF + TXT) ──────────────────────────
        print("\nProcesando páginas emparejadas...")
        for stem in sorted(matched):
            pdf_rel, parent_id, txt_path = matched[stem]
            _process(stem, pdf_rel, parent_id, txt_path)

        # ── Procesar solo-PDF si se solicitó ──────────────────────────────────
        if solo_pdf and pdf_only:
            print(f"\nProcesando {len(pdf_only)} páginas solo-PDF (sin texto aún)...")
            for stem in pdf_only:
                pdf_rel, parent_id = pdfs[stem]
                _process(stem, pdf_rel, parent_id, None)

        # ── Resumen ───────────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("RESUMEN")
        print(f"  Insertados:             {inserted}")
        print(f"  Actualizados:           {updated}")
        print(f"  Omitidos (ya existían): {skipped}")
        print(f"  Errores:                {errors}")
        if pdf_only and not solo_pdf:
            print(f"\n  {len(pdf_only)} páginas sin texto todavía (usa --solo-pdf para cargarlas sin TXT).")


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
