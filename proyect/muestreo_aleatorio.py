#!/usr/bin/env python3
"""
muestreo_aleatorio_pdfs.py
───────────────────────────
Script para seleccionar aleatoriamente 400 PDFs de página (de un total de ~15 500),
dividirlos en 5 grupos de 80, y guardar la lista en archivos de texto para
revisión manual (clasificación: impreso, manuscrito, foto).

Uso:
    python muestreo_aleatorio_pdfs.py               # usa semilla fija 42
    python muestreo_aleatorio_pdfs.py --seed 123    # usa otra semilla
    python muestreo_aleatorio_pdfs.py --no-seed     # aleatoriedad real (sin fijar)

Estructura esperada (la misma que en load_from_reconstrucciones.py):
  proyect/data/pdfs/
      ohcbh_cf_erlXXX/
          Articulos_ohcbh_cf_erlXXX/
              ohcbh_cf_erlXXX_001.pdf
              ohcbh_cf_erlXXX_002.pdf
              ...
"""

import os
import sys
import re
import random
import argparse
from pathlib import Path

# ─── Ruta base ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
PDF_DIR = SCRIPT_DIR / "data" / "pdfs"

# ─── Patrón para identificar carpetas de documentos ──────────────────────
DOC_ID_RE = re.compile(r"^ohcbh_cf_erl(\d+[A-Za-z]?)$")


# ──────────────────────────────────────────────────────────────────────────
# Fase 1: Escaneo de TODOS los PDFs (página a página)
# ──────────────────────────────────────────────────────────────────────────

def scan_all_pdfs() -> list[dict]:
    """
    Recorre PDF_DIR y recopila todos los PDFs de página (los que están dentro
    de cada Articulos_ohcbh_cf_erlXXX/).

    Retorna una lista de diccionarios, cada uno con:
        - 'page_stem'  : identificador único de la página (ej: 'ohcbh_cf_erl1_001')
        - 'rel_path'   : ruta relativa desde PDF_DIR (ej: 'ohcbh_cf_erl1/Articulos_.../ohcbh_cf_erl1_001.pdf')
        - 'parent_id'  : ID del documento padre (ej: 'ohcbh_cf_erl1')
    """
    pdfs = []

    if not PDF_DIR.is_dir():
        print(f"[ERROR] No se encontró el directorio de PDFs: {PDF_DIR}")
        return pdfs

    # Recorremos las carpetas de cada documento padre
    for doc_dir in sorted(PDF_DIR.iterdir()):
        if not doc_dir.is_dir():
            continue
        parent_doc_id = doc_dir.name
        if not DOC_ID_RE.match(parent_doc_id):
            continue  # nos saltamos carpetas que no sigan el patrón

        articulos_dir = doc_dir / f"Articulos_{parent_doc_id}"
        if not articulos_dir.is_dir():
            continue  # si no tiene subcarpeta de artículos, no hay PDFs de página

        # Recorremos los PDFs dentro de la carpeta de artículos
        for pdf_file in sorted(articulos_dir.iterdir()):
            if pdf_file.suffix.lower() != ".pdf" or pdf_file.name == "index.html":
                continue

            page_stem = pdf_file.stem  # ej: "ohcbh_cf_erl1_001"
            rel_path = pdf_file.relative_to(PDF_DIR)  # ruta relativa para localizar el archivo

            pdfs.append({
                "page_stem": page_stem,
                "rel_path": str(rel_path),
                "parent_id": parent_doc_id,
            })

    return pdfs


# ──────────────────────────────────────────────────────────────────────────
# Fase 2: Aleatorización, selección y división en grupos
# ──────────────────────────────────────────────────────────────────────────

def generar_muestras(pdf_list: list[dict], total_muestra: int = 400,
                     num_grupos: int = 5, seed: int | None = 42) -> list[list[dict]]:
    """
    Toma la lista completa de PDFs, la desordena aleatoriamente, selecciona
    los primeros 'total_muestra' y los divide en 'num_grupos' grupos de igual
    tamaño (total_muestra / num_grupos).

    Retorna una lista de listas, donde cada sublista contiene los diccionarios
    de los PDFs de ese grupo.
    """
    if len(pdf_list) < total_muestra:
        print(f"[WARN] Solo se encontraron {len(pdf_list)} PDFs, que son menos de los {total_muestra} solicitados.")
        total_muestra = len(pdf_list)
        # Ajustamos el número de grupos para que sean lo más equilibrados posible
        num_grupos = min(num_grupos, total_muestra)

    # 1. Copiar la lista para no modificar la original
    working_list = pdf_list.copy()

    # 2. Aleatorizar usando una semilla (para reproducibilidad)
    rng = random.Random(seed) if seed is not None else random
    rng.shuffle(working_list)

    # 3. Tomar los primeros 'total_muestra'
    muestra = working_list[:total_muestra]

    # 4. Dividir en 'num_grupos' partes iguales
    grupo_tamano = total_muestra // num_grupos
    grupos = []
    for i in range(0, total_muestra, grupo_tamano):
        grupo = muestra[i:i + grupo_tamano]
        grupos.append(grupo)

    return grupos


# ──────────────────────────────────────────────────────────────────────────
# Fase 3: Guardar los grupos en archivos de texto
# ──────────────────────────────────────────────────────────────────────────

def guardar_grupos(grupos: list[list[dict]], output_dir: Path = None) -> list[Path]:
    """
    Guarda cada grupo en un archivo de texto numerado (grupo_1.txt, grupo_2.txt, ...)
    dentro del directorio de salida.
    Retorna la lista de rutas de los archivos creados.
    """
    if output_dir is None:
        output_dir = SCRIPT_DIR / "muestras_pdfs"
    output_dir.mkdir(exist_ok=True)

    archivos_generados = []

    for idx, grupo in enumerate(grupos, start=1):
        archivo_txt = output_dir / f"grupo_{idx}.txt"
        with open(archivo_txt, "w", encoding="utf-8") as f:
            # Escribimos el encabezado
            f.write("=" * 60 + "\n")
            f.write(f"GRUPO {idx} DE {len(grupos)}\n")
            f.write(f"Total de documentos en este grupo: {len(grupo)}\n")
            f.write("(Seleccionados aleatoriamente de un total de PDFs escaneados)\n")
            f.write("=" * 60 + "\n\n")

            # Escribimos la lista numerada
            for num, item in enumerate(grupo, start=1):
                page_stem = item["page_stem"]
                rel_path = item["rel_path"]
                f.write(f"{num:4d}. {page_stem:40s}  →  {rel_path}\n")

        archivos_generados.append(archivo_txt)

    return archivos_generados


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Selecciona aleatoriamente 400 PDFs de página, los divide en 5 grupos de 80 y guarda las listas."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para el generador aleatorio (por defecto 42). Usa --no-seed para aleatoriedad real.",
    )
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Si se indica, no se fija semilla (aleatoriedad real).",
    )
    args = parser.parse_args()

    # Determinar la semilla
    seed = None if args.no_seed else args.seed

    print("=" * 60)
    print("MUESTREO ALEATORIO DE PDFs PARA REVISIÓN MANUAL")
    print("=" * 60)

    # 1. Escanear todos los PDFs
    print("\n[1] Escaneando todos los PDFs página a página...")
    todos_los_pdfs = scan_all_pdfs()
    print(f"    Total de PDFs encontrados: {len(todos_los_pdfs)}")

    if len(todos_los_pdfs) == 0:
        print("[ERROR] No se encontraron PDFs. Verifica que la carpeta 'data/pdfs' exista y contenga los archivos.")
        sys.exit(1)

    # 2. Generar la muestra y los grupos
    print(f"\n[2] Seleccionando 400 documentos aleatorios (semilla={seed})...")
    grupos = generar_muestras(todos_los_pdfs, total_muestra=400, num_grupos=5, seed=seed)

    total_seleccionados = sum(len(g) for g in grupos)
    print(f"    Seleccionados: {total_seleccionados} documentos.")
    print(f"    Divididos en {len(grupos)} grupos de aproximadamente {len(grupos[0])} documentos cada uno.")

    # 3. Guardar los grupos en archivos de texto
    print("\n[3] Guardando los grupos en archivos de texto...")
    archivos = guardar_grupos(grupos)

    print("\n[4] Archivos generados:")
    for archivo in archivos:
        print(f"    - {archivo}")

    # Resumen final
    print("\n" + "=" * 60)
    print("RESUMEN")
    print(f"  Total de PDFs escaneados:  {len(todos_los_pdfs)}")
    print(f"  Tamaño de la muestra:      {total_seleccionados}")
    print(f"  Número de grupos:          {len(grupos)}")
    print(f"  Documentos por grupo:      {len(grupos[0])} (último grupo: {len(grupos[-1])})")
    print(f"  Archivos de salida:        {', '.join(str(a.name) for a in archivos)}")
    print("=" * 60)
    print("\n¡Listo! Revisa los archivos .txt generados para localizar los PDFs y proceder con la clasificación manual.")


if __name__ == "__main__":
    main()