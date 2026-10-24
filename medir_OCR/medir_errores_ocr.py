#!/usr/bin/env python3
"""Muestrea textos OCR y estima errores frente a una referencia textual.

La unidad es cada archivo de texto del corpus (15 584 páginas/archivos). El
error e_i es la distancia de Levenshtein por carácter: inserciones,
eliminaciones y sustituciones. Para cumplir la comparación con referencia
humana, se debe proporcionar esa colección mediante --human-reference-dir.
Las reconstrucciones Qwen/DeepSeek son automáticas, por lo que solo se usan
como comparación opcional y nunca se etiquetan como transcripción humana.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
OCR_DIR = WORKSPACE_ROOT / "TXT_ohcbh_cf_erl"
CORRECTED_DIR = WORKSPACE_ROOT / "Reconstrucciones_qwen_deepseek"
OCR_COPY_DIR = SCRIPT_DIR / "texto_extraido"
CORRECTED_COPY_DIR = SCRIPT_DIR / "texto_corregido"
RESULTS_DIR = SCRIPT_DIR / "registro_de_errores"
POPULATION_SIZE = 15_584
SAMPLE_SIZE = 100
CONFIDENCE_LEVEL = 0.95


def normalized_key(path: Path, root: Path) -> str:
    """Genera una clave relativa común para archivos con sufijo .txt o .pdf.txt."""
    relative = path.relative_to(root).as_posix()
    if relative.endswith(".pdf.txt"):
        return relative[: -len(".pdf.txt")]
    if relative.endswith(".txt"):
        return relative[: -len(".txt")]
    return relative


def index_text_files(root: Path) -> dict[str, Path]:
    """Indexa archivos .txt recursivamente y rechaza claves duplicadas."""
    if not root.is_dir():
        raise FileNotFoundError(f"No existe el directorio: {root}")

    indexed: dict[str, Path] = {}
    for path in sorted(root.rglob("*.txt")):
        key = normalized_key(path, root)
        if key in indexed:
            raise ValueError(f"Clave duplicada {key!r}: {indexed[key]} y {path}")
        indexed[key] = path
    return indexed


def load_fixed_sample() -> list[dict[str, Path | str]]:
    """Valida y carga los mismos archivos ya preparados en las tres carpetas."""
    folders = {
        "referencias_humanas": SCRIPT_DIR / "referencias_humanas",
        "texto_corregido": CORRECTED_COPY_DIR,
        "texto_extraido": OCR_COPY_DIR,
    }
    indexed = {name: index_text_files(folder) for name, folder in folders.items()}
    errors = []

    for name, files in indexed.items():
        if len(files) != SAMPLE_SIZE:
            errors.append(
                f"{name}: se esperaban {SAMPLE_SIZE} archivos .txt y hay {len(files)}."
            )

    all_keys = set().union(*(files.keys() for files in indexed.values()))
    for key in sorted(all_keys):
        missing_from = [name for name, files in indexed.items() if key not in files]
        if missing_from:
            errors.append(
                f"{key}: falta en {', '.join(missing_from)}."
            )

    if errors:
        raise ValueError(
            "El mapeo de la muestra fija no coincide; no se seleccionaron ni "
            "copiaron documentos nuevos:\n  - " + "\n  - ".join(errors)
        )

    keys = sorted(indexed["referencias_humanas"])
    if len(keys) != SAMPLE_SIZE:
        raise ValueError(
            f"La muestra fija debe tener {SAMPLE_SIZE} identificadores únicos; "
            f"se encontraron {len(keys)}. No se inició una nueva selección."
        )

    return [
        {
            "key": key,
            "reference": indexed["referencias_humanas"][key],
            "corrected": indexed["texto_corregido"][key],
            "ocr": indexed["texto_extraido"][key],
        }
        for key in keys
    ]


def levenshtein_distance(left: str, right: str) -> int:
    """Distancia de Levenshtein exacta con el algoritmo bit-parallel de Myers."""
    if len(left) < len(right):
        left, right = right, left
    pattern_length = len(right)
    if pattern_length == 0:
        return len(left)

    bitmask = (1 << pattern_length) - 1
    high_bit = 1 << (pattern_length - 1)
    character_masks: dict[str, int] = {}
    for index, character in enumerate(right):
        character_masks[character] = character_masks.get(character, 0) | (1 << index)

    positive = bitmask
    negative = 0
    score = pattern_length
    for character in left:
        matching = character_masks.get(character, 0)
        vertical = matching | negative
        horizontal = (((matching & positive) + positive) ^ positive) | matching
        positive_horizontal = negative | ~(horizontal | positive)
        negative_horizontal = positive & horizontal

        if positive_horizontal & high_bit:
            score += 1
        elif negative_horizontal & high_bit:
            score -= 1

        positive_horizontal = ((positive_horizontal << 1) | 1) & bitmask
        negative_horizontal = (negative_horizontal << 1) & bitmask
        positive = (negative_horizontal | ~(vertical | positive_horizontal)) & bitmask
        negative = positive_horizontal & vertical

    return score


def read_text(path: Path) -> str:
    """Lee como UTF-8 y estandariza saltos de línea antes de comparar."""
    return path.read_text(encoding="utf-8", errors="replace").replace(
        "\r\n", "\n"
    ).replace("\r", "\n")


def student_t_critical_95(sample_size: int) -> float:
    """Aproxima el cuantil bilateral t(0.975, n-1) mediante expansión en 1/gl."""
    degrees_freedom = sample_size - 1
    if degrees_freedom < 1:
        raise ValueError("Se necesitan al menos dos observaciones para el IC.")
    z = 1.959963984540054
    df = degrees_freedom
    return (
        z
        + (z**3 + z) / (4 * df)
        + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
        + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * df**3)
    )


def summarize_errors(
    errors: list[int], population_size: int, *, allow_negative: bool = False
) -> dict[str, float | int]:
    """Estima un total y su IC t al 95%, con FPC y diferencias opcionalmente negativas."""
    sample_size = len(errors)
    if sample_size < 2 or sample_size > population_size:
        raise ValueError("El tamaño muestral debe estar entre 2 y N.")

    mean_errors = statistics.mean(errors)
    sample_sd = statistics.stdev(errors)
    fpc = math.sqrt((population_size - sample_size) / (population_size - 1))
    standard_error_mean = sample_sd / math.sqrt(sample_size) * fpc
    estimated_total = population_size * mean_errors
    margin = population_size * student_t_critical_95(sample_size) * standard_error_mean
    return {
        "population_size": population_size,
        "sample_size": sample_size,
        "mean_errors_per_document": mean_errors,
        "sample_standard_deviation": sample_sd,
        "finite_population_correction": fpc,
        "standard_error_of_mean": standard_error_mean,
        "estimated_total_errors": estimated_total,
        "confidence_level": CONFIDENCE_LEVEL,
        "confidence_interval_low": (
            estimated_total - margin
            if allow_negative
            else max(0.0, estimated_total - margin)
        ),
        "confidence_interval_high": estimated_total + margin,
    }


def save_results(
    rows: list[dict[str, str | int | float]],
    summaries: dict[str, dict[str, float | int] | float | None],
    reference_label: str,
) -> None:
    """Escribe comparaciones por archivo y resúmenes antes/después."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "errores_por_documento.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as csv_file:
        fields = [
            "id", "ruta_ocr", "ruta_corregido", "ruta_referencia_usada",
            "caracteres_ocr", "caracteres_corregido", "caracteres_referencia",
            "errores_ocr_e_i", "errores_corregido_e_i",
            "reduccion_errores_e_i", "tasa_error_ocr", "tasa_error_corregido",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with (RESULTS_DIR / "muestra_seleccionada.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=["id", "ruta_ocr", "ruta_corregido"]
        )
        writer.writeheader()
        writer.writerows(sample_rows_from_results(rows))

    report = {
        "population_size": POPULATION_SIZE,
        "sample_size": SAMPLE_SIZE,
        "confidence_level": CONFIDENCE_LEVEL,
        "reference": reference_label,
        "metric": "Distancia de Levenshtein por carácter contra la misma referencia humana",
        "ocr_before_postcorrection": summaries["ocr"],
        "postcorrected": summaries["corrected"],
        "paired_error_reduction": summaries["improvement"],
        "formulas": {
            "estimated_total": "T_hat=N*mean(e_i)",
            "confidence_interval": "N*(mean(e_i) +/- t*s/sqrt(n)*sqrt((N-n)/(N-1)))",
            "paired_reduction": "d_i=e_i_ocr-e_i_corregido; T_hat_reduccion=N*mean(d_i)",
        },
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (RESULTS_DIR / "resumen_errores.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    ocr = summaries["ocr"]
    corrected = summaries["corrected"]
    improvement = summaries["improvement"]
    assert isinstance(ocr, dict) and isinstance(corrected, dict)
    assert isinstance(improvement, dict)
    percent = improvement["improvement_percent"]
    percent_text = "no definido" if percent is None else f"{percent:.2f}%"
    (RESULTS_DIR / "resumen_errores.txt").write_text(
        "Comparación de OCR antes y después de la postcorrección\n"
        f"Referencia humana: {reference_label}\n"
        f"Población N: {POPULATION_SIZE:,} | muestra n: {SAMPLE_SIZE}\n"
        "\n1) OCR original vs. referencia humana\n"
        f"   Media ē: {ocr['mean_errors_per_document']:.2f}\n"
        f"   Desviación estándar s: {ocr['sample_standard_deviation']:.2f}\n"
        f"   Total estimado T̂: {ocr['estimated_total_errors']:.2f}\n"
        f"   IC 95%: [{ocr['confidence_interval_low']:.2f}, "
        f"{ocr['confidence_interval_high']:.2f}]\n"
        "\n2) Texto postcorregido vs. referencia humana\n"
        f"   Media ē: {corrected['mean_errors_per_document']:.2f}\n"
        f"   Desviación estándar s: {corrected['sample_standard_deviation']:.2f}\n"
        f"   Total estimado T̂: {corrected['estimated_total_errors']:.2f}\n"
        f"   IC 95%: [{corrected['confidence_interval_low']:.2f}, "
        f"{corrected['confidence_interval_high']:.2f}]\n"
        "\n3) Mejora pareada (OCR menos postcorregido)\n"
        f"   Reducción media de errores por archivo: {improvement['mean_errors_per_document']:.2f}\n"
        f"   Reducción total estimada: {improvement['estimated_total_errors']:.2f}\n"
        f"   IC 95% de la reducción: [{improvement['confidence_interval_low']:.2f}, "
        f"{improvement['confidence_interval_high']:.2f}]\n"
        f"   Mejora porcentual estimada: {percent_text}\n\n"
        "Los intervalos usan t bilateral al 95% y corrección por población finita; "
        "la reducción se estima con diferencias pareadas por documento.\n",
        encoding="utf-8",
    )


def sample_rows_from_results(
    rows: list[dict[str, str | int | float]],
) -> list[dict[str, str]]:
    """Extrae las columnas de muestra del registro de errores."""
    return [
        {
            "id": str(row["id"]),
            "ruta_ocr": str(row["ruta_ocr"]),
            "ruta_corregido": str(row["ruta_corregido"]),
        }
        for row in rows
    ]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compara la muestra fija de 100 documentos existente en las tres carpetas.",
    )
    return parser


def main() -> int:
    build_argument_parser().parse_args()
    try:
        sample = load_fixed_sample()
        print(f"Mapeo validado: {len(sample)} documentos en las tres carpetas.")
        rows: list[dict[str, str | int | float]] = []
        ocr_errors: list[int] = []
        corrected_errors: list[int] = []
        for item in sample:
            key = str(item["key"])
            reference_path = item["reference"]
            ocr_source = item["ocr"]
            corrected_source = item["corrected"]
            assert isinstance(reference_path, Path)
            assert isinstance(ocr_source, Path) and isinstance(corrected_source, Path)
            ocr_text = read_text(ocr_source)
            corrected_text = read_text(corrected_source)
            reference_text = read_text(reference_path)
            ocr_error_count = levenshtein_distance(ocr_text, reference_text)
            corrected_error_count = levenshtein_distance(corrected_text, reference_text)
            ocr_errors.append(ocr_error_count)
            corrected_errors.append(corrected_error_count)
            rows.append(
                {
                    "id": key,
                    "ruta_ocr": str(ocr_source.relative_to(SCRIPT_DIR)),
                    "ruta_corregido": str(corrected_source.relative_to(SCRIPT_DIR)),
                    "ruta_referencia_usada": str(reference_path),
                    "caracteres_ocr": len(ocr_text),
                    "caracteres_corregido": len(corrected_text),
                    "caracteres_referencia": len(reference_text),
                    "errores_ocr_e_i": ocr_error_count,
                    "errores_corregido_e_i": corrected_error_count,
                    "reduccion_errores_e_i": ocr_error_count - corrected_error_count,
                    "tasa_error_ocr": ocr_error_count / max(len(reference_text), 1),
                    "tasa_error_corregido": corrected_error_count / max(len(reference_text), 1),
                }
            )

        improvements = [raw - post for raw, post in zip(ocr_errors, corrected_errors)]
        ocr_summary = summarize_errors(ocr_errors, POPULATION_SIZE)
        corrected_summary = summarize_errors(corrected_errors, POPULATION_SIZE)
        improvement_summary = summarize_errors(
            improvements, POPULATION_SIZE, allow_negative=True
        )
        improvement_summary["improvement_percent"] = (
            100.0 * improvement_summary["estimated_total_errors"]
            / ocr_summary["estimated_total_errors"]
            if ocr_summary["estimated_total_errors"] != 0
            else None
        )
        summaries: dict[str, dict[str, float | int] | float | None] = {
            "ocr": ocr_summary,
            "corrected": corrected_summary,
            "improvement": improvement_summary,
        }
        reference_label = str(SCRIPT_DIR / "referencias_humanas")
        save_results(rows, summaries, reference_label)
        (RESULTS_DIR / "pendiente_referencia_humana.txt").unlink(missing_ok=True)
        print(
            f"Referencia: {reference_label}\n"
            f"OCR original: ē={ocr_summary['mean_errors_per_document']:.2f}; "
            f"T̂={ocr_summary['estimated_total_errors']:.2f}; "
            f"IC95%=[{ocr_summary['confidence_interval_low']:.2f}, "
            f"{ocr_summary['confidence_interval_high']:.2f}]\n"
            f"Postcorregido: ē={corrected_summary['mean_errors_per_document']:.2f}; "
            f"T̂={corrected_summary['estimated_total_errors']:.2f}; "
            f"IC95%=[{corrected_summary['confidence_interval_low']:.2f}, "
            f"{corrected_summary['confidence_interval_high']:.2f}]\n"
            f"Reducción estimada: {improvement_summary['estimated_total_errors']:.2f} errores "
            f"({improvement_summary['improvement_percent']:.2f}%); "
            f"IC95%=[{improvement_summary['confidence_interval_low']:.2f}, "
            f"{improvement_summary['confidence_interval_high']:.2f}]\n"
            f"Resultados: {RESULTS_DIR}"
        )
        return 0
    except (ValueError, OSError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())