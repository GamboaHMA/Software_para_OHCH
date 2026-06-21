"""
Evaluación y comparación de modelos de topic modeling.

Lee los archivos .csv producidos por topic_modeling.py en Topic_results/
(formato: documento, ruta, tema_dominante, peso) y calcula un conjunto de
métricas cuantitativas que permiten determinar cuán bien separados están
los temas para cada configuración de n_topics.

Métricas calculadas por archivo:
  avg_weight          — peso promedio del tema dominante (confianza media)
  weight_std          — desviación estándar del peso
  weight_p25 / p75    — primer y tercer cuartil del peso
  low_conf_pct        — % documentos con peso < 0.30 (asignación débil)
  high_conf_pct       — % documentos con peso > 0.50 (asignación fuerte)
  n_docs              — total de documentos
  n_topics_total      — temas configurados (inferido del nombre del archivo)
  n_active_topics     — temas que tienen al menos 1 documento como dominante
  coverage_ratio      — n_active_topics / n_topics_total  (1.0 = ideal)
  topic_entropy       — entropía de Shannon normalizada de la distribución
                        de documentos por tema (1.0 = perfectamente balanceado)
  gini_coeff          — coeficiente Gini de la distribución (0 = equilibrado,
                        1 = todo concentrado en un tema)
  balance_score       — 1 − gini_coeff
  composite_score     — puntuación compuesta:
                        0.50·avg_weight + 0.25·balance_score + 0.25·coverage_ratio
                        Valores más altos indican mejor modelo.

Razonamiento detrás de las métricas
-------------------------------------
Un buen modelo de topics requiere:
  1. Confianza alta (avg_weight): cada documento se asigna con certeza a
     un tema → frases claras, vocabulario diferenciado.
  2. Cobertura total (coverage_ratio): todos los temas configurados capturan
     al menos un documento → no hay temas "fantasma".
  3. Balance (balance_score/topic_entropy): los documentos se distribuyen de
     forma razonablemente uniforme → no se forman mega-temas que absorben
     casi todo el corpus.

Uso:
    python topic_experimenter.py
    python topic_experimenter.py --results-dir Topic_results/
    python topic_experimenter.py --output reporte_topics.csv
    python topic_experimenter.py --per-topic
"""

import argparse
import csv
import math
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
DEFAULT_RESULTS_DIR = BASE_DIR / "Topic_results"

LOW_CONF_THRESHOLD  = 0.30
HIGH_CONF_THRESHOLD = 0.50


# ---------------------------------------------------------------------------
# Carga de CSVs
# ---------------------------------------------------------------------------

def _infer_n_topics(filename: str) -> int | None:
    """Extrae el número de temas del nombre del archivo.
    Acepta patrones como: 10_topics_result.csv, topics_10.csv, 10topics.csv, etc.
    Devuelve None si no puede inferirlo."""
    m = re.search(r"(\d+)", filename)
    return int(m.group(1)) if m else None


def load_csv(path: Path) -> list[dict]:
    """Carga un CSV de resultados de topic modeling."""
    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "documento":      row["documento"],
                "ruta":           row.get("ruta", ""),
                "tema_dominante": int(row["tema_dominante"]),
                "peso":           float(row["peso"]),
            })
    return rows


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def _percentile(sorted_values: list[float], p: float) -> float:
    """Percentil p (0–100) de una lista ya ordenada."""
    if not sorted_values:
        return 0.0
    idx = (p / 100) * (len(sorted_values) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_values) - 1)
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _shannon_entropy_normalized(counts: list[int]) -> float:
    """Entropía de Shannon normalizada (0–1) de una distribución de conteos."""
    total = sum(counts)
    if total == 0:
        return 0.0
    active = [c for c in counts if c > 0]
    if len(active) <= 1:
        return 0.0
    entropy = -sum((c / total) * math.log2(c / total) for c in active)
    max_entropy = math.log2(len(active))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def _gini(values: list[float]) -> float:
    """Coeficiente Gini de una lista de valores no negativos."""
    n = len(values)
    if n == 0:
        return 0.0
    sorted_v = sorted(values)
    total = sum(sorted_v)
    if total == 0:
        return 0.0
    cumsum = sum((i + 1) * v for i, v in enumerate(sorted_v))
    return (2 * cumsum) / (n * total) - (n + 1) / n


def compute_metrics(rows: list[dict], n_topics_total: int | None) -> dict:
    """Calcula todas las métricas para un conjunto de filas de un CSV."""
    if not rows:
        return {}

    weights = sorted(r["peso"] for r in rows)
    n_docs  = len(rows)
    avg_w   = _mean(weights)
    std_w   = _std(weights, avg_w)

    # Conteos por tema
    topic_counts: dict[int, int] = {}
    for r in rows:
        t = r["tema_dominante"]
        topic_counts[t] = topic_counts.get(t, 0) + 1

    n_active = len(topic_counts)
    n_total  = n_topics_total if n_topics_total else n_active

    # Para Gini usamos todos los n_total temas (incluyendo los inactivos con 0)
    all_counts = [topic_counts.get(t, 0) for t in range(1, n_total + 1)]

    entropy    = _shannon_entropy_normalized(all_counts)
    gini       = _gini([float(c) for c in all_counts])
    balance    = 1.0 - gini
    coverage   = n_active / n_total if n_total else 0.0

    low_conf  = sum(1 for w in weights if w < LOW_CONF_THRESHOLD)  / n_docs * 100
    high_conf = sum(1 for w in weights if w > HIGH_CONF_THRESHOLD) / n_docs * 100

    composite = 0.50 * avg_w + 0.25 * balance + 0.25 * coverage

    return {
        "n_docs":           n_docs,
        "n_topics_total":   n_total,
        "n_active_topics":  n_active,
        "avg_weight":       round(avg_w, 4),
        "weight_std":       round(std_w, 4),
        "weight_p25":       round(_percentile(weights, 25), 4),
        "weight_p75":       round(_percentile(weights, 75), 4),
        "low_conf_pct":     round(low_conf, 2),
        "high_conf_pct":    round(high_conf, 2),
        "coverage_ratio":   round(coverage, 4),
        "topic_entropy":    round(entropy, 4),
        "gini_coeff":       round(gini, 4),
        "balance_score":    round(balance, 4),
        "composite_score":  round(composite, 4),
    }


def per_topic_stats(rows: list[dict], n_topics_total: int | None) -> list[dict]:
    """Estadísticas (conteo, peso medio, std) por cada tema."""
    topic_weights: dict[int, list[float]] = {}
    for r in rows:
        t = r["tema_dominante"]
        topic_weights.setdefault(t, []).append(r["peso"])

    n_total = n_topics_total if n_topics_total else max(topic_weights, default=0)
    stats = []
    for t in range(1, n_total + 1):
        ws = topic_weights.get(t, [])
        m  = _mean(ws)
        stats.append({
            "tema":       t,
            "n_docs":     len(ws),
            "avg_weight": round(m, 4),
            "weight_std": round(_std(ws, m), 4),
            "min_weight": round(min(ws), 4) if ws else 0.0,
            "max_weight": round(max(ws), 4) if ws else 0.0,
        })
    return stats


# ---------------------------------------------------------------------------
# Impresión de resultados
# ---------------------------------------------------------------------------

def _bar(value: float, width: int = 20, max_val: float = 1.0) -> str:
    """Barra ASCII proporcional a value/max_val."""
    filled = int(round(value / max_val * width)) if max_val else 0
    return "█" * filled + "░" * (width - filled)


def print_summary(results: list[dict]) -> None:
    """Imprime tabla comparativa de todos los modelos, ordenada por composite_score."""
    if not results:
        print("[WARN] No hay resultados para mostrar.")
        return

    sorted_res = sorted(results, key=lambda r: r["metrics"]["composite_score"], reverse=True)

    print(f"\n{'='*80}")
    print("  COMPARACIÓN DE MODELOS  (ordenado por composite_score ↓)")
    print(f"{'='*80}")
    header = (
        f"{'Archivo':<28} {'n_topics':>8} {'avg_w':>6} {'balance':>8} "
        f"{'cov':>5} {'hi_conf%':>8} {'composite':>10}"
    )
    print(header)
    print("-" * 80)

    for r in sorted_res:
        m = r["metrics"]
        bar = _bar(m["composite_score"])
        print(
            f"{r['filename']:<28} {m['n_topics_total']:>8} "
            f"{m['avg_weight']:>6.3f} {m['balance_score']:>8.3f} "
            f"{m['coverage_ratio']:>5.2f} {m['high_conf_pct']:>8.1f} "
            f"{m['composite_score']:>10.4f}  {bar}"
        )

    print(f"\n{'='*80}")
    best = sorted_res[0]
    print(
        f"  Mejor modelo: {best['filename']}  "
        f"(composite={best['metrics']['composite_score']:.4f})"
    )
    print(f"{'='*80}\n")


def print_detailed(results: list[dict], show_per_topic: bool = False) -> None:
    """Imprime métricas detalladas de cada modelo."""
    for r in sorted(results, key=lambda x: x["metrics"]["n_topics_total"]):
        m  = r["metrics"]
        pt = r.get("per_topic", [])

        print(f"\n{'─'*60}")
        print(f"  Archivo : {r['filename']}")
        print(f"  n_docs  : {m['n_docs']}   n_topics_total: {m['n_topics_total']}")
        print(f"{'─'*60}")
        print(f"  Confianza de asignación (peso del tema dominante):")
        print(f"    Media              : {m['avg_weight']:.4f}  {_bar(m['avg_weight'])}")
        print(f"    Std                : {m['weight_std']:.4f}")
        print(f"    P25 / P75          : {m['weight_p25']:.4f} / {m['weight_p75']:.4f}")
        print(f"    Baja conf (<0.30)  : {m['low_conf_pct']:.1f}%")
        print(f"    Alta conf (>0.50)  : {m['high_conf_pct']:.1f}%")
        print(f"  Distribución de temas:")
        print(f"    Temas activos      : {m['n_active_topics']} / {m['n_topics_total']}")
        print(f"    Cobertura          : {m['coverage_ratio']:.4f}  {_bar(m['coverage_ratio'])}")
        print(f"    Entropía norm.     : {m['topic_entropy']:.4f}  {_bar(m['topic_entropy'])}")
        print(f"    Gini               : {m['gini_coeff']:.4f}")
        print(f"    Balance            : {m['balance_score']:.4f}  {_bar(m['balance_score'])}")
        print(f"  Puntuación compuesta : {m['composite_score']:.4f}  {_bar(m['composite_score'])}")

        if show_per_topic and pt:
            print(f"\n  Desglose por tema:")
            print(f"    {'Tema':>5}  {'n_docs':>7}  {'avg_w':>7}  {'std_w':>7}  {'min':>6}  {'max':>6}")
            print(f"    {'─'*5}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*6}")
            for ts in pt:
                bar = "█" * ts["n_docs"] if ts["n_docs"] <= 30 else "█" * 30 + "…"
                print(
                    f"    {ts['tema']:>5}  {ts['n_docs']:>7}  {ts['avg_weight']:>7.4f}  "
                    f"{ts['weight_std']:>7.4f}  {ts['min_weight']:>6.4f}  {ts['max_weight']:>6.4f}  {bar}"
                )


# ---------------------------------------------------------------------------
# Guardado de reporte CSV
# ---------------------------------------------------------------------------

def save_report(results: list[dict], output_path: str) -> None:
    """Guarda un CSV con las métricas de todos los modelos."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "filename", "n_docs", "n_topics_total", "n_active_topics",
        "avg_weight", "weight_std", "weight_p25", "weight_p75",
        "low_conf_pct", "high_conf_pct",
        "coverage_ratio", "topic_entropy", "gini_coeff", "balance_score",
        "composite_score",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(results, key=lambda x: x["metrics"]["n_topics_total"]):
            row = {"filename": r["filename"]}
            row.update(r["metrics"])
            writer.writerow(row)

    print(f"[INFO] Reporte guardado en: {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluación de calidad de modelos de topic modeling."
    )
    p.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Carpeta con los CSVs de resultados (default: Topic_results/)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Ruta CSV para guardar el reporte de métricas (opcional).",
    )
    p.add_argument(
        "--per-topic",
        action="store_true",
        help="Mostrar estadísticas detalladas por tema en cada modelo.",
    )
    p.add_argument(
        "--pattern",
        default="*_topics_result.csv",
        help="Patrón glob para encontrar archivos de resultados (default: *_topics_result.csv).",
    )
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        import sys
        print(f"[ERROR] No existe la carpeta: {results_dir}", file=sys.stderr)
        sys.exit(1)

    csv_files = sorted(results_dir.glob(args.pattern))
    if not csv_files:
        import sys
        print(
            f"[ERROR] No se encontraron archivos con patrón '{args.pattern}' en {results_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[INFO] {len(csv_files)} archivo(s) encontrado(s) en {results_dir}")

    results = []
    for csv_path in csv_files:
        try:
            rows       = load_csv(csv_path)
            n_total    = _infer_n_topics(csv_path.name)
            metrics    = compute_metrics(rows, n_total)
            pt_stats   = per_topic_stats(rows, n_total) if args.per_topic else []
            results.append({
                "filename":  csv_path.name,
                "metrics":   metrics,
                "per_topic": pt_stats,
            })
            print(f"  ✓ {csv_path.name}  ({len(rows)} docs, n_topics={n_total})")
        except Exception as exc:
            print(f"  ✗ {csv_path.name}: {exc}")

    if not results:
        print("[ERROR] No se pudo procesar ningún archivo.")
    else:
        print_summary(results)
        print_detailed(results, show_per_topic=args.per_topic)

        if args.output:
            save_report(results, args.output)
