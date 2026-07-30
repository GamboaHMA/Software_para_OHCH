"""
Evaluación y comparación de modelos de topic modeling.

Lee los archivos .csv producidos por topic_modeling.py en Topic_results/
(formato: documento, ruta, tema_dominante, peso) y calcula un conjunto de
métricas cuantitativas. Opcionalmente, si se proporciona --models-dir, carga
los modelos .pkl para calcular coherencia (C_v) y diversidad de tópicos.

Uso:
    python topic_experimenter.py
    python topic_experimenter.py --results-dir Topic_results/
    python topic_experimenter.py --models-dir Topic_models/ --output reporte.csv
    python topic_experimenter.py --criterio coherence --n-words 15
"""

import argparse
import csv
import math
import pickle
import re
import sys
from pathlib import Path

# Para coherencia (opcional)
try:
    import numpy as np
    from gensim.models.coherencemodel import CoherenceModel
    from gensim.corpora.dictionary import Dictionary
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    CoherenceModel = None
    Dictionary = None

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
DEFAULT_RESULTS_DIR = BASE_DIR / "Topic_results"
DEFAULT_MODELS_DIR = BASE_DIR / "Topic_models"

LOW_CONF_THRESHOLD  = 0.30
HIGH_CONF_THRESHOLD = 0.50


# ---------------------------------------------------------------------------
# Carga de CSVs
# ---------------------------------------------------------------------------

def _infer_n_topics(filename: str) -> int | None:
    """Extrae el número de temas del nombre del archivo."""
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
# Carga de modelos .pkl
# ---------------------------------------------------------------------------

def load_model_pkl(path: Path) -> dict:
    """Carga el estado de un modelo guardado con topic_modeling.py."""
    with path.open("rb") as f:
        return pickle.load(f)


def extract_rows_from_state(state: dict) -> list[dict]:
    """
    A partir del estado de un modelo, genera la lista de filas
    (documento, ruta, tema_dominante, peso) usando la matriz doc_topic_matrix.
    Asume que la matriz ya está normalizada (suma=1 por fila) para NMF.
    """
    doc_paths = state.get("doc_paths", [])
    matrix = state.get("doc_topic_matrix")
    if matrix is None:
        return []
    rows = []
    for i, path_str in enumerate(doc_paths):
        dist = matrix[i]
        dominant = int(dist.argmax()) + 1   # 1-indexed
        weight = float(dist.max())
        rows.append({
            "documento": Path(path_str).name,
            "ruta": str(path_str),
            "tema_dominante": dominant,
            "peso": round(weight, 4),
        })
    return rows


def get_topic_words_from_state(state: dict, n_words: int) -> list[list[str]]:
    """
    Devuelve una lista de listas, cada una con las top-n_words palabras de cada tópico.
    """
    model = state.get("model")
    vectorizer = state.get("vectorizer")
    if model is None or vectorizer is None:
        return []
    feature_names = vectorizer.get_feature_names_out()
    components = model.components_   # shape (n_topics, n_features)
    topic_words = []
    for topic in components:
        top_indices = topic.argsort()[:-n_words-1:-1]
        words = [feature_names[i] for i in top_indices]
        topic_words.append(words)
    return topic_words


def compute_coherence_diversity(state: dict, n_words: int) -> tuple[float | None, float | None]:
    """
    Calcula la coherencia (C_v) y diversidad para un modelo cargado.
    Devuelve (coherence, diversity) o (None, None) si no es posible.
    """
    if not GENSIM_AVAILABLE:
        print("  [WARN] gensim no disponible, no se calculará coherencia.")
        return None, None

    processed_texts = state.get("processed_texts")
    if not processed_texts:
        print("  [WARN] 'processed_texts' no encontrado en el modelo.")
        return None, None

    topic_words = get_topic_words_from_state(state, n_words)
    if not topic_words:
        print("  [WARN] No se pudieron obtener palabras de tópicos.")
        return None, None

    # Tokenizar textos (cada texto es un string de palabras separadas por espacio)
    tokenized_corpus = [text.split() for text in processed_texts]

    # Crear diccionario de Gensim a partir del corpus tokenizado
    dictionary = Dictionary(tokenized_corpus)

    try:
        cm = CoherenceModel(
            topics=topic_words,
            texts=tokenized_corpus,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence = cm.get_coherence()
    except Exception as e:
        print(f"  [WARN] Error calculando coherencia: {e}")
        coherence = None

    # Diversidad: porcentaje de palabras únicas en el conjunto de palabras de todos los tópicos
    all_words = set()
    for topic in topic_words:
        all_words.update(topic)
    total_words = len(topic_words) * n_words
    diversity = len(all_words) / total_words if total_words > 0 else 0.0

    return coherence, diversity


# ---------------------------------------------------------------------------
# Métricas generales (desde CSVs o desde matriz)
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    idx = (p / 100) * (len(sorted_values) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_values) - 1)
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _shannon_entropy_normalized(counts: list[int]) -> float:
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
    n = len(values)
    if n == 0:
        return 0.0
    sorted_v = sorted(values)
    total = sum(sorted_v)
    if total == 0:
        return 0.0
    cumsum = sum((i + 1) * v for i, v in enumerate(sorted_v))
    return (2 * cumsum) / (n * total) - (n + 1) / n


def compute_metrics(rows: list[dict], n_topics_total: int | None,
                    coherence: float | None = None,
                    diversity: float | None = None) -> dict:
    """Calcula todas las métricas para un conjunto de filas."""
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

    all_counts = [topic_counts.get(t, 0) for t in range(1, n_total + 1)]

    entropy    = _shannon_entropy_normalized(all_counts)
    gini       = _gini([float(c) for c in all_counts])
    balance    = 1.0 - gini
    coverage   = n_active / n_total if n_total else 0.0

    low_conf  = sum(1 for w in weights if w < LOW_CONF_THRESHOLD)  / n_docs * 100
    high_conf = sum(1 for w in weights if w > HIGH_CONF_THRESHOLD) / n_docs * 100

    # Puntuación compuesta original
    composite = 0.50 * avg_w + 0.25 * balance + 0.25 * coverage

    metrics = {
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

    # Añadir coherencia y diversidad solo si no son None
    if coherence is not None:
        metrics["coherence"] = round(coherence, 4)
    if diversity is not None:
        metrics["diversity"] = round(diversity, 4)

    return metrics


# ---------------------------------------------------------------------------
# Impresión de resultados
# ---------------------------------------------------------------------------

def _bar(value: float, width: int = 20, max_val: float = 1.0) -> str:
    filled = int(round(value / max_val * width)) if max_val else 0
    return "█" * filled + "░" * (width - filled)


def print_summary(results: list[dict], criterion: str = "composite_score") -> None:
    """Imprime tabla comparativa de todos los modelos, ordenada por criterion."""
    if not results:
        print("[WARN] No hay resultados para mostrar.")
        return

    # Verificar que el criterio exista en los resultados
    sample = results[0]["metrics"]
    if criterion not in sample:
        print(f"[WARN] Criterio '{criterion}' no encontrado. Usando 'composite_score'.")
        criterion = "composite_score"

    sorted_res = sorted(results, key=lambda r: r["metrics"].get(criterion, 0), reverse=True)

    # Determinar columnas a mostrar
    first_metrics = sorted_res[0]["metrics"]
    has_coherence = "coherence" in first_metrics
    has_diversity = "diversity" in first_metrics

    print(f"\n{'='*100}")
    print(f"  COMPARACIÓN DE MODELOS  (ordenado por: {criterion})")
    print(f"{'='*100}")

    # Cabecera
    header = f"{'Archivo':<28} {'n_topics':>8} {'avg_w':>6} {'balance':>8} {'cov':>5}"
    if has_coherence:
        header += f" {'coherence':>10}"
    if has_diversity:
        header += f" {'diversity':>10}"
    header += f" {'composite':>10}  {criterion.upper():>12}"
    print(header)
    print("-" * 100)

    for r in sorted_res:
        m = r["metrics"]
        line = f"{r['filename']:<28} {m['n_topics_total']:>8} {m['avg_weight']:>6.3f} {m['balance_score']:>8.3f} {m['coverage_ratio']:>5.2f}"
        if has_coherence:
            val = m.get('coherence', 'N/A')
            if isinstance(val, float):
                line += f" {val:>10.4f}"
            else:
                line += f" {val:>10}"
        if has_diversity:
            val = m.get('diversity', 'N/A')
            if isinstance(val, float):
                line += f" {val:>10.4f}"
            else:
                line += f" {val:>10}"
        line += f" {m['composite_score']:>10.4f}  "
        val_crit = m.get(criterion, 'N/A')
        if isinstance(val_crit, float):
            line += f"{val_crit:>12.4f}"
        else:
            line += f"{val_crit:>12}"
        print(line)

    print(f"\n{'='*100}")
    best = sorted_res[0]
    best_val = best['metrics'].get(criterion, 'N/A')
    if isinstance(best_val, float):
        best_str = f"{best_val:.4f}"
    else:
        best_str = str(best_val)
    print(
        f"  Mejor modelo según '{criterion}': {best['filename']}  "
        f"({criterion}={best_str})"
    )
    print(f"{'='*100}\n")


def print_detailed(results: list[dict], show_per_topic: bool = False) -> None:
    """Imprime métricas detalladas de cada modelo."""
    for r in sorted(results, key=lambda x: x["metrics"]["n_topics_total"]):
        m  = r["metrics"]
        pt = r.get("per_topic", [])

        print(f"\n{'─'*70}")
        print(f"  Archivo : {r['filename']}")
        print(f"  n_docs  : {m['n_docs']}   n_topics_total: {m['n_topics_total']}")
        print(f"{'─'*70}")
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
        if "coherence" in m:
            print(f"  Coherencia (C_v)    : {m['coherence']:.4f}  {_bar(m['coherence'])}")
        if "diversity" in m:
            print(f"  Diversidad          : {m['diversity']:.4f}  {_bar(m['diversity'])}")
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

def save_report(results: list[dict], output_path: str, criterion: str = "composite_score") -> None:
    """Guarda un CSV con las métricas de todos los modelos, ordenado por criterion."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Obtener todas las claves de métricas de los resultados
    fieldnames = ["filename"]
    if results:
        sample_metrics = results[0]["metrics"]
        # Queremos un orden fijo: primero las básicas, luego coherencia/diversidad si existen
        base_fields = [
            "n_docs", "n_topics_total", "n_active_topics",
            "avg_weight", "weight_std", "weight_p25", "weight_p75",
            "low_conf_pct", "high_conf_pct",
            "coverage_ratio", "topic_entropy", "gini_coeff", "balance_score",
            "composite_score"
        ]
        extra_fields = [k for k in sample_metrics.keys() if k not in base_fields]
        fieldnames.extend(base_fields)
        fieldnames.extend(extra_fields)

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Ordenar según el criterio
        sorted_res = sorted(results, key=lambda r: r["metrics"].get(criterion, 0), reverse=True)
        for r in sorted_res:
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
        "--models-dir",
        default=None,
        help="Carpeta con los modelos .pkl (opcional). Si se da, calcula coherencia y diversidad.",
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
    p.add_argument(
        "--n-words",
        type=int,
        default=10,
        help="Número de palabras por tópico para coherencia y diversidad (default: 10).",
    )
    p.add_argument(
        "--criterio",
        default="composite_score",
        help="Métrica para ordenar y guardar el reporte (ej: composite_score, coherence, diversity, avg_weight).",
    )
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"[ERROR] No existe la carpeta: {results_dir}", file=sys.stderr)
        sys.exit(1)

    # Si se proporciona models-dir, cargamos modelos para calcular coherencia/diversidad
    models_dir = Path(args.models_dir) if args.models_dir else None
    if models_dir and not models_dir.exists():
        print(f"[ERROR] No existe la carpeta de modelos: {models_dir}", file=sys.stderr)
        sys.exit(1)

    # Buscar CSVs
    csv_files = sorted(results_dir.glob(args.pattern))
    if not csv_files:
        print(
            f"[ERROR] No se encontraron archivos con patrón '{args.pattern}' en {results_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[INFO] {len(csv_files)} archivo(s) encontrado(s) en {results_dir}")

    # Lista de modelos .pkl
    pkl_files = []
    if models_dir:
        pkl_files = list(models_dir.glob("*_topics_*.pkl"))
        print(f"[INFO] {len(pkl_files)} modelo(s) .pkl encontrados en {models_dir}")

    results = []

    for csv_path in csv_files:
        try:
            rows = load_csv(csv_path)
            n_total = _infer_n_topics(csv_path.name)
            coherence = None
            diversity = None

            if models_dir and pkl_files and n_total is not None:
                # Buscar .pkl que tenga el número exacto de tópicos (usando regex)
                matching_pkl = None
                for pkl in pkl_files:
                    # Extraer número del nombre del .pkl (ej. "20" de "20_topics_nmf.pkl")
                    m = re.search(r"(\d+)_topics_", pkl.name)
                    if m and int(m.group(1)) == n_total:
                        matching_pkl = pkl
                        break
                if matching_pkl:
                    state = load_model_pkl(matching_pkl)
                    coherence, diversity = compute_coherence_diversity(state, args.n_words)
                    coherence_str = f"{coherence:.4f}" if coherence is not None else "N/A"
                    diversity_str = f"{diversity:.4f}" if diversity is not None else "N/A"
                    print(f"  → Coherencia C_v: {coherence_str}, Diversidad: {diversity_str} para {matching_pkl.name}")
                else:
                    print(f"  ⚠️  No se encontró .pkl para {csv_path.name} (n_topics={n_total})")

            metrics = compute_metrics(rows, n_total, coherence, diversity)
            results.append({
                "filename": csv_path.name,
                "metrics": metrics,
                "per_topic": [],
            })
            print(f"  ✓ {csv_path.name}  ({len(rows)} docs, n_topics={n_total})")
        except Exception as exc:
            print(f"  ✗ {csv_path.name}: {exc}")

    if not results:
        print("[ERROR] No se pudo procesar ningún archivo.")
        sys.exit(1)

    # Imprimir resumen y detalles
    print_summary(results, criterion=args.criterio)
    print_detailed(results, show_per_topic=args.per_topic)

    # Guardar reporte si se solicita
    if args.output:
        save_report(results, args.output, criterion=args.criterio)