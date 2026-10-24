"""Búsqueda HDBSCAN con TopicTuner y exportación de candidatos BERTopic.

Fase 1: calcula embeddings y reducción UMAP una vez, compara configuraciones
HDBSCAN y guarda el tuner, la tabla de búsqueda y sus parámetros.
Fase 2: entrena y evalúa uno de los pares elegidos usando los embeddings
persistidos, sin volver a calcularlos.
"""

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np

from topic_modeling import (
    BERTOPIC_EMBEDDING_MODEL,
    DEFAULT_CORPUS_DIR,
    HistoricalTopicModeler,
)
from topic_experimenter import (
    compute_coherence_diversity,
    compute_metrics,
    extract_rows_from_state,
    save_report,
)

BASE_DIR = Path(__file__).parent
DEFAULT_TUNER_PATH = BASE_DIR / "Topic_models" / "bertopic" / "topic_tuner.joblib"
DEFAULT_SEARCH_CSV = (
    BASE_DIR / "Topic_experimental_results" / "bertopic_topictuner_search.csv"
)
DEFAULT_RESULTS_DIR = BASE_DIR / "Topic_results" / "bertopic"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ajusta HDBSCAN para BERTopic con TopicTuner."
    )
    parser.add_argument(
        "--mode",
        choices=("search", "export"),
        required=True,
        help="search explora parámetros; export entrena y evalúa un candidato.",
    )
    parser.add_argument(
        "--corpus-dir",
        default=str(DEFAULT_CORPUS_DIR),
        help="Carpeta con documentos .txt; aplica el filtro común del proyecto.",
    )
    parser.add_argument(
        "--min-document-words",
        type=int,
        default=50,
        help="Mínimo de palabras alfabéticas por documento (default: 50).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Semilla de UMAP para reproducibilidad."
    )
    parser.add_argument(
        "--tuner-path",
        default=str(DEFAULT_TUNER_PATH),
        help="Ruta del estado TopicTuner guardado/cargado.",
    )
    parser.add_argument(
        "--search-csv",
        default=str(DEFAULT_SEARCH_CSV),
        help="CSV con los resultados HDBSCAN de TopicTuner.",
    )
    parser.add_argument(
        "--cluster-sizes",
        type=int,
        nargs="+",
        default=[20, 40, 60, 80, 100, 150, 200],
        help="Valores de min_cluster_size para search.",
    )
    parser.add_argument(
        "--min-samples-ratios",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.25, 0.50],
        help="Proporciones de min_cluster_size para min_samples (0-1).",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        help="Parámetro del candidato al usar --mode export.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        help="Parámetro del candidato al usar --mode export.",
    )
    parser.add_argument(
        "--n-words",
        type=int,
        default=15,
        help="Palabras por tópico en las métricas C_v/diversidad.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directorio para modelo y asignaciones exportados.",
    )
    return parser


def load_documents(corpus_dir: str, min_document_words: int) -> tuple[list[Path], list[str]]:
    loader = HistoricalTopicModeler(
        corpus_dir=corpus_dir,
        method="bertopic",
        min_document_words=min_document_words,
    )
    loader.load_corpus()
    return loader.doc_paths, loader.raw_texts


def make_tuner(docs: list[str], seed: int, verbose: int):
    try:
        from sentence_transformers import SentenceTransformer
        from topictuner import TopicModelTuner
    except ImportError as exc:
        raise ImportError(
            "Instala las dependencias en este entorno: "
            "pip install topicmodeltuner sentence-transformers"
        ) from exc

    encoder = SentenceTransformer(BERTOPIC_EMBEDDING_MODEL)
    tuner = TopicModelTuner(
        embedding_model=encoder,
        docs=docs,
        reducer_random_state=seed,
        reducer_components=5,
        verbose=verbose,
    )
    print("[INFO] Calculando embeddings con el encoder multilingüe del proyecto…")
    tuner.createEmbeddings()
    print("[INFO] Reduciendo embeddings con UMAP…")
    tuner.reduce()
    return tuner


def run_search(args: argparse.Namespace) -> None:
    _, docs = load_documents(args.corpus_dir, args.min_document_words)
    print(f"[INFO] TopicTuner trabajará con {len(docs)} documentos.")
    tuner = make_tuner(docs, args.seed, verbose=1)

    results = tuner.pseudoGridSearch(
        args.cluster_sizes,
        args.min_samples_ratios,
    )
    results = results.sort_values(
        ["number_uncategorized", "number_of_clusters", "min_cluster_size", "min_samples"]
    )
    search_csv = Path(args.search_csv)
    search_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(search_csv, index=False)

    tuner_path = Path(args.tuner_path)
    tuner_path.parent.mkdir(parents=True, exist_ok=True)
    tuner.save(str(tuner_path), save_docs=True, save_embeddings=True)

    print(f"[INFO] Resultados de búsqueda: {search_csv}")
    print(f"[INFO] Tuner con documentos y embeddings guardado: {tuner_path}")
    print("[INFO] Revisa number_uncategorized y number_of_clusters; TopicTuner no calcula C_v.")
    print("[INFO] Exporta los pares prometedores por separado para medir coherencia y diversidad.")


def write_assignments(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["documento", "ruta", "tema_dominante", "peso"],
        )
        writer.writeheader()
        writer.writerows(rows)


def run_export(args: argparse.Namespace) -> None:
    if args.min_cluster_size is None or args.min_samples is None:
        raise ValueError("--mode export requiere --min-cluster-size y --min-samples.")
    if args.min_cluster_size <= 1 or not 1 <= args.min_samples <= args.min_cluster_size:
        raise ValueError("Se requiere min_cluster_size > 1 y 1 <= min_samples <= min_cluster_size.")

    try:
        from topictuner import TopicModelTuner
    except ImportError as exc:
        raise ImportError("Instala TopicTuner: pip install topicmodeltuner") from exc

    tuner_path = Path(args.tuner_path)
    if not tuner_path.is_file():
        raise FileNotFoundError(f"No existe el estado guardado de TopicTuner: {tuner_path}")
    tuner = TopicModelTuner.load(str(tuner_path))
    if tuner.docs is None or tuner.embeddings is None:
        raise ValueError("El estado guardado debe incluir documentos y embeddings.")
    doc_paths, current_docs = load_documents(args.corpus_dir, args.min_document_words)
    if current_docs != list(tuner.docs):
        raise ValueError(
            "El corpus cambió desde la búsqueda; vuelve a ejecutar --mode search "
            "antes de exportar candidatos."
        )

    topic_model = tuner.getBERTopicModel(args.min_cluster_size, args.min_samples)
    topic_model.top_n_words = args.n_words
    topic_model.calculate_probabilities = False
    print(
        "[INFO] Ajustando BERTopic con "
        f"min_cluster_size={args.min_cluster_size}, min_samples={args.min_samples}…"
    )
    _, probabilities = topic_model.fit_transform(tuner.docs, embeddings=tuner.embeddings)
    assignments = np.asarray(topic_model.topics_)
    topic_info = topic_model.get_topic_info()
    topic_ids = sorted(int(topic_id) for topic_id in topic_info["Topic"])
    n_topics = sum(topic_id != -1 for topic_id in topic_ids)
    doc_topic_matrix = HistoricalTopicModeler._hard_topic_matrix(
        assignments, probabilities, topic_ids
    )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stem = f"bertopic_tmt_mcs_{args.min_cluster_size}_ms_{args.min_samples}"
    model_path = results_dir / f"{stem}.pickle"
    metadata_path = Path(f"{model_path}.metadata.pkl")
    assignments_path = results_dir / f"{stem}_docs.csv"
    metrics_path = results_dir / f"{stem}_metrics.csv"

    topic_model.save(str(model_path), serialization="pickle")
    state = {
        "config": {
            "corpus_dir": str(Path(args.corpus_dir)),
            "n_topics": n_topics,
            "method": "bertopic",
            "n_words": args.n_words,
            "min_df": 2,
            "max_df": 0.90,
            "random_state": args.seed,
            "min_count": 50,
            "embedding_model": BERTOPIC_EMBEDDING_MODEL,
            "min_topic_size": args.min_cluster_size,
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples,
            "min_document_words": args.min_document_words,
            "topic_tuner": True,
            "umap_random_state": int(tuner.reducer_random_state),
        },
        "doc_paths": [str(path) for path in doc_paths],
        "raw_texts": list(tuner.docs),
        "processed_texts": [],
        "vectorizer": None,
        "dtm": None,
        "doc_topic_matrix": doc_topic_matrix,
        "topic_ids": topic_ids,
        "top2vec_reduced": False,
    }
    with metadata_path.open("wb") as stream:
        pickle.dump(state, stream, protocol=pickle.HIGHEST_PROTOCOL)

    metric_state = {**state, "model": topic_model}
    rows = extract_rows_from_state(metric_state)
    write_assignments(rows, assignments_path)
    coherence, diversity = compute_coherence_diversity(metric_state, args.n_words)
    metrics = compute_metrics(rows, n_topics, coherence, diversity, topic_ids)
    save_report(
        [{"filename": model_path.name, "metrics": metrics, "per_topic": []}],
        str(metrics_path),
    )
    print(f"[INFO] Modelo BERTopic guardado: {model_path}")
    print(f"[INFO] Metadatos guardados: {metadata_path}")
    print(f"[INFO] Asignaciones guardadas: {assignments_path}")
    print(f"[INFO] Métricas guardadas: {metrics_path}")
    print(f"[INFO] Tópicos encontrados (sin outliers): {n_topics}; outliers: {int((assignments == -1).sum())}")


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "search":
        run_search(args)
    else:
        run_export(args)


if __name__ == "__main__":
    main()
