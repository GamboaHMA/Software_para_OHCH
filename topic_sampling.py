#!/usr/bin/env python3
"""
Visualización de clústeres temáticos para los modelos mejor evaluados.

Los modelos .pkl deben estar en:
    Topic_models/lda/  y  Topic_models/nmf/

También se pueden visualizar modelos directamente con --model-path, incluidos
los modelos Top2Vec y BERTopic guardados en sus formatos nativos.

Los CSVs de métricas deben estar en (por defecto):
    Topic_experimental_results/

Uso:
    python plot_topic_clusters.py --criterion composite_score
    python plot_topic_clusters.py --criterion coherence

Requiere:
    pandas, matplotlib, seaborn, scikit-learn, pyLDAvis (opcional)
"""

import argparse
import pickle
import re
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

# pyLDAvis (solo para LDA)
try:
    import pyLDAvis
    import pyLDAvis.sklearn
    PYLDAVIS_AVAILABLE = True
except ImportError:
    PYLDAVIS_AVAILABLE = False

# Importamos tu clase de topic modeling
from topic_modeling import HistoricalTopicModeler

# -------------------------------------------------------------------
# Constantes y rutas
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "Topic_models"
RESULTS_DIR = BASE_DIR / "Topic_experimental_results"
OUTPUT_DIR = BASE_DIR / "Topic_visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
COSINE_TOPIC_SIMILARITY_LABEL = "Similitud coseno entre tópicos"

# -------------------------------------------------------------------
# Carga de métricas y selección del mejor modelo
# -------------------------------------------------------------------

def load_metrics_csv(path):
    """Carga un reporte consolidado de métricas generado por topic_experimenter."""
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"El reporte no contiene resultados: {path}")
    if "filename" not in df.columns:
        raise ValueError("El reporte debe incluir la columna 'filename'.")
    df["method"] = df["filename"].astype(str).map(infer_method)
    if "n_topics_total" in df.columns:
        df["n_topics_total"] = pd.to_numeric(df["n_topics_total"], errors="coerce")
    return df


def infer_method(filename):
    """Identifica el algoritmo a partir del nombre del resultado."""
    match = re.search(r"(?:^|_)(top2vec|bertopic|lda|nmf)(?:_|$)", str(filename).lower())
    return match.group(1) if match else "desconocido"

def select_best_model(metrics_df, criterion='composite_score'):
    """Devuelve la fila con el mejor valor del criterio (mayor es mejor)."""
    if criterion not in metrics_df.columns:
        raise ValueError(f"Criterio {criterion} no encontrado")
    valid = metrics_df.dropna(subset=[criterion])
    if valid.empty:
        raise ValueError(f"No hay valores disponibles para el criterio {criterion}")
    best_idx = valid[criterion].idxmax()
    return metrics_df.loc[best_idx]

def get_model_path_from_filename(filename, method):
    """
    Convierte el nombre del archivo CSV (ej. '20_lda_topics_result.csv')
    en la ruta del .pkl correspondiente.
    """
    method = method.lower()
    stem = Path(filename).stem
    exact_matches = [
        path for path in MODELS_DIR.rglob(Path(filename).name)
        if path.is_file() and not path.name.endswith(".metadata.pkl")
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]

    topic_match = re.match(r"(\d+)_", stem)
    if method in ("lda", "nmf") and topic_match:
        pkl_name = f"{topic_match.group(1)}_topics_{method}.pkl"
        candidates = [MODELS_DIR / method / pkl_name, MODELS_DIR / pkl_name]
    elif method == "top2vec":
        candidates = sorted(MODELS_DIR.glob("top2vec*.model"))
    elif method == "bertopic":
        candidates = sorted([
            *MODELS_DIR.rglob("bertopic*.pickle"),
            *MODELS_DIR.rglob("bertopic*.pkl"),
        ])
    else:
        candidates = []
    return next((path for path in candidates if path.exists()), None)


def find_model_metadata(model_path):
    """Busca el sidecar adyacente o en las subcarpetas organizadas por método."""
    adjacent = Path(f"{model_path}.metadata.pkl")
    if adjacent.is_file():
        return adjacent
    matches = list(MODELS_DIR.rglob(f"{model_path.name}.metadata.pkl"))
    method = infer_method(model_path.name)
    method_matches = [path for path in matches if path.parent.name.lower() == method]
    if len(method_matches) == 1:
        return method_matches[0]
    return matches[0] if len(matches) == 1 else None

# -------------------------------------------------------------------
# Visualizaciones
# -------------------------------------------------------------------

def plot_topic_distribution(modeler, title, output_file):
    """Gráfico de barras: número de documentos por tema."""
    if modeler.doc_topic_matrix is None or len(modeler.doc_topic_matrix) == 0:
        print("  [!] No hay asignaciones por documento; se omite la distribución.")
        return
    topics = np.argmax(modeler.doc_topic_matrix, axis=1)
    topic_ids = getattr(modeler, "topic_ids", None)
    labels = [int(topic_ids[index]) for index in topics] if topic_ids else (topics + 1).tolist()
    counts = pd.Series(labels).value_counts().sort_index()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=counts.index, y=counts.values, palette='viridis')
    plt.title(f'Distribución de documentos por tema\n{title}')
    plt.xlabel('Tema')
    plt.ylabel('Número de documentos')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  → Distribución guardada en {output_file}")


def plot_distribution_from_csv(csv_path, title, output_file):
    """Grafica asignaciones existentes cuando el modelo no puede cargarse."""
    if not csv_path.is_file():
        print(f"  [!] No se encontró el CSV de asignaciones: {csv_path}")
        return
    assignments = pd.read_csv(csv_path)
    if "tema_dominante" not in assignments:
        print(f"  [!] El CSV no tiene la columna tema_dominante: {csv_path}")
        return
    counts = assignments["tema_dominante"].value_counts().sort_index()
    plt.figure(figsize=(12, 6))
    plt.bar(counts.index.astype(str), counts.values, color="#4C78A8")
    plt.title(f"Distribución de documentos por tema\n{title}")
    plt.xlabel("Tema / ID del tópico")
    plt.ylabel("Número de documentos")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  → Distribución desde CSV guardada en {output_file}")

def plot_document_projection(modeler, title, output_file, method='pca'):
    """
    Proyección de los documentos en 2D usando PCA o t-SNE.
    Colorea por tema dominante.
    """
    if modeler.doc_topic_matrix is None:
        print("  [!] No hay matriz de tópicos, se omite la proyección.")
        return
    
    X = modeler.doc_topic_matrix
    if X.shape[0] < 2 or X.shape[1] < 2:
        print("  [!] Se requieren al menos dos documentos y dos tópicos; se omite la proyección.")
        return
    topic_columns = np.argmax(X, axis=1)  # 0-indexado
    topic_ids = getattr(modeler, "topic_ids", None)
    topics = np.array([topic_ids[index] for index in topic_columns]) if topic_ids else topic_columns + 1
    
    if method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
    else:
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    
    coords = reducer.fit_transform(X)
    
    df = pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1],
        'tema': topics
    })
    
    plt.figure(figsize=(12, 10))
    n_topics = len(np.unique(topics))
    palette = 'tab20' if n_topics <= 20 else 'tab20b'
    sns.scatterplot(data=df, x='x', y='y', hue='tema', palette=palette, alpha=0.6, legend='full')
    plt.title(f'Proyección de documentos ({method.upper()})\n{title}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  → Proyección ({method}) guardada en {output_file}")


def plot_bertopic_documents(modeler, title, output_file, sample=0.1):
    """Genera la visualización nativa de BERTopic con muestreo por tópico."""
    if modeler.method != "bertopic" or modeler.model is None:
        print("  [!] Se necesita un modelo BERTopic cargado; se omite la visualización.")
        return
    if not 0 < sample <= 1:
        raise ValueError("El muestreo BERTopic debe estar en el intervalo (0, 1].")

    docs = getattr(modeler, "raw_texts", None)
    topics = getattr(modeler.model, "topics_", None)
    if docs is None or topics is None or len(docs) != len(topics):
        print("  [!] Textos y asignaciones BERTopic ausentes o desalineados; se omite.")
        return

    fig = modeler.model.visualize_documents(
        docs,
        topics=topics,
        sample=sample,
        title=f"{title} | muestra {sample:.0%} por tópico",
        hide_annotations=True,
    )
    fig.write_html(str(output_file))
    print(f"  → Visualización BERTopic (sample={sample:.0%}) guardada en {output_file}")

def plot_topic_similarity_matrix(modeler, title, output_file):
    """Mapa de calor usando la representación propia de cada algoritmo."""
    if modeler.model is None:
        if modeler.doc_topic_matrix is None:
            print("  [!] No hay modelo ni matriz documento-tópico; se omite la similitud.")
            return
        # En el fallback se compara el patrón de asignación de documentos a cada
        # tópico; no equivale a la similitud semántica interna del modelo.
        components = np.asarray(modeler.doc_topic_matrix).T
        similarity_label = "Similitud de distribución entre tópicos"
    elif modeler.method in ('lda', 'nmf'):
        components = modeler.model.components_
        similarity_label = COSINE_TOPIC_SIMILARITY_LABEL
    elif modeler.method == 'top2vec':
        components = (
            modeler.model.topic_vectors_reduced
            if modeler._top2vec_reduced
            else modeler.model.topic_vectors
        )
        similarity_label = COSINE_TOPIC_SIMILARITY_LABEL
    else:
        components = getattr(modeler.model, 'topic_embeddings_', None)
        if components is None or len(components) == 0:
            components = getattr(modeler.model, 'c_tf_idf_', None)
        similarity_label = COSINE_TOPIC_SIMILARITY_LABEL

    if components is None or components.shape[0] < 2:
        print("  [!] No hay representaciones de al menos dos temas; se omite la matriz.")
        return

    sim_matrix = cosine_similarity(components)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, annot=False, cmap='coolwarm', square=True,
                xticklabels=[f'T{i+1}' for i in range(sim_matrix.shape[0])],
                yticklabels=[f'T{i+1}' for i in range(sim_matrix.shape[0])])
    plt.title(f'{similarity_label}\n{title}')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  → Matriz de similitud guardada en {output_file}")


def plot_bertopic_stopword_similarity(model_path, output_file):
    """Grafica la similitud c-TF-IDF con stopwords sin persistir cambios al modelo."""
    from sklearn.feature_extraction.text import CountVectorizer
    from spacy.lang.es.stop_words import STOP_WORDS

    from topic_modeling import STOPWORDS_EXTRA

    modeler = HistoricalTopicModeler.load_model(model_path)
    if modeler.method != "bertopic":
        raise ValueError(f"El modelo no es BERTopic: {model_path}")

    topics = getattr(modeler.model, "topics_", None)
    if topics is None or len(topics) != len(modeler.raw_texts):
        raise ValueError(f"No hay asignaciones alineadas con los textos en {model_path}")

    # update_topics reconstruye únicamente las representaciones de los tópicos
    # existentes; no ejecuta fit/fit_transform ni se guarda el objeto modificado.
    modeler.model.update_topics(
        modeler.raw_texts,
        topics=list(topics),
        top_n_words=modeler.n_words,
        vectorizer_model=CountVectorizer(
            stop_words=sorted(set(STOP_WORDS) | STOPWORDS_EXTRA),
        ),
    )
    components = getattr(modeler.model, "c_tf_idf_", None)
    if components is None or components.shape[0] < 2:
        raise ValueError(f"No se generó una matriz c-TF-IDF válida para {model_path}")

    sim_matrix = cosine_similarity(components)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        sim_matrix,
        annot=False,
        cmap="coolwarm",
        square=True,
        xticklabels=[f"T{i + 1}" for i in range(sim_matrix.shape[0])],
        yticklabels=[f"T{i + 1}" for i in range(sim_matrix.shape[0])],
    )
    plt.title(f"{COSINE_TOPIC_SIMILARITY_LABEL} (c-TF-IDF, sin stopwords)\n"
              f"BERTopic | {modeler.n_topics} tópicos")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  → Matriz BERTopic con stopwords guardada en {output_file}")


def visualize_model(model_path, bertopic_sample=0.1):
    """Carga un modelo guardado (incluidos los formatos nativos) y lo grafica."""
    modeler = HistoricalTopicModeler.load_model(model_path)
    title = f"{modeler.method.upper()} | {modeler.n_topics} tópicos"
    prefix = Path(model_path).stem

    plot_topic_distribution(
        modeler, title, OUTPUT_DIR / f"{prefix}_distribucion.png"
    )
    if modeler.method == "bertopic":
        plot_bertopic_documents(
            modeler,
            title,
            OUTPUT_DIR / f"{prefix}_documentos.html",
            sample=bertopic_sample,
        )
    else:
        plot_document_projection(
            modeler, title, OUTPUT_DIR / f"{prefix}_proyeccion_pca.png", method='pca'
        )
    plot_topic_similarity_matrix(
        modeler, title, OUTPUT_DIR / f"{prefix}_similitud_topicos.png"
    )
    if modeler.method == 'lda':
        plot_pyldavis(modeler, OUTPUT_DIR / f"{prefix}_pyLDAvis.html")
    print(f"\n✅ Visualizaciones guardadas en {OUTPUT_DIR}")

def plot_pyldavis(modeler, output_file):
    """Genera la visualización interactiva de pyLDAvis (solo para LDA)."""
    if not PYLDAVIS_AVAILABLE:
        print("  [!] pyLDAvis no instalado, omitiendo.")
        return
    if modeler.method != 'lda':
        print("  [!] pyLDAvis solo está disponible para LDA.")
        return
    
    try:
        vis = pyLDAvis.sklearn.prepare(modeler.model, modeler.dtm, modeler.vectorizer)
        pyLDAvis.save_html(vis, output_file)
        print(f"  → pyLDAvis guardado en {output_file}")
    except Exception as e:
        print(f"  [!] Error generando pyLDAvis: {e}")

def plot_metrics_comparison(df_lda, df_nmf, output_file):
    """Gráfico de líneas comparando composite_score y coherence vs n_topics."""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(df_lda['n_topics_total'], df_lda['composite_score'], 'o-', label='LDA')
    plt.plot(df_nmf['n_topics_total'], df_nmf['composite_score'], 's-', label='NMF')
    plt.xlabel('Número de tópicos')
    plt.ylabel('Puntuación compuesta')
    plt.legend()
    plt.grid(True)
    
    if 'coherence' in df_lda.columns and 'coherence' in df_nmf.columns:
        plt.subplot(1, 2, 2)
        plt.plot(df_lda['n_topics_total'], df_lda['coherence'], 'o-', label='LDA')
        plt.plot(df_nmf['n_topics_total'], df_nmf['coherence'], 's-', label='NMF')
        plt.xlabel('Número de tópicos')
        plt.ylabel('Coherencia (C_v)')
        plt.legend()
        plt.grid(True)
    
    plt.suptitle('Comparación de métricas de evaluación')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  → Gráfico comparativo guardado en {output_file}")

def plot_report_rankings(df, output_file):
    """Grafica todas las filas del reporte, ordenadas por algoritmo y tópicos."""
    metrics = [
        ("composite_score", "Puntuación compuesta"),
        ("coherence", "Coherencia C_v"),
        ("diversity", "Diversidad"),
    ]
    available = [(column, label) for column, label in metrics if column in df and df[column].notna().any()]
    if not available:
        print("  [!] El reporte no contiene métricas graficables.")
        return

    plot_df = df.copy()
    plot_df["modelo"] = plot_df.apply(
        lambda row: f"{row['method'].upper()} {int(row['n_topics_total']) if pd.notna(row.get('n_topics_total')) else '?'}",
        axis=1,
    )
    fig, axes = plt.subplots(len(available), 1, figsize=(14, 4.2 * len(available)), squeeze=False)
    for ax, (metric, label) in zip(axes[:, 0], available):
        data = plot_df.dropna(subset=[metric]).sort_values(metric, ascending=True)
        colors = data["method"].map({"lda": "#4C78A8", "nmf": "#F58518", "top2vec": "#54A24B", "bertopic": "#E45756"}).fillna("#777777")
        ax.barh(data["modelo"], data[metric], color=colors)
        ax.set_xlabel(label)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Comparación de modelos del reporte experimental", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Comparación de modelos guardada en {output_file}")


def plot_metric_trends(df, output_file):
    """Muestra cómo cambian métricas y cobertura según el número de tópicos."""
    metrics = [
        ("composite_score", "Puntuación compuesta"),
        ("coherence", "Coherencia C_v"),
        ("diversity", "Diversidad"),
        ("avg_weight", "Peso dominante medio"),
        ("balance_score", "Balance"),
        ("coverage_ratio", "Cobertura"),
        ("topic_entropy", "Entropía normalizada"),
        ("outlier_pct", "Documentos outlier (%)"),
    ]
    metrics = [(col, label) for col, label in metrics if col in df and df[col].notna().any()]
    topic_df = df[df["method"].isin(["lda", "nmf"])].dropna(subset=["n_topics_total"])
    if not metrics or topic_df.empty:
        print("  [!] No hay datos LDA/NMF suficientes para tendencias según número de tópicos.")
        return

    ncols = 2
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows), squeeze=False)
    for ax, (metric, label) in zip(axes.flat, metrics):
        for method, group in topic_df.groupby("method"):
            group = group.dropna(subset=[metric]).sort_values("n_topics_total")
            if not group.empty:
                ax.plot(group["n_topics_total"], group[metric], marker="o", label=method.upper())
        ax.set_title(label)
        ax.set_xlabel("Número de tópicos")
        ax.grid(alpha=0.25)
        ax.legend()
    for ax in axes.flat[len(metrics):]:
        ax.set_visible(False)
    fig.suptitle("Tendencias de métricas por número de tópicos", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Tendencias guardadas en {output_file}")


def save_report_summary(df, criterion, output_file):
    """Guarda una síntesis legible de los mejores resultados por algoritmo."""
    lines = [f"Resumen de experimentación — criterio: {criterion}", ""]
    for method, group in df.groupby("method", sort=True):
        valid = group.dropna(subset=[criterion]) if criterion in group else pd.DataFrame()
        if valid.empty:
            lines.append(f"{method.upper()}: sin valores para {criterion}.")
            continue
        best = valid.loc[valid[criterion].idxmax()]
        count = int(best["n_topics_total"]) if pd.notna(best.get("n_topics_total")) else "N/D"
        lines.append(
            f"{method.upper()}: {best['filename']} | tópicos={count} | "
            f"{criterion}={best[criterion]:.4f} | coherence={best.get('coherence', float('nan'))} | "
            f"diversity={best.get('diversity', float('nan'))}"
        )
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  → Síntesis guardada en {output_file}")


def visualize_report(report_path, criterion="composite_score", bertopic_sample=0.1):
    """Analiza el reporte completo y visualiza el modelo mejor puntuado por método."""
    df = load_metrics_csv(report_path)
    if criterion not in df.columns:
        raise ValueError(f"El criterio '{criterion}' no existe en el reporte.")

    plot_report_rankings(df, OUTPUT_DIR / "comparacion_modelos.png")
    plot_metric_trends(df, OUTPUT_DIR / "tendencias_metricas.png")
    save_report_summary(df, criterion, OUTPUT_DIR / "resumen_experimentacion.txt")

    best_rows = []
    for method, group in df.groupby("method", sort=True):
        valid = group.dropna(subset=[criterion])
        if valid.empty:
            print(f"[!] {method}: sin valores de {criterion}; no se selecciona.")
            continue
        if method == "bertopic":
            # El reporte de muestreo compara varias configuraciones BERTopic;
            # generar gráficos para todas permite contrastarlas individualmente.
            best_rows.extend(row for _, row in valid.iterrows())
        else:
            best_rows.append(valid.loc[valid[criterion].idxmax()])

    for row in best_rows:
        method = row["method"]
        model_path = get_model_path_from_filename(row["filename"], method)
        if model_path is None:
            print(f"[!] No se encontró el archivo del modelo para {row['filename']} ({method}).")
            continue
        print(f"\n--- Mejor {method.upper()} según {criterion}: {row['filename']} ({row[criterion]:.4f}) ---")
        prefix = Path(row["filename"]).stem
        title = f"{method.upper()} | {row['n_topics_total']} tópicos | {criterion}={row[criterion]:.4f}"
        try:
            modeler = HistoricalTopicModeler.load_model(model_path)
            title = f"{method.upper()} | {modeler.n_topics} tópicos | {criterion}={row[criterion]:.4f}"
        except Exception as exc:
            print(f"  [!] No se pudo visualizar {model_path}: {exc}")
            metadata_path = find_model_metadata(model_path)
            try:
                if metadata_path is None:
                    raise FileNotFoundError(f"No se encontró metadata para {model_path.name}")
                with metadata_path.open("rb") as metadata_file:
                    state = pickle.load(metadata_file)
                matrix = state.get("doc_topic_matrix")
                if matrix is None:
                    raise ValueError("Los metadatos no contienen doc_topic_matrix")
                config = state.get("config", {})
                modeler = SimpleNamespace(
                    method=config.get("method", method),
                    n_topics=config.get("n_topics", int(row["n_topics_total"])),
                    doc_topic_matrix=matrix,
                    topic_ids=state.get("topic_ids", list(range(matrix.shape[1]))),
                    model=None,
                    _top2vec_reduced=state.get("top2vec_reduced", False),
                )
                print("  → Se usarán los metadatos guardados; similitud basada en asignaciones documentales.")
            except Exception as metadata_error:
                print(f"  [!] No se pudieron leer metadatos: {metadata_error}")
                csv_path = BASE_DIR / "Topic_results" / method / str(row["filename"])
                plot_distribution_from_csv(
                    csv_path, title,
                    OUTPUT_DIR / f"{prefix}_distribucion.png",
                )
                continue

        plot_topic_distribution(modeler, title, OUTPUT_DIR / f"{prefix}_distribucion.png")
        if method == "bertopic":
            plot_bertopic_documents(
                modeler,
                title,
                OUTPUT_DIR / f"{prefix}_documentos.html",
                sample=bertopic_sample,
            )
        else:
            plot_document_projection(
                modeler, title, OUTPUT_DIR / f"{prefix}_proyeccion_pca.png"
            )
        plot_topic_similarity_matrix(modeler, title, OUTPUT_DIR / f"{prefix}_similitud_topicos.png")
        if modeler.model is not None and modeler.method == "lda":
            plot_pyldavis(modeler, OUTPUT_DIR / f"{prefix}_pyLDAvis.html")

    print(f"\n✅ Resultados y visualizaciones almacenados en {OUTPUT_DIR}")
# -------------------------------------------------------------------
# Función principal
# -------------------------------------------------------------------

def visualize_best_models(metrics_lda_path, metrics_nmf_path, criterion='composite_score'):
    # Cargar métricas
    df_lda = load_metrics_csv(metrics_lda_path)
    df_nmf = load_metrics_csv(metrics_nmf_path)
    
    # Seleccionar mejores
    best_lda = select_best_model(df_lda, criterion)
    best_nmf = select_best_model(df_nmf, criterion)
    
    print(f"Mejor LDA: {best_lda['filename']} ({criterion}={best_lda[criterion]:.4f})")
    print(f"Mejor NMF: {best_nmf['filename']} ({criterion}={best_nmf[criterion]:.4f})")
    
    # Obtener rutas de los modelos
    path_lda = get_model_path_from_filename(best_lda['filename'], 'lda')
    path_nmf = get_model_path_from_filename(best_nmf['filename'], 'nmf')
    
    if not path_lda.exists():
        print(f"[ERROR] No se encontró el modelo LDA: {path_lda}")
        return
    if not path_nmf.exists():
        print(f"[ERROR] No se encontró el modelo NMF: {path_nmf}")
        return
    
    # Cargar modelos
    print("Cargando modelo LDA...")
    model_lda = HistoricalTopicModeler.load_model(path_lda)
    print("Cargando modelo NMF...")
    model_nmf = HistoricalTopicModeler.load_model(path_nmf)
    
    # Generar visualizaciones
    for model, name in [(model_lda, 'LDA'), (model_nmf, 'NMF')]:
        metric_val = best_lda[criterion] if name == 'LDA' else best_nmf[criterion]
        base_title = f"{name} | {model.n_topics} tópicos | {criterion}={metric_val:.4f}"
        prefix = f"{model.n_topics}topics_{model.method}"
        
        print(f"\n--- Generando gráficos para {name} ({model.n_topics} tópicos) ---")
        
        plot_topic_distribution(model, base_title, OUTPUT_DIR / f"{prefix}_distribucion.png")
        plot_document_projection(model, base_title, OUTPUT_DIR / f"{prefix}_proyeccion_pca.png", method='pca')
        # Si quieres t-SNE, descomenta:
        # plot_document_projection(model, base_title, OUTPUT_DIR / f"{prefix}_proyeccion_tsne.png", method='tsne')
        plot_topic_similarity_matrix(model, base_title, OUTPUT_DIR / f"{prefix}_similitud_topicos.png")
        
        if model.method == 'lda':
            plot_pyldavis(model, OUTPUT_DIR / f"{prefix}_pyLDAvis.html")
    
    plot_metrics_comparison(df_lda, df_nmf, OUTPUT_DIR / "metricas_comparativas.png")
    print("\n✅ ¡Visualizaciones completadas!")

# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualización y análisis del reporte consolidado de topic modeling."
    )
    parser.add_argument(
        "--report",
        default=str(RESULTS_DIR / "report.csv"),
        help="Reporte consolidado de topic_experimenter (default: Topic_experimental_results/report.csv)"
    )
    parser.add_argument(
        "--criterion",
        default="composite_score",
        choices=["composite_score", "coherence", "diversity", "avg_weight", "balance_score"],
        help="Criterio para seleccionar el mejor modelo"
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Ruta directa a un modelo guardado (.pkl, .model o .pickle), "
        "incluidos Top2Vec y BERTopic.",
    )
    parser.add_argument(
        "--bertopic-sample",
        type=float,
        default=0.1,
        help=(
            "Proporción de documentos por tópico que BERTopic muestra en "
            "visualize_documents (0 < valor <= 1; default: 0.1)."
        ),
    )
    args = parser.parse_args()

    if not 0 < args.bertopic_sample <= 1:
        parser.error("--bertopic-sample debe estar en el intervalo (0, 1].")

    if args.model_path:
        visualize_model(Path(args.model_path), bertopic_sample=args.bertopic_sample)
    else:
        visualize_report(
            Path(args.report), args.criterion, bertopic_sample=args.bertopic_sample
        )