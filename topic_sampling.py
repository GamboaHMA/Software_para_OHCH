#!/usr/bin/env python3
"""
Visualización de clústeres temáticos para los modelos mejor evaluados.

Los modelos .pkl deben estar en:
    Topic_models/lda/  y  Topic_models/nmf/

Los CSVs de métricas deben estar en (por defecto):
    Topic_experimental_results/

Uso:
    python plot_topic_clusters.py --criterion composite_score
    python plot_topic_clusters.py --criterion coherence

Requiere:
    pandas, matplotlib, seaborn, scikit-learn, pyLDAvis (opcional)
"""

import argparse
from pathlib import Path

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
RESULTS_DIR = BASE_DIR / "Topic_experimental_results"   # <--- NUEVA CARPETA
OUTPUT_DIR = BASE_DIR / "Topic_visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------------------
# Carga de métricas y selección del mejor modelo
# -------------------------------------------------------------------

def load_metrics_csv(path):
    """Carga el CSV de métricas generado por topic_experimenter."""
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    return pd.read_csv(path)

def select_best_model(metrics_df, criterion='composite_score'):
    """Devuelve la fila con el mejor valor del criterio (mayor es mejor)."""
    if criterion not in metrics_df.columns:
        raise ValueError(f"Criterio {criterion} no encontrado")
    best_idx = metrics_df[criterion].idxmax()
    return metrics_df.loc[best_idx]

def get_model_path_from_filename(filename, method):
    """
    Convierte el nombre del archivo CSV (ej. '20_lda_topics_result.csv')
    en la ruta del .pkl correspondiente.
    """
    parts = filename.split('_')
    n_topics = parts[0]          # ej. '20'
    method = method.lower()
    pkl_name = f"{n_topics}_topics_{method}.pkl"
    model_path = MODELS_DIR / method / pkl_name
    return model_path

# -------------------------------------------------------------------
# Visualizaciones
# -------------------------------------------------------------------

def plot_topic_distribution(modeler, title, output_file):
    """Gráfico de barras: número de documentos por tema."""
    rows = modeler.dominant_topic_per_doc()
    df = pd.DataFrame(rows)
    counts = df['tema_dominante'].value_counts().sort_index()
    
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

def plot_document_projection(modeler, title, output_file, method='pca'):
    """
    Proyección de los documentos en 2D usando PCA o t-SNE.
    Colorea por tema dominante.
    """
    if modeler.doc_topic_matrix is None:
        print("  [!] No hay matriz de tópicos, se omite la proyección.")
        return
    
    X = modeler.doc_topic_matrix
    topics = np.argmax(X, axis=1)  # 0-indexado
    
    if method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
    else:
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    
    coords = reducer.fit_transform(X)
    
    df = pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1],
        'tema': topics + 1
    })
    
    plt.figure(figsize=(12, 10))
    n_topics = modeler.n_topics
    palette = 'tab20' if n_topics <= 20 else 'tab20b'
    sns.scatterplot(data=df, x='x', y='y', hue='tema', palette=palette, alpha=0.6, legend='full')
    plt.title(f'Proyección de documentos ({method.upper()})\n{title}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  → Proyección ({method}) guardada en {output_file}")

def plot_topic_similarity_matrix(modeler, title, output_file):
    """Mapa de calor de similitud coseno entre tópicos."""
    if modeler.model is None:
        print("  [!] Modelo no disponible, se omite la matriz de similitud.")
        return
    
    components = modeler.model.components_
    sim_matrix = cosine_similarity(components)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, annot=False, cmap='coolwarm', square=True,
                xticklabels=[f'T{i+1}' for i in range(sim_matrix.shape[0])],
                yticklabels=[f'T{i+1}' for i in range(sim_matrix.shape[0])])
    plt.title(f'Similitud coseno entre tópicos\n{title}')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  → Matriz de similitud guardada en {output_file}")

def plot_pyLDAvis(modeler, output_file):
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
            plot_pyLDAvis(model, OUTPUT_DIR / f"{prefix}_pyLDAvis.html")
    
    plot_metrics_comparison(df_lda, df_nmf, OUTPUT_DIR / "metricas_comparativas.png")
    print("\n✅ ¡Visualizaciones completadas!")

# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualización de clústeres temáticos para los mejores modelos."
    )
    parser.add_argument(
        "--metrics-lda",
        default=str(RESULTS_DIR / "lda_20_200_20_coherence.csv"),
        help="Ruta al CSV de métricas para LDA"
    )
    parser.add_argument(
        "--metrics-nmf",
        default=str(RESULTS_DIR / "nmf_20_200_20_coherence.csv"),
        help="Ruta al CSV de métricas para NMF"
    )
    parser.add_argument(
        "--criterion",
        default="composite_score",
        choices=["composite_score", "coherence", "avg_weight", "balance_score"],
        help="Criterio para seleccionar el mejor modelo"
    )
    args = parser.parse_args()
    
    visualize_best_models(
        Path(args.metrics_lda),
        Path(args.metrics_nmf),
        args.criterion
    )