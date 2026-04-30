"""
Topic Modeling sobre documentos históricos en español.

Carga todos los .txt de Reconstrucciones_perplexity, los preprocesa con
spaCy (es_core_news_sm) y aplica LDA y NMF con scikit-learn.

Uso rápido:
    python topic_modeling.py

Opciones:
    --corpus-dir   Ruta a la carpeta raíz de los documentos (default: Reconstrucciones_perplexity)
    --n-topics     Número de temas a descubrir (default: 10)
    --method       Algoritmo: lda | nmf  (default: lda)
    --n-words      Palabras por tema al imprimir resultados (default: 15)
    --output       Guarda una tabla doc→tema dominante en este archivo CSV
"""

import argparse
import csv
import glob
import os
import re
from pathlib import Path

import spacy
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
DEFAULT_CORPUS_DIR = BASE_DIR / "Reconstrucciones_perplexity"

# Stopwords adicionales para español histórico (siglos XIX-XX)
STOPWORDS_EXTRA = {
    "así", "aún", "á", "é", "ó", "ú", "í", "ü",
    "ser", "estar", "haber", "tener", "hacer", "ir",
    "más", "muy", "bien", "todo", "todos", "toda", "todas",
    "cada", "entre", "sobre", "bajo", "tras", "ante",
    "sino", "aunque", "pues", "porque", "cuando", "donde",
    "como", "que", "quien", "cual", "cuyo", "cuya",
    "allí", "allá", "aquí", "acá", "ahora", "antes", "después",
    "siempre", "nunca", "jamás", "ya", "aun",
    "sr", "sra", "don", "doña",
}


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class HistoricalTopicModeler:
    """
    Encapsula la carga, preprocesamiento y modelado de temas para un corpus
    de documentos históricos en español.

    Parámetros
    ----------
    corpus_dir : str | Path
        Carpeta raíz que contiene los .txt (se busca de forma recursiva).
    n_topics : int
        Número de temas latentes a descubrir.
    method : {'lda', 'nmf'}
        Algoritmo de topic modeling a utilizar.
    n_words : int
        Palabras representativas por tema que se muestran al imprimir.
    min_df : int
        Frecuencia mínima de documento para incluir un término.
    max_df : float
        Proporción máxima de documentos para incluir un término.
    """

    def __init__(
        self,
        corpus_dir: str | Path = DEFAULT_CORPUS_DIR,
        n_topics: int = 10,
        method: str = "lda",
        n_words: int = 15,
        min_df: int = 2,
        max_df: float = 0.90,
        random_state: int = 42,
    ):
        self.corpus_dir = Path(corpus_dir)
        self.n_topics = n_topics
        self.method = method.lower()
        self.n_words = n_words
        self.min_df = min_df
        self.max_df = max_df
        self.random_state = random_state

        if self.method not in ("lda", "nmf"):
            raise ValueError("method debe ser 'lda' o 'nmf'")

        # Cargado al llamar fit()
        self.doc_paths: list[Path] = []
        self.raw_texts: list[str] = []
        self.processed_texts: list[str] = []
        self.vectorizer = None
        self.dtm = None          # document-term matrix
        self.model = None
        self.doc_topic_matrix = None

        # Modelo lingüístico de spaCy
        print("[INFO] Cargando modelo de spaCy (es_core_news_sm)…")
        self._nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])

    # ------------------------------------------------------------------
    # Carga de documentos
    # ------------------------------------------------------------------

    def load_corpus(self) -> int:
        """
        Carga recursivamente todos los archivos .txt del corpus_dir.
        Devuelve el número de documentos cargados.
        """
        pattern = str(self.corpus_dir / "**" / "*.txt")
        paths = sorted(glob.glob(pattern, recursive=True))

        if not paths:
            raise FileNotFoundError(
                f"No se encontraron archivos .txt en: {self.corpus_dir}"
            )

        self.doc_paths = [Path(p) for p in paths]
        self.raw_texts = []

        for p in self.doc_paths:
            text = p.read_text(encoding="utf-8", errors="replace")
            self.raw_texts.append(text)

        print(f"[INFO] {len(self.doc_paths)} documentos cargados.")
        return len(self.doc_paths)

    # ------------------------------------------------------------------
    # Preprocesamiento
    # ------------------------------------------------------------------

    def _clean_text(self, text: str) -> str:
        """Limpieza básica antes de pasar por spaCy."""
        # Normalizar espacios y saltos de línea
        text = re.sub(r"\s+", " ", text)
        # Eliminar caracteres no alfabéticos excepto acentos y ñ
        text = re.sub(r"[^\w\sáéíóúüñÁÉÍÓÚÜÑ]", " ", text)
        return text.strip()

    def _lemmatize(self, text: str) -> str:
        """
        Tokeniza, lematiza y elimina stopwords con spaCy.
        Solo conserva sustantivos, verbos, adjetivos y adverbios.
        """
        pos_permitidos = {"NOUN", "VERB", "ADJ", "ADV"}
        doc = self._nlp(text[:1_000_000])  # límite de seguridad spaCy
        tokens = []
        for token in doc:
            if (
                token.pos_ in pos_permitidos
                and not token.is_stop
                and not token.is_punct
                and not token.is_space
                and len(token.lemma_) > 2
                and token.lemma_.lower() not in STOPWORDS_EXTRA
            ):
                tokens.append(token.lemma_.lower())
        return " ".join(tokens)

    def preprocess_corpus(self) -> None:
        """Limpia y lematiza todos los documentos del corpus."""
        if not self.raw_texts:
            raise RuntimeError("Primero llama load_corpus().")

        print("[INFO] Preprocesando documentos…")
        self.processed_texts = []
        for i, text in enumerate(self.raw_texts, 1):
            clean = self._clean_text(text)
            lemmatized = self._lemmatize(clean)
            self.processed_texts.append(lemmatized)
            if i % 10 == 0 or i == len(self.raw_texts):
                print(f"  {i}/{len(self.raw_texts)} documentos procesados")

    # ------------------------------------------------------------------
    # Vectorización y modelado
    # ------------------------------------------------------------------

    def _build_vectorizer(self):
        """
        Crea el vectorizador adecuado según el método:
        - LDA  → CountVectorizer (frecuencias brutas)
        - NMF  → TfidfVectorizer (TF-IDF)
        """
        common = {
            "min_df": self.min_df,
            "max_df": self.max_df,
            "max_features": 5000,
            "ngram_range": (1, 2),
        }
        if self.method == "lda":
            return CountVectorizer(**common)
        return TfidfVectorizer(**common)

    def fit(self) -> None:
        """
        Pipeline completo: carga → preprocesa → vectoriza → entrena modelo.
        """
        self.load_corpus()
        self.preprocess_corpus()

        print(f"[INFO] Vectorizando ({self.method.upper()})…")
        self.vectorizer = self._build_vectorizer()
        self.dtm = self.vectorizer.fit_transform(self.processed_texts)

        print(
            f"[INFO] Entrenando modelo con {self.n_topics} temas "
            f"sobre una matriz {self.dtm.shape}…"
        )
        if self.method == "lda":
            self.model = LatentDirichletAllocation(
                n_components=self.n_topics,
                learning_method="batch",
                max_iter=30,
                random_state=self.random_state,
            )
        else:
            self.model = NMF(
                n_components=self.n_topics,
                init="nndsvda",
                max_iter=500,
                random_state=self.random_state,
            )

        self.doc_topic_matrix = self.model.fit_transform(self.dtm)
        print("[INFO] Entrenamiento completado.")

    # ------------------------------------------------------------------
    # Inspección de resultados
    # ------------------------------------------------------------------

    def print_topics(self) -> None:
        """Imprime las palabras más representativas de cada tema."""
        if self.model is None:
            raise RuntimeError("Primero llama fit().")

        feature_names = self.vectorizer.get_feature_names_out()
        print(f"\n{'='*60}")
        print(f"  TEMAS ({self.method.upper()}, n_topics={self.n_topics})")
        print(f"{'='*60}")
        for idx, topic in enumerate(self.model.components_):
            top_indices = topic.argsort()[: -self.n_words - 1 : -1]
            top_words = [feature_names[i] for i in top_indices]
            print(f"\nTema {idx + 1:>2}: {', '.join(top_words)}")

    def dominant_topic_per_doc(self) -> list[dict]:
        """
        Devuelve una lista de dicts con: ruta del documento,
        tema dominante (1-indexado) y su peso.
        """
        if self.doc_topic_matrix is None:
            raise RuntimeError("Primero llama fit().")

        results = []
        for i, (path, dist) in enumerate(
            zip(self.doc_paths, self.doc_topic_matrix)
        ):
            dominant = int(dist.argmax()) + 1
            weight = float(dist.max())
            results.append(
                {
                    "documento": path.name,
                    "ruta": str(path),
                    "tema_dominante": dominant,
                    "peso": round(weight, 4),
                }
            )
        return results

    def print_doc_topics(self, n: int = 20) -> None:
        """Imprime el tema dominante de los primeros *n* documentos."""
        rows = self.dominant_topic_per_doc()
        print(f"\n{'='*60}")
        print(f"  TEMA DOMINANTE POR DOCUMENTO (primeros {n})")
        print(f"{'='*60}")
        for row in rows[:n]:
            print(
                f"  [{row['tema_dominante']:>2}] (peso={row['peso']:.3f})  "
                f"{row['documento']}"
            )

    def save_doc_topics(self, output_path: str) -> None:
        """Guarda la asignación documento→tema en un CSV."""
        rows = self.dominant_topic_per_doc()
        out = Path(output_path)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["documento", "ruta", "tema_dominante", "peso"]
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"[INFO] Resultados guardados en: {out}")

    def topic_distribution_summary(self) -> None:
        """Muestra cuántos documentos tiene cada tema como dominante."""
        rows = self.dominant_topic_per_doc()
        counts: dict[int, int] = {}
        for row in rows:
            t = row["tema_dominante"]
            counts[t] = counts.get(t, 0) + 1

        print(f"\n{'='*60}")
        print("  DISTRIBUCIÓN DE DOCUMENTOS POR TEMA")
        print(f"{'='*60}")
        for t in sorted(counts):
            bar = "█" * counts[t]
            print(f"  Tema {t:>2}: {counts[t]:>3} docs  {bar}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Topic modeling sobre documentos históricos en español."
    )
    p.add_argument(
        "--corpus-dir",
        default=str(DEFAULT_CORPUS_DIR),
        help="Carpeta raíz con los .txt (default: Reconstrucciones_perplexity)",
    )
    p.add_argument(
        "--n-topics",
        type=int,
        default=10,
        help="Número de temas (default: 10)",
    )
    p.add_argument(
        "--method",
        choices=["lda", "nmf"],
        default="lda",
        help="Algoritmo: lda o nmf (default: lda)",
    )
    p.add_argument(
        "--n-words",
        type=int,
        default=15,
        help="Palabras por tema al imprimir (default: 15)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Ruta CSV para guardar doc→tema (opcional)",
    )
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    modeler = HistoricalTopicModeler(
        corpus_dir=args.corpus_dir,
        n_topics=args.n_topics,
        method=args.method,
        n_words=args.n_words,
    )

    modeler.fit()
    modeler.print_topics()
    modeler.topic_distribution_summary()
    modeler.print_doc_topics(n=30)

    if args.output:
        modeler.save_doc_topics(args.output)
