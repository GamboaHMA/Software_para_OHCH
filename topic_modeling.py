"""
Topic Modeling sobre documentos históricos en español.

Carga todos los .txt de un corpus y aplica LDA, NMF, Top2Vec o BERTopic.
LDA/NMF usan texto lematizado; Top2Vec/BERTopic usan el texto original.

Uso rápido:
    python topic_modeling.py

Opciones:
    --corpus-dir   Ruta a la carpeta raíz de los documentos (default: Reconstrucciones_qwen_deepseek)
    --n-topics     Número de temas para LDA/NMF (default: 10)
    --method       Algoritmo: lda | nmf | top2vec | bertopic (default: lda)
    --min-count    Frecuencia mínima para que Top2Vec incluya una palabra (default: 50)
    --embedding-model Encoder para Top2Vec (default: all-MiniLM-L6-v2)
    --min-topic-size Tamaño mínimo de clúster para BERTopic (default: 10)
    --min-document-words Mínimo de palabras alfabéticas por documento (default: 50)
    --n-words      Palabras por tema al imprimir resultados (default: 15)
    --output       Guarda una tabla doc→tema dominante en este archivo CSV
    --load-model   Carga un modelo guardado sin volver a entrenarlo
"""

import argparse
import csv
import glob
import pickle
import re
from pathlib import Path

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
DEFAULT_CORPUS_DIR = BASE_DIR / "Reconstrucciones_qwen_deepseek"
CACHE_DIR = BASE_DIR / ".cache"
MODELS_DIR = BASE_DIR / "Topic_models"
TOP2VEC_VALID_MODELS = {
    "distiluse-base-multilingual-cased",
    "all-MiniLM-L6-v2",
    "paraphrase-multilingual-MiniLM-L12-v2",
}
SENTENCE_TRANSFORMERS_PREFIX = "sentence-transformers/"
BERTOPIC_EMBEDDING_MODEL = (
    SENTENCE_TRANSFORMERS_PREFIX + "paraphrase-multilingual-MiniLM-L12-v2"
)
TOP2VEC_DEFAULT_EMBEDDING_MODEL = BERTOPIC_EMBEDDING_MODEL.split(
    SENTENCE_TRANSFORMERS_PREFIX, 1
)[1]
FIT_REQUIRED_MESSAGE = "Primero llama fit()."

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
    "sr", "sra", "don", "doña", "él", 
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
    method : {'lda', 'nmf', 'top2vec', 'bertopic'}
        Algoritmo de topic modeling a utilizar.
    n_words : int
        Palabras representativas por tema que se muestran al imprimir.
    min_count : int
        Frecuencia mínima de palabra para Top2Vec.
    embedding_model : str
        Encoder de frases/documentos para Top2Vec.
    min_topic_size : int
        Tamaño mínimo de clúster para BERTopic.
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
        min_count: int = 50,
        embedding_model: str = TOP2VEC_DEFAULT_EMBEDDING_MODEL,
        min_topic_size: int = 10,
        min_document_words: int = 50,
    ):
        self.corpus_dir = Path(corpus_dir)
        self.n_topics = n_topics
        self.method = method.lower()
        self.n_words = n_words
        self.min_df = min_df
        self.max_df = max_df
        self.random_state = random_state
        self.min_count = min_count
        self.embedding_model = embedding_model
        self.min_topic_size = min_topic_size
        self.min_document_words = min_document_words

        if self.method not in ("lda", "nmf", "top2vec", "bertopic"):
            raise ValueError("method debe ser 'lda', 'nmf', 'top2vec' o 'bertopic'")
        if self.n_topics < 1:
            raise ValueError("n_topics debe ser un entero mayor que cero")
        if self.min_count < 1:
            raise ValueError("min_count debe ser un entero mayor que cero")
        if self.min_topic_size < 2:
            raise ValueError("min_topic_size debe ser un entero mayor o igual a 2")
        if self.min_document_words < 0:
            raise ValueError("min_document_words no puede ser negativo")

        # Cargado al llamar fit()
        self.doc_paths: list[Path] = []
        self.raw_texts: list[str] = []
        self.processed_texts: list[str] = []
        self.vectorizer = None
        self.dtm = None          # document-term matrix
        self.model = None
        self.doc_topic_matrix = None
        self.topic_ids: list[int] = []
        self._top2vec_reduced = False

        # spaCy solo se carga al preprocesar LDA/NMF.
        self._nlp = None

    def _get_cache_path(self) -> Path:
        """Devuelve la ruta del archivo .pkl de caché para este corpus."""
        CACHE_DIR.mkdir(exist_ok=True)
        safe_name = self.corpus_dir.name.replace(" ", "_")
        return CACHE_DIR / f"{safe_name}_processed.pkl"

    def _build_fingerprint(self) -> list:
        """Huella del corpus: lista ordenada de (nombre, mtime) por archivo."""
        return sorted(
            (p.name, p.stat().st_mtime) for p in self.doc_paths
        )

    # ------------------------------------------------------------------
    # Carga de documentos
    # ------------------------------------------------------------------

    def load_corpus(self) -> int:
        """
        Carga los .txt recursivamente y descarta los documentos vacíos o más
        cortos que min_document_words. Devuelve el número de documentos válidos.
        """
        pattern = str(self.corpus_dir / "**" / "*.txt")
        paths = sorted(glob.glob(pattern, recursive=True))

        if not paths:
            raise FileNotFoundError(
                f"No se encontraron archivos .txt en: {self.corpus_dir}"
            )

        self.doc_paths = []
        self.raw_texts = []
        rejected = []

        for path_string in paths:
            path = Path(path_string)
            text = path.read_text(encoding="utf-8", errors="replace")
            word_count = len(re.findall(r"[^\W\d_]+", text, flags=re.UNICODE))
            if word_count < self.min_document_words:
                rejected.append((path, word_count))
                continue
            self.doc_paths.append(path)
            self.raw_texts.append(text)

        if rejected:
            print(
                f"[INFO] {len(rejected)} documentos descartados por tener menos "
                f"de {self.min_document_words} palabras alfabéticas "
                "(vacíos incluidos)."
            )
            for path, word_count in rejected[:5]:
                print(f"  - {path} ({word_count} palabras)")
            if len(rejected) > 5:
                print(f"  ... y {len(rejected) - 5} más")

        if not self.raw_texts:
            raise ValueError(
                "No quedan documentos después del filtrado. Reduce "
                "--min-document-words o revisa el corpus."
            )

        print(
            f"[INFO] {len(self.doc_paths)} documentos válidos cargados "
            f"(mínimo: {self.min_document_words} palabras)."
        )
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
        if self._nlp is None:
            import spacy

            print("[INFO] Cargando modelo de spaCy (es_core_news_sm)…")
            self._nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])

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
        """Limpia y lematiza todos los documentos del corpus.
        Si existe un caché válido (.pkl) lo carga en su lugar."""
        if not self.raw_texts:
            raise RuntimeError("Primero llama load_corpus().")

        cache_path = self._get_cache_path()
        fingerprint = self._build_fingerprint()

        # Intentar cargar desde caché
        if cache_path.exists():
            try:
                with cache_path.open("rb") as f:
                    cached = pickle.load(f)
                if cached.get("fingerprint") == fingerprint:
                    self.processed_texts = cached["processed_texts"]
                    print(
                        f"[INFO] Preprocesamiento cargado desde caché: {cache_path}"
                    )
                    return
                else:
                    print("[INFO] Caché desactualizado (corpus cambió), reprocesando…")
            except Exception as exc:
                print(f"[WARN] No se pudo leer el caché: {exc}. Reprocesando…")

        print("[INFO] Preprocesando documentos…")
        self.processed_texts = []
        for i, text in enumerate(self.raw_texts, 1):
            clean = self._clean_text(text)
            lemmatized = self._lemmatize(clean)
            self.processed_texts.append(lemmatized)
            if i % 10 == 0 or i == len(self.raw_texts):
                print(f"  {i}/{len(self.raw_texts)} documentos procesados")

        # Guardar caché
        try:
            with cache_path.open("wb") as f:
                pickle.dump(
                    {"fingerprint": fingerprint, "processed_texts": self.processed_texts},
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            print(f"[INFO] Preprocesamiento guardado en caché: {cache_path}")
        except Exception as exc:
            print(f"[WARN] No se pudo guardar el caché: {exc}")

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

    @staticmethod
    def _normalize_embedding_model(model_name: str, *, for_top2vec: bool = False) -> str:
        """Normaliza una cadena de embedding para la versión concreta de Top2Vec."""
        normalized = str(model_name).strip()
        if normalized.startswith(SENTENCE_TRANSFORMERS_PREFIX):
            normalized = normalized.split(SENTENCE_TRANSFORMERS_PREFIX, 1)[1]

        if for_top2vec and normalized not in TOP2VEC_VALID_MODELS:
            raise ValueError(
                "Top2Vec recibió un nombre de embedding no soportado: "
                f"{model_name!r}. Usa uno de: {sorted(TOP2VEC_VALID_MODELS)}"
            )
        return normalized

    def fit(self) -> None:
        """
        Entrena el método seleccionado. LDA/NMF usan textos lematizados y
        Top2Vec/BERTopic usan directamente raw_texts, sin vectorización previa.
        """
        self.load_corpus()
        if self.method in ("lda", "nmf"):
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
            if self.method == "nmf":
                row_sums = self.doc_topic_matrix.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                self.doc_topic_matrix = self.doc_topic_matrix / row_sums
                print("[INFO] Matriz de tópicos de NMF normalizada por filas.")
            self.topic_ids = list(range(self.doc_topic_matrix.shape[1]))

        else:
            texts = self.raw_texts
            empty_docs = [
                str(path) for path, text in zip(self.doc_paths, texts) if not text.strip()
            ]
            if empty_docs:
                raise ValueError(
                    "Top2Vec y BERTopic no admiten documentos vacíos. "
                    f"Ejemplos: {', '.join(empty_docs[:5])}"
                )

            if self.method == "top2vec":
                try:
                    from top2vec import Top2Vec
                except ImportError as exc:
                    raise ImportError(
                        "Para usar Top2Vec instala la dependencia: pip install top2vec"
                    ) from exc

                embedding_model = self._normalize_embedding_model(
                    self.embedding_model,
                    for_top2vec=True,
                )
                print(
                    "[INFO] Entrenando Top2Vec con los textos originales "
                    f"(embedding: {embedding_model})…"
                )
                self.model = Top2Vec(
                    documents=texts,
                    embedding_model=embedding_model,
                    min_count=self.min_count,
                )
                self.n_topics = self.model.get_num_topics()

                topic_words, _, topic_ids = self.model.get_topics(
                    reduced=self._top2vec_reduced
                )
                assignments = (
                    self.model.doc_top_reduced
                    if self._top2vec_reduced
                    else self.model.doc_top
                )
                scores = (
                    self.model.doc_dist_reduced
                    if self._top2vec_reduced
                    else self.model.doc_dist
                )
                self.topic_ids = [int(topic_id) for topic_id in topic_ids]
                self._topic_words = topic_words
                self.doc_topic_matrix = self._hard_topic_matrix(
                    assignments, scores, self.topic_ids
                )

            else:
                try:
                    from bertopic import BERTopic
                except ImportError as exc:
                    raise ImportError(
                        "Para usar BERTopic instala la dependencia: pip install bertopic"
                    ) from exc

                print("[INFO] Entrenando BERTopic con los textos originales…")
                self.model = BERTopic(
                    language="multilingual",
                    embedding_model=BERTOPIC_EMBEDDING_MODEL,
                    min_topic_size=self.min_topic_size,
                    top_n_words=self.n_words,
                    calculate_probabilities=False,
                    verbose=True,
                )
                _, probabilities = self.model.fit_transform(texts)
                assignments = self.model.topics_
                topic_info = self.model.get_topic_info()
                self.topic_ids = sorted(int(topic_id) for topic_id in topic_info["Topic"])
                actual_topics = [topic_id for topic_id in self.topic_ids if topic_id != -1]
                if actual_topics:
                    self.n_topics = len(actual_topics)
                self.doc_topic_matrix = self._hard_topic_matrix(
                    assignments, probabilities, self.topic_ids
                )

        print("[INFO] Entrenamiento completado.")

    @staticmethod
    def _hard_topic_matrix(assignments, scores, topic_ids: list[int]) -> np.ndarray:
        """Crea una matriz doc-tema dispersa a partir de asignaciones duras."""
        assignments = np.asarray(assignments)
        matrix = np.zeros((len(assignments), len(topic_ids)), dtype=float)
        topic_columns = {topic_id: column for column, topic_id in enumerate(topic_ids)}
        score_values = None if scores is None else np.asarray(scores)

        for row, topic_id in enumerate(assignments):
            topic_id = int(topic_id)
            if topic_id not in topic_columns:
                raise ValueError(f"El modelo asignó un tema no registrado: {topic_id}")
            if score_values is None:
                score = 1.0
            elif score_values.ndim == 1:
                score = float(score_values[row])
            else:
                score = float(score_values[row].max())
            matrix[row, topic_columns[topic_id]] = max(
                score, np.finfo(float).eps
            )
        return matrix

    # ------------------------------------------------------------------
    # Persistencia del modelo
    # ------------------------------------------------------------------

    def save_model(self, output_path: "str | Path | None" = None) -> Path:
        """
        Guarda el modelo y los metadatos necesarios para reutilizar resultados.

        Si output_path es None se usa:
            LDA/NMF: Topic_models/{n_topics}_topics_{method}.pkl
            Top2Vec: Topic_models/top2vec_min_count_{min_count}.model
            BERTopic: Topic_models/bertopic_min_topic_size_{min_topic_size}.pickle

        Top2Vec y BERTopic se serializan mediante sus métodos nativos. Un archivo
        lateral .metadata.pkl conserva rutas y matrices para las visualizaciones.
        """
        if self.model is None:
            raise RuntimeError(FIT_REQUIRED_MESSAGE)

        if output_path is None:
            if self.method == "top2vec":
                output_path = MODELS_DIR / (
                    f"top2vec_min_count_{self.min_count}.model"
                )
            elif self.method == "bertopic":
                output_path = MODELS_DIR / (
                    f"bertopic_min_topic_size_{self.min_topic_size}.pickle"
                )
            else:
                output_path = MODELS_DIR / f"{self.n_topics}_topics_{self.method}.pkl"

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "config": {
                "corpus_dir": str(self.corpus_dir),
                "n_topics": self.n_topics,
                "method": self.method,
                "n_words": self.n_words,
                "min_df": self.min_df,
                "max_df": self.max_df,
                "random_state": self.random_state,
                "min_count": self.min_count,
                "embedding_model": self.embedding_model,
                "min_topic_size": self.min_topic_size,
                "min_document_words": self.min_document_words,
            },
            "doc_paths": [str(p) for p in self.doc_paths],
            "raw_texts": self.raw_texts,
            "processed_texts": self.processed_texts,
            "vectorizer": self.vectorizer,
            "dtm": self.dtm,
            "model": self.model,
            "doc_topic_matrix": self.doc_topic_matrix,
            "topic_ids": self.topic_ids,
            "top2vec_reduced": self._top2vec_reduced,
        }

        if self.method in ("top2vec", "bertopic"):
            if self.method == "top2vec":
                self.model.save(str(out))
            else:
                self.model.save(str(out), serialization="pickle")

            metadata_path = Path(f"{out}.metadata.pkl")
            metadata = {key: value for key, value in state.items() if key != "model"}
            with metadata_path.open("wb") as f:
                pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[INFO] Modelo nativo guardado en: {out}")
            print(f"[INFO] Metadatos de visualización guardados en: {metadata_path}")
        else:
            with out.open("wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"[INFO] Modelo guardado en: {out}")
        return out

    @classmethod
    def load_model(cls, path: "str | Path") -> "HistoricalTopicModeler":
        """
        Carga un modelo guardado con save_model(). Los modelos Top2Vec/BERTopic
        se reabren con sus cargadores nativos y recuperan metadatos del sidecar.
        No requiere spaCy ni el corpus original en disco.

        Ejemplo::
            modeler = HistoricalTopicModeler.load_model("Topic_models/10_topics_lda.pkl")
            modeler.print_topics()
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {p}")

        print(f"[INFO] Cargando modelo desde: {p} …")
        metadata_path = Path(f"{p}.metadata.pkl")
        if metadata_path.exists():
            with metadata_path.open("rb") as f:
                state = pickle.load(f)
            method = state["config"]["method"]
            if method == "top2vec":
                try:
                    from top2vec import Top2Vec
                except ImportError as exc:
                    raise ImportError("Para cargar Top2Vec instala: pip install top2vec") from exc
                model = Top2Vec.load(str(p))
            elif method == "bertopic":
                try:
                    from bertopic import BERTopic
                except ImportError as exc:
                    raise ImportError("Para cargar BERTopic instala: pip install bertopic") from exc
                model = BERTopic.load(str(p))
            else:
                raise ValueError(f"Método no válido en metadatos nativos: {method}")
            state["model"] = model
        else:
            with p.open("rb") as f:
                state = pickle.load(f)

        cfg = state["config"]
        # Usamos __new__ para evitar cargar spaCy (no se necesita para inferencia)
        instance = cls.__new__(cls)
        instance.corpus_dir = Path(cfg["corpus_dir"])
        instance.n_topics = cfg["n_topics"]
        instance.method = cfg["method"]
        instance.n_words = cfg["n_words"]
        instance.min_df = cfg["min_df"]
        instance.max_df = cfg["max_df"]
        instance.random_state = cfg["random_state"]
        instance.min_count = cfg.get("min_count", state.get("min_count", 50))
        instance.embedding_model = cfg.get(
            "embedding_model", TOP2VEC_DEFAULT_EMBEDDING_MODEL
        )
        instance.min_topic_size = cfg.get(
            "min_topic_size", state.get("min_topic_size", 10)
        )
        instance.min_document_words = cfg.get("min_document_words", 50)
        instance.doc_paths = [Path(pp) for pp in state["doc_paths"]]
        instance.raw_texts = state["raw_texts"]
        instance.processed_texts = state["processed_texts"]
        instance.vectorizer = state["vectorizer"]
        instance.dtm = state["dtm"]
        instance.model = state["model"]
        instance.doc_topic_matrix = state["doc_topic_matrix"]
        instance.topic_ids = state.get(
            "topic_ids", list(range(instance.doc_topic_matrix.shape[1]))
        )
        instance._top2vec_reduced = state.get("top2vec_reduced", False)
        if "_topic_words" in state:
            instance._topic_words = state["_topic_words"]
        instance._nlp = None  # no se necesita spaCy al cargar un modelo ya entrenado

        print(
            f"[INFO] Modelo listo: {cfg['n_topics']} temas, "
            f"método {cfg['method'].upper()}, "
            f"{len(instance.doc_paths)} documentos."
        )
        return instance

    # ------------------------------------------------------------------
    # Inspección de resultados
    # ------------------------------------------------------------------

    def print_topics(self) -> None:
        """Imprime las palabras más representativas de cada tema."""
        if self.model is None:
            raise RuntimeError(FIT_REQUIRED_MESSAGE)

        print(f"\n{'='*60}")
        print(f"  TEMAS ({self.method.upper()}, n_topics={self.n_topics})")
        print(f"{'='*60}")
        if self.method in ("lda", "nmf"):
            feature_names = self.vectorizer.get_feature_names_out()
            for idx, topic in enumerate(self.model.components_):
                top_indices = topic.argsort()[: -self.n_words - 1 : -1]
                top_words = [feature_names[i] for i in top_indices]
                print(f"\nTema {idx + 1:>2}: {', '.join(top_words)}")
        elif self.method == "top2vec":
            topic_words, _, topic_ids = self.model.get_topics(
                reduced=self._top2vec_reduced
            )
            for idx, words in enumerate(topic_words):
                shown_words = [str(word) for word in words[: self.n_words]]
                print(
                    f"\nTema {idx + 1:>2} (ID Top2Vec {int(topic_ids[idx])}): "
                    f"{', '.join(shown_words)}"
                )
        else:
            for idx, topic_id in enumerate(self.topic_ids):
                words = self.model.get_topic(topic_id)
                if not words:
                    continue
                top_words = [word for word, _ in words[: self.n_words]]
                label = " (outliers)" if topic_id == -1 else f" (ID BERTopic {topic_id})"
                print(f"\nTema {idx + 1:>2}{label}: {', '.join(top_words)}")

    def dominant_topic_per_doc(self) -> list[dict]:
        """
        Devuelve una lista de dicts con: ruta del documento,
        tema dominante (1-indexado) y su peso.
        """
        if self.doc_topic_matrix is None:
            raise RuntimeError(FIT_REQUIRED_MESSAGE)

        results = []
        for path, dist in zip(self.doc_paths, self.doc_topic_matrix):
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
        help="Carpeta raíz con los .txt (default: Reconstrucciones_qwen_deepseek)",
    )
    p.add_argument(
        "--n-topics",
        type=int,
        default=10,
        help="Número de temas para LDA/NMF (default: 10). Top2Vec/BERTopic lo ignoran.",
    )
    p.add_argument(
        "--method",
        choices=["lda", "nmf", "top2vec", "bertopic"],
        default="lda",
        help="Algoritmo: lda, nmf, top2vec o bertopic (default: lda)",
    )
    p.add_argument(
        "--n-words",
        type=int,
        default=15,
        help="Palabras por tema al imprimir (default: 15)",
    )
    p.add_argument(
        "--min-count",
        type=int,
        default=50,
        metavar="N",
        help="Top2Vec: frecuencia mínima de palabra (default: 50).",
    )
    p.add_argument(
        "--embedding-model",
        choices=sorted(TOP2VEC_VALID_MODELS),
        default=TOP2VEC_DEFAULT_EMBEDDING_MODEL,
        help=(
            "Top2Vec: modelo de embeddings. El predeterminado conserva el "
            "mismo encoder multilingüe configurado para BERTopic (default: "
            "paraphrase-multilingual-MiniLM-L12-v2)."
        ),
    )
    p.add_argument(
        "--min-topic-size",
        type=int,
        default=10,
        metavar="N",
        help="BERTopic: tamaño mínimo de los clústeres (default: 10).",
    )
    p.add_argument(
        "--min-document-words",
        type=int,
        default=50,
        metavar="N",
        help="Descarta para todos los métodos documentos con menos de N palabras "
        "alfabéticas (default: 50). Use 0 para desactivar el filtro.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Ruta CSV para guardar doc→tema (opcional). Ignorado si se usa --desde.",
    )
    p.add_argument(
        "--load-model",
        default=None,
        metavar="RUTA",
        help="Carga un modelo guardado (nativo o .pkl) sin volver a entrenarlo.",
    )
    p.add_argument(
        "--desde",
        type=int,
        default=None,
        metavar="N",
        help="Inicio del rango de temas para experimentación (ej: 5).",
    )
    p.add_argument(
        "--hasta",
        type=int,
        default=None,
        metavar="N",
        help="Fin del rango de temas para experimentación, inclusivo (ej: 20).",
    )
    p.add_argument(
        "--unit",
        type=int,
        default=1,
        metavar="U",
        help="Paso del iterador de temas (default: 1). Ej: --desde 5 --hasta 20 --unit 5",
    )
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    if args.load_model:
        if args.desde is not None:
            build_arg_parser().error("--load-model no se combina con --desde/--hasta")
        modeler = HistoricalTopicModeler.load_model(args.load_model)
        modeler.print_topics()
        modeler.topic_distribution_summary()
        modeler.print_doc_topics(n=30)
        if args.output:
            modeler.save_doc_topics(args.output)
        raise SystemExit(0)

    # ── Modo experimentación: iterar sobre rango de n_topics ──────────────────
    if args.desde is not None:
        if args.hasta is None or args.hasta < args.desde:
            import sys
            print("[ERROR] --hasta debe ser >= --desde", file=sys.stderr)
            sys.exit(1)
        if args.method in ("top2vec", "bertopic"):
            build_arg_parser().error(
                "--desde/--hasta solo aplican a LDA/NMF; configure --min-count "
                "o --min-topic-size para Top2Vec/BERTopic"
            )
        if args.unit < 1:
            import sys
            print("[ERROR] --unit debe ser mayor que cero", file=sys.stderr)
            sys.exit(1)

        results_dir = Path("Topic_results")
        results_dir.mkdir(exist_ok=True)

        n_values = list(range(args.desde, args.hasta + 1, args.unit))
        print(
            f"[INFO] Modo experimentación: n_topics ∈ {n_values} "
            f"(desde={args.desde}, hasta={args.hasta}, unit={args.unit})"
        )

        for n in n_values:
            print(f"\n{'='*60}")
            print(f"  EXPERIMENTO: n_topics = {n}")
            print(f"{'='*60}")

            modeler = HistoricalTopicModeler(
                corpus_dir=args.corpus_dir,
                n_topics=n,
                method=args.method,
                n_words=args.n_words,
                min_count=args.min_count,
                embedding_model=args.embedding_model,
                min_topic_size=args.min_topic_size,
                min_document_words=args.min_document_words,
            )
            modeler.fit()
            modeler.print_topics()
            modeler.topic_distribution_summary()

            out_path = results_dir / f"{n}_topics_{args.method}_result.csv"
            modeler.save_doc_topics(str(out_path))
            modeler.save_model(MODELS_DIR / f"{n}_topics_{args.method}.pkl")

    # ── Modo normal: ejecución única con n_topics fijo ────────────────────────
    else:
        modeler = HistoricalTopicModeler(
            corpus_dir=args.corpus_dir,
            n_topics=args.n_topics,
            method=args.method,
            n_words=args.n_words,
            min_count=args.min_count,
            embedding_model=args.embedding_model,
            min_topic_size=args.min_topic_size,
            min_document_words=args.min_document_words,
        )

        modeler.fit()
        modeler.print_topics()
        modeler.topic_distribution_summary()
        modeler.print_doc_topics(n=30)
        modeler.save_model()

        if args.output:
            modeler.save_doc_topics(args.output)
