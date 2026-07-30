"""
chat_service.py
───────────────
Servicio de búsqueda semántica y resumen para el módulo Chat del
Corpus Documental.

Flujo principal
───────────────
1. El usuario envía una consulta histórica en lenguaje natural.
2. Se preprocesa con spaCy (misma pipeline que topic_modeling.py).
3. Se proyecta sobre el modelo LDA/NMF entrenado (archivo .pkl) para obtener
   un vector de distribución de temas (query_topic_vector).
4. Se calcula la similitud coseno entre ese vector y los vectores de todos
   los documentos del corpus entrenado.
5. Se devuelven los documentos cuya similitud supere el umbral indicado.
6. Opcionalmente, se llama a Ollama para generar un resumen histórico
   basado en los documentos más relevantes encontrados (modo streaming SSE).

Compatibilidad con el schema de la DB
──────────────────────────────────────
Los archivos .pkl almacenan doc_paths de la forma:
    .../Reconstrucciones_qwen_deepseek/Articulos_ohcbh_cf_erlXXX/ohcbh_cf_erlXXX_NNN.pdf.txt

El doc_id de la base de datos es la parte del nombre sin ".pdf.txt", es decir:
    ohcbh_cf_erlXXX_NNN

La función `_stem_to_doc_id` extrae ese identificador del path del pkl.
"""

from __future__ import annotations

import os
import pickle
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Generator

import numpy as np

# ---------------------------------------------------------------------------
# Constantes y rutas
# ---------------------------------------------------------------------------

# Raíz del workspace (tres niveles arriba de este archivo:
#   proyect/app/utils/chat_service.py  →  workspace root)
_HERE = Path(os.path.dirname(os.path.abspath(__file__)))
_WORKSPACE_ROOT = _HERE.parent.parent.parent   # .../Soft_para_doc_historica/

# Carpeta que contiene los modelos entrenados (.pkl)
MODELS_ROOT = _WORKSPACE_ROOT / "Topic_models"

# Configuraciones de los modelos Ollama disponibles para el chat
OLLAMA_MODEL_CONFIGS: dict[str, dict] = {
    "salamandra": {
        # Salamandra-7B: ventana de 4096 tokens, inferencia parcial en GPU
        "model":    "hdnh2006/salamandra-7b-instruct:latest",
        "options":  {
            "num_ctx":        4096,
            "temperature":    0.7,   # temperatura más alta para respuestas más fluidas
            "top_p":          0.9,
            "repeat_penalty": 1.1,
            "num_gpu":        22,    # capas en VRAM (GTX 1050 4 GB)
        },
        # Máximo de caracteres de contexto documental enviado al modelo
        "max_context_chars": 3_000,
    },
    "qwen": {
        # Qwen 2.5 3B: ventana de 8192 tokens, cabe íntegro en GPU
        "model":    "qwen2.5:3b",
        "options":  {
            "num_ctx":        8192,
            "temperature":    0.7,
            "top_p":          0.9,
            "repeat_penalty": 1.1,
            "num_gpu":        99,
        },
        "max_context_chars": 6_000,
    },
}

# Por defecto se usa Salamandra
DEFAULT_OLLAMA_MODEL = "salamandra"

# ---------------------------------------------------------------------------
# Caché de modelos pkl (en memoria, keyed por path absoluto)
# Para no recargar el pkl en cada petición se guarda en un dict protegido
# por un lock (puede haber peticiones concurrentes desde distintos hilos Flask).
# ---------------------------------------------------------------------------
_model_cache: dict[str, dict] = {}
_model_cache_lock = threading.Lock()


def list_available_models() -> list[dict]:
    """
    Devuelve una lista de todos los .pkl disponibles en Topic_models/,
    organizados con información de nombre, método, número de temas y ruta.

    Retorna
    -------
    list[dict] con claves: name, method, n_topics, path (relativo a workspace)
    """
    results = []
    if not MODELS_ROOT.exists():
        return results

    # Recorre subcarpetas (lda/, nmf/) y la raíz por compatibilidad
    for pkl_file in sorted(MODELS_ROOT.rglob("*.pkl")):
        stem = pkl_file.stem                       # ej. "100_topics_lda"
        # Intentar extraer n_topics y method del nombre de archivo
        m = re.match(r"(\d+)_topics_(lda|nmf)", stem)
        if m:
            n_topics = int(m.group(1))
            method   = m.group(2).upper()
        else:
            n_topics = None
            method   = "?"

        results.append({
            "name":     stem,
            "method":   method,
            "n_topics": n_topics,
            "path":     str(pkl_file.relative_to(_WORKSPACE_ROOT)),
        })

    return results


def load_topic_model(relative_path: str) -> dict:
    """
    Carga y cachea el modelo pkl indicado.

    Parámetros
    ----------
    relative_path : str
        Ruta relativa al workspace root, ej. "Topic_models/lda/100_topics_lda.pkl"

    Retorna
    -------
    dict con las claves del pkl: vectorizer, model, doc_paths, doc_topic_matrix,
    raw_texts, processed_texts, config.

    Lanza
    -----
    FileNotFoundError si el pkl no existe.
    ValueError si la ruta sale fuera del directorio de modelos (seguridad).
    """
    abs_path = (_WORKSPACE_ROOT / relative_path).resolve()

    # Validación de seguridad: el path debe estar dentro de MODELS_ROOT
    try:
        abs_path.relative_to(MODELS_ROOT.resolve())
    except ValueError:
        raise ValueError(f"Ruta de modelo no permitida: {relative_path}")

    if not abs_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {abs_path}")

    key = str(abs_path)

    with _model_cache_lock:
        if key not in _model_cache:
            with abs_path.open("rb") as f:
                state = pickle.load(f)
            _model_cache[key] = state

    return _model_cache[key]


# ---------------------------------------------------------------------------
# Preprocesamiento de texto (igual que topic_modeling.py)
# ---------------------------------------------------------------------------

# Stopwords extra (copiadas de topic_modeling.py para coherencia)
_STOPWORDS_EXTRA = {
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


@lru_cache(maxsize=1)
def _get_nlp():
    """Carga el modelo de spaCy una sola vez (caché global del proceso)."""
    import spacy
    return spacy.load("es_core_news_sm", disable=["parser", "ner"])


def _clean_text(text: str) -> str:
    """Limpieza básica idéntica a topic_modeling.py."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\sáéíóúüñÁÉÍÓÚÜÑ]", " ", text)
    return text.strip()


def _lemmatize(text: str) -> str:
    """
    Lematiza con spaCy usando los mismos filtros de POS y stopwords que
    se usaron al entrenar los modelos de temas.
    """
    nlp = _get_nlp()
    pos_permitidos = {"NOUN", "VERB", "ADJ", "ADV"}
    doc = nlp(text[:1_000_000])
    tokens = []
    for token in doc:
        if (
            token.pos_ in pos_permitidos
            and not token.is_stop
            and not token.is_punct
            and not token.is_space
            and len(token.lemma_) > 2
            and token.lemma_.lower() not in _STOPWORDS_EXTRA
        ):
            tokens.append(token.lemma_.lower())
    return " ".join(tokens)


def preprocess_query(query: str) -> str:
    """
    Aplica la misma pipeline de limpieza y lematización que se usó al
    entrenar, para que la consulta sea comparable con el vocabulario del
    modelo.
    """
    return _lemmatize(_clean_text(query))


# ---------------------------------------------------------------------------
# Proyección de la consulta en el espacio de temas
# ---------------------------------------------------------------------------

def query_to_topic_vector(query: str, state: dict) -> np.ndarray:
    """
    Convierte la consulta del usuario en un vector de distribución de temas
    usando el vectorizador y el modelo entrenados.

    Parámetros
    ----------
    query : str
        Pregunta histórica en lenguaje natural.
    state : dict
        Estado del modelo pkl cargado con load_topic_model().

    Retorna
    -------
    np.ndarray de shape (n_topics,) con la distribución normalizada de temas.
    Puede ser un vector de ceros si la consulta no contiene términos del vocabulario.
    """
    processed = preprocess_query(query)
    vectorizer = state["vectorizer"]
    model      = state["model"]

    # Proyectar sobre el vocabulario del vectorizador entrenado
    query_dtm = vectorizer.transform([processed])

    # Obtener distribución de temas
    topic_vector = model.transform(query_dtm)[0]   # shape: (n_topics,)

    # Normalizar a suma 1 (igual que en NMF normalizado; LDA ya suma 1 internamente)
    total = topic_vector.sum()
    if total > 0:
        topic_vector = topic_vector / total

    return topic_vector


# ---------------------------------------------------------------------------
# Cálculo de similitud y recuperación de documentos
# ---------------------------------------------------------------------------

def cosine_similarity_1d(a: np.ndarray, b_matrix: np.ndarray) -> np.ndarray:
    """
    Calcula la similitud coseno entre el vector a (1D) y cada fila de b_matrix.
    Devuelve un array 1D de similitudes en [0, 1].
    """
    norm_a = np.linalg.norm(a)
    if norm_a == 0:
        return np.zeros(len(b_matrix))

    norms_b = np.linalg.norm(b_matrix, axis=1)
    # Evitar división por cero en filas nulas
    norms_b = np.where(norms_b == 0, 1e-10, norms_b)

    return (b_matrix @ a) / (norms_b * norm_a)


def _stem_to_doc_id(doc_path_str: str) -> str:
    """
    Extrae el doc_id de la base de datos a partir del path almacenado en el pkl.

    Ejemplos de paths que puede contener el pkl:
        .../Reconstrucciones_qwen_deepseek/Articulos_ohcbh_cf_erl1/ohcbh_cf_erl1_001.pdf.txt
        .../Reconstrucciones_perplexity/ohcbh_cf_erl1/ohcbh_cf_erl1_001.txt
        .../ohcbh_cf_erl1_001.txt

    El doc_id de la DB es el stem del archivo sin extensiones OCR:
        ohcbh_cf_erl1_001
    """
    name = Path(doc_path_str).name          # ej. "ohcbh_cf_erl1_001.pdf.txt"
    # Quitar extensiones conocidas de texto OCR
    for suffix in (".pdf.txt", ".txt"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def find_relevant_documents(
    query: str,
    state: dict,
    threshold: float = 0.30,
    max_results: int = 50,
) -> list[dict]:
    """
    Busca los documentos más similares a la consulta dada.

    Parámetros
    ----------
    query : str
        Consulta histórica del usuario.
    state : dict
        Estado pkl cargado con load_topic_model().
    threshold : float
        Umbral de similitud coseno mínimo [0, 1]. Default 0.30 (30 %).
    max_results : int
        Número máximo de documentos a devolver.

    Retorna
    -------
    list[dict] ordenada de mayor a menor similitud, cada elemento con:
        doc_id      – identificador en la base de datos
        similarity  – similitud coseno redondeada a 4 decimales
        topic_idx   – índice del tema dominante del documento (0-based)
        path        – path original del archivo en el pkl
    """
    query_vector   = query_to_topic_vector(query, state)
    doc_topic_mat  = np.array(state["doc_topic_matrix"])   # (n_docs, n_topics)
    doc_paths      = state["doc_paths"]

    similarities = cosine_similarity_1d(query_vector, doc_topic_mat)

    results = []
    for i, (sim, path_str) in enumerate(zip(similarities, doc_paths)):
        if sim >= threshold:
            results.append({
                "doc_id":    _stem_to_doc_id(str(path_str)),
                "similarity": round(float(sim), 4),
                "topic_idx":  int(np.argmax(doc_topic_mat[i])),
                "path":       str(path_str),
            })

    # Ordenar de mayor a menor similitud
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:max_results]


# ---------------------------------------------------------------------------
# Generación de resumen con Ollama (modo streaming)
# ---------------------------------------------------------------------------

_SUMMARY_SYSTEM = (
    "Eres un asistente especializado en historia. Se te proporciona una pregunta "
    "histórica y fragmentos de documentos del corpus. Tu tarea es redactar una "
    "respuesta académica, concisa y bien estructurada basándote ÚNICAMENTE en los "
    "documentos proporcionados. Cita el identificador del documento cuando lo uses. "
    "Si la información no es suficiente para responder, indícalo con claridad."
)

_SUMMARY_USER_TEMPLATE = (
    "PREGUNTA HISTÓRICA:\n{query}\n\n"
    "DOCUMENTOS RELEVANTES:\n{context}\n\n"
    "Redacta una respuesta histórica fundamentada en los documentos anteriores."
)


def build_context_for_summary(
    results: list[dict],
    state: dict,
    max_chars: int = 4_000,
) -> str:
    """
    Construye el bloque de texto de contexto que se enviará a Ollama para
    el resumen, concatenando fragmentos de los documentos más relevantes.

    Se limita a max_chars para no superar la ventana de contexto del modelo.
    """
    raw_texts = state.get("raw_texts", [])
    doc_paths = [str(p) for p in state["doc_paths"]]

    context_parts = []
    total_chars   = 0

    for item in results:
        # Buscar el texto del documento en el pkl por path
        try:
            idx = doc_paths.index(item["path"])
            doc_text = raw_texts[idx] if idx < len(raw_texts) else ""
        except (ValueError, IndexError):
            doc_text = ""

        if not doc_text:
            continue

        # Tomar solo el inicio del documento para no saturar el contexto
        snippet = doc_text[:800].strip()
        block   = f"[{item['doc_id']}] (similitud: {item['similarity']:.2f})\n{snippet}"

        # Respetar el límite de caracteres
        if total_chars + len(block) > max_chars:
            # Truncar el bloque para llegar justo al límite
            remaining = max_chars - total_chars
            if remaining > 100:  # solo agregar si queda espacio significativo
                context_parts.append(block[:remaining] + "…")
            break

        context_parts.append(block)
        total_chars += len(block)

    return "\n\n---\n\n".join(context_parts)


def stream_summary(
    query: str,
    results: list[dict],
    state: dict,
    ollama_alias: str = DEFAULT_OLLAMA_MODEL,
) -> Generator[str, None, None]:
    """
    Genera el resumen histórico vía Ollama con streaming.

    Devuelve un generador de chunks de texto (strings) que el endpoint Flask
    puede enviar como SSE o transferencia chunked.

    Parámetros
    ----------
    query        : Consulta original del usuario.
    results      : Lista de documentos relevantes (salida de find_relevant_documents).
    state        : Estado pkl del modelo.
    ollama_alias : 'salamandra' o 'qwen'.
    """
    from ollama import chat as ollama_chat

    cfg      = OLLAMA_MODEL_CONFIGS.get(ollama_alias, OLLAMA_MODEL_CONFIGS[DEFAULT_OLLAMA_MODEL])
    context  = build_context_for_summary(results, state, max_chars=cfg["max_context_chars"])

    if not context:
        yield "[No se encontraron fragmentos de texto para los documentos relevantes.]"
        return

    messages = [
        {"role": "system", "content": _SUMMARY_SYSTEM},
        {"role": "user",   "content": _SUMMARY_USER_TEMPLATE.format(
            query=query, context=context
        )},
    ]

    # Llamada con stream=True a Ollama
    stream = ollama_chat(
        model=cfg["model"],
        messages=messages,
        options=cfg["options"],
        stream=True,
    )

    for chunk in stream:
        # Cada chunk tiene message.content con el fragmento de texto generado
        delta = chunk.message.content
        if delta:
            yield delta
