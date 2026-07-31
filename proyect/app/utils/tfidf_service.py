"""
tfidf_service.py
────────────────
Servicio de recuperación de documentos por palabras clave usando TF-IDF.

Diferencias con chat_service.py (topic modeling)
──────────────────────────────────────────────────
- No depende de archivos .pkl entrenados externamente: el índice se construye
  directamente desde los documentos almacenados en la base de datos.
- Usa similitud léxica (coincidencia de términos) en lugar de similitud
  temática (distribución de tópicos). Esto lo hace más adecuado para búsquedas
  de palabras clave específicas, nombres propios, fechas, lugares, etc.
- El índice se guarda en memoria y se reconstruye automáticamente cuando cambia
  el número de documentos en la DB (o a petición explícita del usuario).
- Los umbrales de similitud útiles son más bajos (~5 – 20 %) porque la similitud
  coseno en TF-IDF es más dispersa que en el espacio de tópicos.

Flujo
─────
1. `build_index()` carga todos los documentos de la DB, vectoriza sus textos con
   TfidfVectorizer y guarda la matriz dispersa en memoria.
2. `search(query, threshold, max_results)` vectoriza la consulta con el mismo
   vectorizador y calcula similitud coseno contra la matriz.
3. `stream_summary()` llama a Ollama (streaming) para generar un resumen
   histórico basado en los fragmentos de los documentos recuperados.
"""

from __future__ import annotations

import os
import pickle as _pickle
import re
import threading
from pathlib import Path
from typing import Generator

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

# ---------------------------------------------------------------------------
# Ruta del caché en disco (workspace root, ignorado por git si se desea)
# ---------------------------------------------------------------------------
_HERE       = Path(os.path.dirname(os.path.abspath(__file__)))
_WS_ROOT    = _HERE.parent.parent.parent
_CACHE_FILE = _WS_ROOT / ".tfidf_index_cache.pkl"

# ---------------------------------------------------------------------------
# Configuraciones de los modelos Ollama (idénticas a chat_service)
# ---------------------------------------------------------------------------

OLLAMA_MODEL_CONFIGS: dict[str, dict] = {
    "salamandra": {
        "model":    "hdnh2006/salamandra-7b-instruct:latest",
        "options":  {
            "num_ctx":        4096,
            "temperature":    0.7,
            "top_p":          0.9,
            "repeat_penalty": 1.1,
            "num_gpu":        22,
        },
        "max_context_chars": 3_000,
    },
    "qwen": {
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

DEFAULT_OLLAMA_MODEL = "salamandra"

# ---------------------------------------------------------------------------
# Caché del índice TF-IDF (en memoria + en disco para sobrevivir reinicios)
# ---------------------------------------------------------------------------

_index: dict | None = None
_index_lock = threading.Lock()


def _load_disk_cache(doc_count: int) -> "dict | None":
    """
    Carga el índice desde el archivo de caché en disco si existe y el número
    de documentos coincide con el actual.  Devuelve None si no hay caché válido.
    """
    if not _CACHE_FILE.exists():
        return None
    try:
        with _CACHE_FILE.open("rb") as f:
            cached = _pickle.load(f)
        if cached.get("doc_count") == doc_count:
            print(
                f"[TF-IDF] Índice cargado desde caché ({doc_count} docs, "
                f"{cached['matrix'].shape[1]:,} términos).",
                flush=True,
            )
            return cached
        print("[TF-IDF] Caché obsoleto (número de documentos cambió), reconstruyendo…", flush=True)
    except Exception as exc:
        print(f"[TF-IDF] No se pudo leer el caché en disco: {exc}", flush=True)
    return None


def _save_disk_cache() -> None:
    """Serializa el índice actual a disco para persistir entre reinicios."""
    if _index is None:
        return
    try:
        with _CACHE_FILE.open("wb") as f:
            _pickle.dump(_index, f, protocol=_pickle.HIGHEST_PROTOCOL)
        print(f"[TF-IDF] Índice guardado en caché: {_CACHE_FILE}", flush=True)
    except Exception as exc:
        print(f"[TF-IDF] No se pudo guardar el caché en disco: {exc}", flush=True)


# ---------------------------------------------------------------------------
# Preprocesamiento de texto
# ---------------------------------------------------------------------------

# TfidfVectorizer solo acepta "english" como string; para español hay que
# proporcionar una lista explícita.
_SPANISH_STOP_WORDS = frozenset([
    "a", "al", "algo", "algunas", "algunos", "ante", "antes", "aquí", "aún",
    "aunque", "bajo", "bastante", "bien", "cada", "casi", "como", "con",
    "contra", "cual", "cuando", "de", "del", "desde", "donde", "durante",
    "él", "ella", "ellas", "ellos", "en", "entre", "era", "es", "esa",
    "esas", "ese", "esos", "esta", "estas", "este", "estos", "fue", "han",
    "has", "hasta", "hay", "he", "hecho", "hubo", "igual", "la", "las",
    "le", "les", "lo", "los", "más", "me", "menos", "mi", "mis", "mucho",
    "muchos", "muy", "no", "nos", "nuestro", "o", "otra", "otras", "otro",
    "otros", "para", "pero", "poco", "por", "porque", "que", "quien",
    "san", "se", "ser", "si", "sin", "so", "sobre", "su", "sus", "también",
    "tanto", "te", "tenía", "ti", "todo", "todos", "un", "una", "unas",
    "unos", "ya", "yo", "á", "é", "ó",
])

def _clean(text: str) -> str:
    """
    Limpieza liviana: minúsculas y eliminación de caracteres no alfabéticos
    (excepto acentos y ñ).  NO se lematiza: queremos coincidencia léxica exacta
    para que el usuario pueda buscar por palabras clave tal como las escribe.
    """
    text = text.lower()
    # Mantener letras con acento, ñ y espacios; eliminar todo lo demás
    text = re.sub(r"[^\w\sáéíóúüñ]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Construcción y acceso al índice
# ---------------------------------------------------------------------------

def build_index(force: bool = False) -> dict | None:
    """
    Construye (o devuelve del caché) el índice TF-IDF.

    Parámetros
    ----------
    force : bool
        Si True, reconstruye el índice aunque el número de documentos no haya
        cambiado. Útil para reflejar ediciones de texto sin que se añadan docs.

    Devuelve
    --------
    dict con las claves del índice, o None si la DB no tiene documentos con texto.
    """
    global _index

    # Importaciones diferidas para no romper el módulo fuera del contexto Flask
    from app.models import Document

    with _index_lock:
        doc_count = Document.query.count()

        # 1. Caché en memoria (más rápido, sin I/O)
        if (
            not force
            and _index is not None
            and _index["doc_count"] == doc_count
        ):
            return _index

        # 2. Caché en disco (sobrevive reinicios del proceso)
        if not force:
            cached = _load_disk_cache(doc_count)
            if cached is not None:
                _index = cached
                return _index

        # 3. Construir el índice desde la DB
        docs = (
            Document.query
            .filter(
                (Document.ocr_text_edited.isnot(None) & (Document.ocr_text_edited != ""))
                | (Document.ocr_text.isnot(None) & (Document.ocr_text != ""))
            )
            .all()
        )

        if not docs:
            _index = None
            return None

        # Extraer textos (display_text prefiere la versión editada)
        texts     = [_clean(d.display_text) for d in docs]
        doc_ids   = [d.doc_id  for d in docs]
        db_ids    = [d.id      for d in docs]
        titles    = [d.title   for d in docs]
        date_raws = [d.date_raw or "" for d in docs]
        raw_texts = [d.display_text   for d in docs]  # texto original sin limpiar

        # Construir vectorizador TF-IDF
        # sublinear_tf=True aplica log(1+tf) para reducir el peso de términos
        # muy frecuentes (equivalente a BM25 suavizado).
        # ngram_range=(1,2) permite buscar bigramas (ej. "guerra carlista").
        vectorizer = TfidfVectorizer(
            stop_words=list(_SPANISH_STOP_WORDS),
            min_df=1,
            max_df=0.95,
            ngram_range=(1, 2),
            max_features=80_000,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(texts)

        _index = {
            "vectorizer": vectorizer,
            "matrix":     matrix,
            "doc_ids":    doc_ids,
            "db_ids":     db_ids,
            "titles":     titles,
            "date_raws":  date_raws,
            "raw_texts":  raw_texts,
            "doc_count":  doc_count,
        }

        # Guardar en disco para persistir entre reinicios
        _save_disk_cache()

        return _index


def index_stats() -> dict:
    """
    Devuelve estadísticas del índice actual (sin reconstruirlo).
    Si no hay índice, devuelve doc_count=0.
    """
    from app.models import Document
    total_in_db = Document.query.count()

    if _index is None:
        return {"indexed": 0, "total_in_db": total_in_db, "ready": False}

    return {
        "indexed":      _index["doc_count"],
        "total_in_db":  total_in_db,
        "vocabulary":   _index["matrix"].shape[1],
        "ready":        True,
    }


# ---------------------------------------------------------------------------
# Búsqueda
# ---------------------------------------------------------------------------

def search(
    query: str,
    threshold: float = 0.05,
    max_results: int = 50,
) -> list[dict]:
    """
    Busca documentos por palabras clave usando similitud coseno TF-IDF.

    Parámetros
    ----------
    query : str
        Palabras clave o frase del usuario.
    threshold : float
        Similitud coseno mínima [0, 1]. Default 0.05 (5 %).
        Los valores útiles para TF-IDF suelen estar entre 0.03 y 0.30.
    max_results : int
        Número máximo de resultados a devolver.

    Devuelve
    --------
    list[dict] ordenada de mayor a menor similitud, con claves:
        doc_id, db_id, title, date_raw, similarity, in_db
    """
    idx = build_index()
    if not idx:
        return []

    # Vectorizar la consulta con el mismo vocabulario del índice
    query_clean = _clean(query)
    query_vec   = idx["vectorizer"].transform([query_clean])

    # Similitud coseno entre la consulta y todos los documentos
    # sk_cosine devuelve shape (1, n_docs); [0] lo aplana a 1D
    scores = sk_cosine(query_vec, idx["matrix"])[0]

    results = []
    for i, score in enumerate(scores):
        if score >= threshold:
            results.append({
                "doc_id":     idx["doc_ids"][i],
                "db_id":      idx["db_ids"][i],
                "title":      idx["titles"][i],
                "date_raw":   idx["date_raws"][i],
                "similarity": round(float(score), 4),
                "in_db":      True,
            })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:max_results]


# ---------------------------------------------------------------------------
# Resumen con Ollama (streaming) — usa textos de la DB, no de pkl
# ---------------------------------------------------------------------------

_SUMMARY_SYSTEM = (
    "Eres un asistente especializado en historia. Se te proporciona una consulta "
    "y fragmentos de documentos del corpus histórico. Tu tarea es redactar una "
    "respuesta académica, concisa y bien estructurada basándote ÚNICAMENTE en los "
    "documentos proporcionados. Cita el identificador del documento cuando lo uses. "
    "Si la información no es suficiente para responder, indícalo con claridad."
)

_SUMMARY_USER_TEMPLATE = (
    "CONSULTA:\n{query}\n\n"
    "DOCUMENTOS RELEVANTES:\n{context}\n\n"
    "Redacta una respuesta histórica fundamentada en los documentos anteriores."
)


def build_context(results: list[dict], max_chars: int = 4_000) -> str:
    """
    Construye el bloque de texto de contexto para Ollama a partir de los
    documentos recuperados por TF-IDF.  Usa el texto completo de la DB (no el
    procesado) para que Ollama vea el contenido real del documento.
    """
    idx = _index  # ya cargado al llamar search()
    if not idx:
        return ""

    context_parts = []
    total_chars   = 0

    for item in results:
        # Obtener el texto por db_id
        try:
            pos = idx["db_ids"].index(item["db_id"])
            doc_text = idx["raw_texts"][pos]
        except (ValueError, IndexError):
            doc_text = ""

        if not doc_text:
            continue

        snippet = doc_text[:800].strip()
        block   = f"[{item['doc_id']}] (sim: {item['similarity']:.2f})\n{snippet}"

        if total_chars + len(block) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 100:
                context_parts.append(block[:remaining] + "…")
            break

        context_parts.append(block)
        total_chars += len(block)

    return "\n\n---\n\n".join(context_parts)


def stream_summary(
    query: str,
    results: list[dict],
    ollama_alias: str = DEFAULT_OLLAMA_MODEL,
) -> Generator[str, None, None]:
    """
    Genera el resumen histórico vía Ollama con streaming.

    Parámetros
    ----------
    query        : Consulta original del usuario.
    results      : Lista de documentos relevantes (salida de search()).
    ollama_alias : 'salamandra' o 'qwen'.

    Devuelve
    --------
    Generador de chunks de texto para enviar como SSE.
    """
    from ollama import chat as ollama_chat

    cfg     = OLLAMA_MODEL_CONFIGS.get(ollama_alias, OLLAMA_MODEL_CONFIGS[DEFAULT_OLLAMA_MODEL])
    context = build_context(results, max_chars=cfg["max_context_chars"])

    if not context:
        yield "[No se encontraron fragmentos de texto para los documentos relevantes.]"
        return

    messages = [
        {"role": "system", "content": _SUMMARY_SYSTEM},
        {"role": "user",   "content": _SUMMARY_USER_TEMPLATE.format(
            query=query, context=context
        )},
    ]

    stream = ollama_chat(
        model=cfg["model"],
        messages=messages,
        options=cfg["options"],
        stream=True,
    )

    for chunk in stream:
        delta = chunk.message.content
        if delta:
            yield delta
