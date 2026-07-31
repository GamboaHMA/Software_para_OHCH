"""
topics_service.py
─────────────────
Extrae y formatea la información de los modelos de temas entrenados (.pkl)
para su visualización en la pestaña "Temas" de la aplicación.

Devuelve, para cada tema:
  - Etiqueta legible cargada desde Topic_results/<stem>_labels.json
  - Palabras más representativas (del vector de componentes del modelo)
  - Documentos que más pertenecen al tema (según la matriz doc-topic)
  - Flag de "documento multi-tema" cuando un documento tiene pesos altos
    en más de un tema a la vez

No duplica lógica: reutiliza load_topic_model() y _stem_to_doc_id()
de chat_service.py.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Ruta a Topic_results/ (mismo nivel que Topic_models/)
# ---------------------------------------------------------------------------
_HERE           = Path(os.path.dirname(os.path.abspath(__file__)))   # proyect/app/utils/
_WORKSPACE_ROOT = _HERE.parent.parent.parent                          # workspace root
RESULTS_ROOT    = _WORKSPACE_ROOT / "Topic_results"


def _load_labels(model_path: str) -> dict[int, str]:
    """
    Carga el archivo de etiquetas correspondiente al modelo pkl.

    Convención de nombres:
        pkl  →  Topic_models/lda/100_topics_lda.pkl
        json →  Topic_results/100_topics_lda_labels.json

    El JSON contiene una lista de objetos {topic_id (1-based), label, top_words}.
    Si el label comienza por "[ERROR" se trata como sin etiqueta y se devuelve
    una cadena vacía para ese tema.

    Devuelve
    --------
    dict[int, str]  topic_id (1-based) → label  (vacío si no disponible)
    """
    # Extraer el stem del pkl (ej. "100_topics_lda")
    stem = Path(model_path).stem          # nombre sin extensión

    label_file = RESULTS_ROOT / f"{stem}_labels.json"
    if not label_file.exists():
        return {}

    try:
        data = json.loads(label_file.read_text(encoding="utf-8"))
    except Exception:
        return {}

    labels = {}
    for item in data:
        tid   = int(item.get("topic_id", 0))
        label = str(item.get("label", "")).strip()
        # Descartar etiquetas que son mensajes de error de generación
        if label.startswith("[ERROR"):
            label = ""
        labels[tid] = label

    return labels


def get_topics(
    model_path: str,
    n_words: int = 15,
    top_n_docs: int = 10,
    min_weight: float = 0.20,
    multi_topic_ratio: float = 0.60,
) -> dict:
    """
    Carga un modelo pkl y extrae toda la información de temas para la UI.

    Parámetros
    ----------
    model_path : str
        Ruta relativa al workspace root del archivo .pkl.
    n_words : int
        Número de palabras más representativas por tema.
    top_n_docs : int
        Número máximo de documentos mostrados por tema.
    min_weight : float
        Peso mínimo para que un documento aparezca en la lista de un tema.
        Los documentos con peso < min_weight se omiten aunque sean los top.
    multi_topic_ratio : float
        Un documento se marca como "multi-tema" si tiene otro tema con peso
        >= multi_topic_ratio × max_weight.
        Ej: 0.60 → el segundo tema tiene al menos el 60 % del peso del primero.

    Devuelve
    --------
    dict con:
        config   – configuración del modelo {n_topics, method, n_words, ...}
        n_docs   – número de documentos en el corpus entrenado
        topics   – list[dict] con {idx, words, doc_count, top_docs}
            top_docs es list[dict] con:
                doc_id, weight, other_topics (list[{idx, weight}])
            (sin datos de DB; el enriquecimiento ocurre en la ruta Flask)

    Lanza
    -----
    FileNotFoundError / ValueError desde load_topic_model() si el path es inválido.
    """
    # Reutilizar la carga y caché del chat_service
    from app.utils.chat_service import load_topic_model, _stem_to_doc_id

    state = load_topic_model(model_path)

    # Cargar etiquetas legibles desde Topic_results/<stem>_labels.json
    labels = _load_labels(model_path)   # dict {topic_id_1based: label_str}

    feature_names   = list(state["vectorizer"].get_feature_names_out())
    doc_topic_mat   = np.array(state["doc_topic_matrix"])   # (n_docs, n_topics)
    doc_paths       = [str(p) for p in state["doc_paths"]]
    n_topics        = state["config"]["n_topics"]
    components      = state["model"].components_             # (n_topics, n_features)

    topics = []

    for topic_idx in range(n_topics):
        # ── Palabras más representativas ────────────────────────────────────
        # components[i] es el vector de pesos del tema i sobre el vocabulario.
        # Ordenamos descendente y tomamos los primeros n_words índices.
        comp_vec   = components[topic_idx]
        top_widx   = comp_vec.argsort()[:-n_words - 1: -1]
        top_words  = [feature_names[i] for i in top_widx]

        # ── Documentos más relevantes para este tema ─────────────────────────
        # Columna topic_idx de la matriz: peso de cada documento en este tema.
        col_weights = doc_topic_mat[:, topic_idx]

        # Número de documentos que superan el umbral min_weight
        doc_count = int((col_weights >= min_weight).sum())

        # Indices ordenados de mayor a menor peso para este tema
        sorted_idx = col_weights.argsort()[::-1]

        top_docs = []
        for doc_i in sorted_idx:
            if top_n_docs > 0 and len(top_docs) >= top_n_docs:
                break

            w = float(col_weights[doc_i])
            if w < min_weight:
                break   # el resto tienen aún menos peso

            doc_weights = doc_topic_mat[doc_i]   # distribución del doc sobre todos los temas

            # ── Detectar documentos multi-tema ────────────────────────────
            # Un documento es multi-tema si tiene otro tema con peso ≥ ratio × max.
            # Incluimos todos esos temas secundarios (salvo el tema actual) para
            # mostrarlos como etiquetas en la UI.
            max_w = float(doc_weights.max())
            threshold_w = multi_topic_ratio * max_w

            other_topics = [
                {"idx": int(t), "weight": round(float(doc_weights[t]), 4)}
                for t in np.where(doc_weights >= threshold_w)[0]
                if int(t) != topic_idx
            ]
            # Limitar a los 3 más fuertes para no saturar la UI
            other_topics.sort(key=lambda x: x["weight"], reverse=True)
            other_topics = other_topics[:3]

            top_docs.append({
                "doc_i":        int(doc_i),
                "doc_id":       _stem_to_doc_id(doc_paths[doc_i]),
                "weight":       round(w, 4),
                "other_topics": other_topics,   # [] si el doc pertenece solo a este tema
            })

        # Etiqueta legible (topic_id en el JSON es 1-based = topic_idx + 1)
        label = labels.get(topic_idx + 1, "")

        topics.append({
            "idx":       topic_idx,
            "label":     label,          # cadena vacía si no hay etiqueta disponible
            "words":     top_words,
            "doc_count": doc_count,
            "top_docs":  top_docs,
        })

    return {
        "config":  state["config"],
        "n_docs":  len(doc_paths),
        "topics":  topics,
    }
