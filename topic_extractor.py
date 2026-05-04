"""
Etiquetado automático de temas con Perplexity AI (interfaz web).

Toma la salida de HistoricalTopicModeler y genera una etiqueta semántica
por tema interactuando con perplexity.ai vía Selenium/undetected-chromedriver,
igual que procesar_todos_perplexity.py.  No requiere API key.

Dos variantes:
  top_words  — input: las palabras más relevantes del tema (LDA/NMF)
  top_docs   — input: fragmentos de los documentos con mayor peso en el tema
  both       — corre ambas y devuelve las dos etiquetas para comparar

Uso rápido:
    python topic_extractor.py

Opciones:
    --corpus-dir     Ruta a la carpeta raíz de los documentos
    --n-topics       Número de temas (default: 10)
    --method         lda | nmf (default: lda)
    --label-method   top_words | top_docs | both (default: top_words)
    --n-words        Palabras top por tema (default: 15)
    --top-n-docs     Docs representativos para variante top_docs (default: 3)
    --fragment-chars Caracteres máx. de fragmento por doc (default: 800)
    --delay          Segundos entre consultas (default: 2.0)
    --output         Ruta JSON de salida (opcional)
"""

import argparse
import json
import time
from pathlib import Path

from topic_modeling import DEFAULT_CORPUS_DIR, HistoricalTopicModeler
from procesar_todos_perplexity import PerplexityBot

# Fragmento máximo (caracteres) por documento en la variante top_docs
MAX_FRAGMENT_CHARS = 800

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Variante 1: top-words
# Se le pasan las N palabras más características del tema y se pide una
# etiqueta concisa. Suficiente para la gran mayoría de casos y muy barato
# en tokens.
PROMPT_TOP_WORDS = """\
Eres un especialista en historia hispanohablante (siglos XVI–XX) \
y en análisis de corpus documentales.

Se te proporcionan las palabras clave más representativas de un tema \
latente descubierto mediante análisis estadístico (LDA/NMF) sobre un corpus \
de documentos históricos en español.

Palabras clave del tema:
{words}

Tu tarea: genera una etiqueta temática concisa (2–5 palabras) en español \
que describa con precisión el asunto central que estas palabras representan, \
considerando el contexto histórico hispanohablante.

Responde ÚNICAMENTE con la etiqueta, sin puntuación final ni explicación adicional.\
"""

# Variante 2: top-docs (fragmentos de documentos representativos)
# Se toman los K documentos con mayor peso en el tema, se extrae su
# fragmento inicial y se pide al modelo que infiera el tema. Produce
# etiquetas más precisas para corpus especializados o temáticamente ambiguos.
PROMPT_TOP_DOCS = """\
Eres un especialista en historia hispanohablante (siglos XVI–XX) \
y en análisis de corpus documentales.

Los siguientes fragmentos provienen de los documentos más representativos \
de un tema latente descubierto mediante análisis estadístico (LDA/NMF) \
sobre un corpus de documentos históricos en español.

{fragments}

Tu tarea: a partir de los fragmentos anteriores, genera una etiqueta temática \
concisa (2–5 palabras) en español que describa el asunto principal \
que estos documentos comparten, considerando el contexto histórico hispanohablante.

Responde ÚNICAMENTE con la etiqueta, sin puntuación final ni explicación adicional.\
"""


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class TopicLabeler:
    """
    Genera etiquetas semánticas para los temas descubiertos por
    HistoricalTopicModeler interactuando con perplexity.ai vía Selenium.

    Parámetros
    ----------
    modeler : HistoricalTopicModeler
        Instancia ya entrenada (fit() llamado).
    label_method : {'top_words', 'top_docs'}
        Variante usada por label_all_topics(). Para correr ambas usa
        label_all_both() directamente.
    n_words : int
        Palabras top del tema para la variante top_words.
    top_n_docs : int
        Documentos top para la variante top_docs.
    fragment_chars : int
        Caracteres máximos de fragmento por documento.
    delay : float
        Segundos de espera entre consultas (evita rate-limit).
    """

    def __init__(
        self,
        modeler: HistoricalTopicModeler,
        label_method: str = "top_words",
        n_words: int = 15,
        top_n_docs: int = 3,
        fragment_chars: int = MAX_FRAGMENT_CHARS,
        delay: float = 2.0,
    ):
        if modeler.model is None:
            raise RuntimeError(
                "El modeler debe tener fit() ejecutado antes de crear TopicLabeler."
            )
        if label_method not in ("top_words", "top_docs"):
            raise ValueError("label_method debe ser 'top_words' o 'top_docs'")

        self.modeler = modeler
        self.label_method = label_method
        self.n_words = n_words
        self.top_n_docs = top_n_docs
        self.fragment_chars = fragment_chars
        self.delay = delay

        self._bot: PerplexityBot | None = None

    # ------------------------------------------------------------------
    # Helpers de extracción
    # ------------------------------------------------------------------

    def _get_top_words(self, topic_idx: int) -> list[str]:
        """Devuelve las top-N palabras del tema topic_idx."""
        feature_names = self.modeler.vectorizer.get_feature_names_out()
        topic = self.modeler.model.components_[topic_idx]
        top_indices = topic.argsort()[: -self.n_words - 1 : -1]
        return [feature_names[i] for i in top_indices]

    def _get_top_doc_fragments(self, topic_idx: int) -> list[tuple[str, float]]:
        """
        Devuelve los (fragmento, peso) de los top_n_docs documentos con
        mayor peso en el tema topic_idx.
        """
        col = self.modeler.doc_topic_matrix[:, topic_idx]
        top_doc_indices = col.argsort()[::-1][: self.top_n_docs]
        fragments = []
        for doc_idx in top_doc_indices:
            weight = float(col[doc_idx])
            text = self.modeler.raw_texts[doc_idx]
            fragment = text[: self.fragment_chars].strip()
            fragments.append((fragment, weight))
        return fragments

    # ------------------------------------------------------------------
    # Construcción de prompts
    # ------------------------------------------------------------------

    def _build_prompt_top_words(self, topic_idx: int) -> str:
        words = self._get_top_words(topic_idx)
        return PROMPT_TOP_WORDS.format(words=", ".join(words))

    def _build_prompt_top_docs(self, topic_idx: int) -> str:
        fragments_with_weights = self._get_top_doc_fragments(topic_idx)
        doc_blocks = []
        for i, (frag, weight) in enumerate(fragments_with_weights, 1):
            doc_blocks.append(
                f"--- Documento {i} (peso en el tema: {weight:.3f}) ---\n{frag}"
            )
        return PROMPT_TOP_DOCS.format(fragments="\n\n".join(doc_blocks))

    # ------------------------------------------------------------------
    # Gestión del navegador
    # ------------------------------------------------------------------

    def start_browser(self) -> None:
        """Abre el navegador, navega a Perplexity y espera a que el usuario
        inicie sesión si es necesario (igual que en procesar_todos_perplexity)."""
        print("[INFO] Abriendo navegador en perplexity.ai…")
        self._bot = PerplexityBot(headless=False)
        self._bot.start()
        self._bot.open_home()
        print("[INFO] Navegador listo. Continuando…")

    def stop_browser(self) -> None:
        """Cierra el navegador."""
        if self._bot:
            self._bot.quit()
            self._bot = None

    # ------------------------------------------------------------------
    # Consulta web
    # ------------------------------------------------------------------

    def _query(self, prompt: str) -> str:
        """Envía un prompt a Perplexity vía el navegador y devuelve la respuesta."""
        if self._bot is None:
            raise RuntimeError("Llama start_browser() antes de etiquetar.")
        self._bot.new_chat()
        return self._bot.query(prompt)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def label_all_topics(self) -> list[dict]:
        """
        Genera una etiqueta por tema usando la variante configurada en
        label_method ('top_words' o 'top_docs').

        Devuelve lista de dicts: topic_id, label, top_words.
        """
        n_topics = self.modeler.n_topics
        results = []

        print(
            f"[INFO] Etiquetando {n_topics} temas "
            f"(variante: {self.label_method})…"
        )

        for topic_idx in range(n_topics):
            topic_num = topic_idx + 1
            print(f"  Tema {topic_num:>2}/{n_topics}… ", end="", flush=True)

            if self.label_method == "top_words":
                prompt = self._build_prompt_top_words(topic_idx)
            else:
                prompt = self._build_prompt_top_docs(topic_idx)

            try:
                label = self._query(prompt)
            except Exception as exc:
                label = f"[ERROR: {exc}]"

            top_words = self._get_top_words(topic_idx)
            results.append(
                {
                    "topic_id": topic_num,
                    "label": label,
                    "top_words": top_words,
                }
            )
            print(f'"{label}"')

            if topic_idx < n_topics - 1:
                time.sleep(self.delay)

        return results

    def label_all_both(self) -> list[dict]:
        """
        Corre ambas variantes (top_words y top_docs) y devuelve los resultados
        con label_top_words y label_top_docs para comparación directa.

        Devuelve lista de dicts: topic_id, label_top_words, label_top_docs, top_words.
        """
        n_topics = self.modeler.n_topics
        results = []

        print(f"[INFO] Etiquetando {n_topics} temas con AMBAS variantes…")

        for topic_idx in range(n_topics):
            topic_num = topic_idx + 1
            top_words = self._get_top_words(topic_idx)

            # — Variante top_words —
            print(f"  Tema {topic_num:>2}/{n_topics} [top_words]… ", end="", flush=True)
            try:
                label_tw = self._query(self._build_prompt_top_words(topic_idx))
            except Exception as exc:
                label_tw = f"[ERROR: {exc}]"
            print(f'"{label_tw}"')
            time.sleep(self.delay)

            # — Variante top_docs —
            print(f"  Tema {topic_num:>2}/{n_topics} [top_docs]…  ", end="", flush=True)
            try:
                label_td = self._query(self._build_prompt_top_docs(topic_idx))
            except Exception as exc:
                label_td = f"[ERROR: {exc}]"
            print(f'"{label_td}"')

            results.append(
                {
                    "topic_id": topic_num,
                    "label_top_words": label_tw,
                    "label_top_docs": label_td,
                    "top_words": top_words,
                }
            )

            if topic_idx < n_topics - 1:
                time.sleep(self.delay)

        return results

    def save_labels(self, results: list[dict], output_path: str) -> None:
        """Guarda los resultados en un archivo JSON con formato legible."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Etiquetas guardadas en: {out}")

    def print_labels(self, results: list[dict]) -> None:
        """Imprime los resultados en pantalla."""
        print(f"\n{'='*60}")
        print("  ETIQUETAS DE TEMAS")
        print(f"{'='*60}")
        for r in results:
            if "label" in r:
                # Resultado de label_all_topics()
                print(f"\nTema {r['topic_id']:>2}: {r['label']}")
            else:
                # Resultado de label_all_both()
                print(
                    f"\nTema {r['topic_id']:>2}:\n"
                    f"  [top_words] {r['label_top_words']}\n"
                    f"  [top_docs]  {r['label_top_docs']}"
                )
            print(f"  Palabras: {', '.join(r['top_words'][:10])}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Etiquetado automático de temas con Perplexity AI (web)."
    )
    p.add_argument(
        "--corpus-dir",
        default=None,
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
        help="Algoritmo de topic modeling (default: lda)",
    )
    p.add_argument(
        "--label-method",
        choices=["top_words", "top_docs", "both"],
        default="top_words",
        help="Variante de etiquetado: top_words | top_docs | both (default: top_words)",
    )
    p.add_argument(
        "--n-words",
        type=int,
        default=15,
        help="Palabras top por tema (default: 15)",
    )
    p.add_argument(
        "--top-n-docs",
        type=int,
        default=3,
        help="Documentos top para variante top_docs (default: 3)",
    )
    p.add_argument(
        "--fragment-chars",
        type=int,
        default=800,
        help="Caracteres máx. de fragmento por doc en top_docs (default: 800)",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Segundos entre consultas (default: 2.0)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Ruta JSON para guardar etiquetas (opcional)",
    )
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    corpus_dir = args.corpus_dir or DEFAULT_CORPUS_DIR

    modeler = HistoricalTopicModeler(
        corpus_dir=corpus_dir,
        n_topics=args.n_topics,
        method=args.method,
        n_words=args.n_words,
    )
    modeler.fit()
    modeler.print_topics()

    labeler = TopicLabeler(
        modeler=modeler,
        label_method="top_words" if args.label_method == "both" else args.label_method,
        n_words=args.n_words,
        top_n_docs=args.top_n_docs,
        fragment_chars=args.fragment_chars,
        delay=args.delay,
    )

    labeler.start_browser()

    try:
        if args.label_method == "both":
            results = labeler.label_all_both()
        else:
            results = labeler.label_all_topics()
    finally:
        labeler.stop_browser()

    labeler.print_labels(results)

    if args.output:
        labeler.save_labels(results, args.output)
