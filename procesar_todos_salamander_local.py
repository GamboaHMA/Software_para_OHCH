"""
Procesamiento en lote de todos los .txt en TXT_ohcbh_cf_erl/ usando Ollama
con el modelo Salamandra-7B-Instruct ejecutado localmente.

Hardware objetivo:
  - CPU  : Intel Core i7 8ª gen
  - RAM  : 16 GB
  - GPU  : NVIDIA GTX 1050 (4 GB VRAM)  → el modelo de 4.9 GB no cabe íntegro
           en VRAM; Ollama lo distribuye automáticamente entre GPU y CPU por capas
  - Disco: SSD 1 TB

Por esa razón MAX_CHARS_PER_QUERY es conservador (≈ 500 tokens de entrada),
de modo que el contexto total (instrucciones + texto + respuesta) se mantenga
holgadamente dentro de la ventana de 4 096 tokens del modelo y la inferencia
parcial en CPU no se vuelva prohibitivamente lenta.

Por cada archivo:
  - Aplica limpiar_espacios_basicos al texto crudo
  - Divide en fragmentos si supera MAX_CHARS_PER_QUERY
  - Envía cada fragmento a Ollama con un prompt de reconstrucción OCR
  - Guarda el texto reconstruido en Reconstrucciones_salamander/<subcarpeta>/<archivo>.txt
  - Guarda metadata en Reconstrucciones_salamander/<subcarpeta>/<archivo>.meta.json

Se puede reanudar: si el .txt de salida ya existe se salta ese archivo.

Dependencias:
    pip install ollama
"""

import argparse
import json
import time
import traceback
from pathlib import Path

from ollama import chat, list as ollama_list, ResponseError

from anomalies_checker import limpiar_espacios_basicos

# ══════════════════════════════════════════════════════════════════════════════
# ── Rutas ─────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
INPUT_ROOT  = Path("TXT_ohcbh_cf_erl")
OUTPUT_ROOT = Path("Reconstrucciones_salamander")

# ══════════════════════════════════════════════════════════════════════════════
# ── Modelo y parámetros de inferencia ─────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
MODEL = "hdnh2006/salamandra-7b-instruct:latest"

# Ventana de contexto del modelo (tokens).
# Salamandra-7B usa arquitectura Llama-2 → 4 096 tokens máximo.
NUM_CTX = 4096

# Máximo de caracteres por fragmento enviado al modelo.
# Estimación con español ~ 4 chars/token:
#   2 000 chars → ~ 500 tokens de entrada
#   + instrucciones  ~ 150 tokens
#   + respuesta      ~ 500 tokens (similar al input)
#   ─────────────────────────────────────────
#   Total estimado   ~ 1 150 / 4 096  → margen holgado para GPU+CPU offload
#
# Bajar este valor si el modelo responde muy lento o produce respuestas vacías.
MAX_CHARS_PER_QUERY = 2_000

# Opciones de generación pasadas a Ollama
OLLAMA_OPTIONS = {
    "num_ctx": NUM_CTX,
    "temperature": 0.1,    # baja temperatura → reproducibilidad en corrección OCR
    "top_p": 0.9,
    "repeat_penalty": 1.1,
    # Capas a offloadear a la GPU. Salamandra-7B tiene 32 capas; con 4 GB de VRAM
    # se pueden meter ~20 de forma segura dejando margen para KV-cache y overhead.
    # Si Ollama reporta errores de VRAM, bajar a 16. Si va bien, subir hasta 28.
    "num_gpu": 22,
}

# Segundos de pausa entre archivos (reduce la carga térmica en inferencia por CPU)
DELAY_BETWEEN_FILES = 1

# ══════════════════════════════════════════════════════════════════════════════
# ── Prompt de reconstrucción OCR ──────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
_SYSTEM_MSG = (
    "Eres un experto en transcripción y corrección de documentos históricos en español. "
    "Tu única tarea es corregir errores de OCR sin alterar el contenido ni el estilo. "
    "Devuelve ÚNICAMENTE el texto reconstruido, sin explicaciones, sin comentarios, "
    "sin encabezados, sin delimitadores de bloque de código."
)

_USER_TEMPLATE = (
    "El siguiente texto fue extraído mediante OCR de un documento histórico impreso y "
    "contiene errores típicos: caracteres aislados sin sentido (letras o símbolos sueltos "
    "en líneas propias), palabras partidas con guión al final de línea que deben unirse, "
    "espacios incorrectos dentro de palabras, confusión entre caracteres similares "
    "(l/1, o/0, rn/m, cl/d, etc.) y signos de puntuación mal reconocidos.\n"
    "Reconstruye el texto corrigiendo únicamente los errores de OCR. Conserva el español "
    "histórico, la ortografía de la época (tildes arcaicas, formas antiguas) y el "
    "contenido exacto: NO añadas, inventes ni elimines información.\n\n"
    "TEXTO OCR:\n{text}"
)


def _build_messages(text: str) -> list:
    return [
        {"role": "system", "content": _SYSTEM_MSG},
        {"role": "user",   "content": _USER_TEMPLATE.format(text=text)},
    ]


# ══════════════════════════════════════════════════════════════════════════════
# ── Utilidades de texto ───────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def chunk_text(text: str, max_chars: int = MAX_CHARS_PER_QUERY) -> list:
    """Divide text en fragmentos que quepan dentro de max_chars.
    Intenta cortar en párrafos; si un párrafo solo ya supera el límite lo parte
    en oraciones; si sigue siendo grande lo parte por caracteres."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = text.split("\n\n")
    chunks, current, current_len = [], [], 0

    for para in paragraphs:
        # +2 por el separador \n\n que se añadirá al unir
        needed = len(para) + (2 if current else 0)
        if current_len + needed > max_chars and current:
            chunks.append("\n\n".join(current))
            current, current_len = [], 0
        # Si el párrafo solo ya supera el límite, partirlo en oraciones
        if len(para) > max_chars:
            sentences = para.replace(". ", ".\n").split("\n")
            for sent in sentences:
                needed_s = len(sent) + (2 if current else 0)
                if current_len + needed_s > max_chars and current:
                    chunks.append("\n\n".join(current))
                    current, current_len = [], 0
                current.append(sent)
                current_len += needed_s
        else:
            current.append(para)
            current_len += needed

    if current:
        chunks.append("\n\n".join(current))

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# ── Checkpoint de progreso parcial ───────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def _save_pending(path: Path, total_chunks: int, chunks_done: int, parts: list) -> None:
    """Persiste el estado de procesamiento parcial en un archivo .pending.json."""
    data = {
        "total_chunks":       total_chunks,
        "chunks_done":        chunks_done,
        "reconstructed_parts": parts,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# ── Llamada a Ollama ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def query_ollama(text: str) -> str:
    """Envía un fragmento de texto OCR a Ollama y devuelve la respuesta."""
    response = chat(
        model=MODEL,
        messages=_build_messages(text),
        options=OLLAMA_OPTIONS,
    )
    return response.message.content.strip()


def process_text(raw_text: str, pending_path=None) -> dict:
    """Limpia, divide en fragmentos si es necesario, consulta Salamandra y
    devuelve un dict con 'reconstructed_text' y 'chunks_used'.

    Si se proporciona pending_path (Path), guarda el estado parcial tras cada
    fragmento completado para poder reanudar si Ollama falla a mitad del proceso.
    Al retomar, si el número de chunks coincide con el guardado, continúa desde
    el último chunk completado; en caso contrario, empieza de cero."""
    cleaned = limpiar_espacios_basicos(raw_text)
    fragments = chunk_text(cleaned, MAX_CHARS_PER_QUERY)
    total_chunks = len(fragments)
    start_chunk = 0
    reconstructed_parts = []

    # ── Reanudar desde punto de control si existe ──────────────────────────────
    if pending_path is not None and pending_path.exists():
        try:
            pending = json.loads(pending_path.read_text(encoding="utf-8"))
            if pending.get("total_chunks") == total_chunks:
                start_chunk = int(pending.get("chunks_done", 0))
                reconstructed_parts = list(pending.get("reconstructed_parts", []))
                if start_chunk > 0:
                    print(
                        f"      Reanudando desde fragmento "
                        f"{start_chunk + 1}/{total_chunks}…",
                        flush=True,
                    )
        except Exception:
            # Pending corrupto → empezar de cero
            start_chunk = 0
            reconstructed_parts = []

    # ── Guardar estado inicial del pending (crea el archivo si no existe) ──────
    if pending_path is not None:
        _save_pending(pending_path, total_chunks, start_chunk, reconstructed_parts)

    for i in range(start_chunk, total_chunks):
        fragment = fragments[i]
        if total_chunks > 1:
            print(
                f"      fragmento {i + 1}/{total_chunks} "
                f"({len(fragment)} chars)…",
                flush=True,
            )
        response = query_ollama(fragment)
        reconstructed_parts.append(response)

        # Guardar punto de control tras cada fragmento completado
        if pending_path is not None:
            _save_pending(pending_path, total_chunks, i + 1, reconstructed_parts)

    return {
        "reconstructed_text": "\n\n".join(reconstructed_parts),
        "chunks_used": total_chunks,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ── Verificación de disponibilidad del modelo ─────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def check_model():
    """Lanza RuntimeError si el modelo no está disponible en Ollama."""
    try:
        available = [m.model for m in ollama_list().models]
    except ResponseError as e:
        raise RuntimeError(f"No se pudo conectar con Ollama: {e}")
    if MODEL not in available:
        raise RuntimeError(
            f"Modelo '{MODEL}' no encontrado en Ollama.\n"
            f"Modelos disponibles: {available}\n"
            f"Descárgalo con:  ollama pull {MODEL}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# ── Bucle principal ───────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def main(desde=None, hasta=None):
    stats = {"procesados": 0, "saltados": 0, "errores": 0, "tiempo_total": 0.0}

    all_txt = sorted(INPUT_ROOT.rglob("*.txt"))
    if desde is None:
        desde = 0
    if hasta is None:
        hasta = len(all_txt)
    all_txt = all_txt[desde:hasta]
    total = len(all_txt)


    print(f"Archivos encontrados: {total}\n", flush=True)

    print("Verificando modelo en Ollama…", flush=True)
    check_model()
    print(f"Modelo '{MODEL}' disponible. Iniciando procesamiento…\n", flush=True)

    for idx, src_path in enumerate(all_txt, 1):
        rel         = src_path.relative_to(INPUT_ROOT)
        dst_txt     = OUTPUT_ROOT / rel
        dst_meta    = dst_txt.with_suffix(".meta.json")
        dst_pending = dst_txt.parent / (dst_txt.stem + ".pending.json")

        # Reanudar: saltar solo si el .txt final ya existe
        if dst_txt.exists():
            stats["saltados"] += 1
            continue

        dst_txt.parent.mkdir(parents=True, exist_ok=True)

        # Indicar si se retoma un procesamiento parcial previo
        if dst_pending.exists():
            print(f"[{idx}/{total}] {rel}  [retomando pending]", flush=True)
        else:
            print(f"[{idx}/{total}] {rel}", flush=True)

        t0 = time.perf_counter()
        try:
            raw_text = src_path.read_text(encoding="utf-8", errors="replace")
            result   = process_text(raw_text, pending_path=dst_pending)
            elapsed  = time.perf_counter() - t0

            # Guardar texto reconstruido
            dst_txt.write_text(result["reconstructed_text"], encoding="utf-8")

            # Eliminar el pending ahora que el archivo final está guardado
            if dst_pending.exists():
                dst_pending.unlink()

            # Guardar metadata
            meta = {
                "source":      str(src_path),
                "model":       MODEL,
                "chunks_used": result["chunks_used"],
                "elapsed_s":   round(elapsed, 2),
            }
            dst_meta.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            stats["procesados"]   += 1
            stats["tiempo_total"] += elapsed
            print(f"    OK  ({elapsed:.1f}s)", flush=True)

        except Exception:
            elapsed = time.perf_counter() - t0
            stats["errores"] += 1
            error_msg = traceback.format_exc()
            print(f"    ERROR:\n{error_msg}", flush=True)

            # Guardar traza en meta; el .pending.json conserva el avance parcial
            dst_meta.write_text(
                json.dumps({"error": error_msg}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        if idx < total:
            time.sleep(DELAY_BETWEEN_FILES)

    # ── Resumen final ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Procesados : {stats['procesados']}")
    print(f"Saltados   : {stats['saltados']}")
    print(f"Errores    : {stats['errores']}")
    if stats["procesados"]:
        avg = stats["tiempo_total"] / stats["procesados"]
        print(
            f"Tiempo total: {stats['tiempo_total']:.1f}s  |  "
            f"Promedio: {avg:.1f}s/archivo"
        )
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reconstrucción OCR en lote con Ollama."
    )
    parser.add_argument(
        "--modelo",
        default=MODEL,
        help=(
            "Nombre del modelo en Ollama (p.ej. qwen2.5:7b, "
            "hdnh2006/salamandra-7b-instruct:latest). "
            f"Por defecto: {MODEL}"
        ),
    )
    parser.add_argument(
        "--desde",
        type=int,
        default=0,
        metavar="M",
        help=(
            "Empezar desde el documento número M de la lista global ordenada "
            "(índice 0-based, inclusivo). "
            "Ej: --desde 1000"
        ),
    )
    parser.add_argument(
        "--hasta",
        type=int,
        default=None,
        metavar="N",
        help="Procesar solo los primeros N archivos de la lista ordenada.",
    )
    parser.add_argument(
        "--salida",
        default=str(OUTPUT_ROOT),
        help=f"Carpeta raíz de salida. Por defecto: {OUTPUT_ROOT}",
    )
    args = parser.parse_args()

    # Alias cortos → configuración completa por modelo
    # num_gpu:    capas offloadeadas a GPU (99 = todas las que quepan en VRAM)
    # num_ctx:    ventana de contexto en tokens
    # max_chars:  tamaño máximo de fragmento por consulta (chars)
    #   qwen2.5:3b  → num_ctx 8192 → max_chars 6000 (~1500 tok entrada, margen holgado)
    #   salamandra  → num_ctx 4096 → max_chars 2000 (conservador por GPU+CPU offload)
    _ALIASES = {
        "qwen": {
            "model":       "qwen2.5:3b",
            "output_root": "Reconstrucciones_qwen_deepseek",
            "num_gpu":     99,
            "num_ctx":     8192,
            "max_chars":   6_000,
        },
        "salamandra": {
            "model":       "hdnh2006/salamandra-7b-instruct:latest",
            "output_root": "Reconstrucciones_salamander",
            "num_gpu":     16,
            "num_ctx":     4096,
            "max_chars":   2_000,
        },
    }

    alias = _ALIASES.get(args.modelo)

    # Actualizar globals con los valores resueltos
    MODEL                     = alias["model"]       if alias else args.modelo
    OUTPUT_ROOT               = Path(alias["output_root"] if alias else args.salida)
    OLLAMA_OPTIONS["num_gpu"] = alias["num_gpu"]     if alias else OLLAMA_OPTIONS["num_gpu"]
    OLLAMA_OPTIONS["num_ctx"] = alias["num_ctx"]     if alias else OLLAMA_OPTIONS["num_ctx"]
    MAX_CHARS_PER_QUERY       = alias["max_chars"]   if alias else MAX_CHARS_PER_QUERY

    main(args.desde, args.hasta)
