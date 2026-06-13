"""
Procesamiento en lote de todos los .txt en TXT_ohcbh_cf_erl/ usando la API oficial
de DeepSeek (deepseek-chat / DeepSeek-V3) a través del SDK de OpenAI.

Diferencias clave respecto a los scripts Selenium y Ollama:
  - Usa la API REST de DeepSeek en lugar de Selenium o Ollama.
  - Chunks mucho más grandes: hasta 40 000 chars (~10 000 tokens de entrada).
    La mayoría de archivos queda en UNA sola llamada API.
  - Limitador de tasa compartido entre todos los workers (fcntl, archivo en disco).
    Respeta MAX_RPM, MAX_TPM y MAX_TPH configurables.
  - Retroceso exponencial automático ante 429 (rate limit) y 5xx (servidor).
  - Parámetro --max-tokens-per-session para controlar el gasto diario.
  - Reanudación automática: si el .txt final ya existe se salta el archivo.
    Los .pending.json permiten retomar archivos parcialmente procesados.

Modelo: deepseek-chat (DeepSeek-V3)
  Ventana de contexto : 64 000 tokens
  Salida máxima       :  8 192 tokens
  Precio (jun 2025)   :  $0.27/M tokens entrada (cache miss)
                         $0.07/M tokens entrada (cache hit)
                         $1.10/M tokens salida

Límites de la API (tier estándar de pago):
  RPM  : ~60 requests/min   → usamos MAX_RPM = 50 (conservador)
  TPM  : ~1 000 000 tok/min → usamos MAX_TPM = 500 000 (conservador)
  TPH  : sin límite oficial  → usamos MAX_TPH = 2 000 000 (presupuesto propio)

Uso — una sola terminal:
    python procesar_todos_deepseek_api.py

Uso — múltiples terminales (ej. 4 workers):
    Terminal 0:  python procesar_todos_deepseek_api.py --worker 0 --total-workers 4
    Terminal 1:  python procesar_todos_deepseek_api.py --worker 1 --total-workers 4
    Terminal 2:  python procesar_todos_deepseek_api.py --worker 2 --total-workers 4
    Terminal 3:  python procesar_todos_deepseek_api.py --worker 3 --total-workers 4

Dependencias:
    pip install openai          (ya instalado en el entorno)
"""

import argparse
import fcntl
import json
import time
import traceback
from pathlib import Path

from openai import OpenAI, RateLimitError, APIStatusError, APIConnectionError

from deepseek_api_key import API_KEY
from anomalies_checker import limpiar_espacios_basicos

# ══════════════════════════════════════════════════════════════════════════════
# ── Rutas ─────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
INPUT_ROOT  = Path("TXT_ohcbh_cf_erl")
OUTPUT_ROOT = Path("Reconstrucciones_qwen_deepseek")

# ══════════════════════════════════════════════════════════════════════════════
# ── Modelo y parámetros de la API ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
MODEL            = "deepseek-v4-flash"   # deepseek-v4-pro para mayor calidad
DEEPSEEK_BASE    = "https://api.deepseek.com"

# Tamaño máximo de fragmento enviado en UNA llamada.
# 40 000 chars ≈ 10 000 tokens de entrada; con respuesta similar → ~20 000 tok/llamada,
# muy holgado respecto a la ventana de 64 K del modelo.
MAX_CHARS_PER_QUERY = 40_000

# Máximo de tokens de salida que puede generar el modelo por llamada.
MAX_OUTPUT_TOKENS = 8_192

# Temperatura baja → reproducibilidad en corrección OCR.
TEMPERATURE = 0.1

# ══════════════════════════════════════════════════════════════════════════════
# ── Límites de tasa (compartidos entre todos los workers) ─────────────────────
# ══════════════════════════════════════════════════════════════════════════════
MAX_RPM = 50          # requests por minuto (conservador, límite real ~60)
MAX_TPM = 500_000     # tokens por minuto   (conservador, límite real ~1 000 000)
MAX_TPH = 2_000_000   # tokens por hora (presupuesto propio, sin límite oficial)

# Archivo de estado compartido (todos los workers leen/escriben aquí con fcntl)
RATE_STATE_FILE = OUTPUT_ROOT / ".rate_state.json"

# ══════════════════════════════════════════════════════════════════════════════
# ── Reintentos ────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
MAX_RETRIES      = 6    # intentos máximos por llamada antes de lanzar excepción
RETRY_BASE_DELAY = 5    # segundos base para retroceso exponencial
# Número máximo de turnos de continuación cuando la respuesta se trunca (finish_reason='length').
# Cada continuación le pide al modelo que siga desde donde se quedó.
MAX_CONTINUATIONS = 4

# Pausa entre archivos (segundos) para reducir carga térmica y dar margen a la API.
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
# ── Cliente OpenAI → DeepSeek ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def _make_client() -> OpenAI:
    return OpenAI(api_key=API_KEY, base_url=DEEPSEEK_BASE)


# ══════════════════════════════════════════════════════════════════════════════
# ── Limitador de tasa compartido entre procesos ───────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
class SharedRateLimiter:
    """
    Limita la tasa de llamadas a la API entre múltiples procesos en paralelo.
    Usa fcntl.flock para exclusión mutua al leer/escribir el archivo de estado JSON.

    Controla:
      - Requests por minuto (RPM)
      - Tokens por minuto   (TPM)
      - Tokens por hora     (TPH)
    """

    def __init__(
        self,
        state_path: Path,
        max_rpm: int = MAX_RPM,
        max_tpm: int = MAX_TPM,
        max_tph: int = MAX_TPH,
    ):
        self.path    = state_path
        self.max_rpm = max_rpm
        self.max_tpm = max_tpm
        self.max_tph = max_tph
        state_path.parent.mkdir(parents=True, exist_ok=True)
        if not state_path.exists():
            self._write_state_unlocked(
                state_path,
                {"min_start": time.time(), "hour_start": time.time(),
                 "tokens_min": 0, "req_min": 0, "tokens_hour": 0},
            )

    @staticmethod
    def _write_state_unlocked(path: Path, state: dict) -> None:
        """Escribe el estado en el archivo (debe llamarse con el lock ya adquirido)."""
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _current_state(self, fh) -> dict:
        """Lee el estado actual del archivo ya abierto (lock adquirido)."""
        fh.seek(0)
        try:
            return json.loads(fh.read())
        except (json.JSONDecodeError, ValueError):
            now = time.time()
            return {"min_start": now, "hour_start": now,
                    "tokens_min": 0, "req_min": 0, "tokens_hour": 0}

    def wait_and_register(self, estimated_tokens: int) -> None:
        """
        Bloquea hasta que haya capacidad para `estimated_tokens` tokens.
        Luego registra el uso y libera el lock.
        Usa retroceso con espera corta para no monopolizar el lock entre workers.
        """
        while True:
            wait_secs = 0.0

            # Modo apertura "r+" falla si el archivo no existe; usamos "a+" que lo crea
            with open(self.path, "a+", encoding="utf-8") as fh:
                fcntl.flock(fh, fcntl.LOCK_EX)
                try:
                    now = time.time()
                    s = self._current_state(fh)

                    # Rotar ventana de minuto
                    if now - s["min_start"] >= 60:
                        s.update(min_start=now, tokens_min=0, req_min=0)
                    # Rotar ventana de hora
                    if now - s["hour_start"] >= 3600:
                        s.update(hour_start=now, tokens_hour=0)

                    tpm_ok   = s["tokens_min"] + estimated_tokens <= self.max_tpm
                    rpm_ok   = s["req_min"] + 1 <= self.max_rpm
                    tph_ok   = s["tokens_hour"] + estimated_tokens <= self.max_tph

                    if tpm_ok and rpm_ok and tph_ok:
                        s["tokens_min"]  += estimated_tokens
                        s["req_min"]     += 1
                        s["tokens_hour"] += estimated_tokens
                        fh.seek(0)
                        fh.truncate()
                        fh.write(json.dumps(s, indent=2))
                        return  # cupo reservado, podemos llamar a la API

                    # Calcular cuánto tiempo falta hasta que se libere cupo
                    wait_min  = max(0.0, 61.0 - (now - s["min_start"]))
                    wait_hour = max(0.0, 3601.0 - (now - s["hour_start"])) if not tph_ok else 0.0
                    wait_secs = max(wait_min, wait_hour)

                finally:
                    fcntl.flock(fh, fcntl.LOCK_UN)

            # Dormir en intervalos cortos para ceder la CPU al resto de workers
            print(
                f"    [RateLimit] cuota alcanzada, esperando {wait_secs:.0f}s…",
                flush=True,
            )
            time.sleep(min(wait_secs, 30.0))

    def correct_usage(self, actual_tokens: int, estimated_tokens: int) -> None:
        """
        Ajusta el conteo con los tokens reales informados por la API.
        Solo actúa si la diferencia supera 500 tokens para evitar bloqueos frecuentes.
        """
        diff = actual_tokens - estimated_tokens
        if abs(diff) < 500:
            return
        with open(self.path, "a+", encoding="utf-8") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                s = self._current_state(fh)
                s["tokens_min"]  = max(0, s["tokens_min"]  + diff)
                s["tokens_hour"] = max(0, s["tokens_hour"] + diff)
                fh.seek(0)
                fh.truncate()
                fh.write(json.dumps(s, indent=2))
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)


# ══════════════════════════════════════════════════════════════════════════════
# ── Utilidades de texto ───────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def chunk_text(text: str, max_chars: int = MAX_CHARS_PER_QUERY) -> list:
    """Divide text en fragmentos que quepan dentro de max_chars.

    Intenta cortar en párrafos (doble salto de línea); si un párrafo solo supera
    el límite lo parte en oraciones; si una oración sola sigue siendo demasiado
    grande la parte por caracteres."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = text.split("\n\n")
    chunks, current, current_len = [], [], 0

    for para in paragraphs:
        needed = len(para) + (2 if current else 0)
        if current_len + needed > max_chars and current:
            chunks.append("\n\n".join(current))
            current, current_len = [], 0
        if len(para) > max_chars:
            sentences = para.replace(". ", ".\n").split("\n")
            for sent in sentences:
                needed_s = len(sent) + (2 if current else 0)
                if current_len + needed_s > max_chars and current:
                    chunks.append("\n\n".join(current))
                    current, current_len = [], 0
                # Si la oración sola excede max_chars, partirla por caracteres
                if len(sent) > max_chars:
                    for i in range(0, len(sent), max_chars):
                        chunks.append(sent[i : i + max_chars])
                else:
                    current.append(sent)
                    current_len += needed_s
        else:
            current.append(para)
            current_len += needed

    if current:
        chunks.append("\n\n".join(current))

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# ── Llamada a la API con reintento automático ─────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def _single_call(
    client: OpenAI,
    messages: list,
    estimated_tokens: int,
    rate_limiter: SharedRateLimiter,
    prefix: str = "",
) -> tuple[str, str, int]:
    """
    Hace UNA llamada a la API con retroceso exponencial.
    Devuelve (content, finish_reason, total_tokens).

    Reintentos automáticos ante:
      - 429 RateLimitError   → exceso de tasa reportado por la API
      - 5xx APIStatusError   → errores temporales del servidor
      - APIConnectionError   → red caída o timeout
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_OUTPUT_TOKENS,
            )
            content       = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason
            total_tokens  = response.usage.total_tokens if response.usage else estimated_tokens
            rate_limiter.correct_usage(total_tokens, estimated_tokens)
            return content, finish_reason, total_tokens

        except RateLimitError as e:
            wait = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(
                f"    {prefix}[429 RateLimit] intento {attempt}/{MAX_RETRIES}, "
                f"esperando {wait}s… ({e})",
                flush=True,
            )
            time.sleep(wait)

        except APIStatusError as e:
            if e.status_code == 402:
                raise RuntimeError(
                    "La API de DeepSeek reporta créditos insuficientes (402). "
                    "Recarga el saldo o verifica la API key."
                ) from e
            if e.status_code >= 500:
                wait = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(
                    f"    {prefix}[{e.status_code} ServerError] intento {attempt}/{MAX_RETRIES}, "
                    f"esperando {wait}s… ({e})",
                    flush=True,
                )
                time.sleep(wait)
            else:
                raise  # 4xx que no son 429 → relanzar sin reintentar

        except APIConnectionError as e:
            wait = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(
                f"    {prefix}[ConexiónError] intento {attempt}/{MAX_RETRIES}, "
                f"esperando {wait}s… ({e})",
                flush=True,
            )
            time.sleep(wait)

    raise RuntimeError(f"La llamada a la API falló tras {MAX_RETRIES} intentos.")


def query_api(
    client: OpenAI,
    text: str,
    rate_limiter: SharedRateLimiter,
    prefix: str = "",
) -> tuple[str, int]:
    """
    Envía un fragmento de texto OCR a la API de DeepSeek y devuelve
    (respuesta_texto, total_tokens_usados).

    Si la respuesta se trunca (finish_reason='length'), solicita automáticamente
    hasta MAX_CONTINUATIONS turnos adicionales para obtener el texto completo.
    """
    # Estimación conservadora de tokens (1 token ≈ 3.5 chars en español)
    estimated_tokens = int(len(text) / 3.5) + 500  # +500 por el system prompt
    rate_limiter.wait_and_register(estimated_tokens)

    messages     = _build_messages(text)
    full_content = ""
    total_tokens = 0

    for cont in range(MAX_CONTINUATIONS + 1):
        content, finish_reason, used = _single_call(
            client, messages, estimated_tokens, rate_limiter, prefix
        )
        full_content += content
        total_tokens += used

        if finish_reason != "length":
            break  # respuesta completa

        # La respuesta fue truncada: pedir continuación
        if cont < MAX_CONTINUATIONS:
            print(
                f"    {prefix}[cont.{cont + 1}/{MAX_CONTINUATIONS}] "
                f"respuesta truncada (finish_reason='length'), continuando…",
                flush=True,
            )
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": "Continúa el texto reconstruido exactamente desde donde lo dejaste, sin repetir nada.",
            })
            estimated_tokens = MAX_OUTPUT_TOKENS + 200
            rate_limiter.wait_and_register(estimated_tokens)
        else:
            print(
                f"    {prefix}⚠ Se alcanzó el límite de continuaciones ({MAX_CONTINUATIONS}). "
                f"El texto puede estar incompleto.",
                flush=True,
            )

    return full_content.strip(), total_tokens


# ══════════════════════════════════════════════════════════════════════════════
# ── Checkpoint de progreso parcial ───────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def _save_pending(path: Path, total_chunks: int, chunks_done: int, parts: list) -> None:
    data = {
        "total_chunks":        total_chunks,
        "chunks_done":         chunks_done,
        "reconstructed_parts": parts,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# ── Procesamiento de un archivo ───────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def process_text(
    raw_text: str,
    client: OpenAI,
    rate_limiter: SharedRateLimiter,
    pending_path: Path | None = None,
    prefix: str = "",
) -> dict:
    """
    Limpia, divide en fragmentos si es necesario, consulta la API de DeepSeek y
    devuelve un dict con 'reconstructed_text', 'chunks_used' y 'total_tokens'.

    Si se proporciona pending_path, guarda checkpoint tras cada fragmento completado.
    Si el pending.json existe con el mismo número de chunks, reanuda desde el último.
    """
    cleaned      = limpiar_espacios_basicos(raw_text)
    fragments    = chunk_text(cleaned, MAX_CHARS_PER_QUERY)
    total_chunks = len(fragments)
    start_chunk  = 0
    parts        = []
    total_tokens = 0

    # ── Reanudar desde checkpoint si existe ───────────────────────────────────
    if pending_path is not None and pending_path.exists():
        try:
            pending = json.loads(pending_path.read_text(encoding="utf-8"))
            if pending.get("total_chunks") == total_chunks:
                start_chunk = int(pending.get("chunks_done", 0))
                parts       = list(pending.get("reconstructed_parts", []))
                if start_chunk > 0:
                    print(
                        f"    {prefix}Reanudando desde fragmento "
                        f"{start_chunk + 1}/{total_chunks}…",
                        flush=True,
                    )
        except Exception:
            start_chunk = 0
            parts = []

    # Guardar estado inicial
    if pending_path is not None:
        _save_pending(pending_path, total_chunks, start_chunk, parts)

    for i in range(start_chunk, total_chunks):
        fragment = fragments[i]
        if total_chunks > 1:
            print(
                f"    {prefix}fragmento {i + 1}/{total_chunks} "
                f"({len(fragment)} chars)…",
                flush=True,
            )
        response_text, used_tokens = query_api(
            client, fragment, rate_limiter, prefix=prefix
        )
        parts.append(response_text)
        total_tokens += used_tokens

        if pending_path is not None:
            _save_pending(pending_path, total_chunks, i + 1, parts)

    return {
        "reconstructed_text": "\n\n".join(parts),
        "chunks_used":        total_chunks,
        "total_tokens":       total_tokens,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ── Bucle principal ───────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def main(args: argparse.Namespace) -> None:
    prefix       = f"[W{args.worker}] " if args.total_workers > 1 else ""
    rate_limiter = SharedRateLimiter(
        state_path=RATE_STATE_FILE,
        max_rpm=MAX_RPM,
        max_tpm=MAX_TPM,
        max_tph=MAX_TPH,
    )
    client       = _make_client()

    stats = {
        "procesados": 0, "saltados": 0, "errores": 0,
        "tiempo_total": 0.0, "tokens_total": 0,
    }

    # ── Lista de archivos asignados a este worker ──────────────────────────────
    all_txt_global = sorted(INPUT_ROOT.rglob("*.txt"))
    # Aplicar rango --desde / --hasta
    hasta = args.hasta if args.hasta is not None else len(all_txt_global)
    all_txt = all_txt_global[args.desde:hasta]
    # Reparto entre workers: posiciones worker, worker+T, worker+2T, …
    worker_txt   = all_txt[args.worker :: args.total_workers]
    total_worker = len(worker_txt)

    rango_str = f"#{args.desde}–#{hasta - 1}"
    print(
        f"{prefix}Archivos globales: {len(all_txt_global)} | "
        f"Rango ({rango_str}): {len(all_txt)} | "
        f"Asignados a este worker: {total_worker}\n",
        flush=True,
    )

    print(
        f"{prefix}Modelo: {MODEL}  |  "
        f"Chunk máx: {MAX_CHARS_PER_QUERY:,} chars  |  "
        f"Límites: {MAX_RPM} RPM · {MAX_TPM:,} TPM · {MAX_TPH:,} TPH",
        flush=True,
    )
    if args.max_tokens_session:
        print(
            f"{prefix}Presupuesto de sesión: {args.max_tokens_session:,} tokens\n",
            flush=True,
        )

    print("", flush=True)

    for idx, src_path in enumerate(worker_txt, 1):
        # ── Verificar presupuesto de sesión ───────────────────────────────────
        if args.max_tokens_session and stats["tokens_total"] >= args.max_tokens_session:
            print(
                f"\n{prefix}Presupuesto de sesión alcanzado "
                f"({stats['tokens_total']:,} tokens). Deteniendo worker.",
                flush=True,
            )
            break

        rel         = src_path.relative_to(INPUT_ROOT)
        dst_txt     = OUTPUT_ROOT / rel
        dst_meta    = dst_txt.with_suffix(".meta.json")
        dst_pending = dst_txt.parent / (dst_txt.stem + ".pending.json")

        # Reanudar: saltar solo si el .txt final ya existe
        if dst_txt.exists():
            stats["saltados"] += 1
            continue

        dst_txt.parent.mkdir(parents=True, exist_ok=True)

        retomando = dst_pending.exists()
        print(
            f"{prefix}[{idx}/{total_worker}] {rel}"
            + ("  [retomando pending]" if retomando else ""),
            flush=True,
        )

        t0 = time.perf_counter()
        try:
            raw_text = src_path.read_text(encoding="utf-8", errors="replace")
            result   = process_text(
                raw_text,
                client=client,
                rate_limiter=rate_limiter,
                pending_path=dst_pending,
                prefix=prefix,
            )
            elapsed = time.perf_counter() - t0

            # Guardar texto reconstruido
            dst_txt.write_text(result["reconstructed_text"], encoding="utf-8")

            # Eliminar pending ahora que el archivo final está guardado
            if dst_pending.exists():
                dst_pending.unlink()

            # Guardar metadata
            meta = {
                "source":       str(src_path),
                "model":        MODEL,
                "chunks_used":  result["chunks_used"],
                "total_tokens": result["total_tokens"],
                "elapsed_s":    round(elapsed, 2),
            }
            dst_meta.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            stats["procesados"]   += 1
            stats["tiempo_total"] += elapsed
            stats["tokens_total"] += result["total_tokens"]
            print(
                f"{prefix}    OK  ({elapsed:.1f}s, {result['total_tokens']:,} tokens)",
                flush=True,
            )

        except RuntimeError as e:
            # Errores irrecuperables (sin créditos, etc.) → abortar el worker
            print(f"{prefix}    FATAL: {e}", flush=True)
            stats["errores"] += 1
            dst_meta.write_text(
                json.dumps({"error": str(e)}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            break

        except Exception:
            elapsed = time.perf_counter() - t0
            stats["errores"] += 1
            error_msg = traceback.format_exc()
            print(f"{prefix}    ERROR:\n{error_msg}", flush=True)
            # El pending.json guarda el avance parcial para retomar luego
            dst_meta.write_text(
                json.dumps({"error": error_msg}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        if idx < total_worker:
            time.sleep(DELAY_BETWEEN_FILES)

    # ── Resumen final ──────────────────────────────────────────────────────────
    print(f"\n{prefix}" + "=" * 60)
    print(f"{prefix}Procesados : {stats['procesados']}")
    print(f"{prefix}Saltados   : {stats['saltados']}")
    print(f"{prefix}Errores    : {stats['errores']}")
    print(f"{prefix}Tokens usados (sesión): {stats['tokens_total']:,}")
    if stats["procesados"]:
        avg_t = stats["tiempo_total"] / stats["procesados"]
        avg_tok = stats["tokens_total"] / stats["procesados"]
        print(
            f"{prefix}Tiempo total : {stats['tiempo_total']:.1f}s  "
            f"Promedio: {avg_t:.1f}s/archivo"
        )
        print(f"{prefix}Tokens promedio/archivo: {avg_tok:,.0f}")
    print(f"{prefix}" + "=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# ── Punto de entrada ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reconstrucción OCR en lote usando la API de DeepSeek."
    )
    parser.add_argument(
        "--worker",
        type=int,
        default=0,
        metavar="N",
        help="Índice de este worker (0-based). Por defecto: 0",
    )
    parser.add_argument(
        "--total-workers",
        type=int,
        default=1,
        metavar="T",
        help="Número total de workers en paralelo. Por defecto: 1",
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
        help=(
            "Procesar solo hasta el documento número N (índice 0-based, exclusivo). "
            "Si se omite, procesa hasta el final. "
            "Ej: --hasta 2000  →  procesa los archivos [--desde, 2000)"
        ),
    )
    parser.add_argument(
        "--max-tokens-session",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Máximo de tokens a gastar en esta sesión antes de detenerse. "
            "Útil para controlar el gasto diario. "
            "Ej: --max-tokens-session 1000000  (≈ $0.30 a precios actuales)"
        ),
    )
    parser.add_argument(
        "--max-rpm",
        type=int,
        default=MAX_RPM,
        metavar="N",
        help=f"Límite de requests por minuto compartido entre workers. Por defecto: {MAX_RPM}",
    )
    parser.add_argument(
        "--max-tpm",
        type=int,
        default=MAX_TPM,
        metavar="N",
        help=f"Límite de tokens por minuto compartido entre workers. Por defecto: {MAX_TPM:,}",
    )
    parser.add_argument(
        "--max-tph",
        type=int,
        default=MAX_TPH,
        metavar="N",
        help=f"Presupuesto de tokens por hora. Por defecto: {MAX_TPH:,}",
    )
    args = parser.parse_args()

    if args.worker >= args.total_workers:
        parser.error(
            f"--worker ({args.worker}) debe ser menor que --total-workers ({args.total_workers})"
        )

    # Actualizar límites globales del rate limiter con los args
    MAX_RPM = args.max_rpm
    MAX_TPM = args.max_tpm
    MAX_TPH = args.max_tph

    main(args)
