"""
Procesamiento en lote de todos los .txt en TXT_ohcbh_cf_erl/ usando Perplexity.ai
vía Selenium (interfaz web gratuita).

Por cada archivo:
  - Aplica limpiar_espacios_basicos al texto crudo
  - Divide en fragmentos si supera MAX_CHARS_PER_QUERY
  - Envía cada fragmento a Perplexity.ai con un prompt de reconstrucción OCR
  - Guarda el texto reconstruido en Reconstrucciones_perplexity/<subcarpeta>/<archivo>.txt
  - Guarda metadata en Reconstrucciones_perplexity/<subcarpeta>/<archivo>.meta.json

Se puede reanudar: si el .txt de salida ya existe se salta ese archivo.

Dependencias:
    pip install undetected-chromedriver pyperclip

Nota: Si los selectores de Perplexity dejan de funcionar tras una actualización de la UI,
ajustar las constantes de la sección "── Selectores CSS ──" al inicio del archivo.

Soluciones implementadas:
  - undetected-chromedriver  → evita la detección de Cloudflare
  - clipboard (xclip/pyperclip) + Ctrl+V → activa el botón de envío en React
  - wait_for_human()         → detecta login/Cloudflare y pausa hasta resolución manual
"""

import argparse
import json
import subprocess
import time
import traceback
from pathlib import Path

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    StaleElementReferenceException,
)

try:
    import pyperclip
    _PYPERCLIP = True
except ImportError:
    _PYPERCLIP = False

from anomalies_checker import limpiar_espacios_basicos

# ══════════════════════════════════════════════════════════════════════════════
# ── Rutas ─────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
INPUT_ROOT  = Path("TXT_ohcbh_cf_erl")
OUTPUT_ROOT = Path("Reconstrucciones_perplexity")
PERPLEXITY_URL = "https://www.perplexity.ai/"

# ══════════════════════════════════════════════════════════════════════════════
# ── Parámetros de procesamiento ───────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
# Máximo de caracteres por consulta. Perplexity acepta ~4 000–6 000 chars en la UI.
MAX_CHARS_PER_QUERY = 4_000
# Tiempo máximo (s) esperando que la respuesta deje de cambiar
RESPONSE_TIMEOUT    = 180
# Segundos de pausa después de enviar antes de empezar a leer
DELAY_AFTER_SUBMIT  = 3
# Segundos de espera entre archivos (evita rate-limiting)
DELAY_BETWEEN_FILES = 2
# Número de verificaciones estables seguidas antes de declarar respuesta completa
STABLE_POLLS_NEEDED = 4
# Segundos entre cada sondeo de estabilidad
POLL_INTERVAL       = 2

# ══════════════════════════════════════════════════════════════════════════════
# ── Selectores CSS (actualizar aquí si cambia la UI de Perplexity) ────────────
# ══════════════════════════════════════════════════════════════════════════════
# Área de entrada de texto (se prueban en orden hasta encontrar uno válido)
INPUT_SELECTORS = [
    # Perplexity 2024-2026 usa un contenteditable como campo principal
    "[contenteditable='true'][role='textbox']",
    "[contenteditable='true']",
    "textarea[placeholder]",
    "textarea",
]

# Contenedor(es) de la respuesta generada
RESPONSE_SELECTORS = [
    "[class*='prose']",
    ".prose",
    "[data-testid*='answer']",
    "[data-testid*='response']",
    "[class*='answer']",
    "[class*='response']",
    "div.col-span-8",          # fallback estructural
]

# Indicador de que la respuesta todavía se está generando (botón "detener")
STILL_GENERATING_SELECTORS = [
    "button[aria-label*='Stop']",
    "button[aria-label*='stop']",
    "button[aria-label*='Detener']",
    "[data-testid*='stop']",
    "button[class*='stop']",
]

# ══════════════════════════════════════════════════════════════════════════════
# ── Prompt de reconstrucción OCR ──────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
_PROMPT_INSTRUCCIONES = (
    "Eres un experto en transcripción y corrección de documentos históricos en español. "
    "El siguiente texto fue extraído mediante OCR de un documento histórico impreso y "
    "contiene errores típicos de OCR: caracteres aislados sin sentido (letras o símbolos "
    "sueltos en líneas propias), palabras partidas con guión al final de línea que deben "
    "unirse, espacios incorrectos dentro de palabras, confusión entre caracteres similares "
    "(l/1, o/0, rn/m, cl/d, etc.) y signos de puntuación mal reconocidos. "
    "Tu tarea es reconstruir el texto original corrigiendo únicamente los errores de OCR. "
    "Conserva el español histórico, la ortografía de la época (tildes arcaicas, formas "
    "antiguas) y el contenido exacto: NO añadas, inventes ni elimines información. "
    "Devuelve ÚNICAMENTE el texto reconstruido, sin explicaciones, sin comentarios, "
    "sin encabezados, sin delimitadores de bloque de código."
)


def _build_prompt(text: str) -> str:
    return f"{_PROMPT_INSTRUCCIONES}\n\nTEXTO OCR:\n{text}"


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
# ── Clase PerplexityBot ───────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
# ── Textos que indican que la página requiere intervención humana ─────────────
_HUMAN_REQUIRED_TEXTS = [
    "verify you are human",
    "verifica que eres humano",
    "sign in",
    "inicia sesión",
    "log in",
    "iniciar sesión",
    "create account",
    "just a moment",
    "checking your browser",
    "enable javascript",
    "cloudflare",
    "access denied",
]


def _copy_to_clipboard(text: str) -> bool:
    """Copia text al portapapeles del sistema. Devuelve True si tuvo éxito."""
    # Intentar xclip (Linux)
    try:
        subprocess.run(
            ["xclip", "-selection", "clipboard"],
            input=text.encode("utf-8"),
            check=True,
            timeout=5,
        )
        return True
    except Exception:
        pass
    # Intentar xsel
    try:
        subprocess.run(
            ["xsel", "--clipboard", "--input"],
            input=text.encode("utf-8"),
            check=True,
            timeout=5,
        )
        return True
    except Exception:
        pass
    # pyperclip como último recurso
    if _PYPERCLIP:
        try:
            pyperclip.copy(text)
            return True
        except Exception:
            pass
    return False


class PerplexityBot:
    """Controlador undetected-chromedriver para interactuar con Perplexity.ai."""

    def __init__(self, headless: bool = False):
        self.headless = headless
        self.driver = None

    # ── Ciclo de vida ──────────────────────────────────────────────────────────
    def start(self):
        options = uc.ChromeOptions()
        # headless=True activa el modo sin ventana; normalmente hay que dejarlo False
        # para poder resolver captchas / iniciar sesión manualmente.
        if self.headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1400,900")

        # Obtener la ruta del ChromeDriver vía webdriver-manager (gestiona la descarga
        # con reintentos) y pasársela a uc.Chrome para que lo parchee sin descargarlo
        # de nuevo (evita el error "retrieval incomplete").
        driver_path = None
        try:
            from webdriver_manager.chrome import ChromeDriverManager
            from webdriver_manager.core.os_manager import ChromeType
            driver_path = ChromeDriverManager().install()
        except Exception as e:
            print(f"    ⚠ webdriver-manager no pudo obtener el driver: {e}", flush=True)
            print("      Se intentará sin ruta explícita…", flush=True)

        self.driver = uc.Chrome(
            options=options,
            driver_executable_path=driver_path,
            use_subprocess=True,
        )

    def quit(self):
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None

    # ── Detección de bloqueos ──────────────────────────────────────────────────
    def _page_needs_human(self) -> bool:
        """Devuelve True si la página actual requiere intervención humana
        (Cloudflare CAPTCHA, pantalla de login, etc.)."""
        try:
            title = self.driver.title.lower()
            body  = self.driver.find_element(By.TAG_NAME, "body").text.lower()
        except Exception:
            return False
        combined = title + " " + body[:2000]
        return any(keyword in combined for keyword in _HUMAN_REQUIRED_TEXTS)

    def wait_for_human(self, context: str = ""):
        """Si la página requiere intervención, pausa hasta que el usuario resuelva
        el CAPTCHA / inicie sesión y la página vuelva a ser Perplexity normal."""
        if not self._page_needs_human():
            return
        msg = (
            f"\n{'='*60}\n"
            f"ACCIÓN REQUERIDA{' – ' + context if context else ''}:\n"
            f"  El navegador muestra un CAPTCHA o solicitud de login.\n"
            f"  Resuélvelo manualmente en la ventana del navegador y\n"
            f"  luego pulsa ENTER aquí para continuar.\n"
            f"{'='*60}"
        )
        print(msg, flush=True)
        input("  Pulsa ENTER cuando estés listo…")
        # Esperar a que la página cargue tras la acción manual
        time.sleep(3)
        self._dismiss_overlays()

    # ── Navegación ─────────────────────────────────────────────────────────────
    def open_home(self):
        """Navega a la página principal para iniciar una conversación nueva."""
        self.driver.get(PERPLEXITY_URL)
        time.sleep(3)
        self.wait_for_human("página de inicio")
        self._dismiss_overlays()

    def new_chat(self):
        """Inicia un nuevo chat para no acumular contexto entre archivos."""
        # Intentar hacer clic en botón "Nuevo chat" / "New Thread"
        for selector in [
            "a[href='/']",
            "button[aria-label*='New']",
            "button[aria-label*='nuevo']",
            "[data-testid*='new-thread']",
            "[href*='new']",
        ]:
            try:
                el = self.driver.find_element(By.CSS_SELECTOR, selector)
                el.click()
                time.sleep(2)
                self.wait_for_human("nuevo chat")
                self._dismiss_overlays()
                return
            except (NoSuchElementException, Exception):
                continue
        # Fallback: recargar la página raíz
        self.open_home()

    def _dismiss_overlays(self):
        """Cierra modales de cookies / bienvenida si aparecen."""
        dismiss_selectors = [
            "button[id*='accept']",
            "button[class*='accept']",
            "button[aria-label*='Accept']",
            "button[aria-label*='Close']",
            "button[aria-label*='Cerrar']",
            "[data-testid*='dismiss']",
        ]
        for sel in dismiss_selectors:
            try:
                self.driver.find_element(By.CSS_SELECTOR, sel).click()
                time.sleep(0.5)
            except (NoSuchElementException, Exception):
                pass

    # ── Entrada ────────────────────────────────────────────────────────────────
    def _find_input(self, timeout: int = 20):
        """Devuelve el elemento de entrada de texto."""
        wait = WebDriverWait(self.driver, timeout)
        for selector in INPUT_SELECTORS:
            try:
                el = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
                return el
            except TimeoutException:
                continue
        raise RuntimeError(
            "No se encontró el campo de entrada en Perplexity. "
            "Revisa INPUT_SELECTORS si la UI cambió."
        )

    def _set_input_value(self, element, text: str):
        """Pega el texto en el campo de entrada vía portapapeles del sistema.

        Usar xclip/xsel/pyperclip + Ctrl+V es la única estrategia que:
          - Funciona tanto en <textarea> como en contenteditable
          - Dispara los eventos sintéticos nativos del navegador que React escucha
          - Activa el botón de envío sin requerir ningún hack JS adicional
        """
        element.click()
        time.sleep(0.2)

        # Limpiar el campo primero (Ctrl+A → Delete)
        ActionChains(self.driver).key_down(Keys.CONTROL).send_keys("a").key_up(
            Keys.CONTROL
        ).send_keys(Keys.DELETE).perform()
        time.sleep(0.1)

        # Copiar al portapapeles del sistema
        ok = _copy_to_clipboard(text)

        if ok:
            # Pegar: esto dispara el evento 'paste' nativo → React actualiza el estado
            ActionChains(self.driver).key_down(Keys.CONTROL).send_keys("v").key_up(
                Keys.CONTROL
            ).perform()
            time.sleep(0.3)
        else:
            # Fallback: send_keys (muy lento para textos largos, pero funciona)
            print("    ⚠ Clipboard no disponible, usando send_keys (lento)…", flush=True)
            element.send_keys(text)

    # ── Extracción de respuesta ────────────────────────────────────────────────
    def _is_still_generating(self) -> bool:
        """Devuelve True si el botón de detener (streaming) es visible."""
        for sel in STILL_GENERATING_SELECTORS:
            try:
                el = self.driver.find_element(By.CSS_SELECTOR, sel)
                if el.is_displayed():
                    return True
            except (NoSuchElementException, StaleElementReferenceException):
                pass
        return False

    def _get_response_text(self) -> str:
        """Extrae el texto de la última respuesta visible."""
        for sel in RESPONSE_SELECTORS:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, sel)
                if elements:
                    return elements[-1].text.strip()
            except (NoSuchElementException, StaleElementReferenceException):
                continue
        return ""

    def _wait_for_response(self) -> str:
        """Espera a que la respuesta termine de generarse y la devuelve."""
        deadline = time.time() + RESPONSE_TIMEOUT
        last_text = ""
        stable_count = 0

        # Dar tiempo al modelo para empezar a responder
        time.sleep(DELAY_AFTER_SUBMIT)

        while time.time() < deadline:
            # Si hay indicador de "generando", esperar más
            if self._is_still_generating():
                stable_count = 0
                time.sleep(POLL_INTERVAL)
                last_text = self._get_response_text()
                continue

            current_text = self._get_response_text()
            if current_text and current_text == last_text:
                stable_count += 1
                if stable_count >= STABLE_POLLS_NEEDED:
                    return current_text
            else:
                stable_count = 0
                last_text = current_text

            time.sleep(POLL_INTERVAL)

        # Timeout: devolver lo que tengamos
        print("    ⚠ Timeout esperando respuesta; se guarda lo obtenido hasta ahora.",
              flush=True)
        return last_text or ""

    # ── Método principal ───────────────────────────────────────────────────────
    def query(self, text: str) -> str:
        """Envía un prompt y devuelve la respuesta completa de Perplexity."""
        # Verificar antes de cada consulta si la página requiere acción humana
        self.wait_for_human("antes de enviar consulta")

        input_el = self._find_input()
        self._set_input_value(input_el, text)

        # Pequeña pausa para que React habilite el botón de envío
        time.sleep(0.6)

        # Intentar enviar con Enter (más fiable que buscar el botón)
        input_el.send_keys(Keys.RETURN)

        # Verificar de nuevo por si el Enter disparó un login wall
        time.sleep(1)
        self.wait_for_human("después de enviar consulta")

        return self._wait_for_response()

    def process_text(self, raw_text: str) -> dict:
        """Limpia, divide en fragmentos si es necesario, consulta Perplexity y
        devuelve un dict con 'reconstructed_text' y 'chunks_used'."""
        cleaned = limpiar_espacios_basicos(raw_text)
        fragments = chunk_text(cleaned)
        reconstructed_parts = []

        for i, fragment in enumerate(fragments):
            if len(fragments) > 1:
                print(f"      fragmento {i + 1}/{len(fragments)} "
                      f"({len(fragment)} chars)…", flush=True)
            prompt = _build_prompt(fragment)
            response = self.query(prompt)
            reconstructed_parts.append(response)
            if i < len(fragments) - 1:
                self.new_chat()
                time.sleep(3)

        return {
            "reconstructed_text": "\n\n".join(reconstructed_parts),
            "chunks_used": len(fragments),
        }


# ══════════════════════════════════════════════════════════════════════════════
# ── Bucle principal ───────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Reconstrucción OCR en lote usando Perplexity.ai"
    )
    parser.add_argument(
        "--worker",
        type=int,
        default=0,
        metavar="N",
        help="Índice de este worker (0-based). Ej: 0, 1, 2, 3",
    )
    parser.add_argument(
        "--total-workers",
        type=int,
        default=1,
        metavar="T",
        help="Número total de workers en paralelo. Ej: 4",
    )
    args = parser.parse_args()

    if args.worker >= args.total_workers:
        raise ValueError(
            f"--worker ({args.worker}) debe ser menor que --total-workers ({args.total_workers})"
        )

    stats = {"procesados": 0, "saltados": 0, "errores": 0, "tiempo_total": 0.0}

    all_txt = sorted(INPUT_ROOT.rglob("*.txt"))

    # Cada worker toma los archivos en posiciones: worker, worker+T, worker+2T, …
    # Esto distribuye la carga uniformemente y elimina toda condición de carrera.
    worker_txt = all_txt[args.worker :: args.total_workers]
    total_global = len(all_txt)
    total_worker = len(worker_txt)

    prefix = f"[W{args.worker}]" if args.total_workers > 1 else ""
    print(
        f"{prefix} Archivos totales: {total_global} | "
        f"Asignados a este worker: {total_worker}\n",
        flush=True,
    )
    print(f"{prefix} Abriendo navegador…", flush=True)

    bot = PerplexityBot(headless=False)
    bot.start()
    bot.open_home()
    print(f"{prefix} Navegador listo. Iniciando procesamiento…\n", flush=True)

    try:
        for idx, src_path in enumerate(worker_txt, 1):
            rel      = src_path.relative_to(INPUT_ROOT)
            dst_txt  = OUTPUT_ROOT / rel
            dst_meta = dst_txt.with_suffix(".meta.json")

            # Reanudar: saltar si ya está procesado
            if dst_txt.exists():
                stats["saltados"] += 1
                continue

            dst_txt.parent.mkdir(parents=True, exist_ok=True)
            print(f"{prefix} [{idx}/{total_worker}] {rel}", flush=True)

            t0 = time.perf_counter()
            try:
                raw_text = src_path.read_text(encoding="utf-8", errors="replace")

                result = bot.process_text(raw_text)

                elapsed = time.perf_counter() - t0

                # Guardar texto reconstruido
                dst_txt.write_text(result["reconstructed_text"], encoding="utf-8")

                # Guardar metadata
                meta = {
                    "source": str(src_path),
                    "chunks_used": result["chunks_used"],
                    "elapsed_s": round(elapsed, 2),
                }
                dst_meta.write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
                )

                stats["procesados"] += 1
                stats["tiempo_total"] += elapsed
                print(f"{prefix}     OK  ({elapsed:.1f}s)", flush=True)

            except Exception:
                elapsed = time.perf_counter() - t0
                stats["errores"] += 1
                error_msg = traceback.format_exc()
                print(f"    ERROR:\n{error_msg}", flush=True)

                # Guardar traza para revisar después
                dst_meta.write_text(
                    json.dumps({"error": error_msg}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            # Abrir nuevo chat para el siguiente archivo y esperar
            if idx < total_worker:
                bot.new_chat()
                time.sleep(DELAY_BETWEEN_FILES)

    finally:
        bot.quit()

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
    main()
