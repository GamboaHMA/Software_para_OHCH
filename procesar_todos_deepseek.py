"""
Procesamiento en lote de todos los .txt en TXT_ohcbh_cf_erl/ usando DeepSeek Chat
vía Selenium (interfaz web gratuita).

Por cada archivo:
  - Aplica limpiar_espacios_basicos al texto crudo
  - Divide en fragmentos si supera MAX_CHARS_PER_QUERY
  - Envía cada fragmento a chat.deepseek.com con un prompt de reconstrucción OCR
  - Guarda el texto reconstruido en Reconstrucciones_qwen_deepseek/<subcarpeta>/<archivo>.txt
  - Guarda metadata en Reconstrucciones_qwen_deepseek/<subcarpeta>/<archivo>.meta.json

Se puede reanudar: si el .txt de salida ya existe se salta ese archivo.

Dependencias:
    pip install undetected-chromedriver pyperclip

Nota: Si los selectores de DeepSeek dejan de funcionar tras una actualización de la UI,
ajustar las constantes de la sección "── Selectores CSS ──" al inicio del archivo.

Soluciones implementadas:
  - undetected-chromedriver  → evita la detección de bot
  - clipboard (xclip/pyperclip) + Ctrl+V → activa el botón de envío en React
  - wait_for_human()         → detecta login/CAPTCHA y pausa hasta resolución manual
"""

import argparse
import datetime
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
OUTPUT_ROOT = Path("Reconstrucciones_qwen_deepseek")
SESSIONS_LOG_ROOT = Path("sesiones_deepseek")
DEEPSEEK_URL = "https://chat.deepseek.com/"

# ══════════════════════════════════════════════════════════════════════════════
# ── Parámetros de procesamiento ───────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
# DeepSeek Chat (V3/R1) admite contextos de 64 K tokens en la UI web.
# ~8 000 chars ≈ 2 000–2 500 tokens, cómodo para un único turno de reconstrucción.
MAX_CHARS_PER_QUERY = 8_000
# Tiempo máximo (s) esperando que la respuesta deje de cambiar
RESPONSE_TIMEOUT    = 240
# Segundos de pausa después de enviar antes de empezar a leer
DELAY_AFTER_SUBMIT  = 4
# Segundos de espera entre archivos (evita rate-limiting)
DELAY_BETWEEN_FILES = 3
# Número de verificaciones estables seguidas antes de declarar respuesta completa
STABLE_POLLS_NEEDED = 4
# Segundos entre cada sondeo de estabilidad
POLL_INTERVAL       = 2

# ══════════════════════════════════════════════════════════════════════════════
# ── Selectores CSS (actualizar aquí si cambia la UI de DeepSeek) ──────────────
# ══════════════════════════════════════════════════════════════════════════════
# Área de entrada de texto (se prueban en orden hasta encontrar uno válido)
INPUT_SELECTORS = [
    "textarea#chat-input",
    "textarea[placeholder]",
    "[contenteditable='true'][role='textbox']",
    "[contenteditable='true']",
    "textarea",
]

# Contenedor(es) de la respuesta generada
RESPONSE_SELECTORS = [
    "[class*='ds-markdown']",
    "[class*='markdown']",
    "[class*='prose']",
    ".prose",
    "[data-testid*='message']",
    "[class*='message-content']",
    "[class*='chat-message']",
    "[class*='response']",
    "[class*='assistant']",
]

# Botón de envío del mensaje
SUBMIT_SELECTORS = [
    "button[aria-label*='Send']",
    "button[aria-label*='send']",
    "button[aria-label*='Submit']",
    "button[aria-label*='Enviar']",
    "[data-testid*='send']",
    "[data-testid*='submit']",
    "button[type='submit']",
    "div[role='button'][class*='send']",
]

# Pestaña/botón de modo "Expert" / "DeepThink (R1)" (razonamiento extendido)
EXPERT_MODE_SELECTORS = [
    # Botón de texto exacto más probable en la UI de DeepSeek
    "button[class*='expert']",
    "div[class*='expert']",
    "[data-testid*='expert']",
    # DeepThink R1 aparece como toggle con ese texto
    "button[class*='deepthink']",
    "div[class*='deepthink']",
    "[data-testid*='deepthink']",
    # Fallback: cualquier elemento clicable cuyo texto visible sea "Expert"
    # (se evalúa por texto en _select_expert_mode, no por selector solo)
]

# Indicador de que la respuesta todavía se está generando (botón "detener")
STILL_GENERATING_SELECTORS = [
    "button[aria-label*='Stop']",
    "button[aria-label*='stop']",
    "button[aria-label*='Detener']",
    "[data-testid*='stop']",
    "button[class*='stop']",
    "div[class*='stop']",
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
                current.append(sent)
                current_len += needed_s
        else:
            current.append(para)
            current_len += needed

    if current:
        chunks.append("\n\n".join(current))

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# ── Utilidades de portapapeles ────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
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
    "please login",
    "please log in",
]


def _copy_to_clipboard(text: str) -> bool:
    """Copia text al portapapeles del sistema. Devuelve True si tuvo éxito."""
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
    if _PYPERCLIP:
        try:
            pyperclip.copy(text)
            return True
        except Exception:
            pass
    return False


# ══════════════════════════════════════════════════════════════════════════════
# ── Clase DeepSeekBot ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
class DeepSeekBot:
    """Controlador undetected-chromedriver para interactuar con chat.deepseek.com."""

    def __init__(self, headless: bool = False, expert: bool = False):
        self.headless = headless
        self.expert   = expert
        self.driver = None

    # ── Ciclo de vida ──────────────────────────────────────────────────────────
    def start(self):
        options = uc.ChromeOptions()
        if self.headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1400,900")

        driver_path = None
        try:
            from webdriver_manager.chrome import ChromeDriverManager
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
        (CAPTCHA, pantalla de login, etc.)."""
        try:
            title = self.driver.title.lower()
            body  = self.driver.find_element(By.TAG_NAME, "body").text.lower()
        except Exception:
            return False
        combined = title + " " + body[:2000]
        return any(keyword in combined for keyword in _HUMAN_REQUIRED_TEXTS)

    def wait_for_human(self, context: str = ""):
        """Si la página requiere intervención, pausa hasta que el usuario resuelva
        el CAPTCHA / inicie sesión y la página vuelva a ser DeepSeek normal."""
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
        time.sleep(3)
        self._dismiss_overlays()

    # ── Navegación ─────────────────────────────────────────────────────────────
    def _select_expert_mode(self):
        """Activa la pestaña/opción 'Expert' (DeepThink R1) en el chat actual.
        Solo se ejecuta si self.expert es True (--modo expert)."""
        if not self.expert:
            return
        # 1. Intentar por selector CSS directo
        for sel in EXPERT_MODE_SELECTORS:
            try:
                el = self.driver.find_element(By.CSS_SELECTOR, sel)
                if el.is_displayed():
                    el.click()
                    time.sleep(0.5)
                    print("    ✓ Modo Expert activado (selector CSS)", flush=True)
                    return
            except (NoSuchElementException, Exception):
                continue
        # 2. Buscar por texto visible en botones/divs clicables
        for tag in ("button", "div", "span", "a"):
            try:
                candidates = self.driver.find_elements(By.TAG_NAME, tag)
                for el in candidates:
                    try:
                        txt = el.text.strip().lower()
                        if txt in ("expert", "deepthink", "deepthink (r1)", "deep think"):
                            if el.is_displayed():
                                el.click()
                                time.sleep(0.5)
                                print(
                                    f"    ✓ Modo Expert activado (texto '{el.text.strip()}')",
                                    flush=True,
                                )
                                return
                    except (StaleElementReferenceException, Exception):
                        continue
            except Exception:
                continue
        print("    ⚠ No se encontró el botón Expert/DeepThink; continuando sin él.",
              flush=True)

    def open_home(self):
        """Navega a la página principal para iniciar una conversación nueva."""
        self.driver.get(DEEPSEEK_URL)
        time.sleep(4)
        self.wait_for_human("página de inicio")
        self._dismiss_overlays()
        self._select_expert_mode()

    def new_chat(self):
        """Inicia un nuevo chat para no acumular contexto entre archivos."""
        for selector in [
            "button[aria-label*='New chat']",
            "button[aria-label*='nuevo']",
            "a[href='/']",
            "[data-testid*='new-chat']",
            "[class*='new-chat']",
            "button[class*='new']",
        ]:
            try:
                el = self.driver.find_element(By.CSS_SELECTOR, selector)
                el.click()
                time.sleep(2)
                self.wait_for_human("nuevo chat")
                self._dismiss_overlays()
                self._select_expert_mode()
                return
            except (NoSuchElementException, Exception):
                continue
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
            "[data-testid*='close']",
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
            "No se encontró el campo de entrada en DeepSeek. "
            "Revisa INPUT_SELECTORS si la UI cambió."
        )

    def _set_input_value(self, element, text: str):
        """Pega el texto en el campo de entrada vía portapapeles del sistema."""
        element.click()
        time.sleep(0.2)

        ActionChains(self.driver).key_down(Keys.CONTROL).send_keys("a").key_up(
            Keys.CONTROL
        ).send_keys(Keys.DELETE).perform()
        time.sleep(0.1)

        ok = _copy_to_clipboard(text)

        if ok:
            ActionChains(self.driver).key_down(Keys.CONTROL).send_keys("v").key_up(
                Keys.CONTROL
            ).perform()
            time.sleep(0.3)
        else:
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

    def _click_submit(self) -> bool:
        """Intenta hacer clic en el botón de envío. Devuelve True si tuvo éxito."""
        for sel in SUBMIT_SELECTORS:
            try:
                el = self.driver.find_element(By.CSS_SELECTOR, sel)
                if el.is_displayed() and el.is_enabled():
                    el.click()
                    return True
            except (NoSuchElementException, StaleElementReferenceException, Exception):
                continue
        return False

    def _wait_for_response(self) -> str:
        """Espera a que la respuesta termine de generarse y la devuelve."""
        attempt = 0

        while True:
            attempt += 1
            deadline = time.time() + RESPONSE_TIMEOUT
            last_text = ""
            stable_count = 0

            time.sleep(DELAY_AFTER_SUBMIT)

            while time.time() < deadline:
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

            print(
                f"    ⚠ Timeout (intento {attempt}): sin respuesta. "
                f"Intentando clic en botón de envío…",
                flush=True,
            )
            self.wait_for_human(f"timeout intento {attempt}")
            clicked = self._click_submit()
            if clicked:
                print("    ↺ Botón pulsado, esperando respuesta…", flush=True)
            else:
                print(
                    f"    ⚠ Botón no encontrado, reintentando en {RESPONSE_TIMEOUT}s…",
                    flush=True,
                )

    # ── Método principal ───────────────────────────────────────────────────────
    def query(self, text: str) -> str:
        """Envía un prompt y devuelve la respuesta completa de DeepSeek."""
        self.wait_for_human("antes de enviar consulta")

        input_el = self._find_input()
        self._set_input_value(input_el, text)

        time.sleep(0.6)

        if not self._click_submit():
            input_el.send_keys(Keys.RETURN)

        time.sleep(1)
        self.wait_for_human("después de enviar consulta")

        return self._wait_for_response()

    def process_text(self, raw_text: str) -> dict:
        """Limpia, divide en fragmentos si es necesario, consulta DeepSeek y
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
# ── Registro de sesión por correo ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def _save_session_log(
    correo: str,
    worker: int,
    consultas_realizadas: int,
    max_consultas: int,
    limite_alcanzado: bool,
) -> None:
    """Guarda un JSON con el estado del correo al terminar la sesión."""
    SESSIONS_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    ahora     = datetime.datetime.now()
    restantes = max(0, max_consultas - consultas_realizadas)
    entrada   = {
        "correo":                  correo,
        "worker":                  worker,
        "hora_fin":                ahora.isoformat(timespec="seconds"),
        "consultas_realizadas":    consultas_realizadas,
        "max_consultas":           max_consultas,
        "consultas_restantes_dia": restantes,
        "limite_alcanzado":        limite_alcanzado,
    }
    correo_safe = correo.replace("@", "_at_").replace("/", "_").replace("\\", "_")
    nombre = f"{correo_safe}_W{worker}_{ahora.strftime('%Y-%m-%d_%H-%M-%S')}.json"
    ruta   = SESSIONS_LOG_ROOT / nombre
    ruta.write_text(json.dumps(entrada, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRegistro guardado → {ruta}", flush=True)
    print(
        f"  Correo            : {correo}\n"
        f"  Consultas hechas  : {consultas_realizadas} / {max_consultas}\n"
        f"  Restantes hoy     : {restantes}\n"
        f"  Límite alcanzado  : {'sí' if limite_alcanzado else 'no'}",
        flush=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ── Bucle principal ───────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Reconstrucción OCR en lote usando DeepSeek Chat"
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
    parser.add_argument(
        "--desde",
        type=int,
        default=0,
        metavar="M",
        help=(
            "Empezar desde el documento número M de la lista global ordenada "
            "(índice 0-based, inclusivo). Usar junto con --hasta de salamander para "
            "repartir el trabajo sin solapamientos. Ej: --desde 50"
        ),
    )
    parser.add_argument(
        "--modo",
        choices=["instant", "expert"],
        default="instant",
        help=(
            "Modo de respuesta de DeepSeek: 'instant' (por defecto, respuesta rápida) "
            "o 'expert' (activa DeepThink R1, razonamiento extendido, más lento)."
        ),
    )
    parser.add_argument(
        "--correo",
        type=str,
        default="sin_especificar",
        metavar="EMAIL",
        help="Correo de la cuenta DeepSeek usada en esta terminal. Se registra en el JSON de sesión.",
    )
    parser.add_argument(
        "--max-consultas",
        type=int,
        default=50,
        metavar="N",
        help=(
            "Máximo de consultas a realizar en esta sesión antes de detenerse y "
            "guardar el registro (por defecto 50, límite diario aproximado de la capa "
            "gratuita de DeepSeek por cuenta)."
        ),
    )
    args = parser.parse_args()

    if args.worker >= args.total_workers:
        raise ValueError(
            f"--worker ({args.worker}) debe ser menor que --total-workers ({args.total_workers})"
        )

    stats = {"procesados": 0, "saltados": 0, "errores": 0, "tiempo_total": 0.0}

    all_txt_global = sorted(INPUT_ROOT.rglob("*.txt"))
    all_txt = all_txt_global[args.desde:] if args.desde > 0 else all_txt_global

    # Cada worker toma los archivos en posiciones: worker, worker+T, worker+2T, …
    # Esto distribuye la carga uniformemente y elimina toda condición de carrera.
    worker_txt = all_txt[args.worker :: args.total_workers]
    total_global = len(all_txt_global)
    total_rango  = len(all_txt)
    total_worker = len(worker_txt)

    prefix = f"[W{args.worker}]" if args.total_workers > 1 else ""
    rango_str = f"desde #{args.desde}" if args.desde > 0 else "desde #0"
    print(
        f"{prefix} Archivos globales: {total_global} | "
        f"Rango ({rango_str}): {total_rango} | "
        f"Asignados a este worker: {total_worker}\n",
        flush=True,
    )
    print(f"{prefix} Abriendo navegador… (modo: {args.modo} | correo: {args.correo} | max consultas: {args.max_consultas})", flush=True)

    bot = DeepSeekBot(headless=False, expert=(args.modo == "expert"))
    bot.start()
    bot.open_home()
    print(f"{prefix} Navegador listo. Iniciando procesamiento…\n", flush=True)

    consultas_totales  = 0
    limite_alcanzado   = False

    try:
        for idx, src_path in enumerate(worker_txt, 1):
            # Verificar límite ANTES de iniciar el siguiente archivo
            if consultas_totales >= args.max_consultas:
                limite_alcanzado = True
                print(
                    f"\n{prefix} Límite de {args.max_consultas} consultas alcanzado "
                    f"({consultas_totales} realizadas). Deteniendo este worker.",
                    flush=True,
                )
                break

            rel      = src_path.relative_to(INPUT_ROOT)
            dst_txt  = OUTPUT_ROOT / rel
            dst_meta = dst_txt.with_suffix(".meta.json")

            if dst_txt.exists():
                stats["saltados"] += 1
                continue

            print(f"saltados: {stats['saltados']}")

            dst_txt.parent.mkdir(parents=True, exist_ok=True)
            print(f"{prefix} [{idx}/{total_worker}] {rel}", flush=True)

            t0 = time.perf_counter()
            try:
                raw_text = src_path.read_text(encoding="utf-8", errors="replace")

                result = bot.process_text(raw_text)

                consultas_totales += result["chunks_used"]
                elapsed  = time.perf_counter() - t0

                dst_txt.write_text(result["reconstructed_text"], encoding="utf-8")

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

                dst_meta.write_text(
                    json.dumps({"error": error_msg}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            if idx < total_worker:
                bot.new_chat()
                time.sleep(DELAY_BETWEEN_FILES)

    finally:
        bot.quit()
        _save_session_log(
            correo=args.correo,
            worker=args.worker,
            consultas_realizadas=consultas_totales,
            max_consultas=args.max_consultas,
            limite_alcanzado=limite_alcanzado,
        )

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
