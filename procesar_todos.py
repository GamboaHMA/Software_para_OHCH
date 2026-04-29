"""
Procesamiento en lote de todos los .txt en TXT_ohcbh_cf_erl/.
Por cada archivo:
  - Aplica limpiar_espacios_basicos + StringChecker.correjir_cadena
  - Guarda el texto reconstruido en Reconstrucciones/<subcarpeta>/<archivo>.txt
  - Guarda el log de correcciones en Reconstrucciones/<subcarpeta>/<archivo>.log.json

Se puede reanudar: si el .txt de salida ya existe se salta ese archivo.
"""

import os
import json
import time
import traceback
from pathlib import Path

# Suprimir los print de depuracion internos de StringChecker
import builtins
_real_print = builtins.print


def _silent_print(*args, **kwargs):
    pass


from anomalies_checker import StringChecker, limpiar_espacios_basicos

INPUT_ROOT = Path("TXT_ohcbh_cf_erl")
OUTPUT_ROOT = Path("Reconstrucciones")

# ── Estadisticas globales ──────────────────────────────────────────────────────
stats = {
    "procesados": 0,
    "saltados": 0,
    "errores": 0,
    "tiempo_total": 0.0,
}

print("Cargando modelos…")
checker = StringChecker()
print("Modelos cargados.\n")

# Colectar todos los .txt de entrada
all_txt = sorted(INPUT_ROOT.rglob("*.txt"))
total = len(all_txt)
print(f"Archivos encontrados: {total}\n")

for idx, src_path in enumerate(all_txt, 1):
    # Ruta relativa respecto a INPUT_ROOT para replicar estructura
    rel = src_path.relative_to(INPUT_ROOT)
    dst_txt  = OUTPUT_ROOT / rel
    dst_log  = dst_txt.with_suffix(".log.json")

    # Reanudar: saltar si ya esta procesado
    if dst_txt.exists():
        stats["saltados"] += 1
        continue

    dst_txt.parent.mkdir(parents=True, exist_ok=True)

    print(f"[{idx}/{total}] {rel}", flush=True)

    try:
        raw_text = src_path.read_text(encoding="utf-8", errors="replace")

        cleaned = limpiar_espacios_basicos(raw_text)

        # Silenciar prints internos durante el procesamiento
        builtins.print = _silent_print

        doc = checker.nlp(cleaned)

        anomaly_chain = []
        for token in doc:
            is_valid = checker._spell_cache.get(token.text)
            if is_valid is None:
                is_valid = checker.hobj.spell(token.text)
                checker._spell_cache[token.text] = is_valid
            anomaly_chain.append((token, is_valid))

        # Pre-calentar suggest cache
        for token, valid in anomaly_chain:
            if not valid and token.text not in checker._suggest_cache:
                checker._suggest_cache[token.text] = checker.hobj.suggest(token.text)[:5]

        t0 = time.perf_counter()
        checker.correjir_cadena(anomaly_chain)
        elapsed = time.perf_counter() - t0

        builtins.print = _real_print

        corrected_text = checker.last_corrected_text
        correction_log = checker.correction_log

        # Guardar texto reconstruido
        dst_txt.write_text(corrected_text, encoding="utf-8")

        # Guardar log de correcciones
        dst_log.write_text(
            json.dumps(correction_log, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        stats["procesados"] += 1
        stats["tiempo_total"] += elapsed
        print(f"    OK  ({elapsed:.1f}s)", flush=True)

    except Exception:
        builtins.print = _real_print
        stats["errores"] += 1
        error_msg = traceback.format_exc()
        print(f"    ERROR:\n{error_msg}", flush=True)

        # Guardar traza de error para revisar despues
        dst_log.write_text(
            json.dumps({"error": error_msg}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

# ── Resumen final ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Procesados : {stats['procesados']}")
print(f"Saltados   : {stats['saltados']}")
print(f"Errores    : {stats['errores']}")
if stats["procesados"]:
    avg = stats["tiempo_total"] / stats["procesados"]
    print(f"Tiempo total: {stats['tiempo_total']:.1f}s  |  Promedio: {avg:.1f}s/archivo")
print("=" * 60)
