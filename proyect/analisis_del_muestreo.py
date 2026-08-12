#!/usr/bin/env python3
"""
analisis_muestreo.py
────────────────────
Calcula estadísticas e intervalos de confianza para la muestra de 400 PDFs
clasificados manualmente como 'impresos', 'manuscritos' o 'fotos'.

Uso:
    1. Edita las variables de conteo en la sección "CONFIGURACIÓN" abajo.
    2. Ejecuta: python analisis_muestreo.py

El script genera un informe en consola y lo guarda en 'resultados_muestreo.txt'.
"""

import math
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN — INGRESA AQUÍ TUS CONTEOS MANUALES
# ──────────────────────────────────────────────────────────────────────────

# Población total de documentos (todos los PDFs que tienes)
TOTAL_POBLACION = 15584

# Tamaño total de la muestra revisada (debe ser 400, o el número que hayas usado)
TOTAL_MUESTRA = 400

# --- Conteos TOTALES (sobre toda la muestra de 400) ---
TOTAL_IMPRESOS = 373   # <-- CAMBIA ESTE NÚMERO
TOTAL_MANUSCRITOS = 7 # <-- CAMBIA ESTE NÚMERO
TOTAL_FOTOS = 20       # <-- CAMBIA ESTE NÚMERO

# --- Conteos POR GRUPO (opcional, pero muy recomendado para ver consistencia) ---
# Cada tupla es: (impresos, manuscritos, fotos) para ese grupo de 80.
# Si no separaste los conteos por grupo, puedes dejar la lista vacía: DATOS_GRUPOS = []
DATOS_GRUPOS = [
    (74, 1, 5),   # Grupo 1 (80 documentos)
    (72, 2, 6),   # Grupo 2
    (74, 1, 5),  # Grupo 3
    (77, 1, 2),   # Grupo 4
    (76, 2, 2),   # Grupo 5
]

# Umbral para la prueba de hipótesis (queremos demostrar que la proporción es > UMBRAL)
# Normalmente 0.5 (mayoría simple) o 0.7 (gran mayoría).
UMBRAL_HIPOTESIS = 0.5

# ──────────────────────────────────────────────────────────────────────────
# FUNCIONES ESTADÍSTICAS
# ──────────────────────────────────────────────────────────────────────────

def normal_cdf(x):
    """Función de distribución acumulada de la normal estándar."""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def calcular_estadisticas(impresos, n, N, umbral=0.5):
    """
    Calcula:
        - Proporción muestral (p)
        - Error estándar con corrección por población finita
        - Intervalo de confianza del 95 %
        - Estadístico Z y p-valor para probar H0: p <= umbral vs H1: p > umbral
    """
    p = impresos / n if n > 0 else 0

    # Error estándar (con corrección por población finita)
    se_sin_corregir = math.sqrt(p * (1 - p) / n) if n > 0 and 0 < p < 1 else 0
    fpc = math.sqrt((N - n) / (N - 1)) if N > 1 else 1.0
    se = se_sin_corregir * fpc

    # Intervalo de confianza al 95 % (Z = 1.96)
    z_crit = 1.96
    ci_lower = p - z_crit * se
    ci_upper = p + z_crit * se
    ci_lower = max(0, ci_lower)  # no puede ser menor que 0
    ci_upper = min(1, ci_upper)  # no puede ser mayor que 1

    # Prueba de hipótesis (una cola): H0: p <= umbral
    if se > 0:
        z_stat = (p - umbral) / se
    else:
        z_stat = float('inf') if p > umbral else 0
    p_valor = 1 - normal_cdf(z_stat)  # cola superior

    return {
        'proporcion': p,
        'error_estandar': se,
        'ci_95_lower': ci_lower,
        'ci_95_upper': ci_upper,
        'z_stat': z_stat,
        'p_valor': p_valor,
        'conteo': impresos,
        'n': n
    }

def formatear_porcentaje(valor, decimales=2):
    return f"{valor * 100:.{decimales}f} %"

# ──────────────────────────────────────────────────────────────────────────
# GENERACIÓN DEL INFORME
# ──────────────────────────────────────────────────────────────────────────

def generar_informe():
    # Validación básica
    if TOTAL_IMPRESOS + TOTAL_MANUSCRITOS + TOTAL_FOTOS != TOTAL_MUESTRA:
        print(" ADVERTENCIA: La suma de impresos + manuscritos + fotos no coincide con TOTAL_MUESTRA.")
        print(f"   Suma = {TOTAL_IMPRESOS + TOTAL_MANUSCRITOS + TOTAL_FOTOS}, esperado = {TOTAL_MUESTRA}")
        print("   Se usará la suma como tamaño muestral real.")
        n_real = TOTAL_IMPRESOS + TOTAL_MANUSCRITOS + TOTAL_FOTOS
    else:
        n_real = TOTAL_MUESTRA

    # 1. Estadísticas totales
    stats = calcular_estadisticas(TOTAL_IMPRESOS, n_real, TOTAL_POBLACION, UMBRAL_HIPOTESIS)

    # Preparamos el contenido del informe
    lineas = []
    lineas.append("=" * 70)
    lineas.append("            INFORME ESTADÍSTICO DEL MUESTREO DE PDFs")
    lineas.append("=" * 70)
    lineas.append(f"  Población total (N):          {TOTAL_POBLACION:,} documentos")
    lineas.append(f"  Tamaño de la muestra (n):     {n_real:,} documentos")
    lineas.append(f"  Umbral de prueba (H0):        p ≤ {UMBRAL_HIPOTESIS:.0%}")
    lineas.append("=" * 70)
    lineas.append("")

    # Tabla de conteos totales
    lineas.append("CONTEOS TOTALES")
    lineas.append("-" * 70)
    lineas.append(f"  Impresos:           {TOTAL_IMPRESOS:4d}  ({formatear_porcentaje(TOTAL_IMPRESOS/n_real)})")
    lineas.append(f"  Manuscritos:        {TOTAL_MANUSCRITOS:4d}  ({formatear_porcentaje(TOTAL_MANUSCRITOS/n_real)})")
    lineas.append(f"  Fotos:              {TOTAL_FOTOS:4d}  ({formatear_porcentaje(TOTAL_FOTOS/n_real)})")
    lineas.append("-" * 70)
    lineas.append("")

    # Resultados inferenciales
    lineas.append("ESTADÍSTICA INFERENCIAL (para 'Impresos')")
    lineas.append("-" * 70)
    lineas.append(f"  Proporción muestral (p):                 {formatear_porcentaje(stats['proporcion'], 3)}")
    lineas.append(f"  Error estándar (con corrección por FPC): {stats['error_estandar']:.5f}")
    lineas.append("")
    lineas.append("  Intervalo de confianza al 95 %:")
    lineas.append(f"    Límite inferior: {formatear_porcentaje(stats['ci_95_lower'], 2)}")
    lineas.append(f"    Límite superior: {formatear_porcentaje(stats['ci_95_upper'], 2)}")
    lineas.append("")
    lineas.append(f"  Prueba de hipótesis (H0: p ≤ {UMBRAL_HIPOTESIS:.0%} vs H1: p > {UMBRAL_HIPOTESIS:.0%}):")
    lineas.append(f"    Estadístico Z:    {stats['z_stat']:.4f}")
    lineas.append(f"    p-valor:          {stats['p_valor']:.10f}")
    if stats['p_valor'] < 0.05:
        lineas.append("     Conclusión:     Se RECHAZA H0. La proporción de impresos es SIGNIFICATIVAMENTE mayor.")
    else:
        lineas.append("     Conclusión:     No se rechaza H0. No hay suficiente evidencia.")
    lineas.append("-" * 70)
    lineas.append("")

    # Análisis por grupos (si se proporcionaron)
    if DATOS_GRUPOS:
        lineas.append("ANÁLISIS POR GRUPOS (cada grupo de ~80 documentos)")
        lineas.append("-" * 70)
        lineas.append("  Grupo  Impresos  Manus.  Fotos   % Impresos")
        lineas.append("  -----  --------  ------  -----  -----------")
        proporciones_grupo = []
        for i, (imp, man, fot) in enumerate(DATOS_GRUPOS, start=1):
            total_g = imp + man + fot
            p_g = imp / total_g if total_g > 0 else 0
            proporciones_grupo.append(p_g)
            lineas.append(f"  {i:4d}   {imp:6d}   {man:5d}   {fot:4d}   {formatear_porcentaje(p_g, 1)}")
        # Estadísticas de los grupos
        if proporciones_grupo:
            media_grupos = sum(proporciones_grupo) / len(proporciones_grupo)
            min_grupo = min(proporciones_grupo)
            max_grupo = max(proporciones_grupo)
            rango = max_grupo - min_grupo
            lineas.append("  -----  --------  ------  -----  -----------")
            lineas.append(f"  Media de los grupos:                    {formatear_porcentaje(media_grupos, 1)}")
            lineas.append(f"  Mínimo / Máximo:                        {formatear_porcentaje(min_grupo, 1)} / {formatear_porcentaje(max_grupo, 1)}")
            lineas.append(f"  Rango (diferencia máx - mín):          {formatear_porcentaje(rango, 1)}")
            lineas.append("")
            lineas.append("  Interpretación: La baja variabilidad entre grupos (rango pequeño)")
            lineas.append("     indica que el resultado es consistente en toda la colección.")
        lineas.append("-" * 70)
        lineas.append("")

    # Conclusión final
    lineas.append("CONCLUSIÓN FINAL")
    lineas.append("=" * 70)
    if stats['ci_95_lower'] > UMBRAL_HIPOTESIS:
        lineas.append("  Con un 95 % de confianza, la proporción REAL de documentos impresos")
        lineas.append(f"  en el archivo completo (N={TOTAL_POBLACION:,}) está entre ")
        lineas.append(f"  {formatear_porcentaje(stats['ci_95_lower'])} y {formatear_porcentaje(stats['ci_95_upper'])}.")
        lineas.append("")
        lineas.append(f"  Dado que el límite inferior ({formatear_porcentaje(stats['ci_95_lower'])}) es ")
        lineas.append(f"  MAYOR que el umbral del {UMBRAL_HIPOTESIS:.0%}, podemos afirmar")
        lineas.append("  ESTADÍSTICAMENTE que la GRAN MAYORÍA de los documentos son IMPRESOS.")
    else:
        lineas.append("  Los datos no permiten concluir con seguridad que la mayoría sean impresos.")
        lineas.append("  (El intervalo de confianza cruza o está por debajo del umbral elegido).")
    lineas.append("=" * 70)
    lineas.append("")
    lineas.append("  Informe generado automáticamente por 'analisis_muestreo.py'.")

    # Mostrar en consola
    texto_informe = "\n".join(lineas)
    print(texto_informe)

    # Guardar en archivo
    archivo_salida = Path(__file__).parent / "resultados_muestreo.txt"
    with open(archivo_salida, "w", encoding="utf-8") as f:
        f.write(texto_informe)
    print(f"\nEl informe también se ha guardado en: {archivo_salida}")

# ──────────────────────────────────────────────────────────────────────────
# EJECUCIÓN
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    generar_informe()