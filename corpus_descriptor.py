#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import csv
import statistics
from pathlib import Path

def contar_palabras(texto):
    """Devuelve el número de palabras usando expresión regular (letras, números, _)."""
    return len(re.findall(r'\b\w+\b', texto, flags=re.UNICODE))

def procesar_corpus(directorio):
    """
    Recorre recursivamente el directorio, lee todos los .txt y devuelve:
    - lista de diccionarios con info por documento
    - diccionario con estadísticas globales
    """
    # Búsqueda recursiva de todos los archivos .txt
    archivos = list(Path(directorio).rglob('*.txt'))
    if not archivos:
        print(f"No se encontraron archivos .txt en '{directorio}' (y subcarpetas).")
        return None, None

    datos_documentos = []   # lista de dicts para CSV
    total_palabras = 0
    total_caracteres = 0

    for archivo in archivos:
        try:
            with open(archivo, 'r', encoding='utf-8') as f:
                texto = f.read()
        except UnicodeDecodeError:
            # Fallback a latin-1
            with open(archivo, 'r', encoding='latin-1') as f:
                texto = f.read()

        num_palabras = contar_palabras(texto)
        num_caracteres = len(texto)
        total_palabras += num_palabras
        total_caracteres += num_caracteres

        # Guardar información del documento
        datos_documentos.append({
            'ruta': str(archivo),
            'nombre': archivo.name,
            'palabras': num_palabras,
            'caracteres': num_caracteres,
            'ruta_relativa': str(archivo.relative_to(directorio))  # útil para legibilidad
        })

    # Estadísticas globales
    cantidades = [d['palabras'] for d in datos_documentos]
    # Ordenar para min/max
    doc_min = min(datos_documentos, key=lambda x: x['palabras'])
    doc_max = max(datos_documentos, key=lambda x: x['palabras'])

    estadisticas = {
        'total_archivos': len(datos_documentos),
        'total_palabras': total_palabras,
        'total_caracteres': total_caracteres,
        'promedio_palabras': statistics.mean(cantidades),
        'mediana_palabras': statistics.median(cantidades),
        'desviacion_estandar': statistics.stdev(cantidades) if len(cantidades) > 1 else 0,
        'minimo': (doc_min['nombre'], doc_min['palabras']),
        'maximo': (doc_max['nombre'], doc_max['palabras']),
    }

    return datos_documentos, estadisticas

def guardar_csv(datos, nombre_csv):
    """Guarda la lista de diccionarios en un archivo CSV."""
    if not datos:
        return
    # Definir columnas (orden)
    fieldnames = ['ruta', 'nombre', 'palabras', 'caracteres', 'ruta_relativa']
    with open(nombre_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(datos)
    print(f"Datos guardados en '{nombre_csv}'")

def imprimir_resumen(estadisticas):
    """Imprime estadísticas globales en consola."""
    if not estadisticas:
        return

    print("\n" + "="*60)
    print("RESUMEN DEL CORPUS")
    print("="*60)
    print(f"Número total de archivos .txt: {estadisticas['total_archivos']}")
    print(f"Total de palabras en el corpus: {estadisticas['total_palabras']:,}")
    print(f"Total de caracteres (incluyendo espacios): {estadisticas['total_caracteres']:,}")
    print("\nEstadísticas por documento:")
    print(f"  Promedio de palabras: {estadisticas['promedio_palabras']:.2f}")
    print(f"  Mediana de palabras:   {estadisticas['mediana_palabras']:.2f}")
    print(f"  Desviación estándar:   {estadisticas['desviacion_estandar']:.2f}")
    print(f"  Mínimo de palabras:    {estadisticas['minimo'][1]:,} (en '{estadisticas['minimo'][0]}')")
    print(f"  Máximo de palabras:    {estadisticas['maximo'][1]:,} (en '{estadisticas['maximo'][0]}')")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(
        description="Analiza recursivamente un corpus de archivos .txt y genera estadísticas + CSV."
    )
    parser.add_argument(
        "directorio",
        help="Ruta al directorio raíz (puede ser relativa o absoluta). Ejemplo: ./mis_textos o ."
    )
    parser.add_argument(
        "-o", "--output",
        default="resultados_corpus.csv",
        help="Nombre del archivo CSV de salida (por defecto: resultados_corpus.csv)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.directorio):
        print(f"Error: '{args.directorio}' no es un directorio válido.")
        return

    datos, estadisticas = procesar_corpus(args.directorio)
    if datos is None:
        return

    # Guardar CSV
    guardar_csv(datos, args.output)

    # Mostrar resumen en pantalla
    imprimir_resumen(estadisticas)

if __name__ == "__main__":
    main()