import pathlib
import re
import subprocess
import os

ES_PATH = 'ES'
CORPUS_PATH = 'iula_corpus.txt'
# ARPA_PATH = 'iula_modelo.arpa'
# BIN_PATH = 'iula_modelo.bin'
ARPA_PATH = 'iula_modelo_5.arpa'
BIN_PATH = 'iula_modelo_5.bin'
KENLM_BIN = 'kenlm/build/bin'

CORPUS_HISTORICO_PATH = 'corpus_raw_clean.txt'
CORPUS_COMBINADO_PATH = 'corpus_combinado.txt'
# ARPA_COMBINADO_PATH = 'modelo_combinado.arpa'
# BIN_COMBINADO_PATH = 'modelo_combinado.bin'
ARPA_COMBINADO_PATH = 'modelo_combinado_5.arpa'
BIN_COMBINADO_PATH = 'modelo_combinado_5.bin'

patron_palabras = re.compile(r'\b[a-zA-ZáéíóúÁÉÍÓÚñÑüÜ]+(?:\'[a-zA-Z])?\b', re.IGNORECASE)

if __name__ == '__main__':
    es_path = pathlib.Path(ES_PATH)

    total_palabras = 0
    todas_las_palabras = []
    word_set = set()
    corpus_lines = []

    for carpeta in sorted(es_path.iterdir()):
        if not carpeta.is_dir():
            continue

        plain_files = list(carpeta.glob('*-plain.txt'))
        if not plain_files:
            print(f'No se encontró archivo plain en {carpeta.name}')
            continue

        ruta_plain = plain_files[0]
        print(f'Leyendo {ruta_plain.name}')

        with open(ruta_plain, 'r', encoding='utf-8') as f:
            texto = f.read()

        archivo_de_palabras = patron_palabras.findall(texto)
        todas_las_palabras.extend(archivo_de_palabras)
        total_palabras += len(archivo_de_palabras)
        print(f'Cantidad de palabras para {ruta_plain.name}: {len(archivo_de_palabras)}')

        corpus_lines.append(texto)

    for palabra in todas_las_palabras:
        word_set.add(palabra)

    print(f'\nCantidad total de palabras: {total_palabras}')
    print(f'Cantidad de palabras distintas: {len(word_set)}')

    # Guardar corpus para kenlm
    with open(CORPUS_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(corpus_lines))

    print(f'Corpus guardado en {CORPUS_PATH}')

    # Construir modelo .arpa con lmplz (5-gramas) o 3-grama solo cambiar el parametro
    subprocess.run(
        # f'{os.path.join(KENLM_BIN, "lmplz")} -o 3 < {CORPUS_PATH} > {ARPA_PATH}',
        f'{os.path.join(KENLM_BIN, "lmplz")} -o 5 < {CORPUS_PATH} > {ARPA_PATH}',
        shell=True,
        check=True
    )
    print(f'Modelo ARPA guardado en {ARPA_PATH}')

    # Convertir a binario
    subprocess.run(
        [os.path.join(KENLM_BIN, 'build_binary'), ARPA_PATH, BIN_PATH],
        check=True
    )
    print(f'Modelo binario guardado en {BIN_PATH}')

    # --- Corpus combinado: IULA + histórico ---
    print('\nConstruyendo corpus combinado...')
    with open(CORPUS_COMBINADO_PATH, 'w', encoding='utf-8') as fout:
        with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
            fout.write(f.read())
        fout.write('\n')
        with open(CORPUS_HISTORICO_PATH, 'r', encoding='utf-8') as f:
            fout.write(f.read())
    print(f'Corpus combinado guardado en {CORPUS_COMBINADO_PATH}')

    subprocess.run(
        # f'{os.path.join(KENLM_BIN, "lmplz")} -o 3 < {CORPUS_COMBINADO_PATH} > {ARPA_COMBINADO_PATH}',
        f'{os.path.join(KENLM_BIN, "lmplz")} -o 5 < {CORPUS_COMBINADO_PATH} > {ARPA_COMBINADO_PATH}',
        shell=True,
        check=True
    )
    print(f'Modelo ARPA combinado guardado en {ARPA_COMBINADO_PATH}')

    subprocess.run(
        [os.path.join(KENLM_BIN, 'build_binary'), ARPA_COMBINADO_PATH, BIN_COMBINADO_PATH],
        check=True
    )
    print(f'Modelo binario combinado guardado en {BIN_COMBINADO_PATH}')

