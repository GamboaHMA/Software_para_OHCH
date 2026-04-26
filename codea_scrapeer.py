import sys
import pathlib
import requests
import os
import re
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

NOMBRE_BASE = 'CODEA-'
CARPETA_PATH = 'corpus_escrapeado'
NOMBRE_ARCHIVO = 'urls_id.txt'

URL_LEFT = 'https://corpuscodea.es/corpus/documento.php?documento=CODEA-'
URL_RIGHT = '&loc=undefined&paleografica=off&abreviaturas=off&mayusculas=off&busqueda='

def leer_ids_desde_archivo(nombre_archivo: str) -> list[str]:
    with open(nombre_archivo, 'r', encoding='utf-8') as f:
        # numeros = [int(linea.strip()) for linea in f if linea.strip()] # aqui claramente hay un error, los que empiezan con cero los toma
                                                                         # como numero de 3 cifras y ya no tiene la misma forma de id
        numeros = (linea.strip() for linea in f if linea.strip())
        result = []   # ahora para los que empiezan con 0, debido a que no se escrapearon con el comando anterior
        for num in numeros:
            if num[0] == '0':
                result.append(num)
    print(result)
    return result


def descargar_html(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return resp.text

def extraer_textocritico(html:str, num:str) -> str:
    soup = BeautifulSoup(html, 'lxml')

    soup_tags = soup.find_all(['script', 'style', 'noscript', 'header', 'footer', 'nav'])
    #print(soup(soup_tags))
    # Eliminar cosas que no son contenido relevante
    for tag in soup_tags:
        if tag is not None:
            tag.decompose()
    
    # Si hay varios bloques de textocritico los juntamos
    bloques = soup.find_all(class_='textocritico')

    if not bloques:
        print(f'No se encontraron clases llamadas: textocritico para {num}')
    
    else:
        print(len(bloques))
    
    textos = []

    for b in bloques:
        # Extraemos solo nodos de texto que no estén dentro de un <span>
        partes: list[str] = []
        for nodo in b.find_all(string=True):
            # omitimos contenido de spans (u otros elementos si se desea)
            if nodo.parent and nodo.parent.name == 'span':
                continue
            txt = nodo.strip()
            if txt:
                if txt[0] == '{':   # parche para qutar los { h }
                    continue
                else:
                    partes.append(txt)
        # unir con espacio y sin saltos de línea
        textos.append(" ".join(partes))

    return "\n\n".join(textos)

def guardar_texto_en_archivo(texto: str, nombre_id: int) -> None:
    safe_name = NOMBRE_BASE + str(nombre_id) + '.txt'
    ruta = pathlib.Path(CARPETA_PATH) / safe_name
    ruta.write_text(texto, encoding="utf-8")

    return ruta


def main():
    #url = 'https://corpuscodea.es/corpus/documento.php?documento=CODEA-1047&loc=undefined&paleografica=off&abreviaturas=off&mayusculas=off&busqueda='

    numeros = leer_ids_desde_archivo(nombre_archivo=NOMBRE_ARCHIVO)
    
    for num in numeros:
        url = f'{URL_LEFT}{num}{URL_RIGHT}'

        html = descargar_html(url=url)

        texto_extraido = extraer_textocritico(html=html, num=num)

        print(len(texto_extraido.split()))
        guardar_texto_en_archivo(texto_extraido, num)



if __name__ == "__main__":
    #main()
    carpeta_path = pathlib.Path(CARPETA_PATH)
    files_count = 0
    words_count = 0
    word_set = set()

    patron_palabras = re.compile(r'\b[a-zA-ZáéíóúÁÉÍÓÚñÑüÜ]+(?:\'[a-zA-Z])?\b', re.IGNORECASE)

    total_palabras = 0
    todas_las_palabras = []
    count = 0

    for ruta_a_contar in carpeta_path.glob('*.txt'):
        # if count == 1:
        #     break
        print(f'Leyendo {ruta_a_contar.name}')

        with open(ruta_a_contar, 'r', encoding='utf-8') as f:
            texto = f.read()

        archivo_de_palabras = patron_palabras.findall(texto)
        todas_las_palabras.extend(archivo_de_palabras)
        total_palabras += len(archivo_de_palabras)
        print(f'cantidad de palabras para {ruta_a_contar.name}: {len(archivo_de_palabras)}')
        count += 1

    print(f'cantidad de palabras: {total_palabras}')

    for palabra in todas_las_palabras:
        word_set.add(palabra)
        #print(palabra)
    
    print(len(word_set))

