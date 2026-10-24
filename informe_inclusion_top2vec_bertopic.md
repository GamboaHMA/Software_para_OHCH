# Informe de incorporación de Top2Vec y BERTopic

## 1. Objetivo

Se amplió `topic_modeling.py` para ejecutar cuatro métodos: LDA, NMF, Top2Vec y BERTopic. Los métodos nuevos reciben los textos originales (`raw_texts`) sin pasar por la lematización ni por la vectorización de bolsa de palabras del flujo LDA/NMF.

## 2. Cambios realizados

- Se añadieron `top2vec` y `bertopic` a los métodos aceptados por la clase y por `--method`.
- Se separó el flujo de `fit()`:
  - **LDA/NMF:** carga → limpieza y lematización con spaCy → vectorización con CountVectorizer/TF-IDF → entrenamiento.
  - **Top2Vec/BERTopic:** carga → entrenamiento directamente sobre los textos UTF-8 originales.
- spaCy ahora se carga de forma diferida únicamente cuando LDA/NMF necesitan lematizar. Top2Vec y BERTopic ya no requieren cargar el modelo lingüístico de spaCy.
- BERTopic usa el encoder multilingüe `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Top2Vec utiliza por defecto el mismo encoder (sin el prefijo de namespace que su API no acepta); puede seleccionarse otro encoder compatible con `--embedding-model`. Los modelos se descargan de Hugging Face si no están en caché.
- Se adaptó la extracción de palabras representativas: `components_` para LDA/NMF, `get_topics()` para Top2Vec y `get_topic()` para BERTopic.
- Se crea una matriz documento-tema compatible con los resúmenes y CSV existentes. Para Top2Vec y BERTopic contiene la asignación del tema detectado y su puntuación, no una distribución completa entre todos los temas; las puntuaciones de ambos algoritmos no son directamente comparables con LDA/NMF.
- Top2Vec conserva todos los temas que descubre; el parámetro `--min-count` controla la frecuencia mínima de palabra usada por su vocabulario. BERTopic también conserva los temas descubiertos sin configurar `nr_topics`; `--min-topic-size` controla el tamaño mínimo de sus clústeres. BERTopic puede asignar documentos atípicos al tema `-1`, que se muestra explícitamente.
- Top2Vec se guarda con `Top2Vec.save()` y se carga con `Top2Vec.load()`. BERTopic se guarda con `BERTopic.save(..., serialization="pickle")` y se carga con `BERTopic.load()`. Un sidecar `.metadata.pkl` conserva rutas, textos y matriz documento-tema necesarios para resúmenes y visualizaciones. LDA/NMF mantienen el archivo `.pkl` compatible con el flujo previo.
- Se añadió `--load-model` para cargar el archivo de modelo guardado desde la CLI y mostrar resultados sin ejecutar `fit()`.
- `topic_sampling.py` admite `--model-path` para visualizar directamente modelos cargados mediante el cargador común, además de seleccionar los mejores LDA/NMF por CSV. Para la matriz de similitud utiliza componentes de LDA/NMF, vectores de tema de Top2Vec o embeddings/representaciones c-TF-IDF de BERTopic.
- Los resultados del modo experimental ahora incluyen el nombre del método en el CSV para evitar colisiones entre algoritmos.
- La carga del corpus descarta, para todos los métodos, archivos vacíos o con menos de 50 palabras alfabéticas por defecto. El umbral se puede ajustar con `--min-document-words` o desactivar con `0`, y la lista de rutas/textos se mantiene alineada.

## 3. Justificación metodológica y técnica

Top2Vec y BERTopic usan embeddings y agrupamiento semántico, por lo que conviene conservar las formas originales del texto. La lematización y el filtrado POS podrían borrar variaciones morfológicas, contexto o términos históricos que los embeddings pueden aprovechar. Por eso dichos modelos reciben `raw_texts` tal como fueron leídos, sin limpiar, lematizar ni convertirlos previamente a una matriz Count/TF-IDF.

Se utiliza un encoder multilingüe para evitar depender de un modelo diseñado exclusivamente para inglés. Los imports de Top2Vec y BERTopic se hacen dentro de la rama correspondiente: una persona que solo ejecute LDA/NMF no necesita que esas bibliotecas estén importables en ese entorno.

La persistencia mantiene el formato `.pkl` que ya empleaba el proyecto y guarda el objeto entrenado junto con los datos auxiliares. Los modelos deben cargarse en un entorno Python compatible con las versiones de sus bibliotecas. Como cualquier pickle, no se deben cargar archivos de origen no confiable.

## 4. Uso

Entrenamiento y guardado automáticos del modelo:

```bash
python topic_modeling.py --method top2vec --min-count 25
python topic_modeling.py --method bertopic --min-topic-size 8 --output Topic_results/bertopic.csv
python topic_modeling.py --method bertopic --min-topic-size 8 --min-document-words 40
```

`--n-topics` solo configura LDA/NMF y se ignora en Top2Vec/BERTopic. Los modelos se guardan nativamente en `Topic_models/` con nombres como `top2vec_min_count_25.model` y `bertopic_min_topic_size_8.pickle`; junto a cada uno se crea su sidecar `.metadata.pkl`.

Para Top2Vec, `--min-count` afecta el vocabulario de palabras, no el número de temas. Top2Vec y BERTopic comparten por defecto el encoder multilingüe `paraphrase-multilingual-MiniLM-L12-v2`.

Carga y visualización posterior sin volver a entrenar:

```bash
python topic_modeling.py --load-model Topic_models/top2vec_min_count_25.model
python topic_modeling.py --load-model Topic_models/bertopic_min_topic_size_8.pickle --output Topic_results/bertopic.csv
python topic_sampling.py --model-path Topic_models/top2vec_min_count_25.model
```

La opción `--load-model` no se combina con el modo de experimentación `--desde/--hasta`.

## 5. Paso a paso de la inclusión

1. Se amplió la validación de `method` y las opciones de la CLI.
2. Se hizo diferida la carga de spaCy, conservando su uso en el preprocesamiento existente de LDA/NMF.
3. Se dividió `fit()` por familias: LDA/NMF usan `processed_texts` y vectorizadores; Top2Vec/BERTopic usan directamente `raw_texts`.
4. Se añadió el paso por los parámetros nativos `min_count` y `min_topic_size`, sin limitar el número de temas descubiertos.
5. Se añadieron rutas específicas para imprimir palabras representativas de los cuatro tipos de modelos.
6. Se conservaron LDA/NMF en el formato anterior y se cambiaron Top2Vec/BERTopic a sus APIs nativas de guardado y carga, junto con metadatos laterales.
7. Se adaptó `topic_sampling.py` para visualizar modelos cargados directamente y calcular similitudes desde las representaciones de cada algoritmo.
8. Se incorporó este informe actualizado con metodología, justificación y comandos.
9. Se añadió un filtro común para excluir documentos vacíos o demasiado breves antes del preprocesamiento/modelado, evitando ruido y problemas en los métodos basados en embeddings.

## 6. Dependencias

En el intérprete seleccionado para este workspace se verificó que están instalados Top2Vec 1.0.36 y BERTopic 0.17.4. Si se usa otro entorno, instalar los paquetes requeridos por separado (`pip install top2vec bertopic`) y disponer de sus dependencias de embeddings, incluida `sentence-transformers`. No se añadió un archivo de dependencias porque el proyecto no contiene actualmente un `requirements.txt`, `pyproject.toml` ni `setup.py` en la raíz.
