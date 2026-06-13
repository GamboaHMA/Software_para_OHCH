# Match PDF ↔ TXT y carga en la base de datos

**Fecha:** 12 de junio de 2026  
**Script creado:** `proyect/load_from_reconstrucciones.py`

---

## Contexto

La aplicación web (`proyect/`) permite visualizar documentos históricos mostrando el facsímil en PDF a la izquierda y el texto OCR procesado a la derecha. La base de datos (`proyect/data/db/corpus.db`) espera registros con:

- `pdf_filename` — ruta relativa desde `proyect/data/pdfs/` hacia el PDF
- `ocr_text` — texto completo del documento (o `ocr_text_edited` si fue corregido)
- Metadatos: título, autor, archivo, signatura, clasificación, etc.

Los textos reconstruidos (OCR + limpieza + reconstrucción con IA) se encuentran en `Reconstrucciones_qwen_deepseek/` y los PDFs en `proyect/data/pdfs/`, aún sin vincular a la DB.

---

## Estructura de archivos

### PDFs (`proyect/data/pdfs/`)

```
ohcbh_cf_erlXXX/
    PDF_ohcbh_cf_erlXXX/
        ohcbh_cf_erlXXX_bn.pdf   ← archivo principal (blanco y negro)
```

Cada carpeta de primer nivel (`ohcbh_cf_erlXXX`) corresponde a **un documento**. El PDF a servir está siempre dentro del subdirectorio `PDF_ohcbh_cf_erlXXX/`.

### Textos reconstruidos (`Reconstrucciones_qwen_deepseek/`)

```
Articulos_ohcbh_cf_erlXXX/
    ohcbh_cf_erlXXX_001.pdf.txt   ← página 1 reconstruida
    ohcbh_cf_erlXXX_002.pdf.txt   ← página 2
    ...
    ohcbh_cf_erlXXX_NNN.pdf.meta.json  ← metadatos del procesamiento
```

Cada carpeta agrupa las páginas de **un documento**; los `.txt` se concatenan en orden para formar el texto completo.

---

## Lógica del emparejamiento

El identificador de documento se extrae directamente del nombre de la carpeta:

| Fuente | Nombre de carpeta / archivo | ID extraído |
|--------|----------------------------|-------------|
| PDFs | `ohcbh_cf_erl1` | `ohcbh_cf_erl1` |
| Reconstrucciones | `Articulos_ohcbh_cf_erl1` → eliminar `Articulos_` | `ohcbh_cf_erl1` |
| CSV | `Nombre viejo` = `ohcbh_cf_erl1_bn.pdf` → eliminar `_bn.pdf` | `ohcbh_cf_erl1` |

El match es **por igualdad de ID** (string). No se requiere heurística difusa.

### Casos especiales resueltos

Algunos PDFs tienen nombres atípicos (sin sufijo `_bn` o con letras en minúscula vs. mayúscula en el ID):

| ID de carpeta | PDF real encontrado | Problema |
|--------------|---------------------|---------|
| `ohcbh_cf_erl104` | `ohcbh_cf_erl104.pdf` | Sin sufijo `_bn` |
| `ohcbh_cf_erl14A` | `ohcbh_cf_erl14a_bn.pdf` | Letra en minúscula |
| `ohcbh_cf_erl14B` | `ohcbh_cf_erl14b_bn.pdf` | Letra en minúscula |
| `ohcbh_cf_erl158` | `ohcbh_cf_erl158.pdf` | Sin sufijo `_bn` |

El escáner de PDFs prueba candidatos en este orden: `{id}_bn.pdf` → `{id}.pdf` → `{id.lower()}_bn.pdf` → `{id.lower()}.pdf` → cualquier `.pdf` único en el directorio.

---

## Metadatos del CSV

El archivo `Listado completo Facticia.csv` (separado por `;`) contiene 224 entradas únicas de documentos identificadas por el campo `Nombre viejo` (valor `ohcbh_cf_erlXXX_bn.pdf`).

Mapeo de columnas CSV → campos del modelo `Document`:

| Columna CSV | Campo DB | Ejemplo |
|-------------|----------|---------|
| `Título` | `title` | `La Acera del Louvre` |
| `Autor` | `author` | `Roig de Leuchsenring, Emilio, 1889-1964` |
| `Localización` | `signature` | `F ERL 001` |
| *(fijo)* | `archive` | `OHCBH - Fondo Emilio Roig de Leuchsenring` |
| `Tipo de documento` | `classification` | `Libro`, `Folleto` |
| `Nota General/Observaciones` | `notes` | descripción del volumen |
| `Lugar Fuente` | `location_city` | `La Habana` |
| *(fijo)* | `location_country` | `Cuba` |
| `Fecha Fuente` | `date_raw` / `date_year` | extraído con regex si hay año |

---

## Resultados del dry-run (12 jun 2026)

```
PDFs encontrados en data/pdfs/         : 224
Carpetas TXT en Reconstrucciones/      : 73
────────────────────────────────────────
✓ Emparejados (PDF + TXT)             : 73
⏳ Solo PDF (TXT aún en proceso)       : 151
```

Los 73 documentos ya procesados están listos para cargarse. Los 151 restantes se cargarán progresivamente usando `--solo-pdf` (solo PDF, sin texto) o cuando finalice el procesamiento de Reconstrucciones.

---

## Cómo usar el script

El script se ejecuta **desde el directorio `proyect/`** con el entorno virtual activado:

```bash
cd proyect/
source ../.venv/bin/activate
```

### Casos de uso

**1. Cargar solo los documentos emparejados (PDF + texto) — primera ejecución:**
```bash
python load_from_reconstrucciones.py
```
Inserta los 73 documentos con texto completo. Omite los que ya existan en la DB.

**2. Ver qué se haría sin modificar la DB:**
```bash
python load_from_reconstrucciones.py --dry-run
```

**3. Cargar también los documentos sin texto todavía (solo PDF visible):**
```bash
python load_from_reconstrucciones.py --solo-pdf
```
Útil para que todos los PDFs aparezcan en la web aunque el texto aún no esté listo.

**4. Actualizar registros ya existentes (p.ej. cuando lleguen más textos):**
```bash
python load_from_reconstrucciones.py --force
```
Sobreescribe `ocr_text`, `pdf_filename` y metadatos en registros existentes.

**5. Flujo completo recomendado al finalizar el procesamiento:**
```bash
# Primero carga los emparejados nuevos
python load_from_reconstrucciones.py
# Luego actualiza los que ya estaban cargados solo con PDF
python load_from_reconstrucciones.py --force
```

---

## Descripción paso a paso del script

### 1. `scan_pdfs()` — Escaneo de PDFs
Recorre `proyect/data/pdfs/` buscando carpetas con patrón `ohcbh_cf_erlXXX`. Para cada una, entra al subdirectorio `PDF_{id}/` y localiza el PDF con tolerancia a variaciones de nombre. Devuelve `{doc_id → ruta_relativa}`.

### 2. `scan_reconstrucciones()` — Escaneo de textos
Recorre `Reconstrucciones_qwen_deepseek/` buscando carpetas `Articulos_ohcbh_cf_erlXXX`. Extrae el doc_id, lista todos los `.txt` y los ordena por número de página (extraído con regex del nombre). Devuelve `{doc_id → [lista de rutas]}`.

### 3. `load_csv_metadata()` — Carga de metadatos
Lee el CSV buscando filas cuyo `Nombre viejo` sea `{id}_bn.pdf`. Toma la primera fila por documento y construye `{doc_id → dict de metadatos}`. El CSV tiene ~1M filas (una por imagen/página), de ahí la selección específica por `_bn.pdf`.

### 4. Match
`matched = {id: (pdf_rel, txts)} para id in pdfs si id in txts`

Separado de: `pdf_only` (PDF sin TXT aún) y `txt_only` (TXT sin PDF, anomalía).

### 5. `concat_texts()` — Concatenación
Lee los `.txt` en orden y los une con `\n\n`. Luego aplica `clean_ocr_text()` de la utilidad existente de la app.

### 6. Carga en DB (Flask app context)
Para cada documento del match:
- Si **no existe** → crea `Document` con todos los campos y hace `db.session.add()`.
- Si **existe** y `--force` → actualiza los campos individuales según estén vacíos o no.
- Hace `db.session.commit()` tras cada inserción para evitar pérdida total en caso de error.

---

## Archivos creados / modificados

| Archivo | Acción |
|---------|--------|
| `proyect/load_from_reconstrucciones.py` | **Creado** — script principal |
| `match_para_base_de_datos.md` | **Actualizado** — este reporte |

---

## Notas adicionales

- El campo `ocr_text` almacena el texto reconstruido por IA. Si en el futuro se corrige manualmente, se guarda en `ocr_text_edited` (la app muestra `ocr_text_edited` con preferencia sobre `ocr_text`).
- El script puede re-ejecutarse varias veces sin duplicar registros (por el filtro `filter_by(doc_id=...)` antes de insertar).
- Cuando finalice el procesamiento del resto de los ~151 documentos en `Reconstrucciones_qwen_deepseek/`, basta con ejecutar `python load_from_reconstrucciones.py` nuevamente — solo insertará los nuevos.
