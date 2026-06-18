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
    Articulos_ohcbh_cf_erlXXX/
        ohcbh_cf_erlXXX_001.pdf   ← PDF de la página 1
        ohcbh_cf_erlXXX_002.pdf   ← PDF de la página 2
        ...
```

Cada carpeta `Articulos_ohcbh_cf_erlXXX/` contiene los PDFs de cada **página individual** del documento. Cada página es un archivo `.pdf` independiente.

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

El emparejamiento es **página a página**. La clave de match es el **stem del nombre de archivo** (sin extensión):

| Fuente | Archivo | Clave de match |
|--------|---------|----------------|
| PDF | `Articulos_ohcbh_cf_erl1/ohcbh_cf_erl1_001.pdf` | `ohcbh_cf_erl1_001` |
| TXT | `Articulos_ohcbh_cf_erl1/ohcbh_cf_erl1_001.pdf.txt` | `ohcbh_cf_erl1_001` |

- Del PDF: `page_stem = pdf_file.stem` (Python) — elimina `.pdf`
- Del TXT: `page_stem = txt_name[:-len('.pdf.txt')]` — elimina `.pdf.txt`

Ambos lados producen el mismo identificador de página. No se requiere heurística difusa.

El `doc_id` en la base de datos es el identificador de página (`ohcbh_cf_erl1_001`). Los metadatos se heredan del **documento padre** (`ohcbh_cf_erl1`) consultando el CSV por nombre de carpeta.

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

## Resultados del dry-run (13 jun 2026)

```
Páginas PDF en data/pdfs/.../Articulos_.../ : 15584
Páginas TXT en Reconstrucciones/            :  4305
────────────────────────────────────────────────────
✓ Emparejadas (PDF + TXT)                  :  4305
⏳ Solo PDF (TXT aún en proceso)            : 11279
```

Las 4305 páginas ya reconstruidas están listas para cargarse. Las 11279 restantes se cargarán progresivamente cuando finalice el procesamiento de `Reconstrucciones_qwen_deepseek/`.

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
Recorre `proyect/data/pdfs/ohcbh_cf_erlXXX/Articulos_ohcbh_cf_erlXXX/` para cada documento. Recopila cada `.pdf` individual como una entrada `{page_stem → (ruta_relativa, parent_doc_id)}`. El `parent_doc_id` es el nombre de la carpeta padre (`ohcbh_cf_erlXXX`).

### 2. `scan_reconstrucciones()` — Escaneo de textos
Recorre `Reconstrucciones_qwen_deepseek/Articulos_ohcbh_cf_erlXXX/`. Para cada `.pdf.txt` extrae el `page_stem` eliminando el sufijo `.pdf.txt`. Devuelve `{page_stem → (ruta_al_txt, parent_doc_id)}`.

### 3. `load_csv_metadata()` — Carga de metadatos
Lee el CSV buscando filas cuyo `Nombre viejo` sea `{id}_bn.pdf`. Toma la primera fila por documento y construye `{parent_doc_id → dict de metadatos}`. El CSV tiene ~1M filas (una por imagen/página), de ahí la selección específica por `_bn.pdf`.

### 4. Match
`matched = {page_stem: (pdf_rel, parent_id, txt_path)} para page_stem in pdfs si page_stem in txts`

Separado de: `pdf_only` (página sin TXT aún) y `txt_only` (TXT sin PDF, anomalía).

### 5. `concat_texts()` — Lectura del texto
Lee el único `.txt` correspondiente a la página. Luego aplica `clean_ocr_text()` de la utilidad existente de la app.

### 6. Carga en DB (Flask app context)
Para cada página del match:
- `doc_id` = `page_stem` (p.ej. `ohcbh_cf_erl1_001`)
- Metadatos heredados del documento padre mediante el CSV.
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
