import fitz  # Necesitas instalar: pip install PyMuPDF

# Cambia "tu_capitulo.pdf" por la ruta de tu archivo
doc = fitz.open("estado_del_arte_rev-22-junio.pdf")

for pagina in doc:
    for anotacion in pagina.annots():
        # Tipo 8 = Resaltado (highlight)
        if anotacion.type[0] == 8:
            # Extrae el texto que está debajo del resaltado
            texto_resaltado = pagina.get_textbox(anotacion.rect).strip()
            # Extrae el comentario escrito en el globo
            comentario = anotacion.info.get("content", "").strip()
            
            if texto_resaltado or comentario:
                print(f"--- PÁGINA {pagina.number + 1} ---")
                print(f"TEXTO MARCADO: {texto_resaltado}")
                print(f"COMENTARIO: {comentario}")
                print("-----")
doc.close()