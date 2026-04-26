import spacy, hunspell, kenlm, re
from spacy.tokens import Token
import numpy as np
from difflib import SequenceMatcher
from types import SimpleNamespace
import time
from complemento_hunspell import nombres


# state1: busca anomalias manteniendo contexto izquierdo mientras vaya encontrando tuplas con True en [1], al encontrar go to state2
# state2: ha encontrado la primera anomalia, busca no anomalia, mientras vaya encontrando anomalias las agrega al grupo de anomalias, al encontrar go to state3
# state3: ha encontrado la primera no anomalia despues de la 1era anomalia, ahora va a buscar una nueva anomalia, mientras agrega a contexto derecho las no anomalias 
        # que encuentra, al encontrar go to state4
# state4: ha encontrado la primera anomalia del segundo grupo de anomalias, ahora usa el contexto izquierdo, el primer grupo de anomalias, y el contexto derecho, para
        # la correccion aproximada, una vez se determine la mejor, se forma el nuevo contexto izquierdo, que seria, el antiguo contexto izquierdo mas el grupo de 
        # anomalias correjidas mas el contexto derecho, go to state2

'''
input:
    arreglo de tuplas [token, Bool], donde token es el resultado de hacer spacy.load("spanish")(texto), y Bool es
    si el token pertenece al diccionario de hunspell, con self.hobj.spell(palabra)

output:
    arreglo de tuplas ya correjidas
'''

class StringChecker():
    
    def __init__(self, model='es_core_news_sm', 
                dicc_dir=("/usr/share/hunspell/es_ES.dic", "/usr/share/hunspell/es_ES.aff"),
                lm_model='anagrama_models/modelo_combinado_5.bin',
                lm_left_ctx=5, lm_right_ctx=4):
        """
        lm_left_ctx:  numero maximo de palabras del contexto izquierdo que se pasan al LM.
                      Para un modelo trigrama (-o 3) el lookback efectivo es 2, con 3 se
                      cubre ese limite sin desperdiciar computo. Subir a 5 si se usa -o 5.
        lm_right_ctx: idem para el contexto derecho.
        """
        self.nlp = spacy.load(model)
        self.lm_model = kenlm.LanguageModel(lm_model)
        self.lm_left_ctx = lm_left_ctx
        self.lm_right_ctx = lm_right_ctx
        self.hobj = hunspell.HunSpell(dicc_dir[0], dicc_dir[1])
        self.correction_log: list[dict] = []
        self.last_original_text: str = ''
        self.last_corrected_text: str = ''
        # Caches persistentes entre documentos para evitar llamadas repetidas a hunspell
        self._spell_cache: dict[str, bool] = {}
        self._suggest_cache: dict[str, list[str]] = {}

        self.complementar_diccionario_hunspell()


    def complementar_diccionario_hunspell(self):
        # agregando nombres propios
        for nombre in nombres:
            self.hobj.add(nombre)


    def normalizar_cadena(self, anomaly_chain: list) -> list:
        """
        Fusiona rachas de 2+ tokens de un solo caracter alfabetico en un unico
        token-anomalia. Esto resuelve casos OCR como 'a y u d a' -> token 'ayuda'
        marcado como anomalia, para que la maquina de estados lo trate como bloque.
        
        Los tokens de un solo caracter no-alfabetico (puntuacion, numeros) se dejan
        tal cual para no perder informacion estructural.
        """
        result = []
        i = 0
        while i < len(anomaly_chain):
            token, valid = anomaly_chain[i]
            text = token.text.strip()

            # ¿Inicio de una racha de letras sueltas?
            if len(text) == 1 and text.isalpha():
                run = [text]
                j = i + 1
                while j < len(anomaly_chain):
                    next_tok, _ = anomaly_chain[j]
                    next_text = next_tok.text.strip()
                    if len(next_text) == 1 and next_text.isalpha():
                        run.append(next_text)
                        j += 1
                    else:
                        break

                if len(run) >= 2:
                    # Fusionar toda la racha como un unico token anomalia
                    merged_text = ''.join(run)
                    fake_token = SimpleNamespace(text=merged_text)
                    result.append((fake_token, False))
                    i = j
                else:
                    # Racha de 1: dejar el token original con su validez
                    result.append((token, valid))
                    i += 1
            else:
                result.append((token, valid))
                i += 1

        return result


    def correjir_cadena(self, anomaly_chain: list) -> list:
        self.last_original_text = ' '.join(t[0].text for t in anomaly_chain)
        self.correction_log = []

        anomaly_chain = self.normalizar_cadena(anomaly_chain)
        n = len(anomaly_chain)
        result = []
        left_context = []
        i = 0

        # state1: avanzar tokens validos al contexto izquierdo
        while i < n and anomaly_chain[i][1]:
            left_context.append(anomaly_chain[i])
            result.append(anomaly_chain[i])
            i += 1

        # state2+state3+state4 en bucle iterativo (evita recursion mutua)
        while i < n:
            # state2: recolectar grupo de anomalias consecutivas
            anom_group = []
            while i < n and not anomaly_chain[i][1]:
                anom_group.append(anomaly_chain[i])
                i += 1

            # state3: recolectar contexto derecho hasta la proxima anomalia
            right_context = []
            while i < n and anomaly_chain[i][1]:
                right_context.append(anomaly_chain[i])
                i += 1

            # state4: corregir y actualizar
            corrected = self.check_anomaly_group(left_context, anom_group, right_context)
            corrected_tuples = [(SimpleNamespace(text=w), True) for w in corrected]

            result.extend(corrected_tuples)
            result.extend(right_context)

            # Mantener solo la ventana util para el LM, evita crecimiento ilimitado
            left_context = (left_context + corrected_tuples + right_context)[-self.lm_left_ctx:]

        self.last_corrected_text = ' '.join(t[0].text for t in result)
        return result
    
    def check_anomaly_group(self, left_context, anomaly_group, right_context):

        # Ventana de contexto limitada al maximo util para el orden del modelo n-grama
        left_context_str = self.get_str_for_tuple(left_context[-self.lm_left_ctx:])
        right_context_str = self.get_str_for_tuple(right_context[:self.lm_right_ctx])
        anomaly_group_str = self.get_str_for_tuple(anomaly_group)

        # Cadena limpia: solo letras espaniol, sin espacios (fusiona letras sueltas OCR)
        text = re.sub(r'[^a-záéíóúüñ]', '', anomaly_group_str.lower())
        n = len(text)

        dp = np.full(n + 1, float('-inf'), dtype=float)
        back = np.full(n + 1, -1, dtype=int)
        words_at = [''] * (n + 1)
        seg_info_at = [None] * (n + 1)
        # prev_str_cache[j] = string reconstruido de palabras 0..j en O(1)
        # se actualiza despues de fijar back[i]/words_at[i] para cada i
        prev_str_cache = [''] * (n + 1)
        dp[0] = 0.0

        print("TEXT:", text)
        print("=" * 60)

        for i in range(1, n + 1):
            for j in range(max(0, i - 20), i):
                palabra_raw = text[j:i]

                # --- Hunspell score con cache ---
                is_valid = self._spell_cache.get(palabra_raw)
                if is_valid is None:
                    is_valid = self.hobj.spell(palabra_raw)
                    self._spell_cache[palabra_raw] = is_valid

                if is_valid:
                    spell_score = 10.0
                    palabra = palabra_raw
                    current_sugs = []
                else:
                    sugs = self._suggest_cache.get(palabra_raw)
                    if sugs is None:
                        sugs = self.hobj.suggest(palabra_raw)[:5]
                        self._suggest_cache[palabra_raw] = sugs
                    current_sugs = sugs

                    max_sug = float('-inf')
                    the_sug = None
                    for sug in sugs:
                        sim = self.similitud(palabra_raw, sug)
                        if sim > max_sug:
                            the_sug = sug
                            max_sug = sim

                    if the_sug is None or max_sug <= 0:
                        spell_score = -10.0
                        palabra = palabra_raw
                    else:
                        spell_score = max_sug
                        palabra = the_sug

                # --- LM score: prev_str en O(1) desde cache incremental ---
                prev_words_str = prev_str_cache[j]
                anom_str = (prev_words_str + ' ' + palabra).strip()
                cadena_para_lm = (left_context_str + ' ' + anom_str + ' ' + right_context_str).strip()
                lm_score = self.lm_model.score(cadena_para_lm)

                # --- Penalizacion por segmentos muy cortos ---
                if len(palabra) <= 1:
                    length_penalty = -5.0
                elif len(palabra) == 2:
                    length_penalty = -1.0
                else:
                    length_penalty = 0.0

                total_score = spell_score * 0.3 + lm_score * 0.7 + length_penalty
                acumulado = dp[j] + total_score

                if acumulado > dp[i]:
                    dp[i] = acumulado
                    back[i] = j
                    words_at[i] = palabra
                    seg_info_at[i] = {
                        'anomalia': text[j:i],
                        'suggest': current_sugs,
                        'candidata': palabra,
                    }

            # Actualizar prev_str_cache[i] en O(1) una vez fijado back[i]
            if back[i] >= 0:
                prefix = prev_str_cache[back[i]]
                prev_str_cache[i] = (prefix + ' ' + words_at[i]).strip() if prefix else words_at[i]

        resultado, details = self.reconstruir_segmentacion(n, back, words_at, seg_info_at)
        self.correction_log.append({
            'original': anomaly_group_str,
            'correcciones': details,
        })
        print("Reconstruccion final:", resultado)
        return resultado

    
    def reconstruir_segmentacion(
        self, n: int, back: np.ndarray, words_at: list, seg_info_at: list = None
    ) -> tuple[list[str], list[dict]]:
        """Reconstruye la lista de palabras corregidas y los detalles de cada segmento.

        Returns:
            (palabras, details) donde details es una lista de dicts con claves
            'anomalia', 'suggest' y 'candidata' para cada segmento elegido.
        """
        palabras = []
        details = []
        i = n
        while i > 0:
            j = back[i]
            if j < 0:
                break
            palabras.append(words_at[i])
            if seg_info_at is not None and seg_info_at[i] is not None:
                details.append(seg_info_at[i])
            i = j
        return list(reversed(palabras)), list(reversed(details))


    def get_str_for_tuple(self, tuple_list: list[tuple[Token, bool]]) -> str:
        return ' '.join(t[0].text for t in tuple_list)


    def similitud(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()
    


def limpiar_espacios_basicos(text:str) -> str:
    # 1. Unir cortes de linea con guion al final "pala-\nbra" -> "palabra"
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)

    text = text.replace('-', ' ')

    # 2. Reemplazar saltos de linea internos por espacio (no se toca dobles \n\n)
    # linea simple -> espacio
     
    text = re.sub(r'(?<!\n)\s*\n\s*(?!\n)', ' ', text)

    # Colapsar multiples espacios /tab a uno solo
    text = re.sub(r'\s+', ' ', text)

    return text.strip() 

str_ch = StringChecker()  # usa lm_left_ctx=3, lm_right_ctx=3 por defecto

new_text = '''
El\contecimiento de " l a acera" 
sigue llenando con sus ecos clamo-
rosos toda la actualidad: fué un es-
pectáculo original, algo distinto á 
lo corriente, y fué una fiesta impo-
nente, resonante y rebosante, un 
grande y senado éxito. 
La "acer?, del Louvre" es una 
institución histórica, algo real y 
positivo que ya existía, que exis-
tía de antaño, con sus anales lle-
nes de actos y de fechas, llenos de 
vida: ahora lo que se ha hecho ha 
sido daJle "existencia oficial". 
La "acera del Louvre" ha veni-
do, como institución, á pesar en la 
vida pública, á la manera que en 
algunos países llegan ciertas per-
sonalidades á asumir el cargo de 
Senador: "por derecho propio". 
Sólo-que en eso de senadores por 
derecho propio suele ser la mu-
cha edad uno de los títulos, y en 
el derecho propio de "la acera" 
el título es el de "su eterna juven-
tud". 
La "acera" es realmente una 
institución, una institución histó-
rica nacional; en aquellos muros, 
aquellas paredes y aquellas colum-
nas está escrita toda entera la 
historia de Cuba, la historia poli 
tica, la historia social, la historia 
mundana, la historia regocijada y 
galante: " l a acera" ha, sido cam 
po propio de héroes, de patriotas, 
I de hidalgos, de caballeros y de 
galanss.v 
Ese pedacito encantado de la 
I urbr íabanera tiene escritas en las 
¡ene' , sus memorias. Y 
hay en 
i' 
xs compendiadas páginas heroi 
as, páginas patrióticas, páginas 
I caballerescas. 
' " L a acerf 
tiene sus aniversa-' 
« L a 
acera", 
como todos 
los! 
nos, sus ^iemcirides patrióticas, ! g r a n ¿ e g compendios, ha tenido sus 
sus efemérides luctuosas, tiene sus ¿pocas variadas y ha inspirado va-; 
héroes, tiene sus mártires, tiene l i a,¿ a s impresiones, 
sus glorias, tiene sus muertos, tde- 
j j a habido épocas en las que se 
ne la estela de sus arrestos y ga- 
l e b a tenido miedo á " l a acera" 
llardías, tiene sus locuras y sus y ¿pocas en que " l a acera" atraía 
arrebatos, tiene sus alegrías, tiene irre3istiblemente. Y por lo general 
sus tristezas: tiene de todo menos siempre, ahora como antes, la ace-
página de indignidad, menos vile- 
r a infUnde cierto temor y tiene 
za, menos traiciones, menos alevo- 
cjerto resorte, "algo que llama, 
sías, menos incidías y arterias. Lo 
q u e tira como de un cordelito": 
que es indigno ó vil, de allí se "ha 
8S s u característica: á " l a acera" 
despegado" prontamente. 
>-3e i a teme y se la ama". 
La acera ha dado á Cuba muchos 
p a s a como con el mar, ccmo con 
hombres. 
Y los ha dado en tocias 
ei m a r vivo: cuando está eneres 
la= esferas y para todas las maniT pado, cuando las olas se levantan 
festaciones de ia patria actividad, imponentes, cuando parece que 
Ha dado guerreros, duelistas, es-
critores, generales, poetas, hom-
bres de ciencia, patriotas, catedrá-
ticos, oradores, trovadores, pinto, 
res. Y ha dado á torrentes gente 
galante, animada, decidora y ena-
morada, enamorada de "la diosa'" 
á que allí se rinde culto 
jer. J> 
La acera tiene en su escudo, es-
crita en campo azul esta divisa: 
"entereza, hidalguía y amor". 
Y tiene " l a acera" una fuerza,! 
una gran fuerza, la fuerza de algo I 
que vale más que las armas, más 
que el talento, más que el dinero, 
más que la palabra, más que el sa-
ber : tiene la fuerza dominadora é 
incontrastable de la SIMPATIA 
De ahí la tremenda atracción de 
aquel sitio, de ahí la fuerza de 
imán de aquel lugar: de ahí el for-
midable arrastre de " l a acera". 
amenaza tragarnos, nos alejamos, 
huimos, pero sólo unos pasos y pa-
ra volver á acercarnos en seguida. 
Sobretodo nc abandonamos el lu-
gar. 
Tiene aquel espectáculo peligro; 
pero tiene belleza, tiene atracción, 
l a mu~ftiene vida. 
~ 
En otra cosa se parece la ace-
ra al mar: es en su "eterna juven-
tud." 
En el mar las olas se suceden 
lunas á otras; pasan días, meses, 
sños, ciclos, y el mar "no enveje-
ce": cada vez que uno se acerca 
al litoral encuentra allí las olas 
frescas, üjaevas, asuf.es, alegres, 
juguetonas, formidables y siem-
pre en juvenil actividad dibujan-
do como eterna sonrisa de sus la-
bios la blanca espumilla de 
las 
crestas. 
Así es "la acera": la acera es 
días, 
anos, ciclos, generaciones y de 
allí brota el mismo aroma de pri-
mavera, la misma palpación de 
. La siriipatía es la gran palanca, 
el gran peder humano: la simpatía eternamente joven; pasan 
, 
. 
,, 
ovino 
«tii/ilna 
nrpYl Aram rinñ? 
es 
la varita magica . 
'''

# Reusar los modelos ya cargados en str_ch en vez de cargar instancias duplicadas
nlp = str_ch.nlp
hobj = str_ch.hobj

new_text = limpiar_espacios_basicos(new_text)
doc_ = nlp(new_text)

anomaly_chain = []
for token in doc_:
    is_valid = str_ch._spell_cache.get(token.text)
    if is_valid is None:
        is_valid = hobj.spell(token.text)
        str_ch._spell_cache[token.text] = is_valid
    anomaly_chain.append((token, is_valid))

# Pre-calentar el cache de suggest con los tokens anomalos completos
for token, valid in anomaly_chain:
    if not valid and token.text not in str_ch._suggest_cache:
        str_ch._suggest_cache[token.text] = hobj.suggest(token.text)[:5]

start = time.perf_counter()
anomaly_chain_res = str_ch.correjir_cadena(anomaly_chain)
print()
result = str_ch.last_corrected_text

print(result)

corrections = str_ch.correction_log

end = time.perf_counter()

print(f"tiempo: {end - start:.6f} seg")

print()