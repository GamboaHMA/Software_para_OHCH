import nltk

nltk.download()

print(nltk.data.path)

from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

print(stopwords.fileids())

print(wn.langs())

# synsets_bank = wn.synsets('bank')

# for syn in synsets_bank:
#     print(syn, syn.lemma_names('spa'))

# print(stopwords.words('spanish'))

text = '''Para construir un estimador compuesto, los transformadores suelen combinarse con otros transformadores o con predictores
        (como clasificadores o regresores). La herramienta más común para componer estimadores es una canalización. Las canalizaciones 
        requieren que todos los pasos, excepto el último, sean un transformador.'''

w = nltk.tokenize.word_tokenize(text=text)

for w_ in w:
    print(w_)

w2 = [w_ for w_ in w if w_ not in stopwords.words('spanish')]

print("+++++++++++++++++++++++++")

for w2_ in w2:
    print(w2_)

n_w = []

for w_ in w:
    if w_ not in w2:
        n_w.append(w_)

for n_w_ in n_w:
    print(n_w_)

print(n_w)
print('++++++++++++++++++')
# print(stopwords.words('spanish'))
# print(nltk.__version__)

lmt = WordNetLemmatizer()

lmt.lemmatize('running', pos='v')