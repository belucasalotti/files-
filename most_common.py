# %% markdown
# Analisis y Visualización sobre DFs de Sarcasmo y No Sarcasmo

# %%
import nltk
import numpy as np
import pandas as pd
import seaborn as sn

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# %% markdown
Unimos todos los dataframes para trabajar directamente con
todos los datos con los que disponemos

# %%
df0 = pd.read_csv("../diplodatos2020-deteccion_sarcasmo/sarcasm_v2/sarcasm_v2/GEN-sarc-notsarc.csv")
df1 = pd.read_csv("../diplodatos2020-deteccion_sarcasmo/sarcasm_v2/sarcasm_v2/HYP-sarc-notsarc.csv")
df2 = pd.read_csv("../diplodatos2020-deteccion_sarcasmo/sarcasm_v2/sarcasm_v2/RQ-sarc-notsarc.csv")

df = pd.concat([df0, df1, df2], ignore_index=True)

# %% markdown
## Analisis de palabras más comunes.

Primero definamos funciones que nos van a servir.

# %%
flatten_list = lambda nested_list: [
    el for sublist in nested_list for el in sublist]

lemmatizer = WordNetLemmatizer()
get_lemm_tokens = lambda tokens: [lemmatizer.lemmatize(t) for t in tokens]

get_token_freq = lambda tokens: nltk.FreqDist(tokens)

stop_words = set(stopwords.words('english'))
rm_stop_words = lambda tokens: [t for t in tokens if t not in stop_words]

# %%
def compare_freq(most_common_freq, cmp_freq,
                 mc_label, cmp_label,
                 mc_color, cmp_color):
    """
    This function compares the frequency of the most common tokens
    of `most_common_freq` with the frequency they have in `cmp_freq`.
    """
    most_common = most_common_freq.most_common(30)

    most_common_words = [x for x, y in most_common]
    most_common_freqs = [y/len(most_common_freq) for x, y in most_common]

    cmp_freq_in_mc = [cmp_freq[x]/len(cmp_freq) for x, y in most_common]

    most_common = sn.lineplot(
        x=most_common_words,
        y=cmp_freq_in_mc,
        label=cmp_label,
        sort=False,
        color=cmp_color
    )
    not_sarc_gr = sn.lineplot(
        x=most_common_words,
        y=most_common_freqs,
        label=mc_label,
        sort=False,
        color=mc_color
    )
    rot_lab_ns = most_common.set_xticklabels(
        labels=most_common_words, rotation=90)

# %% markdown
Definamos variables que también serán de utilidad.

# %%
sarc_df = df["sarc" == df["class"]]
not_sarc_df = df["notsarc" == df["class"]]

# %%
sarc_token_texts = [nltk.word_tokenize(snts) for snts in sarc_df['text']]
not_sarc_token_texts = [
    nltk.word_tokenize(snts) for snts in not_sarc_df['text']]

# %%
sarc_token_list = flatten_list(sarc_token_texts)
not_sarc_token_list = flatten_list(not_sarc_token_texts)

# %%
lower_sarc_tokens = [token.lower() for token in sarc_token_list]
lower_not_sarc_tokens = [token.lower() for token in not_sarc_token_list]

# %%
sarc_token_freq = get_token_freq(lower_sarc_tokens)
not_sarc_token_freq = get_token_freq(lower_not_sarc_tokens)

# %% markdown
Veamos los primeros analisis

# %% markdown
Palabras mas frecuentes en textos con sarcasmo:

# %%
sarc_token_freq.plot(30, cumulative=False)

# %% markdown
Palabras mas frecuentes en textos sin sarcasmo:

# %%
not_sarc_token_freq.plot(30, cumulative=False)

# %% markdown
Si comparamos ambos gráficos podemos ver como articulos como "the", "and" o
símbolos como el signo de puntuación o el apóstrofe, aparecen prácticamente con
la misma frecuencia en ambos casos, por lo cuál no nos darían información
relevante al análisis de sarcasmo.

# %% markdown
Ahora pasaremos a realizar una comparación gráfica entre la frecuencia de las
palabras en textos sarcasticos y no sarcasticos, donde podremos visualizar más
facilmente relaciones, picos y diferencias entre palabras en ambos tipos de
textos

# %% markdown
Palabras más frecuentes en textos con sarcasmo comparadas con la frecuencia de
las mismas en textos sin sarcasmo:

# %%
compare_freq(
    sarc_token_freq, not_sarc_token_freq,
    "sarcasm", "not sarcasm",
    "blue", "red",
)

# %% markdown
Palabras más frecuentes en textos sin sarcasmo comparadas con la frecuencia de
las mismas en textos con sarcasmo:

# %%
compare_freq(
    not_sarc_token_freq, sarc_token_freq,
    "not sarcasm", "sarcasm",
    "red", "blue",
)

# %% markdown
Observamos que en los textos de sarcasmo, palabras como "you", y signos de
excalamación ("!"), o pregunta ("?"), experimentan picos de crecimiento, por
sobre las mismas palabras en los textos de no sarcasmo. Es decir, se
repiten con mas frecuencia en los textos de Sarcasmo.
Con esta información, podemos realizar dos hipótesis, que iremos contrarestando
a lo largo del trabajo. Hipótesis 1): el uso de la palabra "you" con frecuencia
se debe a que las personas sercástics se refieren a otras personas cuando hacen
comentarios de este tipo. Los signos de admiración se emplean en los comentarios
sarcásticos ya que son comentarios efusivos, hechos con emocionalidad. Los
signos de interrogación se repiten mas en los textos sarcásticos ya que la
mayoría de estos comentarios son hechos en modo de pregunta hacia otra persona.
Hipótesis 2): los datos están sucios por signos de admiración y puntuación
que no corresponden al comentario original, o que no representan énfasis o
interrogación en el comentario.

# %%
not_sarc_token_freq['!']
sarc_token_freq['!']
len(sarc_token_freq)
len(not_sarc_token_freq)

# %% markdown
Tomando como ejemplo el signo de admiración, podemos ver como aparece
triplicado en los textos de sarcasmo, si lo comparamos con el numero de veces
que aparece en los textos de no sarcasmo. Además, la cantidad de palabras dentro
de los textos con sarcasmo es menor, por lo cuál resulta importante destacar que
el signo de admiración aparece triplicado en una menor cantidad total de palabras.

# %% markdown
### Analisis con lematización

# %% markdown
Con la lematizacion se intenta llegar a la raíz de cada palabra, de manera tal
que tengamos una familia de palabras para cada texto (sarc y no sarc), donde
cada familia contenga todas las palabras dentro de si misma.
Luego compararemos cada familia de palabras para ver sus similitudes y diferencias.

# %%
sarc_lemm_tokens = get_lemm_tokens(sarc_token_list)
not_sarc_lemm_tokens = get_lemm_tokens(not_sarc_token_list)

# %%
sarc_lemm_freq = get_token_freq(lower_sarc_tokens)
not_sarc_lemm_freq = get_token_freq(lower_not_sarc_tokens)

# %%
compare_freq(
    sarc_lemm_freq, not_sarc_lemm_freq,
    "sarcasm", "not sarcasm",
    "blue", "red",
)

# %%
compare_freq(
    not_sarc_lemm_freq, sarc_lemm_freq,
    "not sarcasm", "sarcasm",
    "red", "blue",
)

# %% markdown
Una vez realizada la lematización y observados los gráficos obtenidos, no
encontramos diferencias o similitudes que nos resultaran útiles para nuestra
investigación.

# %% markdown
### Analisis con lematización y sin Stopwords

# %%
sarc_lemm_nsw_tokens = rm_stop_words(sarc_lemm_tokens)
not_sarc_lemm_nsw_tokens = rm_stop_words(not_sarc_lemm_tokens)

# %%
sarc_lemm_nsw_freq = get_token_freq(sarc_lemm_nsw_tokens)
not_sarc_lemm_nsw_freq = get_token_freq(not_sarc_lemm_nsw_tokens)

# %%
compare_freq(
    sarc_lemm_nsw_freq, not_sarc_lemm_nsw_freq,
    "sarcasm", "not sarcasm",
    "blue", "red",
)

# %%
compare_freq(
    not_sarc_lemm_nsw_freq, sarc_lemm_nsw_freq,
    "not sarcasm", "sarcasm",
    "red", "blue",
)

# %% markdown
Una vez eliminadas las stopwords, podemos observar en ambos gráficos
que las palabras en cada uno de los textos ya no presentan tantos picos, sino
que la frecuencia de las mismas se vuelve similar en textos sarcásticos y no
sarcásticos.

# %% markdown
### Analisis del Uso de Mayusculas en Sarcasmo y No Sarcasmo

# %%
# the "I" token was really messing the graphs
sarc_upper_tokens = [w for w in sarc_token_list if w.isupper() and w != "I"]
not_sarc_upper_tokens = [
    w for w in not_sarc_token_list if w.isupper() and w != "I"]

# %%
"{} upper tokens from {} sarcastic ones".format(
    len(sarc_upper_tokens),
    len(sarc_token_list),
)

# %%
"while there are {} upper tokens from {} non-sarcastic ones".format(
    len(not_sarc_upper_tokens),
    len(not_sarc_token_list)
)

# %%
sarc_upper_freq = get_token_freq(sarc_upper_tokens)
not_sarc_upper_freq = get_token_freq(not_sarc_upper_tokens)

# %%
compare_freq(
    sarc_upper_freq, not_sarc_upper_freq,
    "sarcasm", "not sarcasm",
    "blue", "red",
)

# %%
compare_freq(
    not_sarc_upper_freq, sarc_upper_freq,
    "not sarcasm", "sarcasm",
    "red", "blue",
)

# %%
sarc_upper_texts = [w.isupper() for w in sarc_df["text"]]
not_sarc_upper_texts = [w.isupper() for w in not_sarc_df["text"]]

# %% markdown
Porcentaje de aparición de palabras en mayúscula en textos con sarcasmo y sin:

# %%
len(sarc_upper_tokens)/len(sarc_token_list)
len(not_sarc_upper_tokens)/len(not_sarc_token_list)

# %% markdown
Nos pareció importante conocer si existen palabras en ambos textos (sarc y no
sarc) que estén escritas en mayúscula. Sin embargo, luego de los análisis
realizados no encontramos información muy relevante para nuestra investigación.

# %% markdown
### Analisis de tipo de palabras

# %% markdown
En este punto realizaremos un análisis de palabras separándolas por grupos:
sustantivos, adjetivos y adverbios. Intentaremos conocer la relación de la
frecuencia de aparición de palabras que forman parte de cada uno de éstos grupos
en textos sarcásticos y no sarcásticos.

# %%
get_pos_tag = lambda text: nltk.pos_tag(text)

# %%
is_noun = lambda tag: tag == 'NN'
is_adjetive = lambda tag: tag == 'JJ'
is_adverb = lambda tag: tag == 'RB'

# %%
sarc_tagged_tokens = get_pos_tag(sarc_token_list)
not_sarc_tagged_tokens = get_pos_tag(not_sarc_token_list)

# %% markdown
#### Sustantivos

# %%
sarc_nouns = [n for n, t in sarc_tagged_tokens if is_noun(t)]
not_sarc_nouns = [n for n, t in not_sarc_tagged_tokens if is_noun(t)]

# %%
sarc_nouns_freq = get_token_freq(sarc_nouns)
not_sarc_nouns_freq = get_token_freq(not_sarc_nouns)

# %%
compare_freq(
    sarc_nouns_freq, not_sarc_nouns_freq,
    "sarcasm", "not sarcasm",
    "blue", "red",
)

# %%
compare_freq(
    not_sarc_nouns_freq, sarc_nouns_freq,
    "not sarcasm", "sarcasm",
    "red", "blue",
)

# %% markdown
En el caso de los sustantivos, vemos que aparecen con mayor frecuencia en los
textos no sarcásticos. Pero en este punto es importante aclarar que los textos
no sarcásticos contienen mas palabras que los sarcásticos.

# %%
len(sarc_nouns)/len(sarc_token_list)
len(not_sarc_nouns)/len(not_sarc_token_list)

# %% markdown
Al haber calculado el porcentaje de aparición de sustantivos en cada tipo de
texto podemos ver que son muy similares, en el caso del sarcasmo de un 12,6% y
el del no sarcasmo de un 13,1%.


# %% markdown
#### Adjetivos

# %%
sarc_adj = [n for n, t in sarc_tagged_tokens if is_adjetive(t)]
not_sarc_adj = [n for n, t in not_sarc_tagged_tokens if is_adjetive(t)]

# %%
sarc_adj_freq = get_token_freq(sarc_adj)
not_sarc_adj_freq = get_token_freq(not_sarc_adj)

# %%
compare_freq(
    sarc_adj_freq, not_sarc_adj_freq,
    "sarcasm", "not sarcasm",
    "blue", "red",
)

# %%
compare_freq(
    not_sarc_adj_freq, sarc_adj_freq,
    "not sarcasm", "sarcasm",
    "red", "blue",
)

# %% markdown
Habiendo observado los gráficos, en el caso de los adjetivos no podemos sacar
conclusiones que nos ayuden a diferenciar textos sarcásticos de no sarcásticos.

# %% markdown
#### Adverbs

# %%
sarc_adv = [n for n, t in sarc_tagged_tokens if is_adverb(t)]
not_sarc_adv = [n for n, t in not_sarc_tagged_tokens if is_adverb(t)]

# %%
sarc_adv_freq = get_token_freq(sarc_adv)
not_sarc_adv_freq = get_token_freq(not_sarc_adv)

# %%
compare_freq(
    sarc_adv_freq, not_sarc_adv_freq,
    "sarcasm", "not sarcasm",
    "blue", "red",
)

# %%
compare_freq(
    not_sarc_adv_freq, sarc_adv_freq,
    "not sarcasm", "sarcasm",
    "red", "blue",
)

# %% markdown
En el caso de los advervios, observamos que la frecuencia de aparición de los
mismos en textos de sarcasmo y no sarcasmo es bastante similar. Por lo tanto,
tampoco obtenemos acá información relevante. Si nos parece importante destacar
el caso del "not", en los textos que no tienen sarcasmo se utiliza mucho mas
que en los que sí lo tienen.
