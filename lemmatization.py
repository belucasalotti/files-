# %% markdown
# Lematización

# %%
import spacy
import numpy as np
import pandas as pd
import seaborn as sn

from collections import Counter

# %%
nlp = spacy.load('en_core_web_sm')

# %% markdown
Unimos todos los dataframes para trabajar directamente con
todos los datos con los que disponemos

# %%
df0 = pd.read_csv("../diplodatos2020-deteccion_sarcasmo/sarcasm_v2/sarcasm_v2/GEN-sarc-notsarc.csv")
df1 = pd.read_csv("../diplodatos2020-deteccion_sarcasmo/sarcasm_v2/sarcasm_v2/HYP-sarc-notsarc.csv")
df2 = pd.read_csv("../diplodatos2020-deteccion_sarcasmo/sarcasm_v2/sarcasm_v2/RQ-sarc-notsarc.csv")

df = pd.concat([df0, df1, df2], ignore_index=True)

# %% markdown
Definamos variables que serán de utilidad.

# %%
sarc_df = df["sarc" == df["class"]]
not_sarc_df = df["notsarc" == df["class"]]

# %% markdown
Funciones útiles

# %%
flatten_list = lambda nested_list: [
    el for sublist in nested_list for el in sublist]

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

# %%
sarc_token_texts = [nlp(snts.lower()) for snts in sarc_df['text']]
not_sarc_token_texts = [
    nlp(snts.lower()) for snts in not_sarc_df['text']]

# %%
sarc_token_list = flatten_list(sarc_token_texts)
not_sarc_token_list = flatten_list(not_sarc_token_texts)

# %% markdown
## Lematización

# %%
sarc_lemm_tokens = [word.lemma_ for word in sarc_token_list]
not_sarc_lemm_tokens = [word.lemma_ for word in not_sarc_token_list]

sarc_lemm_freq = Counter(sarc_lemm_tokens)
not_sarc_lemm_freq = Counter(not_sarc_lemm_tokens)

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
