# %%
import pandas as pd
import spacy

# %%
nlp = spacy.load('en_core_web_sm')

# %%
df = pd.read_csv("./sarcasm_v2/GEN-sarc-notsarc.csv")

# %%
sarc_df = df["sarc" == df["class"]]["text"]
notsarc_df = df["notsarc" == df["class"]]["text"]

# %%
def get_docs_bigrams(texts):
    docs_bigrams = []
    for text in texts:
        doc = nlp(text)
        sent_bigrams = []
        for sent in doc.sents:
            sent_bigrams.append(
                [[sent[ind], sent[ind + 1]] for ind in range(len(sent)-1)]
            )
        docs_bigrams.append(sent_bigrams)
    return docs_bigrams

# %%
# bigrams = get_docs_bigrams(sarc_df)

# %%
def get_docs_trigrams(texts):
    docs_trigrams = []
    for text in texts:
        doc = nlp(text)
        sent_trigrams = []
        for sent in doc.sents:
            sent_trigrams.append(
                [
                    [sent[ind], sent[ind + 1], sent[ind + 2]]
                    for ind in range(len(sent)-2)
                ]
            )
        docs_trigrams.append(sent_trigrams)
    return docs_trigrams

# %%
# trigrams = get_docs_trigrams(sarc_df)

# %%
from spacy import displacy

def display_dep(doc):
    displacy.render(
        doc, style="dep")

# %%
EXCLUDE_DEPS = ["punct", "ccmp", "prep", "pobj", "X", "space", "", "ROOT"]
def get_doc_dep_bigrams(texts):
    docs_bigrams = []
    for text in texts:
        doc = nlp(text)
        sent_trigrams = []
        for sent in doc.sents:
            for token in sent:
                if token.dep_ in EXCLUDE_DEPS:
                    continue
                sent_trigrams.append([token.text, token.head.text, token.dep_])
            docs_bigrams.append(sent_trigrams)
    return docs_bigrams

doc_dep_bigrams = get_doc_dep_bigrams(sarc_df)
