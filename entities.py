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
def get_corpus_entities(texts):
    docs_entities = {}
    for text in texts:
        doc = nlp(text)
        for ent in doc.ents:
            doc_ent_set = docs_entities.get(ent.label_, set())
            doc_ent_set.add(ent.text)
            docs_entities[ent.label_] = doc_ent_set
    return docs_entities

# %%
get_corpus_entities(sarc_df)
