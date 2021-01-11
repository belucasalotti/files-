# %%
import math
import pandas as pd
import spacy
import numpy as np
import seaborn as sn

# %%
nlp = spacy.load('en_core_web_sm')

# %%
df = pd.read_csv("./sarcasm_v2/GEN-sarc-notsarc.csv")

# %%
sarc_df = df["sarc" == df["class"]]["text"]
notsarc_df = df["notsarc" == df["class"]]["text"]

# %%
# tf
def get_doc_tf(text):
    doc = nlp(text)
    doc_tf = {}

    tokens = [
        token for token in doc
        if not (token.is_stop or token.is_punct or token.is_space)
    ]
    token_count = len(tokens)

    for t in tokens:
        doc_tf[t.lower_] = doc_tf.get(t.lower_, 0) + 1

    for t in doc_tf:
        doc_tf[t] = doc_tf[t] / token_count

    return doc_tf

# %%
# idf
def get_docs_idf(texts):
    docs_idf = {}
    doc_count = len(texts)

    for text in texts:
        doc = nlp(text)

        # important to notice this is a set (not an array)
        # and therefore tokens will only appear once
        tokens = {
            token for token in doc
            if not (token.is_stop or token.is_punct or token.is_space)
        }
        token_count = len(tokens)

        for t in tokens:
            docs_idf[t.lower_] = docs_idf.get(t.lower_, 0) + 1

    for token in docs_idf:
        docs_idf[token] = math.log(doc_count/docs_idf[token])

    return docs_idf

# %%
# tf-idf
def get_doc_tfidf(doc_tf, docs_idf):
    doc_tfidf = {
        k: doc_tf[k] * docs_idf[k]
        for k in doc_tf
    }

    return doc_tfidf

# %%
def get_tfidf_sum(token, docs_tfidf):
    tfidf_sum = 0

    for doc_tfidf in docs_tfidf:
        tfidf_sum += doc_tfidf.get(token, 0)

    return tfidf_sum

# %%
def get_above_avg_tokens(docs_tfidf):
    # get the tokens that have a tf-idf above average of each document
    above_avg_tfidf = []

    for tfidf in docs_tfidf:
        for token in tfidf:
            above_avg_tokens = {}
            if tfidf[token] > np.average(list(tfidf.values())):
                above_avg_tokens[token] = tfidf[token]
        # if no token is above average we wont be taking it into account
        if above_avg_tokens != {}:
            above_avg_tfidf.append(above_avg_tokens)

    return above_avg_tfidf

# %%
def get_sum_tfidf(above_avg_tfidf):
    tokens_tfidf_sum = {}

    for doc in above_avg_tfidf:
        for token in doc:
            curr_token_value = tokens_tfidf_sum.get(token, 0)
            tokens_tfidf_sum[token] = curr_token_value + doc[token]
    # sort important tokens
    sorted_tfidf_sum = sorted(
        tokens_tfidf_sum.items(), key=lambda x: x[1], reverse=True)

    return sorted_tfidf_sum

# %%
def lineplot_graph(tuples, label):
    x = [x for x, y in tuples]
    y = [y for x, y in tuples]

    graph = sn.lineplot(
        x=x,
        y=y,
        label=label,
        sort=False,
    )
    x_labels_fix = graph.set_xticklabels(
        labels=x, rotation=90)


# %%
sarc_docs_idf = get_docs_idf(sarc_df)
notsarc_docs_idf = get_docs_idf(notsarc_df)

# %%
sarc_docs_tf = [get_doc_tf(doc) for doc in sarc_df]
notsarc_docs_tf = [get_doc_tf(doc) for doc in notsarc_df]

# %%
sarc_docs_tfidf = [
    get_doc_tfidf(doc_tf, sarc_docs_idf)
    for doc_tf in sarc_docs_tf
]
notsarc_docs_tfidf = [
    get_doc_tfidf(doc_tf, notsarc_docs_idf)
    for doc_tf in notsarc_docs_tf
]

# %%
ab_avg_sarc_tokens = get_above_avg_tokens(sarc_docs_tfidf)
ab_avg_notsarc_tokens = get_above_avg_tokens(notsarc_docs_tfidf)

# %%
sarc_tfidf_sum = get_sum_tfidf(ab_avg_sarc_tokens)
notsarc_tfidf_sum = get_sum_tfidf(ab_avg_notsarc_tokens)

# %%
x_sarc = [x for x,y in sarc_tfidf_sum]
y_sarc = [y for x,y in sarc_tfidf_sum]
sarc_label = "SARC TF-IDF"

x_notsarc = [x for x,y in notsarc_tfidf_sum]
y_notsarc = [y for x,y in notsarc_tfidf_sum]
notsarc_label = "NOT SARC TF-IDF"

# %%
lineplot_graph(
    sarc_tfidf_sum[:15], sarc_label,
)
# %%
lineplot_graph(
    notsarc_tfidf_sum[:15], sarc_label,
)
