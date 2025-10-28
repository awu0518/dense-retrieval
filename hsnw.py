# Implementation of query search using hsnw. The dataset setup is based off of this: https://github.com/facebookresearch/faiss/wiki/Faster-search
# conda install -c pytorch faiss-cpu
# pip install h5py 
# 350 The melting point of a substance is the temperature at which it changes from. a solid to a liquid.
# #  a liquid to a solid. a gas to a solid. a solid to a gas. The boiling point of a substance is the temperature at which it changes from. 
# a liquid to a gas. a liquid to a solid. a gas to a solid. a solid to a liquid. When a gas is compressed it changes state into a. liquid.
from read_h5 import load_h5_embeddings
import numpy as np
import faiss
import pandas as pd

def build_hnsw_ip(X, M=8, efC=100, efS=100):
    d = X.shape[1]
    index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = efC
    index.hnsw.efSearch = efS
    # Optional: train() not required for HNSWFlat
    index.add(X)  # X must be float32
    return index

def search_hnsw(index, Q, topk=1000, efS=None):
    if efS is not None: index.hnsw.efSearch = efS
    scores, I = index.search(Q.astype('float32'), topk)  # dot-product scores
    return I, scores

def load_qrels(qrels_path):
    """
    Loads qrels (query_id, passage_id, relevance) into a lookup dictionary.
    Example row format: 244678\t7077728\t1
    """
    df = pd.read_csv(qrels_path, sep='\t', header=None, names=['qid', 'pid', 'label'])
    qrels = {(int(qid), int(pid)): int(label)
             for qid, pid, label in zip(df.qid, df.pid, df.label)}
    return qrels

def get_relevance_label(qrels_dict, query_id, doc_id):
    """
    Returns (label, found) for a given (query_id, doc_id) pair.
    - label: relevance label if found, else 0
    - found: True if pair exists in qrels file, False otherwise
    """

    # Convert numpy or string arrays to plain ints
    if isinstance(query_id, (list, tuple, np.ndarray)):
        query_id = int(query_id[0])
    else:
        query_id = int(query_id)

    if isinstance(doc_id, (list, tuple, np.ndarray)):
        doc_id = int(doc_id[0])
    else:
        doc_id = int(doc_id)

    # Lookup
    key = (query_id, doc_id)
    if key in qrels_dict:
        return qrels_dict[key], True
    else:
        return 0, False

dids, doc_embeddings = load_h5_embeddings(r"C:\Users\akani\Documents\dense-retrieval\ms_marco\msmarco_passages_embeddings_subset.h5")
qid, q_embeddings = load_h5_embeddings(r"C:\Users\akani\Documents\dense-retrieval\ms_marco\msmarco_queries_dev_eval_embeddings.h5")
doc_embeddings = np.array(doc_embeddings).astype("float32") # because of how faiss works
q_embeddings = np.array(q_embeddings).astype("float32") # because of how faiss works

d = len(doc_embeddings[0]) # dimension of each vector


index = build_hnsw_ip(doc_embeddings)

inds, scores = search_hnsw(index, q_embeddings, 1000)


# print(scores[:20])

docIDS = []
# for i in range(10):
# print(inds[0])
for i in range(1000):
    docIDS.append(dids[inds[i]])

# print(docIDS)

qrels1 = load_qrels("ms_marco/qrels.eval.one.tsv")
qrels2 = load_qrels("ms_marco/qrels.eval.two.tsv")
qdev = load_qrels("ms_marco/qrels.dev.tsv")
print("qrels1")
for i in range(500):
    label, found = get_relevance_label(qrels1, q_embeddings[0], docIDS[i])
    if found:
        print(label)  # → 1
print("qrels2")

for i in range(500):
    label, found = get_relevance_label(qrels2, q_embeddings[0], docIDS[i])
    if found:
        print(label)  # → 1
print("qdev")

for i in range(500):
    label, found = get_relevance_label(qdev, q_embeddings[0], docIDS[i])
    if found:
        print(label)  # → 1