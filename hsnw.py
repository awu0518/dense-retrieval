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
import pytrec_eval
TOPK = 100

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
    Loads a TREC-style qrels file:
        <query_id> <unused> <doc_id> <relevance>
    Returns:
        dict[str, dict[str, int]] in the format:
        { "qid": { "docid": relevance, ... }, ... }
    """
    df = pd.read_csv(
        qrels_path,
        sep=r'\s+',  # handles both tabs and spaces
        header=None,
        names=['qid', 'unused', 'docid', 'label']
    )

    qrels = {}
    for qid, docid, label in zip(df.qid, df.docid, df.label):
        qid, docid = str(qid), str(docid)
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][docid] = int(label)
    
    return qrels

def get_relevance_label(qrels_dict, query_id, doc_id):
    """
    Returns (relevance_score, found) for a given (query_id, doc_id) pair.
    - label: relevance_score if found, else 0
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
    
def format_run(qids, docids, inds, scores, topk=1000):
    """
    The response from the  
        dict[str, dict[str, int]] in the format:
        { "qid": { "docid": score, ... }, ... }
    """
    run = {}
    for i, qid in enumerate(qids):
        qid = str(int(qid)) if not isinstance(qid, str) else qid
        run[qid] = {}
        for rank in range(topk):
            doc_idx =  [i, rank]
            score = float(scores[i, rank])
            run[qid][str(int(docids[doc_idx]))] = score
    return run

dids, doc_embeddings = load_h5_embeddings(r"ms_marco\msmarco_passages_embeddings_subset.h5")
qid, q_embeddings = load_h5_embeddings(r"ms_marco\msmarco_queries_dev_eval_embeddings.h5")
doc_embeddings = np.array(doc_embeddings).astype("float32") # because of how faiss works
q_embeddings = np.array(q_embeddings).astype("float32") # because of how faiss works


index = build_hnsw_ip(doc_embeddings)

inds, scores = search_hnsw(index, q_embeddings, TOPK)



docIDS = []
 
for i in range(TOPK):
    docIDS.append(dids[inds[i]])

# print(docIDS)
qrels = {}
qrels1 = load_qrels("ms_marco/qrels.dev.tsv")
# qrels2 = load_qrels("ms_marco/qrels.eval.two.tsv")
# qdev = load_qrels("ms_marco/qrels.dev.tsv")
evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg_cut.10'})  
print(len(inds))
run = format_run(qid, dids, inds, scores, TOPK)
results = evaluator.evaluate(run)
print(len(results.items()))
for query_id, metrics in results.items():
    print(f"{query_id}: {metrics}")
##
# After a search, i have the docids and the docid score for each query. \
# i need a dict of that queryid mapped to the docid and relavacne score from the files
# i need a dict of the queryid mapped to the docid and the score calculated by distance 
