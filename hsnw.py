# Implementation of query search using hsnw. The dataset setup is based off of this: https://github.com/facebookresearch/faiss/wiki/Faster-search
# conda install -c pytorch faiss-cpu
# pip install h5py pytrec_eval
# 350 The melting point of a substance is the temperature at which it changes from. a solid to a liquid.
# #  a liquid to a solid. a gas to a solid. a solid to a gas. The boiling point of a substance is the temperature at which it changes from. 
# a liquid to a gas. a liquid to a solid. a gas to a solid. a solid to a liquid. When a gas is compressed it changes state into a. liquid.
from read_h5 import load_h5_embeddings
import numpy as np
import faiss
import pandas as pd
import pytrec_eval
import time

TOPK = 100
M = 4
EFS = 150
EFC = 150


def build_hnsw_ip(X, M=8, efC=100, efS=100):
    d = X.shape[1]
    index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = efC
    index.hnsw.efSearch = efS
    # Optional: train() not required for HNSWFlat
    index.add(X)  # X must be float32
    return index

def search_hnsw(index, q_embeddings, topk=1000):
    scores, I = index.search(q_embeddings.astype('float32'), topk)  # dot-product scores
    return I, scores


import pandas as pd

def load_qrels(qrels_path):
    """
    Loads a TREC-style or simplified TSV qrels file.
    Supports formats:
        1) <query_id> <unused> <doc_id> <relevance>
        2) <query_id> <doc_id> <relevance>
    Returns:
        dict[str, dict[str, int]] in the format:
        { "qid": { "docid": relevance, ... }, ... }
    """
    # Read file (handles both space/tab separators)
    df = pd.read_csv(qrels_path, sep=r'\s+|\t+', header=None, engine='python')

    # Detect number of columns
    if len(df.columns) == 4:
        df.columns = ['qid', 'unused', 'docid', 'label']
    elif len(df.columns) == 3:
        df.columns = ['qid', 'docid', 'label']
    else:
        raise ValueError(f"Unexpected number of columns ({len(df.columns)}) in {qrels_path}")

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
    
def format_run(qids, dids, inds, scores, topk=1000):
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
            doc_ind = inds[i, rank]
            score = float(scores[i, rank])
            run[qid][str(int(dids[doc_ind]))] = score
    return run

dids, doc_embeddings = load_h5_embeddings(r"ms_marco\msmarco_passages_embeddings_subset.h5")
qid, q_embeddings = load_h5_embeddings(r"ms_marco\msmarco_queries_dev_eval_embeddings.h5")
doc_embeddings = np.array(doc_embeddings).astype("float32") # because of how faiss works
q_embeddings = np.array(q_embeddings).astype("float32") # because of how faiss works

start_build = time.time()
index = build_hnsw_ip(doc_embeddings)
end_build = time.time()
print(f"Index built in {end_build - start_build:.2f} seconds\n")
start_build = time.time()
inds, scores = search_hnsw(index, q_embeddings, TOPK)
end_build = time.time()
print(f"Index search in {end_build - start_build:.2f} seconds\n")
eval_files = ["ms_marco/qrels.eval.one.tsv", "ms_marco/qrels.eval.two.tsv","ms_marco/qrels.dev.tsv"]
for fname in eval_files:
    qrels = {}
    qrels = load_qrels(fname)
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg_cut.10', 'ndcg_cut.100', 'recip_rank', 'recall_100'})  
    run = format_run(qid, dids, inds, scores, TOPK)
    results = evaluator.evaluate(run)
    metrics = {m: [] for m in ['map', 'ndcg_cut_10', 'ndcg_cut_100', 'recip_rank', 'recall_100']}
    for query_metrics in results.values():
        for m in metrics:
            if m in query_metrics:
                metrics[m].append(query_metrics[m])
    for m, vals in metrics.items():
        avg = np.mean(vals) if vals else 0
        print(f"{m}: {avg}")
