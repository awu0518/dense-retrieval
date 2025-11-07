# Implementation of query search using hsnw. The dataset setup is based off of this: https://github.com/facebookresearch/faiss/wiki/Faster-search
# conda install -c pytorch faiss-cpu
# pip install h5py pytrec_eval
# run awk '{print $1 "\t0\t" $2 "\t" $3}' qrels.dev.tsv > qrels.dev.clean.tsv to get the file in the right format for trec_eval

# 786436	0	8597447	0.7097718715667725
# 786436	0	6032770	0.6237262487411499
# 786436	0	8197603	0.6012924313545227
# 786436	0	7654539	0.5569262504577637
# 786436	0	3371951	0.5499846339225769
from read_h5 import load_h5_embeddings
import numpy as np
import faiss
import pandas as pd
import pytrec_eval
import time

TOPK = 100
M = 6
EFC = 200
EFS = 150


def build_hnsw_ip(X, M=8, efC=100, efS=100):
    d = X.shape[1]
    index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = efC
    index.hnsw.efSearch = efS
    index.add(X) 
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

    # number of columns
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

index = build_hnsw_ip(doc_embeddings, M=M, efC=EFC, efS=EFS)
inds, scores = search_hnsw(index, q_embeddings, TOPK)
eval_files = ["ms_marco/qrels.eval.one.tsv", "ms_marco/qrels.eval.two.tsv","ms_marco/qrels.dev.tsv"]
for fname in eval_files:
    qrels = {}
    qrels = load_qrels(fname)
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg_cut.10', 'ndcg_cut.100', 'recip_rank', 'recall_100'})  
    run = format_run(qid, dids, inds, scores, TOPK) # formatting the results of the search for comparisons
    results = evaluator.evaluate(run) 

    # from here we just need to print out the results
    metrics = {m: [] for m in ['ndcg_cut_10', 'ndcg_cut_100', 'recip_rank', 'recall_100']}
    for query_metrics in results.values():
        for m in metrics:
            if m in query_metrics:
                metrics[m].append(query_metrics[m])
    for m, vals in metrics.items():
        avg = np.mean(vals) if vals else 0
        print(f"{m}: {avg}")
    print("------------------------------------------------")
    # this code is to output our results to a file so we can run trec_eval
    # I used trec_eval just for the MAP@10 and MAP@100 since this library doesn't
    # support it
    if (fname == "ms_marco/qrels.dev.tsv"):
        output_path = "output.tsv"
        with open(output_path, "w") as f:
            for i, q in enumerate(qid):
                q_str = str(int(q))
                for rank in range(TOPK):
                    doc_index = inds[i, rank]
                    doc_str = str(int(dids[doc_index]))
                    score = float(scores[i, rank])
                    f.write(f"{q_str}\tQ0\t{doc_str}\t{rank+1}\t{score}\tEMB\n")
########