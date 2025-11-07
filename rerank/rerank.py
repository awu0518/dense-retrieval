import h5py
import csv
import numpy as np
from collections import defaultdict

def load_h5_embeddings(file_path="../ms_marco/msmarco_passages_embeddings_subset.h5", id_key='id', embedding_key='embedding'):
    print(f"Loading data from {file_path}...")
    with h5py.File(file_path, 'r') as f:
        ids = np.array(f[id_key]).astype(str)
        embeddings = np.array(f[embedding_key]).astype(np.float32)  

    print(f"Loaded {len(ids)} embeddings.")
    return ids, embeddings

def l2norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def zscore(x):
    m = x.mean()
    s = x.std()
    if s == 0 or np.isnan(s):
        # fallback: min-max normalization
        xmin, xmax = x.min(), x.max()
        if xmax == xmin:
            return np.zeros_like(x)  # all values same → no signal
        return (x - xmin) / (xmax - xmin)
    return (x - m) / s

def main():
    queryIDs, queryEmbeddings = load_h5_embeddings("dense-retrieval/ms_marco/msmarco_queries_dev_eval_embeddings.h5")
    docIDs, docEmbeddings = load_h5_embeddings("dense-retrieval/ms_marco/msmarco_passages_embeddings_subset.h5")
    
    q_index = {qid:i for i, qid in enumerate(queryIDs)}
    d_index = {pid:i for i, pid in enumerate(docIDs)}

    queryEmbeddings = l2norm(queryEmbeddings)
    docEmbeddings = l2norm(docEmbeddings)

    files = [("dense-retrieval/buildBM25Index/bm25.dev.tsv", "dense-retrieval/rerank/rerank1.dev.tsv"),
             ("dense-retrieval/buildBM25Index/bm25.eval.one.tsv", "dense-retrieval/rerank/rerank1.eval.one.tsv"),
             ("dense-retrieval/buildBM25Index/bm25.eval.two.tsv", "dense-retrieval/rerank/rerank1.eval.two.tsv")]
    
    for filePath, newFilePath in files:
        cand = defaultdict(list)   # qid -> [(docid, bm25)]
        with open(filePath, "r", encoding="utf-8") as f:
            for line in f:
                qid, _, docid, rank, score, tag = line.strip().split()
                cand[qid].append((docid, float(score)))

        out_lines = []
        for qid, pairs in cand.items():
            if qid not in q_index:
                continue
            qv = queryEmbeddings[q_index[qid]]
            rows, bm25Scores = [], []
            for docid, bm25 in pairs:
                if docid in d_index: 
                    rows.append(docid)
                    bm25Scores.append(bm25)

            if not rows: continue

            D = docEmbeddings[[d_index[d] for d in rows]]
            sims = (D @ qv)  # cosine similarity
            
            bm25Scores = np.array(bm25Scores, dtype=np.float32)
            sims = np.array(sims, dtype=np.float32)

            bm25_n = zscore(bm25Scores)
            sims_n = zscore(sims)
            
            ALPHA = 0.975
            hybrid = ALPHA * sims_n + (1 - ALPHA) * bm25_n

            order = np.argsort(-hybrid)
            for r, i in enumerate(order, start=1):
                docid = rows[i]
                score = float(hybrid[i]) 
                out_lines.append(f"{qid} Q0 {docid} {r} {score:.6f} HYB")

        with open(newFilePath, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines))


if __name__ == '__main__':
    main()