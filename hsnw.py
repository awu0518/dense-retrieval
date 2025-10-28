# Implementation of query search using hsnw. The dataset setup is based off of this: https://github.com/facebookresearch/faiss/wiki/Faster-search
# conda install -c pytorch faiss-cpu
# pip install read_h5 sentence-transformers
# 350 The melting point of a substance is the temperature at which it changes from. a solid to a liquid.
# #  a liquid to a solid. a gas to a solid. a solid to a gas. The boiling point of a substance is the temperature at which it changes from. 
# a liquid to a gas. a liquid to a solid. a gas to a solid. a solid to a liquid. When a gas is compressed it changes state into a. liquid.
from read_h5 import load_h5_embeddings
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

num_neighbhors = 4 # the number of nearest neighbors to retrieve during a search.
num_clusters = 100 # num of clusters

ids, embeddings = load_h5_embeddings(r"C:\Users\akani\Documents\dense-retrieval\ms_marco\msmarco_passages_embeddings_subset.h5")
embeddings = np.array(embeddings).astype("float32") # because of how faiss works

d = len(embeddings[0]) # dimension of each vector


quantizer = faiss.IndexFlatL2(d)  # the other index
index = faiss.IndexIVFFlat(quantizer, d, num_clusters)
index.train(embeddings)
index.add(embeddings)


model = SentenceTransformer("all-MiniLM-L6-v2")

query = "The melting point of a substance is the temperature at which it changes from. a solid to a liquid."

# Convert text to a vector
xq = model.encode([query], normalize_embeddings=True).astype("float32")

D, I = index.search(xq, num_neighbhors)

print(I)  # indices of the top matches
print(D)  # similarity scores