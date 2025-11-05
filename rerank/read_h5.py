import h5py
import numpy as np


def load_h5_embeddings(file_path="../ms_macro/msmarco_passages_embeddings_subset.h5", id_key='id', embedding_key='embedding'):
    """
    Load IDs and embeddings from an HDF5 file.

    Parameters:
    - id_key: Dataset name for the IDs inside the HDF5 file.
    - embedding_key: Dataset name for the embeddings inside the HDF5 file.

    Returns:
    - ids: Numpy array of IDs (as strings).
    - embeddings: Numpy array of embeddings (as float32).
    """
    print(f"Loading data from {file_path}...")
    with h5py.File(file_path, 'r') as f:
        ids = np.array(f[id_key]).astype(str)
        embeddings = np.array(f[embedding_key]).astype(np.float32)  

    print(f"Loaded {len(ids)} embeddings.")
    return ids, embeddings

# usage:
file_path = r"dense-retrieval/ms_marco/msmarco_passages_embeddings_subset.h5"
ids, embeddings = load_h5_embeddings(file_path)

# Print first 5 IDs and their corresponding embeddings
print("Sample IDs and embeddings:")
for i in range(min(5, len(ids))):
    print(f"ID: {ids[i]}, Embedding: {embeddings[i]}")
