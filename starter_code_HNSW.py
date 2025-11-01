import faiss
import h5py
import numpy as np
import os
import requests

def evaluate_hnsw():

    with h5py.File("../sift-128-euclidean.hdf5", "r") as f:
        train_data = f["train"][:]  # Database embeddings
        test_data = f["test"][:]  # Query embeddings
    
    # Dimension of the vectors
    d = 128  # SIFT vectors are 128-dimensional
    
    # Create the HNSW index with parameters M=16, efConstruction=200, and efSearch=200
    M = 16
    efConstruction = 200
    efSearch = 200
    
    # Create HNSW index
    index = faiss.IndexHNSWFlat(d, M)
    
    # Set the efConstruction parameter
    index.hnsw.efConstruction = efConstruction
    
    # Add the training data (database embeddings) to the index
    index.add(train_data)
    
    # Set efSearch for search query
    index.hnsw.efSearch = efSearch
    
    # Perform the query using the first test vector (query vector)
    query_vector = test_data[0].reshape(1, -1)  # Reshape query vector to match the input shape
    
    # Perform the search (top 10 nearest neighbors)
    k = 10
    distances, indices = index.search(query_vector, k)
    
    # Verify that the number of neighbors returned is correct (should be 10)
    assert len(indices[0]) == 10, "Number of neighbors returned is not 10"
    
    # Save the output to the file (output.txt)
    with open('./output.txt', 'w') as f:
        for idx in indices[0]:
            f.write(f"{idx}\n")

if __name__ == "__main__":
    evaluate_hnsw()
