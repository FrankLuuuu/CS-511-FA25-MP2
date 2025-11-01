import faiss
import h5py
import numpy as np
import os
import requests

def evaluate_hnsw():

    # start your code here
    # download data, build index, run query
    with h5py.File("./sift-128-euclidean.hdf5", "r") as f:
        train_data = f["train"][:]  # Train data (database embeddings)
        test_data = f["test"][:]  # Test data (query embeddings)

    d = 128  # The dimension of the SIFT vectors

    # Create the index
    index = faiss.IndexHNSWFlat(d, 16)  # HNSW with M=16

    # Set the efConstruction parameter
    index.hnsw.efConstruction = 200

    # Train the index (this step is required for the index to learn the vector space)
    index.add(train_data)  # Add the training data to the index

    # write the indices of the 10 approximate nearest neighbours in output.txt, separated by new line in the same directory
    query_vector = test_data[0].reshape(1, -1)  # Reshape query vector to match the index's input shape

    # Perform the search (return the indices of the top 10 nearest neighbors)
    k = 10
    distances, indices = index.search(query_vector, k)

    # Write the indices of the top 10 nearest neighbors to a file
    with open('./output.txt', 'w') as f:
        for idx in indices[0]:
            f.write(f"{idx}\n")

if __name__ == "__main__":
    evaluate_hnsw()
