import faiss
import numpy as np

class FAISSIndex:
    """
    A class to handle vector-based similarity search using FAISS (Facebook AI Similarity Search).

    Args:
        embedding_dim (int): The dimensionality of the embeddings.
        use_gpu (bool): Whether to use GPU for faster computations. Defaults to False.
    """
    def __init__(self, embedding_dim, use_gpu=False):
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu
        # Create a FAISS index for L2 distance (Euclidean distance)
        self.index = faiss.IndexFlatL2(embedding_dim)
        
        # If GPU is specified, transfer the index to GPU
        if self.use_gpu:
            res = faiss.StandardGpuResources()  # Allocate resources for GPU
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)  # Move index to GPU
    
    def add_embeddings(self, embeddings):
        """
        Adds embeddings to the FAISS index.

        Args:
            embeddings (np.ndarray): A numpy array of shape (n_samples, embedding_dim) containing the embeddings.
        
        Example:
            embeddings = np.random.random((100, 128))  # 100 samples, 128-dimensional embeddings
            faiss_index.add_embeddings(embeddings)
        """
        self.index.add(embeddings)  # Add the embeddings to the index
        print(f"Number of embeddings indexed: {self.index.ntotal}")  # Display the total number of embeddings in the index
    
    def search(self, query_embedding, k=1):
        """
        Searches the FAISS index for the k-nearest neighbors of the query embedding.

        Args:
            query_embedding (np.ndarray): A numpy array of shape (1, embedding_dim) containing the query embedding.
            k (int): The number of nearest neighbors to return. Defaults to 1.

        Returns:
            distances (np.ndarray): A numpy array containing the distances of the nearest neighbors.
            indices (np.ndarray): A numpy array containing the indices of the nearest neighbors.
        
        Example:
            query_embedding = np.random.random((1, 128))  # Single 128-dimensional query embedding
            distances, indices = faiss_index.search(query_embedding, k=5)
        """
        distances, indices = self.index.search(query_embedding, k)  # Perform the search in the index
        return distances, indices  # Return distances and indices of the nearest neighbors
