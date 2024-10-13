from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class EmbeddingGenerator:
    """
    A class to generate embeddings for text using a pre-trained transformer model.

    Args:
        model_name (str): The name of the pre-trained model to use. Defaults to 'sentence-transformers/all-MiniLM-L6-v2'.
        device (str): The device to run the model on (either 'cpu' or 'cuda'). Defaults to 'cpu'.
    """
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', device='cpu'):
        # Load the tokenizer and model from Hugging Face's pre-trained models
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device
        self.model.to(self.device)  # Move the model to the specified device (CPU or GPU)
    
    def get_embeddings_in_batches(self, text_list, batch_size=16):
        """
        Generates embeddings for a list of texts in batches.

        Args:
            text_list (list of str): A list of text strings to generate embeddings for.
            batch_size (int): The size of each batch for processing. Defaults to 16.

        Returns:
            np.ndarray: A numpy array containing the embeddings for each text.
        
        Raises:
            ValueError: If the input text_list is not a list or contains non-string elements.
        """
        # Ensure that text_list is a list of strings
        if not isinstance(text_list, list):
            raise ValueError("text_list must be a list of strings.")
        if not all(isinstance(text, str) for text in text_list):
            raise ValueError("All elements in text_list must be strings.")

        embeddings = []  # List to store the resulting embeddings
        
        # Process the text in batches
        for i in range(0, len(text_list), batch_size):
            batch_text = text_list[i:i+batch_size]  # Slice the text list into batches
            inputs = self.tokenizer(batch_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)  # Tokenize the text
            
            # Generate embeddings without computing gradients
            with torch.no_grad():
                outputs = self.model(**inputs)  # Get the model's output
                # Compute the mean of the hidden states for each token to get sentence-level embeddings
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.extend(batch_embeddings)  # Append the batch embeddings to the list
        
        # Return the embeddings as a numpy array
        return np.array(embeddings)
