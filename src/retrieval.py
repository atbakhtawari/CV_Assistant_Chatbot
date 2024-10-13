import os

def retrieve_resumes(query, embedding_generator, faiss_index, df_resumes, k=3):
    """
    Retrieves resumes from a dataset based on a query string by searching for the most similar embeddings using FAISS.

    Args:
        query (str): The search query to find similar resumes.
        embedding_generator (EmbeddingGenerator): An instance of the EmbeddingGenerator class to convert the query into embeddings.
        faiss_index (FAISSIndex): An instance of the FAISSIndex class used to search for the nearest neighbors in the embedding space.
        df_resumes (pd.DataFrame): A DataFrame containing resume file information, including file paths.
        k (int): The number of resumes to retrieve based on similarity. Defaults to 3.

    Returns:
        dict: A dictionary with file names as keys and their corresponding resume contents as values.
    
    Raises:
        ValueError: If the query is not a string.
    
    Example:
        resume_data = retrieve_resumes("Python developer", embedding_generator, faiss_index, df_resumes, k=5)
        print(resume_data)
    """
    if not isinstance(query, str):
        raise ValueError("Query must be a string.")

    # Generate embedding for the query string
    query_embedding = embedding_generator.get_embeddings_in_batches([query])
    
    # Search the FAISS index for the k-nearest resumes
    distances, indices = faiss_index.search(query_embedding, k)
    
    # Retrieve corresponding resumes from the DataFrame using the indices
    retrieved_resumes = df_resumes.iloc[indices[0]]

    # Load resume contents
    # text_directory = os.path.join("data", "resumes", "txts")
    # print("Text Directory",text_directory)

    # Prepare a dictionary to store the contents of retrieved resumes
    resume_contents = {}
    
    # Iterate over the retrieved resumes and load the content from the corresponding files
    for file_name in retrieved_resumes['file_name']:
        # print("File Name",file_name)
        # file_path = os.path.join(text_directory, file_name)
        # print("File Name",file_path)
        # Check if the resume file exists
        if not os.path.exists(file_name):
            print(f"File not found: {file_name}. Skipping...")
            continue
        
        # Try to open and read the resume file
        try:
            with open(file_name, "r", encoding="utf-8") as file:
                resume_contents[file_name] = file.read()
        except Exception as e:
            print(f"Error reading file {file_name}: {e}")
    
    return resume_contents


            