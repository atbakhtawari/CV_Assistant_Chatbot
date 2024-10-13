import os
from PyPDF2 import PdfReader
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize NLTK modules
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def extract_text_from_pdf(pdf_file_path):
    """
    Extracts text from a given PDF file.

    Args:
        pdf_file_path (str): The file path of the PDF.

    Returns:
        str: The extracted text from the PDF.
    """
    reader = PdfReader(pdf_file_path)
    text = ""
    for page in reader.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text
    return text

def preprocess_text(text, important_words):
    """
    Preprocesses the given text by lowercasing, removing special characters, 
    and filtering out stopwords except for important words.

    Args:
        text (str): The input text to preprocess.
        important_words (set): A set of words that are important and should not be removed.

    Returns:
        str: The preprocessed text.
    """
    # Convert text to lowercase
    text = text.lower()
    # Remove special characters (retain only words and spaces)
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespaces
    text = " ".join(text.split())
    
    # Get stopwords, excluding important words
    stop_words = set(stopwords.words('english')).difference(important_words)
    
    # Lemmatize words and filter out stopwords
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    
    return " ".join(words)

def process_resumes(pdf_directory, text_directory, important_words):
    """
    Processes all PDF resumes in the given directory by extracting and 
    preprocessing their text, then saving the processed text to corresponding .txt files.

    Args:
        pdf_directory (str): Directory containing the PDF resumes.
        text_directory (str): Directory to save the processed text files.
        important_words (set): A set of words that should not be removed during preprocessing.

    Returns:
        list: A list of dictionaries containing file names and their preprocessed text.
    """
    # Create the output directory if it does not exist
    if not os.path.exists(text_directory):
        os.makedirs(text_directory)
    
    resume_data = []
    
    # Iterate over all files in the PDF directory
    for pdf_filename in os.listdir(pdf_directory):
        pdf_path = os.path.join(pdf_directory, pdf_filename)
        
        # Skip non-PDF files
        if not pdf_filename.endswith(".pdf"):
            print(f"Skipping non-PDF file: {pdf_filename}")
            continue
        
        # Define the corresponding text filename
        text_filename = os.path.join(text_directory, pdf_filename.replace(".pdf", ".txt"))
    
        # Skip if the text file already exists
        if os.path.exists(text_filename):
            print(f"Text file for {pdf_filename} already exists. Skipping...")
            continue
    
        # Check if the PDF file exists
        if not os.path.exists(pdf_path):
            print(f"PDF file not found: {pdf_path}")
            continue
    
        try:
            # Extract and preprocess text from the PDF
            text = extract_text_from_pdf(pdf_path)
            preprocessed_text = preprocess_text(text, important_words)
            
            # Save the preprocessed text to a .txt file
            with open(text_filename, "w", encoding="utf-8") as text_file:
                text_file.write(preprocessed_text)
                print(f"Processed and saved text for {text_filename}.")
            
            # Append the processed resume data to the list
            resume_data.append({"file_name": text_filename, "preprocessed_resume": preprocessed_text})
        
        except Exception as e:
            print(f"Error processing {text_filename}: {e}")
    
    return resume_data
