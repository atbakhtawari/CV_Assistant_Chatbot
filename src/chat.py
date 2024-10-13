from dotenv import load_dotenv
import os
from groq import Groq

# Load environment variables from a .env file
load_dotenv()

# Retrieve the GROQ API key from the environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Raise an error if the API key is not found
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables.")

class ChatCompletion:
    """
    A class to interact with the Groq API to create chat completions using a specified model.
    
    Attributes:
        client (Groq): The Groq client initialized with the API key.
    """
    
    def __init__(self, api_key):
        """
        Initializes the ChatCompletion instance by setting up the Groq client.
        
        Args:
            api_key (str): The API key for authentication with the Groq API.
        """
        # Initialize the Groq client using the provided API key
        self.client = Groq(api_key=api_key)
    
    def create_completion(self, messages, model="llama3-70b-8192"):
        """
        Sends a request to the Groq API to generate a chat completion based on the provided messages.
        
        Args:
            messages (list): A list of messages (e.g., user prompts) to be passed for completion.
            model (str): The model to be used for generating the chat completion. Default is "llama3-70b-8192".
        
        Returns:
            str: The content of the chat completion returned by the Groq API.
        """
        # Make a request to the Groq API to create a chat completion
        response = self.client.chat.completions.create(
            messages=messages,
            model=model,
        )
        
        # Extract and print the content of the chat completion from the API response
        chat_completion = response.choices[0].message.content
        print("Response", chat_completion)
        
        # Return the content of the chat completion
        return chat_completion
