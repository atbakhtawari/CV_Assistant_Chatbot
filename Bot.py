import os
import streamlit as st
import pandas as pd
import numpy as np
import faiss
from dotenv import load_dotenv
from src.data_processing import process_resumes
from src.embedding import EmbeddingGenerator
from src.indexing import FAISSIndex
from src.retrieval import retrieve_resumes
from src.chat import ChatCompletion
import logging

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()
def display_message(role, content, avatar_emoji):
    avatar_class = "user-avatar" if role == "user" else "assistant-avatar"
    bubble_class = "user-bubble" if role == "user" else "assistant-bubble"
    st.markdown(f"""
        <div class="chat-message">
            <div class="chat-avatar {avatar_class}">{avatar_emoji}</div>
            <div class="chat-bubble {bubble_class}">{content}</div>
        </div>
    """, unsafe_allow_html=True)
IMPORTANT_WORDS = {# Job Titles
    'lead', 'developer', 'engineer', 'manager', 'architect', 'scientist', 'analyst', 'consultant',
    'specialist', 'administrator', 'director', 'technician', 'executive', 'officer', 'coordinator',
    'strategist', 'consultant', 'intern', 'freelancer', 'contractor',

    # Programming Languages
    'python', 'java', 'javascript', 'typescript', 'c', 'c++', 'c#', 'ruby', 'php', 'swift',
    'kotlin', 'go', 'golang', 'rust', 'scala', 'perl', 'r', 'matlab', 'sql', 'html', 'css',
    'sass', 'less', 'bash', 'shell', 'powershell', 'dart', 'elixir', 'haskell', 'clojure',
    'fortran', 'visual basic', 'vb.net',

    # Cloud Platforms & Tools
    'aws', 'azure', 'gcp', 'google cloud', 'cloud', 's3', 'lambda', 'ec2', 'kubernetes',
    'docker', 'jenkins', 'terraform', 'cloudformation', 'ansible', 'puppet', 'chef', 'helm',
    'prometheus', 'grafana',

    # Databases
    'sql', 'mysql', 'postgresql', 'mongodb', 'dynamodb', 'oracle', 'cassandra', 'hbase',
    'redis', 'couchdb', 'neo4j', 'firebase', 'elasticsearch', 'solr', 'mariadb',

    # Frameworks & Libraries
    'react', 'angular', 'vue', 'django', 'flask', 'spring', 'springboot', 'hibernate', 'express',
    'node', 'nextjs', 'nestjs', 'gatsby', 'redux', 'graphql', 'jquery', 'bootstrap', 'tailwind',
    'material ui', 'tensorflow', 'pytorch', 'scikit-learn', 'keras', 'pandas', 'numpy',
    'matplotlib', 'seaborn', 'plotly', 'd3', 'threejs',

    # DevOps & Automation
    'ci/cd', 'continuous integration', 'continuous delivery', 'jenkins', 'teamcity', 'circleci',
    'travis', 'bitbucket', 'github actions', 'ansible', 'terraform', 'puppet', 'saltstack',
    'chef', 'prometheus', 'grafana', 'nagios', 'new relic', 'datadog', 'elk stack', 'elastic stack',

    # Data & Analytics
    'big data', 'hadoop', 'spark', 'hive', 'pig', 'kafka', 'flume', 'sqoop', 'airflow',
    'databricks', 'tableau', 'powerbi', 'qlikview', 'looker', 'data science', 'machine learning',
    'deep learning', 'nlp', 'artificial intelligence', 'ai', 'bi', 'business intelligence',

    # Networking & Security
    'vpn', 'firewall', 'tcp', 'udp', 'ipsec', 'bgp', 'ospf', 'vpn', 'ssl', 'tls', 'ddos', 'penetration testing',
    'vulnerability scanning', 'splunk', 'wireshark', 'nmap', 'burpsuite', 'kali linux', 'aws security',
    'cybersecurity', 'iam', 'identity and access management',

    # Methodologies
    'agile', 'scrum', 'kanban', 'waterfall', 'devops', 'sre', 'site reliability engineering',
    'tdd', 'bdd', 'test driven development', 'behavior driven development', 'continuous integration',
    'continuous delivery', 'gitops',

    # Operating Systems
    'linux', 'windows', 'macos', 'ubuntu', 'debian', 'centos', 'redhat', 'fedora', 'arch',
    'kali', 'suse', 'rhel',

    # Testing Tools
    'selenium', 'junit', 'pytest', 'mocha', 'jest', 'cypress', 'karma', 'protractor', 'chai',
    'unittest', 'postman', 'soapui', 'loadrunner', 'jmeter', 'gatling', 'newman',

    # API & Integration
    'rest', 'graphql', 'soap', 'json', 'xml', 'grpc', 'openapi', 'swagger', 'oauth', 'jwt',
    'api gateway', 'kong', 'nginx', 'haproxy', 'reverse proxy', 'http', 'websocket', 'microservices',
    'service-oriented architecture', 'soa',

    # Version Control & Collaboration Tools
    'git', 'github', 'gitlab', 'bitbucket', 'svn', 'mercurial', 'jira', 'confluence', 'trello',
    'asana', 'slack', 'teams', 'zoom',

    # AI/ML & Data Science Terms
    'machine learning', 'deep learning', 'artificial intelligence', 'ai', 'data science',
    'nlp', 'natural language processing', 'computer vision', 'reinforcement learning', 'unsupervised learning',
    'supervised learning', 'classification', 'regression', 'clustering', 'random forest', 'decision tree',
    'logistic regression', 'svm', 'support vector machine', 'pca', 'dimensionality reduction',
    'time series', 'forecasting', 'gradient boosting', 'xgboost', 'lightgbm', 'hyperparameter tuning',
    'cross-validation', 'ensemble methods', 'bagging', 'stacking', 'bayesian methods',

    # Front-end Development
    'html', 'css', 'javascript', 'typescript', 'react', 'angular', 'vue', 'sass', 'less',
    'webpack', 'gulp', 'babel', 'parcel', 'npm', 'yarn', 'es6', 'typescript', 'redux',

    # Back-end Development
    'node', 'express', 'spring', 'django', 'flask', 'graphql', 'rest api', 'rpc', 'grpc',
    'microservices', 'monolithic', 'mvc', 'soap',

    # Miscellaneous Tech Skills
    'ci', 'cd', 'devops', 'docker', 'kubernetes', 'linux', 'terraform', 'bash', 'powershell', 'shell',
    'cloudformation', 'ansible', 'jenkins', 'git', 'bash', 'powershell', 'agile', 'scrum', 'jira', 'confluence'

}

pdf_directory = os.path.join("data", "resumes", "pdfs")
text_directory = os.path.join("data", "resumes", "txts")
embeddings_path = os.path.join("data", "embeddings", "resume_embeddings.npy")
metadata_path = os.path.join("data", "metadata", "resume_metadata.csv")

st.markdown("""
    <style>
        .chat-message {
            display: flex;
            align-items: center;
            margin: 10px 0;
        }
        .chat-avatar {
            width: 40px;
            height: 40px;
            margin-right: 10px;
            border-radius: 50%;
            background-color: #f0f4ff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
        }
        .user-avatar {
            background-color: #ddefff;
        }
        .assistant-avatar {
            background-color: #c8e6c9;
        }
        .chat-bubble {
            background-color: #f0f4ff;
            padding: 10px;
            border-radius: 8px;
            max-width: 80%;
            color: #000;
            word-wrap: break-word;
        }
        .user-bubble {
            background-color: #f0f4ff;
        }
        .assistant-bubble {
            background-color: #e8f5e9;
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.write('**Menu**')
sidebar_options = st.sidebar.radio(
    "",
    ["Resume Search", "About", "Conversation History"]
)

if sidebar_options == "About":
    st.title("About")
    st.write("**Resume Search Assistant** is a tool that helps you find the most relevant resumes based on specific skill sets. It uses machine learning to process resumes and find the best match for your needs. Powered by FAISS for efficient searching and a language model for generating responses.")
    st.write("Designed to save time for HR professionals by quickly finding qualified candidates.")

elif sidebar_options == "Conversation History":
    st.title("Conversation History")
    st.write("Early Searches:")
    if st.session_state.history:
        for message in st.session_state.history :
            if message['role'] == 'user':
                display_message("user", message["content"], "🧑🏻")
            else:
                display_message("assistant", message["content"], "💻")
    else:
        st.write("No conversation history available.")

if 'history' not in st.session_state:
    st.session_state.history = []


if sidebar_options == "Resume Search":
    st.title("Resume Search Assistant")
    st.write("Welcome to the Resume Search Assistant! Enter a query to find the most relevant resume.")
    user_query = st.chat_input("Enter your query (e.g., I am looking for a Java Developer'):")

    if user_query:
        st.session_state.history.append({'role': 'user', 'content': user_query})
        #display_message("user", user_query, "🧑🏻")
        
        try:
            st.write("Processing your query, please wait...")

            if not os.path.exists(embeddings_path) or not os.path.exists(metadata_path):
                st.write("Processing resumes and generating embeddings...")
                resume_data = process_resumes(pdf_directory, text_directory, IMPORTANT_WORDS)
                df_resumes = pd.DataFrame(resume_data)
                embedding_generator = EmbeddingGenerator()
                embeddings = embedding_generator.get_embeddings_in_batches(df_resumes['preprocessed_resume'].tolist())
                np.save(embeddings_path, embeddings)
                df_resumes[['file_name']].to_csv(metadata_path, index=False)
            else:
                st.write("Loading existing embeddings and metadata...")
                embeddings = np.load(embeddings_path)
                df_resumes = pd.read_csv(metadata_path)

            embedding_dim = embeddings.shape[1] 
            faiss_index = FAISSIndex(embedding_dim)
            faiss_index.add_embeddings(embeddings)  
            embedding_generator = EmbeddingGenerator()

            chat = ChatCompletion(api_key=os.getenv("GROQ_API_KEY"))  
            st.write("Retrieving the most relevant resume...")
            retrieved_resumes = retrieve_resumes(user_query, embedding_generator, faiss_index, df_resumes, k=1)

            messages = [
                {
                    "role": "system", 
                    "content": """
                    You are an AI assistant that finds the most relevant person based on their skill set.
                    INSTRUCTIONS:
                    - Only output 1 Most relevant resume.
                    - Only respond from the RESUME_SUMMARY provided below.
                    - Output should be well-structured and human-readable.
                    - Do not include any information or content which is not relevant to matching relevant person.
                    """
                },
                {
                    "role": "assistant", 
                    "content": f"""
                    Use the following RESUME_SUMMARY that is in python dictionary format to create your response:{retrieved_resumes}
                    """
                },
                {
                    "role": "user", 
                    "content": user_query
                }
            ]

            response = chat.create_completion(messages)
            st.session_state.history.append({'role': 'assistant', 'content': response})
            #display_message("assistant", response, "💻")

        except Exception as e:
            logger.error(f"An error occurred: {e}")
            st.error(f"An unexpected error occurred: {e}. Please try again later.")

    for message in st.session_state.history:
        if message['role'] == 'user':
            display_message("user", message["content"], "🧑🏻")
        else:
            display_message("assistant", message["content"], "💻")
