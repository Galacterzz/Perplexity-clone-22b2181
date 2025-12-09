# src/config.py
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

# Load environment variables from .env file (for local development)
load_dotenv()

def get_env_variable(var_name: str) -> str:
    """
    Retrieve environment variable, prioritizing Streamlit secrets 
    if available, then falling back to os.getenv.
    """
    # 1. Check Streamlit Secrets (for Cloud Deployment)
    if var_name in st.secrets:
        return st.secrets[var_name]
    
    # 2. Check OS Environment Variables (for Local Dev)
    return os.getenv(var_name)

# --- API Keys and Configuration ---
MISTRAL_API_KEY = get_env_variable("MISTRAL_API_KEY")
BRAVE_SEARCH_API_KEY = get_env_variable("BRAVE_SEARCH_API_KEY")

# --- LangSmith Configuration (Optional) ---
# We use os.environ here because LangChain libraries look for these specific env vars automatically.
# We map them from secrets to env vars if they exist.
if "LANGCHAIN_API_KEY" in st.secrets:
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
if "LANGCHAIN_TRACING" in st.secrets:
    os.environ["LANGCHAIN_TRACING"] = st.secrets["LANGCHAIN_TRACING"]
if "LANGSMITH_PROJECT" in st.secrets:
    os.environ["LANGSMITH_PROJECT"] = st.secrets["LANGSMITH_PROJECT"]

# Check for missing keys
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY not found. Please set it in .env or Streamlit Secrets.")
if not BRAVE_SEARCH_API_KEY:
    raise ValueError("BRAVE_SEARCH_API_KEY not found. Please set it in .env or Streamlit Secrets.")
    
# --- Model Names ---
MISTRAL_LLM_MODEL = "mistral-small" # Or another suitable Mistral model
MISTRAL_EMBED_MODEL = "mistral-embed"

# --- Search Configuration ---
BRAVE_SEARCH_COUNT = 5 # Number of results to fetch from Brave Search

# --- Conversation Memory Configuration ---
MAX_HISTORY_LENGTH = 3 # Maximum number of query-response pairs to keep in memory

# --- LLM and Embedding Initializers ---
# Initialize the Mistral LLM with specific parameters
llm = ChatMistralAI(
    model=MISTRAL_LLM_MODEL,
    temperature=0.1, # Lower temperature for more factual, consistent answers
    max_retries=3,
    api_key=MISTRAL_API_KEY
)

# Initialize the Mistral Embedding model
embeddings = MistralAIEmbeddings(
    model=MISTRAL_EMBED_MODEL,
    api_key=MISTRAL_API_KEY
)

# --- FAISS Vector Store Placeholder ---
# FAISS initialization will happen dynamically in the retrieval component
# based on the documents retrieved for the current query.
# A global variable or a class instance can manage it later.
# For now, we just define its type hint or placeholder if needed globally.
# vector_store = None # Or initialize an empty one if necessary for imports
