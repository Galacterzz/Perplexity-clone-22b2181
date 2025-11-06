# src/config.py
import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

# Load environment variables from .env file
load_dotenv()

# --- API Keys and Configuration ---
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY not found in environment variables.")
if not BRAVE_SEARCH_API_KEY:
    raise ValueError("BRAVE_SEARCH_API_KEY not found in environment variables.")

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