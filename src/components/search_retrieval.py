# src/components/search_retrieval.py
import logging
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from ..config import BRAVE_SEARCH_API_KEY, BRAVE_SEARCH_COUNT, embeddings

# Configure logging for this module
logger = logging.getLogger(__name__)

# --- Search Component ---
def perform_search(query: str) -> List[Dict[str, str]]:
    """
    Performs a web search using the Brave Search API via direct requests.

    Args:
        query: The search query string.

    Returns:
        A list of dictionaries containing search results with 'title', 'url', and 'snippet'.
        Returns an empty list if the search fails.
    """
    logger.info(f"Performing web search for query: {query}")
    search_url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_SEARCH_API_KEY,
    }
    params = {
        "q": query,
        "count": BRAVE_SEARCH_COUNT, # Number of results to fetch
    }

    try:
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        search_data = response.json()
        results = search_data.get("web", {}).get("results", [])

        # Extract relevant information from each result
        formatted_results = []
        for result in results:
            formatted_results.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": result.get("description", ""), # Using 'description' as snippet
            })
        logger.info(f"Found {len(formatted_results)} search results.")
        return formatted_results
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error during search: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error during search: {e}")
        return []

# --- Retrieval Component ---
def fetch_and_parse_documents(search_results: List[Dict[str, str]]) -> List[Document]:
    """
    Fetches content from URLs in search results and parses them into LangChain Document objects.

    Args:
        search_results: A list of dictionaries containing 'url' and 'title'.

    Returns:
        A list of LangChain Document objects containing the parsed content and metadata.
    """
    logger.info(f"Parsing content from {len(search_results)} URLs.")
    documents = []
    for result in search_results:
        url = result.get("url")
        title = result.get("title")
        if not url:
            logger.warning("Skipping result with no URL.")
            continue

        try:
            logger.debug(f"Fetching content from: {url}")
            response = requests.get(url, timeout=10) # Add a timeout
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove script and style elements that don't contain useful content
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text content, trying to get the main content if possible
            # This is a basic approach; libraries like trafilatura might be better for complex sites
            text = soup.get_text(separator=' ', strip=True)
            if not text:
                logger.warning(f"No text found on page: {url}")
                continue # Skip empty pages

            # Create a LangChain Document with content and metadata
            doc = Document(
                page_content=text,
                metadata={
                    "source": url,
                    "title": title,
                    "url": url # Also store URL in metadata for citation
                }
            )
            documents.append(doc)
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch or parse URL {url}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error parsing URL {url}: {e}")

    logger.info(f"Parsed {len(documents)} documents successfully.")
    return documents

def create_and_store_embeddings(documents: List[Document]) -> Optional[FAISS]:
    """
    Chunks documents, generates embeddings, and creates a FAISS vector store.

    Args:
        documents: A list of LangChain Document objects.

    Returns:
        A FAISS vector store instance, or None if no documents are provided.
    """
    if not documents:
        logger.warning("No documents provided for embedding.")
        return None

    logger.info(f"Processing {len(documents)} documents for embedding and storage.")

    # --- Chunking ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Target size for each chunk
        chunk_overlap=200,    # Overlap to maintain context between chunks
        separators=["\n\n", "\n", " ", ""] # Try these separators in order
    )
    split_docs = text_splitter.split_documents(documents)
    logger.info(f"Split documents into {len(split_docs)} chunks.")

    # --- Embedding and Storage ---
    try:
        # Create FAISS index from the split documents and their embeddings
        # The embeddings object is imported from config.py
        vector_store = FAISS.from_documents(split_docs, embeddings)
        logger.info("Successfully created and populated FAISS vector store.")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating FAISS vector store: {e}")
        return None

def retrieve_relevant_context(vector_store: FAISS, query: str, top_k: int = 5) -> List[Document]:
    """
    Retrieves the top-k most relevant document chunks from the FAISS store for a given query.

    Args:
        vector_store: The FAISS vector store instance.
        query: The user's query string.
        top_k: The number of top results to retrieve.

    Returns:
        A list of LangChain Document objects containing the relevant context.
    """
    if not vector_store:
        logger.warning("Vector store is None, cannot retrieve context.")
        return []

    logger.info(f"Retrieving top {top_k} relevant chunks for query: {query}")
    try:
        # Perform similarity search
        relevant_docs = vector_store.similarity_search(query, k=top_k)
        logger.info(f"Retrieved {len(relevant_docs)} relevant chunks.")
        return relevant_docs
    except Exception as e:
        logger.error(f"Error during similarity search: {e}")
        return []
