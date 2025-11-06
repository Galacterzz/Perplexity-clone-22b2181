# src/graph_nodes/retrieval_node.py
import logging
from typing import Dict, Any, List
from langchain.schema import Document
from ..components.search_retrieval import fetch_and_parse_documents, create_and_store_embeddings, retrieve_relevant_context
from ..config import llm # Importing llm for potential future use in retrieval logic (e.g., reranking)

# Configure logging for this module
logger = logging.getLogger(__name__)

def retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node to fetch, parse, embed, and retrieve relevant context.

    This node takes the 'web_results' from the state, fetches content from the URLs,
    parses the HTML into text, chunks the text, embeds the chunks, stores them in a
    FAISS vector store, and then retrieves the most relevant chunks based on the
    latest user query found in the 'messages' history.

    Args:
        state: The current state dictionary managed by LangGraph.

    Returns:
        A dictionary containing the updated 'retrieved_docs', 'context_chunks',
        and the 'should_continue' flag.
    """
    logger.info("---EXECUTING RETRIEVAL NODE---")

    web_results = state.get("web_results", [])
    if not web_results:
        logger.warning("No web results provided to retrieval node. Cannot process.")
        return {"retrieved_docs": [], "context_chunks": [], "should_continue": state.get("should_continue", False)}

    # Extract the query from the last message in the history
    messages = state.get("messages", [])
    if not messages:
        logger.error("No messages found in state for query during retrieval.")
        return {"retrieved_docs": [], "context_chunks": [], "should_continue": False}

    last_message = messages[-1]
    query = last_message.content if hasattr(last_message, 'content') else str(last_message)
    logger.info(f"Extracted query for retrieval: {query}")

    # Step 1: Fetch and parse documents from URLs
    logger.info("Fetching and parsing documents...")
    parsed_docs = fetch_and_parse_documents(web_results)

    # Step 2: Create embeddings and store in FAISS
    logger.info("Creating embeddings and storing in FAISS...")
    vector_store = create_and_store_embeddings(parsed_docs)

    # Step 3: Retrieve relevant context from the vector store
    context_chunks = []
    if vector_store:
        logger.info("Retrieving relevant context...")
        # Retrieve top 5 chunks by default
        context_chunks = retrieve_relevant_context(vector_store, query, top_k=5)
    else:
        logger.warning("Vector store was not created, skipping context retrieval.")

    # Update the state
    logger.info(f"Retrieval node completed. Retrieved {len(context_chunks)} relevant chunks.")
    # 'should_continue' is passed through; if search node set it to False due to no results,
    # this node won't run, so the flag remains False from the previous state or search node.
    # If this node runs successfully, it means search node set it to True.
    return {
        "retrieved_docs": parsed_docs,
        "context_chunks": context_chunks,
        "should_continue": state.get("should_continue", False) # Propagate the flag from search
    }