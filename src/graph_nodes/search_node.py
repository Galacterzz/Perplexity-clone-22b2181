# src/graph_nodes/search_node.py (This should already be correct from Segment 3)
import logging
from typing import Dict, Any
from langchain.schema import BaseMessage
from ..components.search_retrieval import perform_search # Import from the correct path

# Configure logging for this module
logger = logging.getLogger(__name__)

def search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node to perform a web search based on the latest user query.

    This node extracts the most recent user message from the 'messages' list
    in the state, performs a search using the Brave Search API, and stores
    the results in the 'web_results' field of the state.

    Args:
        state: The current state dictionary managed by LangGraph.

    Returns:
        A dictionary containing the updated 'web_results' and a flag
        'should_continue' indicating whether the graph should proceed.
    """
    logger.info("---EXECUTING SEARCH NODE---")

    # Extract the last message from the history
    messages = state.get("messages", [])
    if not messages:
        logger.error("No messages found in state for search.")
        return {"web_results": [], "should_continue": False}

    # Assuming the last message is the user's query
    last_message = messages[-1]
    query = last_message.content if hasattr(last_message, 'content') else str(last_message)
    logger.info(f"Extracted query for search: {query}")

    # Perform the search using the function from search_retrieval.py
    search_results = perform_search(query)

    # Determine if there are results to continue with
    should_continue = len(search_results) > 0
    if not should_continue:
        logger.warning("No search results found, terminating workflow.")

    logger.info(f"Search node completed. Found {len(search_results)} results. Should continue: {should_continue}")
    return {
        "web_results": search_results,
        "should_continue": should_continue
    }
