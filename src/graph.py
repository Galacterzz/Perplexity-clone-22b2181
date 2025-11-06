# src/graph.py
import logging
from typing import Dict, Any
from langgraph.graph import StateGraph
from langgraph.graph import END
from .graph_nodes.search_node import search_node
from .graph_nodes.retrieval_node import retrieval_node
from .graph_nodes.generation_node import generation_node
from .components.state_management import State

# Configure logging for this module
logger = logging.getLogger(__name__)

def should_continue(state: Dict[str, Any]) -> str:
    """
    Conditional edge function to determine the next node based on the 'should_continue' flag.

    This function checks the 'should_continue' value in the state.
    If True, it indicates that search results were found and the workflow should proceed
    to the retrieval node. If False, it means no results were found or an error occurred,
    and the workflow should skip retrieval and go directly to the generation node
    (which will likely return an error message or indicate no results).

    Args:
        state: The current state dictionary.

    Returns:
        The name of the next node ("retrieval_node" or "generation_node").
    """
    should_cont = state.get("should_continue", False)
    logger.info(f"Conditional logic: should_continue is {should_cont}")
    if should_cont:
        logger.info("Routing to retrieval node.")
        return "retrieval_node"
    else:
        logger.info("Routing to generation node (or terminating).")
        return "generation_node" # Generation node handles the final state update and termination

def create_graph() -> StateGraph:
    """
    Creates and compiles the LangGraph state machine for the Perplexity clone.

    This function defines the workflow by adding nodes, setting the entry point,
    adding conditional edges from the search node, and connecting the retrieval
    node directly to the generation node. It then compiles the graph.

    Returns:
        The compiled LangGraph executable.
    """
    logger.info("---COMPILING LANGGRAPH---")

    # Define a StateGraph with the state structure defined in state_management.py
    workflow = StateGraph(State)

    # Add nodes to the graph
    # Each node is a function that takes the state and returns updates to the state
    workflow.add_node("search_node", search_node)
    workflow.add_node("retrieval_node", retrieval_node)
    workflow.add_node("generation_node", generation_node)

    # Set the starting point of the graph
    workflow.set_entry_point("search_node")

    # Add conditional edge from search_node
    # The 'should_continue' flag from search_node's output determines the next step
    workflow.add_conditional_edges(
        "search_node",      # The node from which the conditional edge originates
        should_continue,    # The function that decides the next node
        {
            "retrieval_node": "retrieval_node", # If should_continue returns "retrieval_node"
            "generation_node": "generation_node" # If should_continue returns "generation_node"
        }
    )

    # Add a direct edge from retrieval_node to generation_node
    # After retrieval, the process always proceeds to generation
    workflow.add_edge("retrieval_node", "generation_node")

    # Add a direct edge from generation_node to END
    # The generation node sets should_continue to False, signaling the end of this cycle
    workflow.add_edge("generation_node", END)

    # Compile the graph into an executable LangGraph object
    compiled_graph = workflow.compile()
    logger.info("---LANGGRAPH COMPILED SUCCESSFULLY---")
    return compiled_graph

# Optional: Example of how to run the graph (for testing purposes)
# if __name__ == "__main__":
#     import pprint
#     from langchain.schema import HumanMessage
#     # Create the graph
#     app = create_graph()
#
#     # Define an initial state for testing
#     initial_state = {
#         "messages": [HumanMessage(content="What is the latest news about AI?")],
#         "web_results": [],
#         "retrieved_docs": [],
#         "context_chunks": [],
#         "response": "",
#         "should_continue": True
#     }
#
#     # Run the graph
#     final_state = app.invoke(initial_state)
#     pprint.pprint(final_state)