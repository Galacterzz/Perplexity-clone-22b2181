# src/graph_nodes/generation_node.py
import logging
from typing import Dict, Any, List
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from ..config import llm

# Configure logging for this module
logger = logging.getLogger(__name__)

def generation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node to generate the final response using the LLM.

    This node formats the retrieved context and the user's query into a prompt
    using a predefined template. It then invokes the Mistral LLM to generate
    the response. The response is appended to the 'messages' history in the state.
    Citations are formatted based on the metadata of the retrieved context chunks.

    Args:
        state: The current state dictionary managed by LangGraph.

    Returns:
        A dictionary containing the updated 'messages' list with the new AI response
        and a final 'should_continue' flag set to False to terminate the graph.
    """
    logger.info("---EXECUTING GENERATION NODE---")

    # Extract components from the state
    messages = state.get("messages", [])
    context_chunks = state.get("context_chunks", [])
    web_results = state.get("web_results", []) # For potential fallback citation logic if context_chunks metadata isn't sufficient

    if not messages:
        logger.error("No messages found in state for generation.")
        return {"response": "Error: No input query found.", "should_continue": False}

    # Extract the user's query from the last message
    last_message = messages[-1]
    query = last_message.content if hasattr(last_message, 'content') else str(last_message)

    # Prepare context string and citations
    context_text = ""
    citations_map = {} # Map citation number to source URL
    if context_chunks:
        for i, chunk in enumerate(context_chunks):
            # Use the source URL from the chunk's metadata for citation
            source_url = chunk.metadata.get("url", "N/A")
            context_text += f"\n\n[Source {i+1}]: {source_url}\nContent: {chunk.page_content}\n"
            citations_map[i+1] = source_url # Store citation number to URL mapping
    else:
        logger.warning("No context chunks provided to generation node. LLM will respond without context.")
        # Even without context, we can still pass the query, but the LLM might not find an answer.
        context_text = "No relevant information was found on the web for this query."

    # Define the prompt template
    # The prompt instructs the LLM to use the context and provide inline citations [Source X].
    prompt_template = PromptTemplate.from_template(
        "You are an AI assistant that provides helpful answers based on the provided context.\n"
        "Use the context below to answer the user's query. Ensure your answer is accurate and well-structured.\n"
        "Cite the sources used in your answer using the format [Source X], where X is the source number provided in the context.\n"
        "If the context does not contain enough information to answer the query, state so clearly.\n"
        "Context from web search:\n{context}\n\nQuery: {query}\n\nAnswer:"
    )

    # Format the prompt with context and query
    formatted_prompt = prompt_template.format(context=context_text, query=query)

    # Invoke the LLM
    logger.info("Invoking LLM to generate response...")
    try:
        response_message = llm.invoke([HumanMessage(content=formatted_prompt)])
        response_text = response_message.content
    except Exception as e:
        logger.error(f"Error invoking LLM: {e}")
        response_text = "Sorry, I encountered an error while processing your request."

    # --- Citation Formatting ---
    # Replace [Source X] placeholders in the response with clickable markdown links [X](URL)
    # Iterate through the citations map to replace placeholders
    final_response_text = response_text
    for num, url in citations_map.items():
        placeholder = f"[Source {num}]"
        # Create a markdown link [X](URL)
        # Using the number only for the link text for cleaner display
        link = f"[{num}]({url})"
        final_response_text = final_response_text.replace(placeholder, link)

    # Append the final response to the messages history
    ai_message = AIMessage(content=final_response_text)
    updated_messages = messages + [ai_message]

    logger.info("Generation node completed. Response generated and added to messages.")
    # Set should_continue to False to signal the end of the graph execution
    return {
        "messages": updated_messages,
        "response": final_response_text, # Store the raw response with formatted citations
        "should_continue": False
    }