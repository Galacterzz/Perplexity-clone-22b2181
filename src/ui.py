# src/ui.py
import streamlit as st
from langchain.schema import HumanMessage, AIMessage
from .graph import create_graph
from .config import MAX_HISTORY_LENGTH

def main_ui():
    # Configure the Streamlit page
    st.set_page_config(
        page_title="Perplexity Clone",
        page_icon="🔍",
        layout="wide", # Use wide layout for better space utilization
        initial_sidebar_state="auto",
    )

    # Initialize the LangGraph application
    @st.cache_resource
    def get_graph():
        """Caches the compiled LangGraph to avoid recompilation on every run."""
        return create_graph()

    app = get_graph()

    # Initialize session state for conversation history if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- Streamlit UI Layout ---
    st.title("🔍 Perplexity Clone")
    st.caption("Powered by LangGraph, Mistral AI, Brave Search, and FAISS")
    st.caption("Sometimes the search may not work as websites are starting to use anti scrapping policies")
    st.caption("May take time as storing and retriveing the embedding")

    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message.type): # Use 'type' which is 'human' or 'ai'
            st.markdown(message.content)

    # Chat input for user query
    if prompt := st.chat_input("Ask anything..."):
        # Add user's message to session state history
        st.session_state.messages.append(HumanMessage(content=prompt))

        # Display user's message immediately
        with st.chat_message("human"):
            st.markdown(prompt)

        # Prepare the initial state for the LangGraph
        # Pass the current conversation history to the graph
        initial_state = {
            "messages": st.session_state.messages.copy(), # Pass a copy of the history
            "web_results": [],
            "retrieved_docs": [],
            "context_chunks": [],
            "response": "",
            "should_continue": True
        }

        # --- Invoke the LangGraph ---
        # Use st.spinner to show a loading indicator while the graph processes
        with st.spinner("Searching and generating response..."):
            try:
                # Run the graph with the initial state
                final_state = app.invoke(initial_state)

                # Extract the updated messages from the final state
                updated_messages = final_state.get("messages", [])

                # --- Manage Conversation History Length ---
                # Limit the history to MAX_HISTORY_LENGTH pairs (Human, AI)
                # Each pair consists of one HumanMessage and one AIMessage.
                # Find the last AI message to identify the end of the latest pair.
                ai_message_indices = [i for i, msg in enumerate(updated_messages) if isinstance(msg, AIMessage)]
                if ai_message_indices:
                    last_ai_index = ai_message_indices[-1]
                    # Keep messages from the start up to and including the last AI message
                    truncated_history = updated_messages[:last_ai_index + 1]
                else:
                    # If no AI message was added (e.g., error in generation), keep history as is
                    truncated_history = updated_messages

                # Enforce the maximum length of message pairs (Human + AI)
                # Calculate max number of messages: MAX_HISTORY_LENGTH pairs * 2 messages per pair
                max_messages = MAX_HISTORY_LENGTH * 2
                if len(truncated_history) > max_messages:
                    # Keep the last 'max_messages' entries
                    st.session_state.messages = truncated_history[-max_messages:]
                else:
                    st.session_state.messages = truncated_history

            except Exception as e:
                st.error(f"An error occurred while processing your request: {e}")
                # Optionally, add an error message to the chat history
                error_msg = "Sorry, I encountered an error while processing your request. Please try again."
                st.session_state.messages.append(AIMessage(content=error_msg))
                # Re-display the error message in the chat
                with st.chat_message("ai"):
                    st.markdown(error_msg)

        # Display the AI's response from the updated session state
        # The last message in the state should now be the AI's response
        if st.session_state.messages and isinstance(st.session_state.messages[-1], AIMessage):
            ai_response = st.session_state.messages[-1].content
            with st.chat_message("ai"):
                # Use st.markdown to render the response, which includes markdown links for citations
                st.markdown(ai_response)

    # Optional: Add a button to clear the chat history
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun() # Rerun the script to clear the UI

    # Optional: Display some information about the tools used in the sidebar
    st.sidebar.title("About this App")
    st.sidebar.info(
        "This is a Perplexity-like application built with:\n"
        "- **LangGraph** for workflow orchestration\n"
        "- **Mistral AI** for LLM and embeddings\n"
        "- **Brave Search** for web search\n"
        "- **FAISS** for vector storage\n"
        "- **Streamlit** for the UI"
    )

# Note: The main_ui function is defined here.
# The actual execution happens in main.py when streamlit run main.py is called.
