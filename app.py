"""
Streamlit UI for the Perplexity-like application.
Updated for Streamlit v1.48.0 (August 2025).
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

try:
    from src.components.orchestrator import PerplexityWorkflow
    from src.utils.config import config
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Make sure you have installed all dependencies and the src directory is properly structured.")
    st.stop()

# Streamlit page configuration with new v1.48 options
st.set_page_config(
    page_title="🔍 Perplexity Clone",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .query-box {
        margin: 1rem 0;
    }
    .source-item {
        background-color: #f0f2f6;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .memory-stats {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'workflow' not in st.session_state:
    try:
        # Validate configuration
        config.validate_keys()
        st.session_state.workflow = PerplexityWorkflow()
        st.session_state.initialized = True
    except ValueError as e:
        st.session_state.initialized = False
        st.session_state.error = str(e)

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Main UI
st.markdown('<h1 class="main-header">🔍 Perplexity-like Research Assistant</h1>', unsafe_allow_html=True)

# Sidebar configuration and status
with st.sidebar:
    st.header("⚙️ Configuration & Status")
    
    # Check initialization status
    if not st.session_state.get('initialized', False):
        st.error("❌ Application not properly initialized")
        if 'error' in st.session_state:
            st.error(f"Error: {st.session_state.error}")
        
        st.markdown("### 📋 Required Environment Variables")
        st.code("""
# Copy to your .env file:
MISTRAL_API_KEY=your_mistral_api_key_here
BRAVE_SEARCH_API_KEY=your_brave_search_api_key_here

# Optional:
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=perplexity-clone
LANGSMITH_TRACING=true
        """, language="bash")
        
        st.markdown("### 🚀 Setup Instructions")
        st.markdown("""
        1. Get your Mistral API key from [console.mistral.ai](https://console.mistral.ai)
        2. Get your Brave Search API key from [brave.com/search/api](https://brave.com/search/api) (2k free requests/month)
        3. Create a `.env` file with the above variables
        4. Restart the application
        """)
        
        st.stop()
    
    st.success("✅ Application initialized successfully")
    
    # Memory management
    st.markdown("### 🧠 Memory Management")
    
    if st.button("🗑️ Clear Conversation Memory", type="secondary"):
        if 'workflow' in st.session_state:
            st.session_state.workflow.get_memory().clear_memory()
            st.session_state.messages = []
            st.success("Memory cleared!")
            st.rerun()
    
    # Display memory stats
    if 'workflow' in st.session_state:
        try:
            memory_stats = st.session_state.workflow.get_memory().get_memory_usage()
            st.markdown('<div class="memory-stats">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Conversations", memory_stats['total_conversations'])
            with col2:
                st.metric("Memory Usage", f"{memory_stats['memory_used_percent']}%")
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Could not load memory stats: {e}")
    
    # System information
    with st.expander("📊 System Information", expanded=False):
        if 'workflow' in st.session_state:
            try:
                stats = st.session_state.workflow.get_workflow_stats()
                st.json(stats)
            except Exception as e:
                st.error(f"Could not load system stats: {e}")

# Main chat interface
st.markdown("### 💬 Chat Interface")

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input using new Streamlit v1.48 chat features
if prompt := st.chat_input("Ask me anything... (e.g., 'What causes rainbows?')"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching and analyzing..."):
            try:
                # Run the workflow
                result = st.session_state.workflow.run(prompt)
                
                # Check for errors
                if result.get('error'):
                    st.error(f"❌ Error: {result['error']}")
                    response_text = f"I apologize, but I encountered an error: {result['error']}"
                else:
                    response_text = result.get('formatted_response', 'No response generated')
                
                # Display the response
                st.markdown(response_text)
                
                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
                # Display sources in an expandable section
                sources = result.get('sources', [])
                if sources:
                    with st.expander(f"📚 View {len(sources)} Sources", expanded=False):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"""
                            <div class="source-item">
                                <strong>{i}. <a href="{source.get('url', '#')}" target="_blank">{source.get('title', 'Unknown Title')}</a></strong><br>
                                <small>{source.get('snippet', 'No description available')}</small>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Display additional debug info if needed
                if st.checkbox("🔍 Show Debug Information", key="debug_info"):
                    with st.expander("Debug Information", expanded=False):
                        st.json({
                            "search_results_count": len(result.get('search_results', [])),
                            "successful_scrapes": len([r for r in result.get('scrape_results', []) if r.get('success')]),
                            "embeddings_count": len(result.get('embeddings', [])),
                            "context_length": len(result.get('context', '')),
                            "has_error": bool(result.get('error'))
                        })
                        
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                error_response = f"I apologize, but I encountered an unexpected error: {str(e)}"
                st.markdown(error_response)
                st.session_state.messages.append({"role": "assistant", "content": error_response})

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    🔍 Perplexity Clone - Built with Streamlit v1.48.0, LangChain, LangGraph v0.6.5, and Mistral AI<br>
    <a href="https://github.com/your-repo" target="_blank">View Source Code</a>
</div>
""", unsafe_allow_html=True)
