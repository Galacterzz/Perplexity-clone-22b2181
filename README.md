## 🔍 Perplexity Clone

A fully functional, modern search-and-answer assistant built with the latest AI technologies. This project searches the web, scrapes content, and generates intelligent, cited responses to user queries.

## 🎥 Video Walkthrough

Note: This is a placeholder. To add your own video, record a demo, upload it to a platform like YouTube or Loom, and replace the link and placeholder image above.

## 🚀 Features

- Web Search & Scraping: Powered by Brave Search API and BeautifulSoup.
- AI Chat Interface: Mistral AI for high-quality text generation and embeddings.
- Vector Search: FAISS (CPU) for fast, in-memory semantic similarity matching.
- Inline Citations: Every answer includes clickable markdown citations referencing the source URLs.
- Conversation Memory: Maintains context across multiple exchanges (default: 3 pairs).
- Modern UI: Built with Streamlit.
- Workflow Orchestration: Uses LangGraph to create a robust, stateful processing pipeline.
- Observability: Optional LangSmith integration for tracing and debugging.

## 📦 Tech Stack

- UI: Streamlit (1.50.0)
-A I Framework: LangChain (0.3.27) + LangGraph (0.6.8)
- LLM & Embeddings: langchain-mistralai (0.2.12)
- Search: brave-search (0.1.8)
- Scraping: BeautifulSoup (4.14.2) + Requests (2.32.5)
- Vector DB: FAISS (CPU) (1.12.0)
- Tracing: LangSmith (0.4.32) (Optional)
- Utilities: python-dotenv (1.1.1)

## 🛠️ Installation

### 1. Clone & Setup
```
# Clone the repository
git clone [https://github.com/your-username/perplexity-clone-22b2181.git](https://github.com/your-username/perplexity-clone-22b2181.git)

# Navigate to the project directory
cd perplexity-clone-22b2181

# Create a Python virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate

# Install the required dependencies
pip install -r requirements.txt
```

### 2. Get API Keys

You will need API keys from the following services:
- Mistral AI:
  - Go to console.mistral.ai
  - Create an account and generate an API key.
- Brave Search (2k free requests/month):
  - Go to brave.com/search/api
  - Sign up and get your API key.
- LangSmith (Optional, for tracing):
  - Go to smith.langchain.com
  - Create an account and generate an API key.

### 3. Configure Environment

Edit the `.env` file in the project root with the API keys you obtained:
```
# .env
MISTRAL_API_KEY="your-mistral-api-key-here"
BRAVE_SEARCH_API_KEY="your-brave-search-api-key-here"

# Optional: for LangSmith tracing
LANGCHAIN_TRACING=true 
LANGCHAIN_API_KEY="your-langsmith-api-key-here" 
LANGSMITH_ENDPOINT=[https://api.smith.langchain.com](https://api.smith.langchain.com)
LANGSMITH_PROJECT=Perplexity-Clone
```

### 4. Run the App
```
streamlit run main.py
```

## 🔧 Usage

- **Basic Usage**: Open the web interface, type your question (e.g., "What is LangGraph?"), and get an AI-generated answer with source citations.
- **Follow-up**: Continue the conversation with follow-up questions. The app maintains chat history.
- **Clear History**: Use the "Clear Chat History" button in the sidebar to reset the conversation.

## 🏗️ Architecture

This project uses LangGraph to manage a stateful workflow. The graph defines a cycle for processing a user's query from search to a final answer.

Workflow Steps
1. Search (search_node): The user's latest query is taken from the state. The perform_search function is called to query the Brave Search API and retrieve a list of relevant URLs and snippets. These are added to the state as web_results.
2. Conditional Routing (should_continue): The graph checks if the web_results list is empty.
  - If results are found: The graph transitions to the retrieval_node.
  - If no results are found: The graph skips retrieval and transitions directly to the generation_node to inform the user.
3. Scrape & Embed (retrieval_node): This node performs three steps:
  - Fetch & Parse: fetch_and_parse_documents scrapes the content from the URLs using requests and BeautifulSoup.
  - Embed & Store: create_and_store_embeddings chunks the scraped text and uses MistralAIEmbeddings to create a FAISS vector store in memory.
  - Retrieve: retrieve_relevant_context performs a similarity search on the FAISS store using the user's query to find the most relevant context chunks.
4. Generate (generation_node): The retrieved context chunks and the user's query are formatted into a detailed prompt. This prompt is sent to the ChatMistralAI LLM, which generates a comprehensive answer and includes [Source X] placeholders.
5. Format & Respond: The generation_node formats the LLM's raw text. It replaces the [Source X] placeholders with clickable markdown links (e.g., [1](https://example.com)) based on the metadata of the retrieved chunks. This final, formatted response is added to the chat history, and the graph flow ends.

## 📁 Project Structure
```
perplexity-clone-22b2181/
│
├── .env                  # Environment variables (API keys)
├── main.py               # Main entry point to run the Streamlit app
├── requirements.txt      # Python dependencies
├── README.md             # This file
│
└── src/
    ├── __init__.py
    ├── config.py         # API key loading, LLM and Embedding model initialization
    ├── graph.py          # LangGraph workflow definition and compilation
    ├── ui.py             # Streamlit user interface logic
    │
    ├── components/
    │   ├── __init__.py
    │   ├── search_retrieval.py # Functions for Brave search, scraping, and FAISS
    │   └── state_management.py # Defines the 'State' TypedDict for the graph
    │
    └── graph_nodes/
        ├── __init__.py
        ├── generation_node.py  # Node for generating the final LLM response
        ├── retrieval_node.py   # Node for scraping, embedding, and retrieving context
        └── search_node.py      # Node for performing the initial web search
```

## 🔮 Extending the App

- Add New LLM Providers: Modify src/config.py to initialize a different LLM (e.g., ChatOpenAI) and update the llm variable.
- Custom Search Engines: Replace the perform_search function in src/components/search_retrieval.py with your own search logic.
- Persistent Memory: Modify src/ui.py to save st.session_state.messages to a database (like SQLite or Redis) instead of just in-session memory.

## 🤝 Contributing

Fork the repository, create a feature branch, make your changes, and submit a pull request.

## 📄 License

This project is open-sourced. Please add a LICENSE file (e.g., MIT License) to define the terms of use.

## 🙏 Acknowledgments

- Mistral AI for powerful LLM and embedding capabilities.
- Brave Search for a privacy-focused search API.
- LangChain & LangGraph for the amazing AI application framework.
- Streamlit for the simple and powerful web framework.
