# 🔍 Perplexity-like Research Assistant

A fully functional, modern search-and-answer assistant built with the latest AI technologies. Search the web, scrape content, and get intelligent responses with inline citations.

## 🚀 Features

✅ **Web Search & Scraping**: Powered by Brave Search API + BeautifulSoup  
✅ **AI Chat Interface**: Mistral AI for embeddings and text generation  
✅ **Vector Search**: FAISS for semantic similarity matching  
✅ **Inline Citations**: Every answer includes source references  
✅ **Conversation Memory**: Maintains context across 3+ exchanges  
✅ **Modern UI**: Built with Streamlit 1.48.0  
✅ **Workflow Orchestration**: LangGraph 0.6.5 for robust pipelines  
✅ **Observability**: Optional LangSmith integration for tracing  

## 📦 Tech Stack (August 2025)

- **UI**: Streamlit 1.48.0
- **AI Framework**: LangChain 0.3.x + LangGraph 0.6.5
- **LLM**: Mistral AI (latest models)
- **Search**: Brave Search API (2k free requests/month)
- **Scraping**: BeautifulSoup 4.13.4
- **Vector DB**: FAISS 1.9.0
- **Tracing**: LangSmith (optional)

## 🛠️ Installation

### 1. Clone & Setup

```bash
git clone <your-repo-url>
cd perplexity_clone
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get API Keys

**Mistral AI** (Free tier available):
1. Go to [console.mistral.ai](https://console.mistral.ai)
2. Create account and generate API key

**Brave Search** (2k free requests/month):
1. Go to [brave.com/search/api](https://brave.com/search/api)  
2. Sign up and get your API key

**LangSmith** (Optional, for tracing):
1. Go to [smith.langchain.com](https://smith.langchain.com)
2. Create account and generate API key

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys:
```

```bash
MISTRAL_API_KEY=your_mistral_api_key_here
BRAVE_SEARCH_API_KEY=your_brave_search_api_key_here

# Optional (for tracing):
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=perplexity-clone
LANGSMITH_TRACING=true
```

### 4. Test Your Setup

```bash
python test_setup.py
```

### 5. Run the App

```bash
streamlit run app.py
```

Open http://localhost:8501 and start asking questions!

## 🔧 Usage

### Basic Usage
1. Open the web interface
2. Type your question (e.g., "What causes rainbows?")
3. Get an AI-generated answer with source citations
4. Continue the conversation with follow-up questions

### Advanced Features
- **Memory Management**: Clear conversation history via sidebar
- **Source Inspection**: Expand to see all sources used
- **Debug Mode**: Toggle debug info to see internal metrics
- **System Stats**: View vector store and memory usage

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │───▶│  LangGraph      │───▶│  Mistral AI     │
│                 │    │  Orchestrator   │    │  (LLM + Embed)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                         ┌─────────────────┐    ┌─────────────────┐
                         │  Brave Search   │    │  FAISS Vector   │
                         │  + Scraper      │    │  Store          │
                         └─────────────────┘    └─────────────────┘
```

### Workflow Steps
1. **Search**: Query Brave Search API for relevant URLs
2. **Scrape**: Extract content using BeautifulSoup 
3. **Embed**: Create Mistral embeddings and store in FAISS
4. **Context**: Retrieve most relevant content chunks
5. **Generate**: Use Mistral AI to synthesize response
6. **Format**: Add inline citations and source list

## 🐛 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Make sure you're in the virtual environment
source venv/bin/activate  # or venv\Scripts\activate
pip install -r requirements.txt
```

**API Key Errors**
```bash
# Check your .env file has the correct keys
python test_setup.py
```

**Search Not Working**
- Verify Brave Search API key is valid
- Check you haven't exceeded rate limits (2k/month free)

**Empty Responses**
- Check Mistral API key and quota
- Verify internet connection for web scraping

**Memory/Performance Issues**
- Reduce MAX_SEARCH_RESULTS in .env
- Clear conversation memory frequently
- Use lighter embedding models

## 🔬 Testing Individual Components

```bash
# Test search engine
python -c "from src.components.search_engine import BraveSearchEngine; print(BraveSearchEngine().search_sync('test', 1))"

# Test LLM engine  
python -c "from src.components.llm_engine import MistralLLMEngine; print(len(MistralLLMEngine().create_single_embedding('test')))"

# Test web scraper
python -c "from src.components.web_scraper import WebScraper; print(WebScraper().scrape_url('https://httpbin.org/html')['success'])"
```

## 📁 Project Structure

```
perplexity_clone/
├── app.py                    # Main Streamlit app
├── requirements.txt          # Dependencies
├── .env.example             # Environment template
├── test_setup.py           # Setup validation script
└── src/
   ├── components/
   │   ├── search_engine.py    # Brave Search integration
   │   ├── web_scraper.py      # BeautifulSoup scraper
   │   ├── llm_engine.py       # Mistral AI client
   │   ├── vector_store.py     # FAISS operations
   │   └── orchestrator.py     # LangGraph workflow
   └── utils/
       ├── config.py           # Configuration management
       ├── memory.py           # Conversation memory
       └── citations.py        # Citation formatting
```

## 🚀 Deployment

### Local Network Access
```bash
streamlit run app.py --server.address 0.0.0.0
```

### Docker (Optional)
```dockerfile
FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

## 🔮 Extending the App

### Add New LLM Providers
1. Create new engine in `src/components/`
2. Update orchestrator to use new engine
3. Add configuration options

### Custom Search Engines
1. Implement search interface in `src/components/search_engine.py`
2. Add API configuration to `src/utils/config.py`

### Enhanced Memory
1. Replace in-memory storage with persistent database
2. Implement conversation embeddings for better context

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable  
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- [Mistral AI](https://mistral.ai) for powerful LLM capabilities
- [Brave Search](https://brave.com/search/api/) for privacy-focused search
- [LangChain](https://langchain.com) for the AI application framework
- [Streamlit](https://streamlit.io) for the amazing web framework

---

Built with ❤️ using the latest AI technologies (August 2025)
