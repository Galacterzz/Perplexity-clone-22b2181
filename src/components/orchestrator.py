"""
Orchestrator workflow using LangGraph v0.6.5.
Updated for the latest LangGraph API (August 2025).
"""

import logging
from typing import List, Dict, Any, Optional, TypedDict
from langgraph.graph import StateGraph, START, END
from langsmith import traceable
from ..utils.config import config
from ..utils.memory import ConversationMemory
from ..utils.citations import CitationFormatter
from .search_engine import BraveSearchEngine
from .web_scraper import WebScraper
from .llm_engine import MistralLLMEngine
from .vector_store import FAISSVectorStore

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define state schema for LangGraph v0.6+
class WorkflowState(TypedDict):
    """State schema for the workflow."""
    query: str
    search_results: List[Dict[str, Any]]
    scrape_results: List[Dict[str, Any]]
    embeddings: List[List[float]]
    context: str
    response: str
    formatted_response: str
    sources: List[Dict[str, str]]
    error: Optional[str]

class PerplexityWorkflow:
    """Main workflow orchestrator using LangGraph v0.6.5."""
    
    def __init__(self):
        """Initialize all components and build the workflow graph."""
        self.search_engine = BraveSearchEngine()
        self.web_scraper = WebScraper()
        self.llm_engine = MistralLLMEngine()
        self.vector_store = FAISSVectorStore()
        self.memory = ConversationMemory(memory_limit=config.memory_limit)
        
        # Build workflow graph
        self.workflow = self._build_workflow()
        
        logger.info("Initialized PerplexityWorkflow with LangGraph v0.6.5")
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with the new v0.6 API."""
        # Create StateGraph with proper typing
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("search", self._search_node)
        workflow.add_node("scrape", self._scrape_node) 
        workflow.add_node("embed", self._embed_node)
        workflow.add_node("context_retrieval", self._context_retrieval_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("format_response", self._format_response_node)
        
        # Define the workflow edges
        workflow.add_edge(START, "search")
        workflow.add_edge("search", "scrape")
        workflow.add_edge("scrape", "embed")
        workflow.add_edge("embed", "context_retrieval")
        workflow.add_edge("context_retrieval", "generate")
        workflow.add_edge("generate", "format_response")
        workflow.add_edge("format_response", END)
        
        # Compile the workflow
        return workflow.compile()
    
    @traceable
    def run(self, query: str) -> Dict[str, Any]:
        """Run the workflow for a given query."""
        logger.info(f"Running workflow for query: {query}")
        
        # Clear vector store for each new top-level query
        self.vector_store.clear()
        
        # Initial state
        initial_state: WorkflowState = {
            "query": query,
            "search_results": [],
            "scrape_results": [],
            "embeddings": [],
            "context": "",
            "response": "",
            "formatted_response": "",
            "sources": [],
            "error": None
        }
        
        try:
            # Run the workflow
            final_state = self.workflow.invoke(initial_state)
            return dict(final_state)
        except Exception as e:
            logger.error(f"Error running workflow: {str(e)}")
            return {
                **initial_state,
                "error": str(e),
                "formatted_response": f"I apologize, but I encountered an error: {str(e)}"
            }
    
    # Node implementations with proper typing and error handling
    @traceable
    def _search_node(self, state: WorkflowState) -> WorkflowState:
        """Perform web search."""
        try:
            query = state["query"]
            search_results = self.search_engine.search_sync(query)
            return {**state, "search_results": search_results}
        except Exception as e:
            logger.error(f"Search node error: {str(e)}")
            return {**state, "error": f"Search failed: {str(e)}"}
    
    @traceable
    def _scrape_node(self, state: WorkflowState) -> WorkflowState:
        """Scrape content from URLs."""
        try:
            urls = self.search_engine.extract_urls(state["search_results"])
            if not urls:
                logger.warning("No URLs found in search results")
                return {**state, "scrape_results": []}
            
            scrape_results = self.web_scraper.scrape_multiple_urls(urls)
            return {**state, "scrape_results": scrape_results}
        except Exception as e:
            logger.error(f"Scrape node error: {str(e)}")
            return {**state, "error": f"Scraping failed: {str(e)}"}
    
    @traceable 
    def _embed_node(self, state: WorkflowState) -> WorkflowState:
        """Create embeddings and store in vector database."""
        try:
            successful_scrapes = self.web_scraper.filter_successful_scrapes(state["scrape_results"])
            
            if not successful_scrapes:
                logger.warning("No successful scrapes to embed")
                return {**state, "embeddings": [], "sources": []}
            
            # Prepare texts and metadata
            texts = [scrape["content"] for scrape in successful_scrapes]
            metadata = [
                {
                    "title": scrape["title"], 
                    "url": scrape["url"],
                    "snippet": scrape["metadata"].get("description", "")[:200]
                }
                for scrape in successful_scrapes
            ]
            
            # Create embeddings
            embeddings = self.llm_engine.create_embeddings(texts)
            
            if embeddings:
                # Add to vector store
                self.vector_store.add_documents(texts, embeddings, metadata)
                
                # Limit sources for display
                sources = metadata[:config.max_search_results]
            else:
                sources = []
            
            return {
                **state, 
                "embeddings": embeddings,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"Embed node error: {str(e)}")
            return {**state, "error": f"Embedding failed: {str(e)}"}
    
    @traceable
    def _context_retrieval_node(self, state: WorkflowState) -> WorkflowState:
        """Retrieve relevant context for the query."""
        try:
            query = state["query"]
            
            # Create query embedding
            query_embedding = self.llm_engine.create_single_embedding(query)
            
            if not query_embedding:
                logger.warning("Failed to create query embedding")
                return {**state, "context": ""}
            
            # Search for similar documents
            similar_docs = self.vector_store.search(query_embedding, k=5)
            
            # Combine contexts
            context = "\n\n".join([doc["text"] for doc in similar_docs])
            
            return {**state, "context": context}
        except Exception as e:
            logger.error(f"Context retrieval node error: {str(e)}")
            return {**state, "error": f"Context retrieval failed: {str(e)}"}
    
    @traceable
    def _generate_node(self, state: WorkflowState) -> WorkflowState:
        """Generate response using LLM."""
        try:
            query = state["query"]
            context = state["context"]
            sources = state["sources"]
            
            # Get conversation history
            conversation_history = self.memory.get_recent_context(include_sources=True)
            
            # Generate response
            response = self.llm_engine.generate_response(
                query=query,
                context=context,
                sources=sources,
                conversation_history=conversation_history
            )
            
            # Add to memory
            self.memory.add_conversation(query, response, sources)
            
            return {**state, "response": response}
        except Exception as e:
            logger.error(f"Generate node error: {str(e)}")
            return {**state, "error": f"Response generation failed: {str(e)}"}
    
    @traceable
    def _format_response_node(self, state: WorkflowState) -> WorkflowState:
        """Format final response with citations."""
        try:
            response = state["response"]
            sources = state["sources"]
            
            # Format response with citations
            formatted_response = CitationFormatter.format_response_with_sources(response, sources)
            
            return {**state, "formatted_response": formatted_response}
        except Exception as e:
            logger.error(f"Format response node error: {str(e)}")
            return {
                **state, 
                "formatted_response": state["response"] or "Response formatting failed",
                "error": f"Formatting failed: {str(e)}"
            }
    
    def get_memory(self) -> ConversationMemory:
        """Get the conversation memory instance."""
        return self.memory
    
    def get_workflow_stats(self) -> Dict[str, Any]:
        """Get workflow statistics."""
        return {
            "vector_store_stats": self.vector_store.get_stats(),
            "memory_stats": self.memory.get_memory_usage(),
            "model_info": self.llm_engine.get_model_info()
        }
