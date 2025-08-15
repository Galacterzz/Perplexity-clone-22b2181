"""
Configuration management for the Perplexity-like application.
Updated for latest library versions (August 2025).
"""

import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config(BaseModel):
    """Application configuration settings."""
    
    # API Keys
    mistral_api_key: str = Field(..., description="Mistral AI API key")
    brave_search_api_key: str = Field(..., description="Brave Search API key")  
    langsmith_api_key: Optional[str] = Field(None, description="LangSmith API key")
    
    # LangSmith Settings
    langsmith_project: str = Field(default="perplexity-clone", description="LangSmith project name")
    langsmith_tracing: bool = Field(default=True, description="Enable LangSmith tracing")
    
    # Search Settings
    max_search_results: int = Field(default=5, description="Maximum search results to fetch")
    max_content_length: int = Field(default=2000, description="Maximum content length per page")
    
    # Chunking Settings
    chunk_size: int = Field(default=500, description="Text chunk size for embeddings")
    chunk_overlap: int = Field(default=50, description="Overlap between text chunks")
    
    # Memory Settings
    memory_limit: int = Field(default=3, description="Maximum conversation history to keep")
    
    # Vector Store Settings
    embedding_dimension: int = Field(default=1024, description="Mistral embedding dimension")
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        return cls(
            mistral_api_key=os.getenv("MISTRAL_API_KEY", ""),
            brave_search_api_key=os.getenv("BRAVE_SEARCH_API_KEY", ""),
            langsmith_api_key=os.getenv("LANGSMITH_API_KEY"),
            langsmith_project=os.getenv("LANGSMITH_PROJECT", "perplexity-clone"),
            langsmith_tracing=os.getenv("LANGSMITH_TRACING", "true").lower() == "true",
            max_search_results=int(os.getenv("MAX_SEARCH_RESULTS", "5")),
            max_content_length=int(os.getenv("MAX_CONTENT_LENGTH", "2000")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
            memory_limit=int(os.getenv("MEMORY_LIMIT", "3")),
        )
    
    def validate_keys(self) -> bool:
        """Validate that required API keys are present."""
        if not self.mistral_api_key:
            raise ValueError("MISTRAL_API_KEY is required")
        if not self.brave_search_api_key:
            raise ValueError("BRAVE_SEARCH_API_KEY is required")
        return True
    
    def setup_langsmith(self) -> None:
        """Setup LangSmith tracing if configured."""
        if self.langsmith_tracing and self.langsmith_api_key:
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGSMITH_API_KEY"] = self.langsmith_api_key
            os.environ["LANGSMITH_PROJECT"] = self.langsmith_project

# Global config instance
config = Config.from_env()
config.setup_langsmith()
