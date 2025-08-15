"""
LLM Engine using Mistral AI for embeddings and text generation.
Updated for Mistral AI client v1.2.0 (August 2025).
"""

import logging
from typing import List, Dict, Any, Optional
from mistralai import Mistral
import numpy as np
from ..utils.config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MistralLLMEngine:
    """Mistral AI integration for embeddings and text generation."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Mistral client."""
        self.api_key = api_key or config.mistral_api_key
        if not self.api_key:
            raise ValueError("Mistral API key is required")
        
        # Initialize Mistral client with v1.0+ API
        self.client = Mistral(api_key=self.api_key)
        
        # Model configurations - updated model names for 2025
        self.embedding_model = "mistral-embed"
        self.chat_model = "mistral-large-latest"
        
        logger.info("Initialized Mistral LLM Engine with v1.0+ client")
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a list of texts."""
        try:
            if not texts:
                return []
            
            logger.info(f"Creating embeddings for {len(texts)} texts")
            
            # Call Mistral embeddings API with v1.0+ syntax
            response = self.client.embeddings.create(
                model=self.embedding_model,
                inputs=texts
            )
            
            # Extract embeddings from response
            embeddings = [data.embedding for data in response.data]
            
            logger.info(f"Successfully created {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            return []
    
    def create_single_embedding(self, text: str) -> List[float]:
        """Create embedding for a single text."""
        embeddings = self.create_embeddings([text])
        return embeddings[0] if embeddings else []
    
    def generate_response(
        self, 
        query: str, 
        context: str, 
        sources: List[Dict[str, str]],
        conversation_history: str = ""
    ) -> str:
        """Generate response using Mistral chat model."""
        try:
            # Construct system prompt
            system_prompt = self._build_system_prompt()
            
            # Construct user message with context
            user_message = self._build_user_message(query, context, sources, conversation_history)
            
            logger.info(f"Generating response for query: {query}")
            
            # Call Mistral chat API with v1.0+ syntax
            response = self.client.chat.complete(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            # Extract generated text from response
            generated_text = response.choices[0].message.content
            logger.info("Successfully generated response")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I apologize, but I encountered an error while generating a response: {str(e)}"
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for the AI assistant."""
        return """You are an AI research assistant that provides comprehensive, accurate, and well-sourced answers to user questions. Your responses should be:

1. **Informative and Comprehensive**: Provide detailed explanations that fully address the user's question
2. **Well-Structured**: Organize information logically with clear sections when appropriate
3. **Source-Aware**: Reference the provided sources naturally in your response
4. **Accurate**: Only use information from the provided context and sources
5. **Conversational**: Maintain a helpful and engaging tone
6. **Honest**: If you cannot answer something based on the provided information, say so

When responding:
- Use the context and sources provided to answer the question
- Integrate information from multiple sources when relevant
- Be specific and provide examples when possible
- If the sources don't contain enough information to fully answer the question, acknowledge this limitation
- Maintain conversation context when relevant to the current query"""
    
    def _build_user_message(
        self, 
        query: str, 
        context: str, 
        sources: List[Dict[str, str]], 
        conversation_history: str
    ) -> str:
        """Build user message with context and sources."""
        
        # Format sources information
        sources_info = ""
        if sources:
            sources_list = []
            for i, source in enumerate(sources, 1):
                title = source.get('title', 'Unknown Title')
                url = source.get('url', '')
                snippet = source.get('snippet', '')
                sources_list.append(f"Source {i}: {title} ({url})\n{snippet}")
            sources_info = "\n\n".join(sources_list)
        
        # Build complete message
        message_parts = []
        
        if conversation_history:
            message_parts.append(f"Previous conversation context:\n{conversation_history}")
        
        if context:
            message_parts.append(f"Relevant information from web search:\n{context}")
        
        if sources_info:
            message_parts.append(f"Sources:\n{sources_info}")
        
        message_parts.append(f"User Question: {query}")
        
        message_parts.append("""Please provide a comprehensive answer based on the information above. Reference specific sources naturally in your response where relevant.""")
        
        return "\n\n".join(message_parts)
    
    def chunk_text(self, text: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None) -> List[str]:
        """Chunk text into smaller pieces for processing."""
        chunk_size = chunk_size or config.chunk_size
        overlap = overlap or config.chunk_overlap
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at word boundary
            if end < len(text):
                # Look for last space before the end
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        logger.info(f"Chunked text into {len(chunks)} pieces")
        return chunks
    
    def validate_api_key(self) -> bool:
        """Validate that the API key is working."""
        try:
            # Test with a simple embedding call
            test_embedding = self.create_single_embedding("test")
            return len(test_embedding) > 0
        except Exception as e:
            logger.error(f"API key validation failed: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the models being used."""
        return {
            "embedding_model": self.embedding_model,
            "chat_model": self.chat_model,
            "embedding_dimension": str(config.embedding_dimension),
            "client_version": "v1.2.0+"
        }
