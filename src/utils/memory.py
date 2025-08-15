"""
Memory management for handling conversation history and context.
Updated for latest library versions (August 2025).
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import json

class ConversationMemory:
    """Manages conversation history with configurable memory limits."""
    
    def __init__(self, memory_limit: int = 3):
        """
        Initialize conversation memory.
        
        Args:
            memory_limit: Maximum number of conversation turns to remember
        """
        self.memory_limit = memory_limit
        self.conversations: List[Dict[str, Any]] = []
        self.current_sources: List[Dict[str, str]] = []
    
    def add_conversation(
        self, 
        query: str, 
        response: str, 
        sources: List[Dict[str, str]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a new conversation turn to memory."""
        conversation = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "sources": sources,
            "metadata": metadata or {}
        }
        
        self.conversations.append(conversation)
        
        # Maintain memory limit
        if len(self.conversations) > self.memory_limit:
            self.conversations = self.conversations[-self.memory_limit:]
    
    def get_recent_context(self, include_sources: bool = False) -> str:
        """Get recent conversation context for LLM."""
        if not self.conversations:
            return ""
        
        context_parts = []
        for conv in self.conversations[-2:]:  # Last 2 conversations for context
            context_parts.append(f"User: {conv['query']}")
            if include_sources and conv['sources']:
                source_info = ", ".join([f"{s['title']} ({s['url']})" for s in conv['sources'][:2]])
                context_parts.append(f"Assistant (using sources: {source_info}): {conv['response']}")
            else:
                context_parts.append(f"Assistant: {conv['response']}")
        
        return "\n\n".join(context_parts)
    
    def get_all_conversations(self) -> List[Dict[str, Any]]:
        """Get all stored conversations."""
        return self.conversations.copy()
    
    def clear_memory(self) -> None:
        """Clear all conversation history."""
        self.conversations.clear()
        self.current_sources.clear()
    
    def set_current_sources(self, sources: List[Dict[str, str]]) -> None:
        """Set sources for the current query being processed."""
        self.current_sources = sources
    
    def get_current_sources(self) -> List[Dict[str, str]]:
        """Get sources for the current query."""
        return self.current_sources.copy()
    
    def export_history(self) -> str:
        """Export conversation history as JSON string."""
        return json.dumps(self.conversations, indent=2)
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get current memory usage statistics."""
        return {
            "total_conversations": len(self.conversations),
            "memory_limit": self.memory_limit,
            "memory_used_percent": int((len(self.conversations) / self.memory_limit) * 100) if self.memory_limit > 0 else 0
        }
