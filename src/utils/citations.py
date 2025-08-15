"""
Citation and source formatting utilities.
Updated for latest library versions (August 2025).
"""

from typing import List, Dict, Any
import re

class CitationFormatter:
    """Handles formatting of citations and sources."""
    
    @staticmethod
    def format_sources_list(sources: List[Dict[str, str]]) -> str:
        """Format sources into a numbered list for display."""
        if not sources:
            return "No sources available."
        
        formatted_sources = []
        for i, source in enumerate(sources, 1):
            title = source.get('title', 'Unknown Title')
            url = source.get('url', '#')
            snippet = source.get('snippet', '')
            
            # Truncate snippet if too long
            if len(snippet) > 150:
                snippet = snippet[:147] + "..."
            
            source_text = f"**{i}. [{title}]({url})**"
            if snippet:
                source_text += f"\n   {snippet}"
            
            formatted_sources.append(source_text)
        
        return "\n\n".join(formatted_sources)
    
    @staticmethod
    def add_inline_citations(text: str, sources: List[Dict[str, str]]) -> str:
        """Add inline citations to text based on content matching."""
        if not sources:
            return text
        
        # Simple citation insertion at the end of sentences
        sentences = text.split('. ')
        
        # Add citations to every few sentences
        cited_sentences = []
        for i, sentence in enumerate(sentences):
            cited_sentences.append(sentence)
            
            # Add citation every 2-3 sentences
            if i > 0 and (i + 1) % 2 == 0 and i < len(sentences) - 1:
                # Cycle through available sources
                source_num = (i // 2) % len(sources) + 1
                cited_sentences[-1] += f" [{source_num}]"
        
        return '. '.join(cited_sentences)
    
    @staticmethod
    def create_source_dict(title: str, url: str, snippet: str = "") -> Dict[str, str]:
        """Create a standardized source dictionary."""
        return {
            "title": title.strip(),
            "url": url.strip(),
            "snippet": snippet.strip()
        }
    
    @staticmethod
    def clean_snippet(snippet: str, max_length: int = 200) -> str:
        """Clean and truncate snippet text."""
        if not snippet:
            return ""
        
        # Clean up whitespace and newlines
        cleaned = re.sub(r'\s+', ' ', snippet.strip())
        
        # Truncate if too long
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length - 3] + "..."
        
        return cleaned
    
    @staticmethod
    def format_response_with_sources(response: str, sources: List[Dict[str, str]]) -> str:
        """Format complete response with inline citations and sources list."""
        if not sources:
            return response
        
        # Add inline citations
        cited_response = CitationFormatter.add_inline_citations(response, sources)
        
        # Add sources section
        sources_section = CitationFormatter.format_sources_list(sources)
        
        complete_response = f"{cited_response}\n\n---\n\n**Sources:**\n\n{sources_section}"
        
        return complete_response
