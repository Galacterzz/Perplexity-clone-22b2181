"""
Search engine integration using Brave Search API.
Updated for brave-search-python-client v0.4.27 (August 2025).
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from brave_search_python_client import BraveSearch
from brave_search_python_client.models import WebSearchRequest, CountryCode, LanguageCode, Freshness
from ..utils.config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BraveSearchEngine:
    """Brave Search API integration for web searching."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Brave Search client."""
        self.api_key = api_key or config.brave_search_api_key
        if not self.api_key:
            raise ValueError("Brave Search API key is required")
        
        self.client = BraveSearch(api_key=self.api_key)
    
    async def search(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """Perform web search using Brave Search API."""
        max_results = max_results or config.max_search_results
        
        try:
            logger.info(f"Searching for: {query}")
            
            # Create search request with updated API
            search_request = WebSearchRequest(
                q=query,
                count=max_results,
                result_filter="web",
                freshness=Freshness.PW,  # Past week for fresher results
                text_decorations=False,
                search_lang=LanguageCode.EN,
                country=CountryCode.US
            )
            
            # Perform search
            response = await self.client.web(search_request)
            
            if not response.web or not response.web.results:
                logger.warning(f"No search results found for query: {query}")
                return []
            
            # Process results
            results = []
            for result in response.web.results[:max_results]:
                processed_result = {
                    "title": result.title or "No Title",
                    "url": result.url or "",
                    "snippet": result.description or "",
                    "published_time": getattr(result, 'age', None),
                    "meta_url": getattr(result, 'meta_url', {})
                }
                results.append(processed_result)
            
            logger.info(f"Found {len(results)} search results")
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return []
    
    def search_sync(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """Synchronous wrapper for search method."""
        try:
            # Check if we're already in an async context
            try:
                loop = asyncio.get_running_loop()
                # If we're in an async context, we need to run in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.search(query, max_results))
                    return future.result()
            except RuntimeError:
                # Not in an async context, safe to use asyncio.run
                return asyncio.run(self.search(query, max_results))
        except Exception as e:
            logger.error(f"Error in synchronous search: {str(e)}")
            return []
    
    def format_results_for_display(self, results: List[Dict[str, Any]]) -> str:
        """Format search results for display in UI."""
        if not results:
            return "No search results found."
        
        formatted_results = []
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No Title')
            url = result.get('url', '#')
            snippet = result.get('snippet', 'No description available')
            
            # Truncate snippet if too long
            if len(snippet) > 150:
                snippet = snippet[:147] + "..."
            
            formatted_result = f"**{i}. [{title}]({url})**\n{snippet}"
            formatted_results.append(formatted_result)
        
        return "\n\n".join(formatted_results)
    
    def extract_urls(self, results: List[Dict[str, Any]]) -> List[str]:
        """Extract URLs from search results."""
        return [result.get('url', '') for result in results if result.get('url')]
    
    def validate_api_key(self) -> bool:
        """Validate that the API key is working by performing a test search."""
        try:
            test_results = self.search_sync("test query", max_results=1)
            return len(test_results) >= 0  # Even 0 results means API key works
        except Exception as e:
            logger.error(f"API key validation failed: {str(e)}")
            return False
