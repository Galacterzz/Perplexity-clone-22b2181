"""
Web scraping component using BeautifulSoup.
Updated for beautifulsoup4 v4.13.4 (August 2025).
"""

import requests
import logging
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import re
from ..utils.config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebScraper:
    """Web scraper using BeautifulSoup for content extraction."""
    
    def __init__(self):
        """Initialize web scraper with default settings."""
        self.session = requests.Session()
        
        # Set user agent to avoid being blocked
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'
        })
        
        # Request timeout settings
        self.timeout = 10
        self.max_content_length = config.max_content_length
    
    def scrape_url(self, url: str) -> Dict[str, Any]:
        """Scrape content from a single URL."""
        try:
            logger.info(f"Scraping URL: {url}")
            
            # Make request with timeout
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                logger.warning(f"Non-HTML content type for {url}: {content_type}")
                return {"url": url, "content": "", "title": "", "error": "Non-HTML content"}
            
            # Parse HTML with lxml parser for better performance
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Extract title
            title = self._extract_title(soup)
            
            # Extract main content
            content = self._extract_content(soup)
            
            # Clean and truncate content
            content = self._clean_content(content)
            
            # Extract metadata
            metadata = self._extract_metadata(soup, url)
            
            result = {
                "url": url,
                "title": title,
                "content": content,
                "metadata": metadata,
                "success": True
            }
            
            logger.info(f"Successfully scraped {len(content)} characters from {url}")
            return result
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout while scraping {url}")
            return {"url": url, "content": "", "title": "", "error": "Timeout", "success": False}
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error while scraping {url}: {str(e)}")
            return {"url": url, "content": "", "title": "", "error": str(e), "success": False}
        
        except Exception as e:
            logger.error(f"Unexpected error while scraping {url}: {str(e)}")
            return {"url": url, "content": "", "title": "", "error": str(e), "success": False}
    
    def scrape_multiple_urls(self, urls: List[str], delay: float = 1.0) -> List[Dict[str, Any]]:
        """Scrape content from multiple URLs with rate limiting."""
        results = []
        
        for i, url in enumerate(urls):
            if i > 0:
                time.sleep(delay)  # Rate limiting
            
            result = self.scrape_url(url)
            results.append(result)
        
        successful_scrapes = len([r for r in results if r.get('success', False)])
        logger.info(f"Scraped {successful_scrapes}/{len(urls)} URLs successfully")
        
        return results
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title from HTML."""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().strip()
        
        # Fallback to h1 tags
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text().strip()
        
        return "No Title Found"
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from HTML."""
        # Remove script, style, and other non-content tags
        for tag in soup(["script", "style", "nav", "footer", "aside", "header", "noscript"]):
            tag.decompose()
        
        # Try to find main content areas
        content_selectors = [
            'article',
            '[role="main"]',
            'main',
            '.content',
            '.main-content',
            '.article-content',
            '.post-content',
            '#content',
            '#main'
        ]
        
        content_text = ""
        
        # Try each selector to find main content
        for selector in content_selectors:
            content_element = soup.select_one(selector)
            if content_element:
                content_text = content_element.get_text()
                break
        
        # Fallback to body content if no main content area found
        if not content_text:
            body = soup.find('body')
            if body:
                content_text = body.get_text()
            else:
                content_text = soup.get_text()
        
        return content_text
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize extracted content."""
        if not content:
            return ""
        
        # Replace multiple whitespace with single space
        content = re.sub(r'\s+', ' ', content)
        
        # Remove extra newlines
        content = re.sub(r'\n\s*\n', '\n', content)
        
        # Strip leading/trailing whitespace
        content = content.strip()
        
        # Truncate if too long
        if len(content) > self.max_content_length:
            content = content[:self.max_content_length - 3] + "..."
        
        return content
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, str]:
        """Extract metadata from HTML."""
        metadata = {"url": url}
        
        # Extract description from meta tags
        description_meta = soup.find('meta', attrs={'name': 'description'}) or \
                          soup.find('meta', attrs={'property': 'og:description'})
        if description_meta:
            metadata['description'] = description_meta.get('content', '')
        
        # Extract author
        author_meta = soup.find('meta', attrs={'name': 'author'}) or \
                     soup.find('meta', attrs={'property': 'article:author'})
        if author_meta:
            metadata['author'] = author_meta.get('content', '')
        
        # Extract publication date
        date_meta = soup.find('meta', attrs={'property': 'article:published_time'}) or \
                   soup.find('meta', attrs={'name': 'date'})
        if date_meta:
            metadata['published_date'] = date_meta.get('content', '')
        
        # Extract site name
        site_meta = soup.find('meta', attrs={'property': 'og:site_name'})
        if site_meta:
            metadata['site_name'] = site_meta.get('content', '')
        else:
            # Extract from URL
            parsed_url = urlparse(url)
            metadata['site_name'] = parsed_url.netloc
        
        return metadata
    
    def filter_successful_scrapes(self, scrape_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out failed scrapes and return only successful ones."""
        successful = [result for result in scrape_results if result.get('success', False) and result.get('content')]
        logger.info(f"Filtered to {len(successful)} successful scrapes from {len(scrape_results)} total")
        return successful
