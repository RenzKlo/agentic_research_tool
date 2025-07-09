"""
Web Research Tool for the Agentic AI system.
"""

import asyncio
import aiohttp
import requests
from typing import Dict, Any, List, Optional, Union
from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
from bs4 import BeautifulSoup
import json
import logging
from datetime import datetime
from urllib.parse import urljoin, urlparse
import time

logger = logging.getLogger(__name__)


class WebResearchInput(BaseModel):
    """Input schema for web research operations"""
    operation: str = Field(description="Operation to perform: search, scrape, analyze, news, academic")
    query: Optional[str] = Field(None, description="Search query (required for search, news, academic operations)")
    url: Optional[str] = Field(None, description="URL to scrape or analyze (required for scrape, analyze operations)")
    num_results: int = Field(default=5, description="Number of results to return")
    extract_type: str = Field(default="text", description="Type of extraction: text, links, images, tables")
    css_selector: Optional[str] = Field(default=None, description="CSS selector for specific elements")
    analysis_type: str = Field(default="summary", description="Type of analysis: summary, sentiment, keywords")


class WebResearchTool(BaseTool):
    """Tool for web research including search, scraping, and analysis."""
    
    name = "web_research"
    description = """
    Perform web research operations. Call with operation and parameters as separate arguments.
    
    Available operations:
    - To search web: call with operation="search", query="your search terms", num_results=5
    - To search news: call with operation="news", query="your search terms", num_results=5
    - To search academic: call with operation="academic", query="your search terms", num_results=5
    - To scrape URL: call with operation="scrape", url="https://example.com", extract_type="text"
    - To analyze content: call with operation="analyze", url="https://example.com", analysis_type="summary"
    
    Always provide the operation parameter first, then the specific parameters needed for that operation.
    """
    
    # Class-level configuration
    _config = {
        "tavily_api_key": None,
        "session": None,
        "headers": {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    }
    
    def __init__(self, tavily_api_key: Optional[str] = None):
        super().__init__()
        # Store configuration in class-level dict to avoid Pydantic issues
        self._config["tavily_api_key"] = tavily_api_key
    
    def _run(self, tool_input: str) -> Dict[str, Any]:
        """Execute web research operation synchronously. Handles both old and new LangChain formats."""
        try:
            # Handle different input formats from LangChain
            if isinstance(tool_input, str):
                # Old LangChain format - try to parse the string
                return self._handle_string_input(tool_input)
            elif isinstance(tool_input, dict):
                # New format with proper parameters
                operation = tool_input.get("operation", "")
                query = tool_input.get("query")
                url = tool_input.get("url")
                num_results = tool_input.get("num_results", 5)
                extract_type = tool_input.get("extract_type", "text")
                css_selector = tool_input.get("css_selector")
                analysis_type = tool_input.get("analysis_type", "summary")
                
                return self._execute_operation(operation, query, url, num_results, extract_type, css_selector, analysis_type)
            else:
                # Handle the named parameter format
                return self._execute_operation(
                    operation=tool_input,
                    query=None,
                    url=None,
                    num_results=5,
                    extract_type="text",
                    css_selector=None,
                    analysis_type="summary"
                )
        except Exception as e:
            logger.error(f"Error in web research tool: {str(e)}")
            return {"error": str(e), "success": False}
    
    def _handle_string_input(self, input_str: str) -> Dict[str, Any]:
        """Handle string input from older LangChain versions."""
        input_str = input_str.strip()
        
        # Try to detect the operation from the input string
        if input_str.startswith("search") or "search for" in input_str.lower():
            # Extract query from search request
            if input_str.startswith("search "):
                query = input_str[7:].strip()
            elif "search for" in input_str.lower():
                query = input_str.lower().split("search for", 1)[1].strip()
            else:
                query = input_str
            return self._execute_operation("search", query, None, 5, "text", None, "summary")
            
        elif input_str.startswith("news") or "news about" in input_str.lower():
            # Extract query from news request
            if input_str.startswith("news "):
                query = input_str[5:].strip()
            elif "news about" in input_str.lower():
                query = input_str.lower().split("news about", 1)[1].strip()
            else:
                query = input_str
            return self._execute_operation("news", query, None, 5, "text", None, "summary")
            
        elif input_str.startswith("academic") or "academic papers" in input_str.lower():
            # Extract query from academic request
            if input_str.startswith("academic "):
                query = input_str[9:].strip()
            else:
                query = input_str
            return self._execute_operation("academic", query, None, 5, "text", None, "summary")
            
        elif input_str.startswith("scrape") or input_str.startswith("http"):
            # Handle scrape operation
            if input_str.startswith("scrape "):
                url = input_str[7:].strip()
            else:
                url = input_str
            return self._execute_operation("scrape", None, url, 5, "text", None, "summary")
            
        elif input_str.startswith("analyze"):
            # Handle analyze operation
            if input_str.startswith("analyze "):
                url = input_str[8:].strip()
            else:
                url = input_str
            return self._execute_operation("analyze", None, url, 5, "text", None, "summary")
        else:
            # Default to search if we can't determine the operation
            return self._execute_operation("search", input_str, None, 5, "text", None, "summary")
    
    def _execute_operation(self, 
                          operation: str, 
                          query: Optional[str] = None,
                          url: Optional[str] = None,
                          num_results: int = 5,
                          extract_type: str = "text",
                          css_selector: Optional[str] = None,
                          analysis_type: str = "summary") -> Dict[str, Any]:
        """Execute the actual web research operation."""
        try:
            if operation == "search":
                if not query:
                    raise ValueError("Query is required for search operation")
                return self._search_web(query, num_results, "general")
            elif operation == "scrape":
                if not url:
                    raise ValueError("URL is required for scrape operation")
                return self._scrape_url(url, extract_type, css_selector)
            elif operation == "analyze":
                if not url:
                    raise ValueError("URL is required for analyze operation")
                return self._analyze_content(url, analysis_type)
            elif operation == "news":
                if not query:
                    raise ValueError("Query is required for news operation")
                return self._search_news(query, num_results)
            elif operation == "academic":
                if not query:
                    raise ValueError("Query is required for academic operation")
                return self._search_academic(query, num_results)
            else:
                raise ValueError(f"Unknown operation: {operation}")
        except Exception as e:
            logger.error(f"Error in web research operation {operation}: {str(e)}")
            return {"error": str(e), "success": False}
    
    async def _arun(self, 
                    operation: str, 
                    query: Optional[str] = None,
                    url: Optional[str] = None,
                    num_results: int = 5,
                    extract_type: str = "text",
                    css_selector: Optional[str] = None,
                    analysis_type: str = "summary") -> Dict[str, Any]:
        """Execute web research operation asynchronously."""
        try:
            if operation == "search":
                if not query:
                    raise ValueError("Query is required for search operation")
                return await self._search_web_async(query, num_results, "general")
            elif operation == "scrape":
                if not url:
                    raise ValueError("URL is required for scrape operation")
                return await self._scrape_url_async(url, extract_type, css_selector)
            elif operation == "analyze":
                if not url:
                    raise ValueError("URL is required for analyze operation")
                return await self._analyze_content_async(url, analysis_type)
            elif operation == "news":
                if not query:
                    raise ValueError("Query is required for news operation")
                return await self._search_news_async(query, num_results)
            elif operation == "academic":
                if not query:
                    raise ValueError("Query is required for academic operation")
                return await self._search_academic_async(query, num_results)
            else:
                raise ValueError(f"Unknown operation: {operation}")
        except Exception as e:
            logger.error(f"Error in async web research operation {operation}: {str(e)}")
            return {"error": str(e), "success": False}
    
    def _search_web(self, query: str, num_results: int = 5, search_type: str = "general") -> Dict[str, Any]:
        """Search the web using Tavily API or fallback to basic search."""
        try:
            if self._config["tavily_api_key"]:
                return self._tavily_search(query, num_results, search_type)
            else:
                return self._basic_search(query, num_results)
        except Exception as e:
            logger.error(f"Error in web search: {str(e)}")
            return {"error": str(e), "success": False}
    
    def _tavily_search(self, query: str, num_results: int = 5, search_type: str = "general") -> Dict[str, Any]:
        """Search using Tavily API."""
        try:
            url = "https://api.tavily.com/search"
            payload = {
                "api_key": self._config["tavily_api_key"],
                "query": query,
                "search_depth": "basic",
                "include_answer": True,
                "include_images": False,
                "include_raw_content": False,
                "max_results": num_results
            }
            
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            results = []
            for result in data.get("results", []):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0)
                })
            
            return {
                "success": True,
                "query": query,
                "answer": data.get("answer", ""),
                "results": results,
                "search_type": search_type,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error in Tavily search: {str(e)}")
            return {"error": str(e), "success": False}
    
    def _basic_search(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """Basic search using DuckDuckGo (as fallback)."""
        try:
            # This is a simplified implementation
            # In a real implementation, you might use libraries like duckduckgo-search
            import urllib.parse
            
            encoded_query = urllib.parse.quote(query)
            search_url = f"https://duckduckgo.com/?q={encoded_query}"
            
            response = requests.get(search_url, headers=self._config["headers"], timeout=30)
            response.raise_for_status()
            
            # This is a placeholder - actual implementation would parse search results
            return {
                "success": True,
                "query": query,
                "answer": f"Search results for: {query}",
                "results": [
                    {
                        "title": f"Search result for {query}",
                        "url": search_url,
                        "content": f"This is a basic search result for {query}. For better results, configure Tavily API key.",
                        "score": 0.5
                    }
                ],
                "search_type": "basic",
                "timestamp": datetime.now().isoformat(),
                "note": "Basic search mode. Configure Tavily API key for better results."
            }
        
        except Exception as e:
            logger.error(f"Error in basic search: {str(e)}")
            return {"error": str(e), "success": False}
    
    def _scrape_url(self, url: str, extract_type: str = "text", css_selector: Optional[str] = None) -> Dict[str, Any]:
        """Scrape content from a URL."""
        try:
            response = requests.get(url, headers=self._config["headers"], timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            if extract_type == "text":
                content = self._extract_text(soup, css_selector)
            elif extract_type == "links":
                content = self._extract_links(soup, url)
            elif extract_type == "images":
                content = self._extract_images(soup, url)
            elif extract_type == "tables":
                content = self._extract_tables(soup)
            else:
                content = soup.get_text()
            
            return {
                "success": True,
                "url": url,
                "extract_type": extract_type,
                "content": content,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {str(e)}")
            return {"error": str(e), "success": False}
    
    def _extract_text(self, soup: BeautifulSoup, css_selector: Optional[str] = None) -> str:
        """Extract text content from HTML."""
        if css_selector:
            elements = soup.select(css_selector)
            return "\n".join([elem.get_text().strip() for elem in elements])
        else:
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text()
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract links from HTML."""
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(base_url, href)
            links.append({
                "text": link.get_text().strip(),
                "url": absolute_url
            })
        return links
    
    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract images from HTML."""
        images = []
        for img in soup.find_all('img', src=True):
            src = img['src']
            absolute_url = urljoin(base_url, src)
            images.append({
                "alt": img.get('alt', ''),
                "src": absolute_url
            })
        return images
    
    def _extract_tables(self, soup: BeautifulSoup) -> List[List[List[str]]]:
        """Extract tables from HTML."""
        tables = []
        for table in soup.find_all('table'):
            table_data = []
            for row in table.find_all('tr'):
                row_data = []
                for cell in row.find_all(['td', 'th']):
                    row_data.append(cell.get_text().strip())
                if row_data:
                    table_data.append(row_data)
            if table_data:
                tables.append(table_data)
        return tables
    
    def _analyze_content(self, url: str, analysis_type: str = "summary") -> Dict[str, Any]:
        """Analyze web content."""
        try:
            # First scrape the content
            scrape_result = self._scrape_url(url, "text")
            
            if not scrape_result.get("success"):
                return scrape_result
            
            content = scrape_result.get("content", "")
            
            if analysis_type == "summary":
                analysis = self._summarize_content(content)
            elif analysis_type == "sentiment":
                analysis = self._analyze_sentiment(content)
            elif analysis_type == "keywords":
                analysis = self._extract_keywords(content)
            else:
                analysis = {"error": f"Unknown analysis type: {analysis_type}"}
            
            return {
                "success": True,
                "url": url,
                "analysis_type": analysis_type,
                "analysis": analysis,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error analyzing content from {url}: {str(e)}")
            return {"error": str(e), "success": False}
    
    def _summarize_content(self, content: str) -> Dict[str, Any]:
        """Create a basic summary of content."""
        sentences = content.split('.')
        word_count = len(content.split())
        char_count = len(content)
        
        # Simple summary - first few sentences
        summary = '. '.join(sentences[:3]) + '.' if len(sentences) > 3 else content
        
        return {
            "summary": summary,
            "word_count": word_count,
            "character_count": char_count,
            "sentence_count": len(sentences)
        }
    
    def _analyze_sentiment(self, content: str) -> Dict[str, Any]:
        """Analyze sentiment of content (basic implementation)."""
        # This is a very basic sentiment analysis
        # In a real implementation, you might use libraries like TextBlob or VADER
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'positive', 'happy']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'negative', 'sad', 'angry', 'disappointed']
        
        words = content.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count > negative_count:
            sentiment = "positive"
        elif negative_count > positive_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "sentiment": sentiment,
            "positive_score": positive_count,
            "negative_score": negative_count,
            "total_words": len(words)
        }
    
    def _extract_keywords(self, content: str) -> Dict[str, Any]:
        """Extract keywords from content (basic implementation)."""
        # This is a very basic keyword extraction
        # In a real implementation, you might use libraries like NLTK or spaCy
        
        # Common stop words
        stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'])
        
        words = content.lower().split()
        word_freq = {}
        
        for word in words:
            # Remove punctuation
            word = ''.join(char for char in word if char.isalnum())
            if word and word not in stop_words and len(word) > 2:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "keywords": sorted_words[:20],  # Top 20 keywords
            "total_unique_words": len(word_freq),
            "total_words": len(words)
        }
    
    def _search_news(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """Search for news articles."""
        # Add "news" to the query for better results
        news_query = f"{query} news recent"
        return self._search_web(news_query, num_results, "news")
    
    def _search_academic(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """Search for academic papers."""
        # Add academic terms to the query
        academic_query = f"{query} research paper study academic"
        return self._search_web(academic_query, num_results, "academic")
    
    # Async versions
    async def _search_web_async(self, query: str, num_results: int = 5, search_type: str = "general") -> Dict[str, Any]:
        """Async version of web search."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._search_web, query, num_results, search_type
        )
    
    async def _scrape_url_async(self, url: str, extract_type: str = "text", css_selector: Optional[str] = None) -> Dict[str, Any]:
        """Async version of URL scraping."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._scrape_url, url, extract_type, css_selector
        )
    
    async def _analyze_content_async(self, url: str, analysis_type: str = "summary") -> Dict[str, Any]:
        """Async version of content analysis."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._analyze_content, url, analysis_type
        )
    
    async def _search_news_async(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """Async version of news search."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._search_news, query, num_results
        )
    
    async def _search_academic_async(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """Async version of academic search."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._search_academic, query, num_results
        )
    
    def get_available_operations(self) -> List[str]:
        """Get list of available operations."""
        return ["search", "scrape", "analyze", "news", "academic"]
    
    def get_operation_description(self, operation: str) -> str:
        """Get description of a specific operation."""
        descriptions = {
            "search": "Search the web for information on any topic",
            "scrape": "Extract content from web pages (text, links, images, tables)",
            "analyze": "Analyze web content for insights (summary, sentiment, keywords)",
            "news": "Search for recent news articles",
            "academic": "Search for academic papers and research"
        }
        return descriptions.get(operation, "Unknown operation")
    
    def __del__(self):
        """Clean up resources."""
        if self._config.get("session"):
            try:
                asyncio.create_task(self._config["session"].close())
            except:
                pass
