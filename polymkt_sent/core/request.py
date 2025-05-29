"""
Enhanced HTTP request handling for polymarket-sentiment.

This module provides improved request reliability with:
- Better default headers to avoid blocking
- Exponential backoff retry logic
- Response validation 
- Configurable timeouts and rate limiting
"""

import time
import random
import logging
from typing import Optional, Dict, Any, List
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup


# Default headers that work well with Nitter instances
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}

logger = logging.getLogger(__name__)


class EnhancedRequestSession:
    """Enhanced requests session with retry logic and validation."""
    
    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 5,
        backoff_factor: float = 1.0,
        min_response_size: int = 2000,
        retry_delays: Optional[List[float]] = None
    ):
        """
        Initialize the enhanced session.
        
        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            backoff_factor: Exponential backoff factor
            min_response_size: Minimum expected response size in bytes
            retry_delays: Custom retry delays (overrides backoff_factor)
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.min_response_size = min_response_size
        self.retry_delays = retry_delays or [1, 2, 4, 8, 16]
        
        # Create session with retry strategy
        self.session = requests.Session()
        
        # Set up retry strategy for connection issues
        retry_strategy = Retry(
            total=max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=backoff_factor,
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        self.session.headers.update(DEFAULT_HEADERS)
        
    def update_headers(self, headers: Dict[str, str]) -> None:
        """Update session headers."""
        self.session.headers.update(headers)
        
    def get_with_validation(
        self,
        url: str,
        instance_host: Optional[str] = None,
        cookies: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Optional[BeautifulSoup]:
        """
        Make a GET request with validation and retry logic.
        
        Args:
            url: URL to request
            instance_host: Host header value (for Nitter instances)
            cookies: Optional cookies
            **kwargs: Additional requests arguments
            
        Returns:
            BeautifulSoup object if successful, None otherwise
        """
        # Update Host header for Nitter instances
        if instance_host:
            self.session.headers.update({"Host": instance_host})
            
        # Default cookies for Nitter
        request_cookies = {"hlsPlayback": "on", "infiniteScroll": ""}
        if cookies:
            request_cookies.update(cookies)
            
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Attempting request to {url} (attempt {attempt + 1})")
                
                response = self.session.get(
                    url,
                    cookies=request_cookies,
                    timeout=self.timeout,
                    **kwargs
                )
                
                # Validate response
                if self._validate_response(response, url):
                    soup = BeautifulSoup(response.text, "lxml")
                    if self._validate_content(soup, url):
                        logger.debug(f"Successful request to {url}")
                        return soup
                        
                # If validation fails, treat as retry-able error
                logger.warning(f"Response validation failed for {url}")
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error for {url}: {e}")
                
            # Wait before retry (except on last attempt)
            if attempt < self.max_retries:
                delay = self._get_retry_delay(attempt)
                logger.debug(f"Waiting {delay}s before retry")
                time.sleep(delay)
                
        logger.error(f"All retry attempts failed for {url}")
        return None
        
    def _validate_response(self, response: requests.Response, url: str) -> bool:
        """Validate HTTP response."""
        if not response.ok:
            logger.warning(f"HTTP {response.status_code} for {url}")
            return False
            
        if len(response.content) < self.min_response_size:
            logger.warning(f"Response too small ({len(response.content)} bytes) for {url}")
            return False
            
        return True
        
    def _validate_content(self, soup: BeautifulSoup, url: str) -> bool:
        """Validate parsed content for Nitter-specific patterns."""
        if soup is None:
            return False
            
        # Check for Nitter error pages
        if soup.find("div", class_="error-panel"):
            error_msg = soup.find("div", class_="error-panel").find("span")
            if error_msg:
                logger.warning(f"Nitter error for {url}: {error_msg.text.strip()}")
            return False
            
        # Check for protected account
        if soup.find("div", class_="timeline-header timeline-protected"):
            logger.warning(f"Protected account for {url}")
            return False
            
        # For user pages, ensure we have some content
        if "/search?" not in url and not soup.find_all("div", class_="timeline-item"):
            logger.warning(f"No timeline content found for {url}")
            return False
            
        return True
        
    def _get_retry_delay(self, attempt: int) -> float:
        """Get delay before retry with jitter."""
        if attempt < len(self.retry_delays):
            base_delay = self.retry_delays[attempt]
        else:
            # Exponential backoff for attempts beyond configured delays
            base_delay = self.retry_delays[-1] * (2 ** (attempt - len(self.retry_delays) + 1))
            
        # Add random jitter (±20%)
        jitter = random.uniform(0.8, 1.2)
        return base_delay * jitter


def create_session_from_config(config: Dict[str, Any]) -> EnhancedRequestSession:
    """Create session from configuration dictionary."""
    scraping_config = config.get("scraping", {})
    rate_limit_config = config.get("rate_limit", {})
    
    return EnhancedRequestSession(
        timeout=scraping_config.get("timeout_sec", 30),
        max_retries=scraping_config.get("max_retry", 5),
        min_response_size=2000,  # Configurable in future versions
        retry_delays=rate_limit_config.get("retry_delays", [1, 2, 4, 8, 16])
    )
