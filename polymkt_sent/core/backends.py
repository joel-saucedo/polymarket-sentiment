"""
Scraper backends for polymarket-sentiment.

This module provides multiple scraping backends:
- SNSScraper (primary): Using snscrape library
- NitterScraper (fallback): Using Nitter instances

The backends are designed to be interchangeable with a common interface.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    import snscrape.modules.twitter as sntwitter
    SNSCRAPE_AVAILABLE = True
except ImportError:
    SNSCRAPE_AVAILABLE = False

from polymkt_sent.core.storage import TweetModel
from polymkt_sent.nitter import Nitter


logger = logging.getLogger(__name__)


@dataclass
class ScrapingResult:
    """Result from a scraping operation."""
    tweets: List[TweetModel]
    success: bool
    error_message: Optional[str] = None
    backend_used: Optional[str] = None


class ScraperBackend(ABC):
    """Abstract base class for scraper backends."""
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the backend name."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available."""
        pass
    
    @abstractmethod
    async def scrape_user_tweets(
        self,
        username: str,
        max_tweets: int = 100
    ) -> ScrapingResult:
        """Scrape tweets from a user timeline."""
        pass
    
    @abstractmethod
    async def scrape_search_tweets(
        self,
        query: str,
        max_tweets: int = 100
    ) -> ScrapingResult:
        """Scrape tweets from a search query."""
        pass


class SNSScrapeBackend(ScraperBackend):
    """Primary scraper backend using snscrape."""
    
    def __init__(self):
        """Initialize SNSScrape backend."""
        self.name = "snscrape"
        
    def get_name(self) -> str:
        return self.name
    
    def is_available(self) -> bool:
        """Check if snscrape is available."""
        return SNSCRAPE_AVAILABLE
    
    async def scrape_user_tweets(
        self,
        username: str,
        max_tweets: int = 100
    ) -> ScrapingResult:
        """Scrape tweets from a user timeline using snscrape."""
        if not self.is_available():
            return ScrapingResult(
                tweets=[],
                success=False,
                error_message="snscrape not available",
                backend_used=self.name
            )
        
        try:
            # Run snscrape in thread pool to avoid blocking
            tweets = await asyncio.to_thread(
                self._scrape_user_sync,
                username,
                max_tweets
            )
            
            logger.info(f"SNSScrape: Scraped {len(tweets)} tweets from user {username}")
            
            return ScrapingResult(
                tweets=tweets,
                success=True,
                backend_used=self.name
            )
            
        except Exception as e:
            logger.error(f"SNSScrape error for user {username}: {e}")
            return ScrapingResult(
                tweets=[],
                success=False,
                error_message=str(e),
                backend_used=self.name
            )
    
    async def scrape_search_tweets(
        self,
        query: str,
        max_tweets: int = 100
    ) -> ScrapingResult:
        """Scrape tweets from a search query using snscrape."""
        if not self.is_available():
            return ScrapingResult(
                tweets=[],
                success=False,
                error_message="snscrape not available",
                backend_used=self.name
            )
        
        try:
            # Run snscrape in thread pool to avoid blocking
            tweets = await asyncio.to_thread(
                self._scrape_search_sync,
                query,
                max_tweets
            )
            
            logger.info(f"SNSScrape: Scraped {len(tweets)} tweets for query '{query}'")
            
            return ScrapingResult(
                tweets=tweets,
                success=True,
                backend_used=self.name
            )
            
        except Exception as e:
            logger.error(f"SNSScrape error for query '{query}': {e}")
            return ScrapingResult(
                tweets=[],
                success=False,
                error_message=str(e),
                backend_used=self.name
            )
    
    def _scrape_user_sync(self, username: str, max_tweets: int) -> List[TweetModel]:
        """Synchronous user scraping for thread pool execution."""
        tweets = []
        
        # Create the scraper for user timeline
        scraper = sntwitter.TwitterUserScraper(username)
        
        # Get tweets up to max_tweets
        for i, tweet in enumerate(scraper.get_items()):
            if i >= max_tweets:
                break
                
            # Convert to TweetModel
            tweet_model = self._convert_sns_tweet(tweet, username)
            if tweet_model:
                tweets.append(tweet_model)
        
        return tweets
    
    def _scrape_search_sync(self, query: str, max_tweets: int) -> List[TweetModel]:
        """Synchronous search scraping for thread pool execution."""
        tweets = []
        
        # Create the scraper for search
        scraper = sntwitter.TwitterSearchScraper(query)
        
        # Get tweets up to max_tweets
        for i, tweet in enumerate(scraper.get_items()):
            if i >= max_tweets:
                break
                
            # Convert to TweetModel
            tweet_model = self._convert_sns_tweet(tweet, None)
            if tweet_model:
                tweets.append(tweet_model)
        
        return tweets
    
    def _convert_sns_tweet(self, tweet, default_username: Optional[str] = None) -> Optional[TweetModel]:
        """Convert snscrape tweet object to TweetModel."""
        try:
            # Extract tweet ID from URL if available
            tweet_id = str(tweet.id) if hasattr(tweet, 'id') else "unknown"
            
            # Get username - use from tweet or fallback to default
            username = tweet.user.username if hasattr(tweet, 'user') and tweet.user else default_username or "unknown"
            
            # Get user ID
            user_id = str(tweet.user.id) if hasattr(tweet, 'user') and tweet.user else username
            
            # Parse timestamp
            timestamp = tweet.date if hasattr(tweet, 'date') else datetime.now(timezone.utc)
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            
            # Get engagement metrics
            likes = tweet.likeCount if hasattr(tweet, 'likeCount') and tweet.likeCount else 0
            retweets = tweet.retweetCount if hasattr(tweet, 'retweetCount') and tweet.retweetCount else 0
            replies = tweet.replyCount if hasattr(tweet, 'replyCount') and tweet.replyCount else 0
            
            # Get content
            content = tweet.rawContent if hasattr(tweet, 'rawContent') else (
                tweet.content if hasattr(tweet, 'content') else ""
            )
            
            return TweetModel(
                tweet_id=tweet_id,
                user_id=user_id,
                username=username,
                content=content,
                timestamp=timestamp,
                likes=likes,
                retweets=retweets,
                replies=replies,
                source_instance="snscrape"
            )
            
        except Exception as e:
            logger.warning(f"Failed to convert snscrape tweet: {e}")
            return None


class NitterBackend(ScraperBackend):
    """Fallback scraper backend using Nitter instances."""
    
    def __init__(self, mirrors: List[str]):
        """Initialize Nitter backend with mirror list."""
        self.name = "nitter"
        self.mirrors = mirrors
        self._nitter_instances = {}
        
    def get_name(self) -> str:
        return self.name
    
    def is_available(self) -> bool:
        """Check if Nitter is available."""
        return len(self.mirrors) > 0
    
    def _get_nitter_for_mirror(self, mirror: str) -> Nitter:
        """Get or create a Nitter instance for a specific mirror."""
        if mirror not in self._nitter_instances:
            self._nitter_instances[mirror] = Nitter(
                instances=[mirror],
                log_level=0,  # Minimal logging
                skip_instance_check=True
            )
        return self._nitter_instances[mirror]
    
    async def scrape_user_tweets(
        self,
        username: str,
        max_tweets: int = 100
    ) -> ScrapingResult:
        """Scrape tweets from a user timeline using Nitter."""
        if not self.is_available():
            return ScrapingResult(
                tweets=[],
                success=False,
                error_message="No Nitter mirrors available",
                backend_used=self.name
            )
        
        # Try each mirror until one succeeds
        for mirror in self.mirrors:
            try:
                nitter = self._get_nitter_for_mirror(mirror)
                
                # Run Nitter scraping in thread pool
                raw_tweets = await asyncio.to_thread(
                    nitter.get_tweets,
                    username,
                    mode="user",
                    number=max_tweets
                )
                
                # Convert to TweetModel objects
                tweets = []
                for raw_tweet in raw_tweets.get("tweets", []):
                    tweet = self._convert_nitter_tweet(raw_tweet, username, mirror)
                    if tweet:
                        tweets.append(tweet)
                
                logger.info(f"Nitter ({mirror}): Scraped {len(tweets)} tweets from user {username}")
                
                return ScrapingResult(
                    tweets=tweets,
                    success=True,
                    backend_used=f"{self.name}:{mirror}"
                )
                
            except Exception as e:
                logger.warning(f"Nitter mirror {mirror} failed for user {username}: {e}")
                continue
        
        # All mirrors failed
        return ScrapingResult(
            tweets=[],
            success=False,
            error_message="All Nitter mirrors failed",
            backend_used=self.name
        )
    
    async def scrape_search_tweets(
        self,
        query: str,
        max_tweets: int = 100
    ) -> ScrapingResult:
        """Scrape tweets from a search query using Nitter."""
        if not self.is_available():
            return ScrapingResult(
                tweets=[],
                success=False,
                error_message="No Nitter mirrors available",
                backend_used=self.name
            )
        
        # Try each mirror until one succeeds
        for mirror in self.mirrors:
            try:
                nitter = self._get_nitter_for_mirror(mirror)
                
                # Run Nitter scraping in thread pool
                raw_tweets = await asyncio.to_thread(
                    nitter.get_tweets,
                    query,
                    mode="term",
                    number=max_tweets
                )
                
                # Convert to TweetModel objects
                tweets = []
                for raw_tweet in raw_tweets.get("tweets", []):
                    tweet = self._convert_nitter_tweet(raw_tweet, None, mirror)
                    if tweet:
                        tweets.append(tweet)
                
                logger.info(f"Nitter ({mirror}): Scraped {len(tweets)} tweets for query '{query}'")
                
                return ScrapingResult(
                    tweets=tweets,
                    success=True,
                    backend_used=f"{self.name}:{mirror}"
                )
                
            except Exception as e:
                logger.warning(f"Nitter mirror {mirror} failed for query '{query}': {e}")
                continue
        
        # All mirrors failed
        return ScrapingResult(
            tweets=[],
            success=False,
            error_message="All Nitter mirrors failed",
            backend_used=self.name
        )
    
    def _convert_nitter_tweet(
        self,
        raw_tweet: Dict[str, Any],
        default_username: Optional[str],
        mirror: str
    ) -> Optional[TweetModel]:
        """Convert Nitter raw tweet to TweetModel."""
        try:
            # Extract tweet ID from link
            tweet_id = raw_tweet.get("link", "").split("/")[-1] or "unknown"
            
            # Get username
            username = (
                raw_tweet.get("user", {}).get("username") or 
                default_username or 
                "unknown"
            )
            
            # Parse timestamp
            date_str = raw_tweet.get("date", "")
            try:
                from dateutil import parser
                timestamp = parser.parse(date_str)
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
            except Exception:
                timestamp = datetime.now(timezone.utc)
            
            return TweetModel(
                tweet_id=tweet_id,
                user_id=username,  # Use username as user_id for Nitter
                username=username,
                content=raw_tweet.get("text", ""),
                timestamp=timestamp,
                likes=raw_tweet.get("stats", {}).get("likes", 0),
                retweets=raw_tweet.get("stats", {}).get("retweets", 0),
                replies=raw_tweet.get("stats", {}).get("comments", 0),
                source_instance=f"nitter:{mirror}"
            )
            
        except Exception as e:
            logger.warning(f"Failed to convert Nitter tweet: {e}")
            return None


class BackendManager:
    """Manages multiple scraper backends with failover logic."""
    
    def __init__(self, mirrors: List[str]):
        """Initialize backend manager."""
        self.backends = []
        
        # Add SNSScrape as primary backend
        sns_backend = SNSScrapeBackend()
        if sns_backend.is_available():
            self.backends.append(sns_backend)
            logger.info("SNSScrape backend available (primary)")
        else:
            logger.warning("SNSScrape backend not available")
        
        # Add Nitter as fallback backend
        nitter_backend = NitterBackend(mirrors)
        if nitter_backend.is_available():
            self.backends.append(nitter_backend)
            logger.info(f"Nitter backend available with {len(mirrors)} mirrors (fallback)")
        else:
            logger.warning("Nitter backend not available")
        
        if not self.backends:
            raise ValueError("No scraper backends available")
    
    async def scrape_user_tweets(
        self,
        username: str,
        max_tweets: int = 100
    ) -> ScrapingResult:
        """Scrape user tweets using the first available backend."""
        for backend in self.backends:
            logger.debug(f"Trying {backend.get_name()} backend for user {username}")
            result = await backend.scrape_user_tweets(username, max_tweets)
            
            if result.success:
                logger.info(f"Successfully scraped {len(result.tweets)} tweets using {result.backend_used}")
                return result
            else:
                logger.warning(f"Backend {backend.get_name()} failed: {result.error_message}")
        
        # All backends failed
        return ScrapingResult(
            tweets=[],
            success=False,
            error_message="All backends failed",
            backend_used="none"
        )
    
    async def scrape_search_tweets(
        self,
        query: str,
        max_tweets: int = 100
    ) -> ScrapingResult:
        """Scrape search tweets using the first available backend."""
        for backend in self.backends:
            logger.debug(f"Trying {backend.get_name()} backend for query '{query}'")
            result = await backend.scrape_search_tweets(query, max_tweets)
            
            if result.success:
                logger.info(f"Successfully scraped {len(result.tweets)} tweets using {result.backend_used}")
                return result
            else:
                logger.warning(f"Backend {backend.get_name()} failed: {result.error_message}")
        
        # All backends failed
        return ScrapingResult(
            tweets=[],
            success=False,
            error_message="All backends failed",
            backend_used="none"
        )
    
    def get_available_backends(self) -> List[str]:
        """Get list of available backend names."""
        return [backend.get_name() for backend in self.backends]
