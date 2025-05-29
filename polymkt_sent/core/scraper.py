"""
Async batch scraper for polymarket-sentiment.

This module provides:
- Async batch scraping of multiple accounts/search terms
- Rate limiting and error handling
- Integration with storage layer
- Configurable concurrency and retry logic
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
import random
import yaml
from pathlib import Path

from polymkt_sent.core.request import EnhancedRequestSession
from polymkt_sent.core.storage import TweetModel, TweetStorage, StorageConfig
from polymkt_sent.nitter import Nitter


logger = logging.getLogger(__name__)


@dataclass
class ScrapingTarget:
    """Configuration for a scraping target (user or search term)."""
    
    target_type: str  # "user" or "search"
    value: str  # username or search query
    weight: float = 1.0
    description: str = ""
    last_scraped: Optional[datetime] = None
    error_count: int = 0
    
    @property
    def is_user(self) -> bool:
        """Check if this is a user target."""
        return self.target_type == "user"
    
    @property
    def is_search(self) -> bool:
        """Check if this is a search target."""
        return self.target_type == "search"


@dataclass
class ScrapingConfig:
    """Configuration for the async batch scraper."""
    
    # Mirrors and rate limiting
    mirrors: List[str] = field(default_factory=lambda: [
        "https://nitter.net",
        "https://nitter.pufe.org", 
        "https://nitter.hu"
    ])
    per_mirror_per_minute: int = 15
    max_concurrent_requests: int = 5
    retry_delays: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    
    # Scraping parameters
    batch_size: int = 50
    min_interval_sec: int = 60
    max_retry: int = 5
    timeout_sec: int = 30
    
    # Data freshness
    max_age_hours: int = 6  # Skip if scraped within this window
    tweets_per_target: int = 100  # Max tweets per target per run


class AsyncBatchScraper:
    """Async batch scraper for multiple targets."""
    
    def __init__(
        self,
        storage: TweetStorage,
        config: ScrapingConfig,
        targets: List[ScrapingTarget]
    ):
        """Initialize the async batch scraper."""
        self.storage = storage
        self.config = config
        self.targets = targets
        
        # Rate limiting state
        self._mirror_usage: Dict[str, List[datetime]] = {}
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "tweets_scraped": 0,
            "targets_processed": 0,
            "last_run": None
        }
        
        # Nitter instances (one per mirror)
        self._nitter_instances = {}
        
    def _get_nitter_for_mirror(self, mirror: str) -> Nitter:
        """Get or create a Nitter instance for a specific mirror."""
        if mirror not in self._nitter_instances:
            self._nitter_instances[mirror] = Nitter(
                log_level=0,  # Minimal logging
                skip_instance_check=True  # We manage mirrors ourselves
            )
            # Override the instance list with just our mirror
            self._nitter_instances[mirror].instances = [mirror]
            
        return self._nitter_instances[mirror]
    
    async def _wait_for_rate_limit(self, mirror: str) -> None:
        """Wait if we're hitting rate limits for a mirror."""
        now = datetime.now()
        
        # Initialize usage tracking for this mirror
        if mirror not in self._mirror_usage:
            self._mirror_usage[mirror] = []
        
        # Clean old entries (older than 1 minute)
        cutoff = now - timedelta(minutes=1)
        self._mirror_usage[mirror] = [
            ts for ts in self._mirror_usage[mirror] if ts > cutoff
        ]
        
        # Check if we need to wait
        if len(self._mirror_usage[mirror]) >= self.config.per_mirror_per_minute:
            # Calculate wait time until oldest request expires
            oldest = self._mirror_usage[mirror][0]
            wait_time = 60 - (now - oldest).total_seconds()
            
            if wait_time > 0:
                logger.debug(f"Rate limit hit for {mirror}, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
        
        # Record this request
        self._mirror_usage[mirror].append(now)
    
    def _should_skip_target(self, target: ScrapingTarget) -> bool:
        """Check if we should skip this target (recently scraped)."""
        if target.last_scraped is None:
            return False
        
        age = datetime.now(timezone.utc) - target.last_scraped
        return age.total_seconds() < (self.config.max_age_hours * 3600)
    
    async def _scrape_user_tweets(
        self,
        target: ScrapingTarget,
        mirror: str
    ) -> List[TweetModel]:
        """Scrape tweets from a user account."""
        nitter = self._get_nitter_for_mirror(mirror)
        tweets = []
        
        try:
            # Wait for rate limit
            await self._wait_for_rate_limit(mirror)
            
            # Scrape user timeline
            raw_tweets = await asyncio.to_thread(
                nitter.get_tweets,
                target.value,
                mode="user",
                number=self.config.tweets_per_target
            )
            
            # Convert to TweetModel objects
            for raw_tweet in raw_tweets.get("tweets", []):
                try:
                    tweet = TweetModel(
                        tweet_id=raw_tweet.get("link", "").split("/")[-1],
                        user_id=target.value,  # Use username as user_id for now
                        username=target.value,
                        content=raw_tweet.get("text", ""),
                        timestamp=self._parse_tweet_date(raw_tweet.get("date", "")),
                        likes=raw_tweet.get("stats", {}).get("likes", 0),
                        retweets=raw_tweet.get("stats", {}).get("retweets", 0),
                        replies=raw_tweet.get("stats", {}).get("comments", 0),
                        source_instance=mirror
                    )
                    tweets.append(tweet)
                except Exception as e:
                    logger.warning(f"Failed to parse tweet: {e}")
                    continue
            
            logger.info(f"Scraped {len(tweets)} tweets from user {target.value}")
            
        except Exception as e:
            logger.error(f"Failed to scrape user {target.value} from {mirror}: {e}")
            raise
        
        return tweets
    
    async def _scrape_search_tweets(
        self,
        target: ScrapingTarget,
        mirror: str
    ) -> List[TweetModel]:
        """Scrape tweets from a search query."""
        nitter = self._get_nitter_for_mirror(mirror)
        tweets = []
        
        try:
            # Wait for rate limit
            await self._wait_for_rate_limit(mirror)
            
            # Scrape search results
            raw_tweets = await asyncio.to_thread(
                nitter.get_tweets,
                target.value,
                mode="term",
                number=self.config.tweets_per_target
            )
            
            # Convert to TweetModel objects
            for raw_tweet in raw_tweets.get("tweets", []):
                try:
                    tweet = TweetModel(
                        tweet_id=raw_tweet.get("link", "").split("/")[-1],
                        user_id=raw_tweet.get("user", {}).get("username", "unknown"),
                        username=raw_tweet.get("user", {}).get("username", "unknown"),
                        content=raw_tweet.get("text", ""),
                        timestamp=self._parse_tweet_date(raw_tweet.get("date", "")),
                        likes=raw_tweet.get("stats", {}).get("likes", 0),
                        retweets=raw_tweet.get("stats", {}).get("retweets", 0),
                        replies=raw_tweet.get("stats", {}).get("comments", 0),
                        source_instance=mirror
                    )
                    tweets.append(tweet)
                except Exception as e:
                    logger.warning(f"Failed to parse search tweet: {e}")
                    continue
            
            logger.info(f"Scraped {len(tweets)} tweets for search '{target.value}'")
            
        except Exception as e:
            logger.error(f"Failed to scrape search '{target.value}' from {mirror}: {e}")
            raise
        
        return tweets
    
    def _parse_tweet_date(self, date_str: str) -> datetime:
        """Parse tweet date string to datetime."""
        try:
            # Handle various date formats from Nitter
            # This is a simplified version - real implementation would be more robust
            from dateutil import parser
            return parser.parse(date_str)
        except Exception:
            # Fallback to current time
            return datetime.now(timezone.utc)
    
    async def _scrape_target(self, target: ScrapingTarget) -> int:
        """Scrape a single target with retry logic."""
        async with self._semaphore:
            tweets_count = 0
            
            # Always increment targets_processed counter
            self.stats["targets_processed"] += 1
            
            # Skip if recently scraped
            if self._should_skip_target(target):
                logger.debug(f"Skipping {target.value} (recently scraped)")
                return 0
            
            # Try each mirror with exponential backoff
            for attempt in range(self.config.max_retry):
                mirror = random.choice(self.config.mirrors)
                
                try:
                    # Scrape based on target type
                    if target.is_user:
                        tweets = await self._scrape_user_tweets(target, mirror)
                    else:
                        tweets = await self._scrape_search_tweets(target, mirror)
                    
                    # Store tweets in batch
                    if tweets:
                        stored_count = self.storage.insert_tweets_batch(tweets)
                        tweets_count = stored_count
                        
                        logger.info(
                            f"Stored {stored_count} tweets for {target.value}"
                        )
                    
                    # Update target state
                    target.last_scraped = datetime.now(timezone.utc)
                    target.error_count = 0
                    
                    # Success - break retry loop
                    self.stats["successful_requests"] += 1
                    break
                    
                except Exception as e:
                    target.error_count += 1
                    self.stats["failed_requests"] += 1
                    
                    # Wait with exponential backoff before retry
                    if attempt < self.config.max_retry - 1:
                        delay = self.config.retry_delays[
                            min(attempt, len(self.config.retry_delays) - 1)
                        ]
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {target.value}: {e}. "
                            f"Retrying in {delay}s..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"All {self.config.max_retry} attempts failed for {target.value}: {e}"
                        )
            
            self.stats["total_requests"] += 1
            return tweets_count
    
    async def scrape_all_targets(self) -> Dict[str, Any]:
        """Scrape all configured targets concurrently."""
        start_time = datetime.now(timezone.utc)
        logger.info(f"Starting batch scrape of {len(self.targets)} targets")
        
        # Reset stats
        self.stats.update({
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "tweets_scraped": 0,
            "targets_processed": 0,
            "last_run": start_time
        })
        
        # Create tasks for all targets
        tasks = [
            self._scrape_target(target)
            for target in self.targets
        ]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        total_tweets = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {i} failed: {result}")
            else:
                total_tweets += result
        
        # Update final stats
        self.stats["tweets_scraped"] = total_tweets
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()
        
        summary = {
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": duration,
            "targets_total": len(self.targets),
            "targets_processed": self.stats["targets_processed"],
            "tweets_scraped": total_tweets,
            "success_rate": (
                self.stats["successful_requests"] / max(self.stats["total_requests"], 1)
            ),
            "tweets_per_second": total_tweets / max(duration, 1)
        }
        
        logger.info(
            f"Batch scrape completed: {total_tweets} tweets from "
            f"{self.stats['targets_processed']}/{len(self.targets)} targets "
            f"in {duration:.1f}s"
        )
        
        return summary


def load_scraping_config(config_path: str = "config/scraper.yml") -> ScrapingConfig:
    """Load scraping configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return ScrapingConfig(
            mirrors=data.get("mirrors", []),
            per_mirror_per_minute=data.get("rate_limit", {}).get("per_mirror_per_minute", 15),
            max_concurrent_requests=data.get("rate_limit", {}).get("max_concurrent_requests", 5),
            retry_delays=data.get("rate_limit", {}).get("retry_delays", [1, 2, 4, 8, 16]),
            batch_size=data.get("scraping", {}).get("batch_size", 50),
            min_interval_sec=data.get("scraping", {}).get("min_interval_sec", 60),
            max_retry=data.get("scraping", {}).get("max_retry", 5),
            timeout_sec=data.get("scraping", {}).get("timeout_sec", 30)
        )
    except Exception as e:
        logger.warning(f"Failed to load config from {config_path}: {e}. Using defaults.")
        return ScrapingConfig()


def load_scraping_targets(accounts_path: str = "config/accounts.yml") -> List[ScrapingTarget]:
    """Load scraping targets from YAML file."""
    targets = []
    
    try:
        with open(accounts_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Load user targets
        for user in data.get("users", []):
            targets.append(ScrapingTarget(
                target_type="user",
                value=user["name"],
                weight=user.get("weight", 1.0),
                description=user.get("description", "")
            ))
        
        # Load search targets
        for term in data.get("terms", []):
            targets.append(ScrapingTarget(
                target_type="search",
                value=term["query"],
                weight=term.get("weight", 1.0),
                description=term.get("description", "")
            ))
        
        logger.info(f"Loaded {len(targets)} scraping targets")
        
    except Exception as e:
        logger.error(f"Failed to load targets from {accounts_path}: {e}")
    
    return targets
