"""
Async batch scraper for polymarket-sentiment.

This module provides:
- Async batch scraping of multiple accounts/search terms
- Rate limiting and error handling
- Integration with storage layer
- Configurable concurrency and retry logic
- Support for multiple backends (snscrape primary, Nitter fallback)
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
from polymkt_sent.core.backends import BackendManager, ScrapingResult
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
    
    # Backend preferences
    preferred_backend: str = "snscrape"  # "snscrape" or "nitter" or "auto"
    enable_backend_fallback: bool = True  # Use fallback if primary fails


class AsyncBatchScraper:
    """Async batch scraper for multiple targets with multiple backend support."""
    
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
        
        # Initialize backend manager
        self.backend_manager = BackendManager(config.mirrors)
        
        # Rate limiting state (for Nitter fallback)
        self._mirror_usage: Dict[str, List[datetime]] = {}
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "tweets_scraped": 0,
            "targets_processed": 0,
            "last_run": None,
            "backend_usage": {}  # Track which backends were used
        }
        
        logger.info(f"AsyncBatchScraper initialized with backends: {self.backend_manager.get_available_backends()}")
        
        # Nitter instances (one per mirror) - kept for backward compatibility
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
    
    async def _scrape_target_with_backend(
        self,
        target: ScrapingTarget
    ) -> List[TweetModel]:
        """Scrape a target using the backend manager."""
        self.stats["total_requests"] += 1
        
        try:
            # Use backend manager to scrape
            if target.is_user:
                result = await self.backend_manager.scrape_user_tweets(
                    target.value,
                    self.config.tweets_per_target
                )
            elif target.is_search:
                result = await self.backend_manager.scrape_search_tweets(
                    target.value,
                    self.config.tweets_per_target
                )
            else:
                raise ValueError(f"Unknown target type: {target.target_type}")
            
            if result.success:
                self.stats["successful_requests"] += 1
                self.stats["tweets_scraped"] += len(result.tweets)
                
                # Track backend usage
                backend_used = result.backend_used or "unknown"
                self.stats["backend_usage"][backend_used] = \
                    self.stats["backend_usage"].get(backend_used, 0) + 1
                
                logger.info(
                    f"Successfully scraped {len(result.tweets)} tweets from "
                    f"{target.target_type} '{target.value}' using {backend_used}"
                )
                
                return result.tweets
            else:
                self.stats["failed_requests"] += 1
                logger.error(
                    f"Failed to scrape {target.target_type} '{target.value}': "
                    f"{result.error_message}"
                )
                return []
                
        except Exception as e:
            self.stats["failed_requests"] += 1
            logger.error(f"Exception while scraping {target.target_type} '{target.value}': {e}")
            return []
    
    async def _scrape_target(self, target: ScrapingTarget) -> int:
        """Scrape a single target using the backend manager."""
        async with self._semaphore:
            tweets_count = 0
            
            # Always increment targets_processed counter
            self.stats["targets_processed"] += 1
            
            # Skip if recently scraped
            if self._should_skip_target(target):
                logger.debug(f"Skipping {target.value} (recently scraped)")
                return 0
            
            # Use the new backend system
            tweets = await self._scrape_target_with_backend(target)
            
            # Store tweets in batch if any were found
            if tweets:
                stored_count = self.storage.insert_tweets_batch(tweets)
                tweets_count = stored_count
                
                logger.info(f"Stored {stored_count} tweets for {target.value}")
                
                # Update target state on success
                target.last_scraped = datetime.now(timezone.utc)
                target.error_count = 0
            else:
                # Increment error count if no tweets found
                target.error_count += 1
            
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
            "last_run": start_time,
            "backend_usage": {}
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
            "tweets_per_second": total_tweets / max(duration, 1),
            "backend_usage": self.stats["backend_usage"],
            "available_backends": self.backend_manager.get_available_backends()
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
            min_interval_sec=data.get("scraping", {}).get("min_interval_sec", 30),
            max_retry=data.get("scraping", {}).get("max_retry", 5),
            timeout_sec=data.get("scraping", {}).get("timeout_sec", 30),
            tweets_per_target=data.get("scraping", {}).get("tweets_per_target", 100),
            preferred_backend=data.get("backend", {}).get("preferred", "snscrape"),
            enable_backend_fallback=data.get("backend", {}).get("enable_fallback", True)
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
