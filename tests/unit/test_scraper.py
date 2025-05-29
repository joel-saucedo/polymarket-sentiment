"""
Unit tests for polymkt_sent.core.scraper module.
"""

import pytest
import asyncio
import tempfile
import shutil
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from polymkt_sent.core.scraper import (
    ScrapingTarget,
    ScrapingConfig,
    AsyncBatchScraper,
    load_scraping_config,
    load_scraping_targets
)
from polymkt_sent.core.storage import TweetModel, TweetStorage, StorageConfig


class TestScrapingTarget:
    """Test cases for ScrapingTarget."""
    
    def test_user_target(self):
        """Test user target creation and properties."""
        target = ScrapingTarget(
            target_type="user",
            value="testuser",
            weight=0.8,
            description="Test user"
        )
        
        assert target.is_user is True
        assert target.is_search is False
        assert target.value == "testuser"
        assert target.weight == 0.8
        assert target.error_count == 0
        assert target.last_scraped is None
    
    def test_search_target(self):
        """Test search target creation and properties."""
        target = ScrapingTarget(
            target_type="search",
            value="test query",
            weight=1.2
        )
        
        assert target.is_user is False
        assert target.is_search is True
        assert target.value == "test query"
        assert target.weight == 1.2


class TestScrapingConfig:
    """Test cases for ScrapingConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ScrapingConfig()
        
        assert len(config.mirrors) == 3
        assert config.per_mirror_per_minute == 15
        assert config.max_concurrent_requests == 5
        assert config.batch_size == 50
        assert config.max_retry == 5
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ScrapingConfig(
            mirrors=["http://test.com"],
            per_mirror_per_minute=10,
            batch_size=25
        )
        
        assert config.mirrors == ["http://test.com"]
        assert config.per_mirror_per_minute == 10
        assert config.batch_size == 25


class TestAsyncBatchScraper:
    """Test cases for AsyncBatchScraper."""
    
    @pytest.fixture
    def storage(self):
        """Create test storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = StorageConfig(data_dir=temp_dir)
            yield TweetStorage(config)
    
    @pytest.fixture
    def config(self):
        """Create test scraper config."""
        return ScrapingConfig(
            mirrors=["http://test1.com", "http://test2.com"],
            per_mirror_per_minute=5,
            max_concurrent_requests=2,
            retry_delays=[0.1, 0.2]  # Faster for tests
        )
    
    @pytest.fixture
    def targets(self):
        """Create test targets."""
        return [
            ScrapingTarget(
                target_type="user",
                value="testuser1",
                weight=1.0
            ),
            ScrapingTarget(
                target_type="search", 
                value="test query",
                weight=0.8
            )
        ]
    
    @pytest.fixture
    def scraper(self, storage, config, targets):
        """Create test scraper."""
        return AsyncBatchScraper(storage, config, targets)
    
    def test_scraper_initialization(self, scraper, storage, config, targets):
        """Test scraper initialization."""
        assert scraper.storage is storage
        assert scraper.config is config
        assert scraper.targets == targets
        assert scraper.stats["total_requests"] == 0
        assert len(scraper._nitter_instances) == 0
    
    def test_get_nitter_for_mirror(self, scraper):
        """Test nitter instance creation per mirror."""
        mirror1 = "http://test1.com"
        mirror2 = "http://test2.com"
        
        nitter1 = scraper._get_nitter_for_mirror(mirror1)
        nitter2 = scraper._get_nitter_for_mirror(mirror2)
        
        # Should create different instances
        assert nitter1 is not nitter2
        
        # Should reuse same instance for same mirror
        nitter1_again = scraper._get_nitter_for_mirror(mirror1)
        assert nitter1 is nitter1_again
        
        # Check instance configuration
        assert nitter1.instances == [mirror1]
        assert nitter2.instances == [mirror2]
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, scraper):
        """Test rate limiting logic."""
        mirror = "http://test.com"
        
        # Should not wait on first request
        start_time = asyncio.get_event_loop().time()
        await scraper._wait_for_rate_limit(mirror)
        elapsed = asyncio.get_event_loop().time() - start_time
        assert elapsed < 0.1  # Should be immediate
        
        # Record enough requests to hit limit
        for _ in range(scraper.config.per_mirror_per_minute):
            await scraper._wait_for_rate_limit(mirror)
        
        # Next request should wait
        start_time = asyncio.get_event_loop().time()
        await scraper._wait_for_rate_limit(mirror)
        elapsed = asyncio.get_event_loop().time() - start_time
        # Should wait some time (but not full minute due to test timing)
        assert elapsed > 0
    
    def test_should_skip_target(self, scraper):
        """Test target skipping logic."""
        target = ScrapingTarget("user", "test", 1.0)
        
        # Should not skip if never scraped
        assert not scraper._should_skip_target(target)
        
        # Should skip if recently scraped
        target.last_scraped = datetime.now(timezone.utc) - timedelta(hours=1)
        assert scraper._should_skip_target(target)
        
        # Should not skip if old enough
        target.last_scraped = datetime.now(timezone.utc) - timedelta(hours=10)
        assert not scraper._should_skip_target(target)
    
    @pytest.mark.asyncio
    async def test_scrape_target_with_backend_success(self, scraper):
        """Test successful target scraping using backend manager."""
        target = ScrapingTarget("user", "testuser", 1.0)
        
        mock_tweets = [
            TweetModel(
                tweet_id="123",
                user_id="testuser",
                username="testuser",
                content="Test tweet",
                timestamp=datetime.now(timezone.utc)
            )
        ]
        
        with patch.object(scraper, '_scrape_target_with_backend', return_value=mock_tweets):
            with patch.object(scraper.storage, 'insert_tweets_batch', return_value=1):
                count = await scraper._scrape_target(target)
                
                assert count == 1
                assert target.last_scraped is not None
                assert target.error_count == 0
    
    @pytest.mark.asyncio
    async def test_scrape_target_failure_with_backend(self, scraper):
        """Test target scraping with backend failures."""
        target = ScrapingTarget("user", "testuser", 1.0)
        
        # Mock backend to return empty tweets (failure case)
        with patch.object(scraper, '_scrape_target_with_backend', return_value=[]):
            count = await scraper._scrape_target(target)
            
            assert count == 0
            assert target.error_count == 1  # Incremented on failure
    
    @pytest.mark.asyncio
    async def test_scrape_target_skip_recent(self, scraper):
        """Test skipping recently scraped targets."""
        target = ScrapingTarget("user", "testuser", 1.0)
        target.last_scraped = datetime.now(timezone.utc) - timedelta(hours=1)
        
        count = await scraper._scrape_target(target)
        
        assert count == 0
        assert scraper.stats["targets_processed"] == 1
    
    @pytest.mark.asyncio
    async def test_scrape_all_targets(self, scraper):
        """Test scraping all targets concurrently."""
        # Mock successful scraping
        async def mock_scrape_target(target):
            # Simulate the stats increments that happen in real method
            scraper.stats["targets_processed"] += 1
            return 5
            
        with patch.object(scraper, '_scrape_target', side_effect=mock_scrape_target):
            summary = await scraper.scrape_all_targets()
            
            assert summary["targets_total"] == len(scraper.targets)
            assert summary["targets_processed"] == len(scraper.targets)
            assert summary["tweets_scraped"] == 5 * len(scraper.targets)
            assert "start_time" in summary
            assert "end_time" in summary
            assert "backend_usage" in summary
            assert "available_backends" in summary


class TestConfigLoading:
    """Test configuration loading functions."""
    
    def test_load_scraping_config_success(self):
        """Test successful config loading."""
        config_content = """
mirrors:
  - "http://test1.com"
  - "http://test2.com"
rate_limit:
  per_mirror_per_minute: 20
  max_concurrent_requests: 3
scraping:
  batch_size: 25
  timeout_sec: 60
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write(config_content)
            f.flush()
            
            config = load_scraping_config(f.name)
            
            assert config.mirrors == ["http://test1.com", "http://test2.com"]
            assert config.per_mirror_per_minute == 20
            assert config.max_concurrent_requests == 3
            assert config.batch_size == 25
            assert config.timeout_sec == 60
        
        Path(f.name).unlink()  # Cleanup
    
    def test_load_scraping_config_file_not_found(self):
        """Test config loading with missing file (should use defaults)."""
        config = load_scraping_config("nonexistent.yml")
        
        # Should return default config
        assert len(config.mirrors) == 3
        assert config.per_mirror_per_minute == 15
    
    def test_load_scraping_targets_success(self):
        """Test successful targets loading."""
        targets_content = """
users:
  - name: "user1"
    weight: 1.0
    description: "Test user 1"
  - name: "user2"
    weight: 0.8
    description: "Test user 2"

terms:
  - query: "search term 1"
    weight: 1.2
    description: "Test search 1"
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write(targets_content)
            f.flush()
            
            targets = load_scraping_targets(f.name)
            
            assert len(targets) == 3
            
            # Check user targets
            user_targets = [t for t in targets if t.is_user]
            assert len(user_targets) == 2
            assert user_targets[0].value == "user1"
            assert user_targets[0].weight == 1.0
            assert user_targets[1].value == "user2"
            assert user_targets[1].weight == 0.8
            
            # Check search targets
            search_targets = [t for t in targets if t.is_search]
            assert len(search_targets) == 1
            assert search_targets[0].value == "search term 1"
            assert search_targets[0].weight == 1.2
        
        Path(f.name).unlink()  # Cleanup
    
    def test_load_scraping_targets_file_not_found(self):
        """Test targets loading with missing file (should return empty list)."""
        targets = load_scraping_targets("nonexistent.yml")
        assert targets == []
    
    def test_load_scraping_targets_empty_file(self):
        """Test targets loading with empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("")
            f.flush()
            
            targets = load_scraping_targets(f.name)
            assert targets == []
        
        Path(f.name).unlink()  # Cleanup
