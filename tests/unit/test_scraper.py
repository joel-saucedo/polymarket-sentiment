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
    async def test_scrape_user_tweets_success(self, scraper):
        """Test successful user tweet scraping."""
        target = ScrapingTarget("user", "testuser", 1.0)
        mirror = "http://test.com"
        
        # Mock the nitter response
        mock_tweets = {
            "tweets": [
                {
                    "link": "https://twitter.com/user/status/123456789",
                    "text": "Test tweet content",
                    "date": "2024-01-01 12:00:00",
                    "stats": {"likes": 5, "retweets": 2, "comments": 1}
                },
                {
                    "link": "https://twitter.com/user/status/987654321",
                    "text": "Another test tweet",
                    "date": "2024-01-01 13:00:00",
                    "stats": {"likes": 10, "retweets": 3, "comments": 0}
                }
            ]
        }
        
        with patch.object(scraper, '_get_nitter_for_mirror') as mock_get_nitter:
            mock_nitter = Mock()
            mock_nitter.get_tweets.return_value = mock_tweets
            mock_get_nitter.return_value = mock_nitter
            
            tweets = await scraper._scrape_user_tweets(target, mirror)
            
            assert len(tweets) == 2
            assert tweets[0].tweet_id == "123456789"
            assert tweets[0].content == "Test tweet content"
            assert tweets[0].username == "testuser"
            assert tweets[0].likes == 5
            assert tweets[1].tweet_id == "987654321"
            
            # Verify nitter was called correctly
            mock_nitter.get_tweets.assert_called_once_with(
                "testuser",
                mode="user",
                number=scraper.config.tweets_per_target
            )
    
    @pytest.mark.asyncio
    async def test_scrape_search_tweets_success(self, scraper):
        """Test successful search tweet scraping."""
        target = ScrapingTarget("search", "test query", 1.0)
        mirror = "http://test.com"
        
        # Mock the nitter response
        mock_tweets = {
            "tweets": [
                {
                    "link": "https://twitter.com/user1/status/111111111",
                    "text": "Search result tweet",
                    "date": "2024-01-01 14:00:00",
                    "user": {"username": "user1"},
                    "stats": {"likes": 15, "retweets": 5, "comments": 2}
                }
            ]
        }
        
        with patch.object(scraper, '_get_nitter_for_mirror') as mock_get_nitter:
            mock_nitter = Mock()
            mock_nitter.get_tweets.return_value = mock_tweets
            mock_get_nitter.return_value = mock_nitter
            
            tweets = await scraper._scrape_search_tweets(target, mirror)
            
            assert len(tweets) == 1
            assert tweets[0].tweet_id == "111111111"
            assert tweets[0].content == "Search result tweet"
            assert tweets[0].username == "user1"
            assert tweets[0].likes == 15
            
            # Verify nitter was called correctly
            mock_nitter.get_tweets.assert_called_once_with(
                "test query",
                mode="term",
                number=scraper.config.tweets_per_target
            )
    
    @pytest.mark.asyncio
    async def test_scrape_target_success(self, scraper):
        """Test successful target scraping."""
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
        
        with patch.object(scraper, '_scrape_user_tweets', return_value=mock_tweets):
            with patch.object(scraper.storage, 'insert_tweets_batch', return_value=1):
                count = await scraper._scrape_target(target)
                
                assert count == 1
                assert target.last_scraped is not None
                assert target.error_count == 0
                assert scraper.stats["successful_requests"] == 1
    
    @pytest.mark.asyncio
    async def test_scrape_target_failure_with_retry(self, scraper):
        """Test target scraping with failures and retries."""
        target = ScrapingTarget("user", "testuser", 1.0)
        
        # Mock to fail twice then succeed
        call_count = 0
        def mock_scrape_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Network error")
            return [TweetModel(
                tweet_id="123",
                user_id="testuser", 
                username="testuser",
                content="Test tweet",
                timestamp=datetime.now(timezone.utc)
            )]
        
        with patch.object(scraper, '_scrape_user_tweets', side_effect=mock_scrape_side_effect):
            with patch.object(scraper.storage, 'insert_tweets_batch', return_value=1):
                count = await scraper._scrape_target(target)
                
                assert count == 1
                assert call_count == 3  # Failed twice, succeeded on third
                assert target.error_count == 0  # Reset on success
                assert scraper.stats["successful_requests"] == 1
                assert scraper.stats["failed_requests"] == 2
    
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
            scraper.stats["total_requests"] += 1
            scraper.stats["successful_requests"] += 1
            return 5
            
        with patch.object(scraper, '_scrape_target', side_effect=mock_scrape_target):
            summary = await scraper.scrape_all_targets()
            
            assert summary["targets_total"] == len(scraper.targets)
            assert summary["targets_processed"] == len(scraper.targets)
            assert summary["tweets_scraped"] == 5 * len(scraper.targets)
            assert summary["success_rate"] == 1.0
            assert "start_time" in summary
            assert "end_time" in summary
    
    def test_parse_tweet_date(self, scraper):
        """Test tweet date parsing."""
        # Test with valid date string
        date_str = "2024-01-01 12:00:00"
        parsed_date = scraper._parse_tweet_date(date_str)
        assert isinstance(parsed_date, datetime)
        
        # Test with invalid date string (should fallback to current time)
        invalid_date = "invalid date"
        parsed_date = scraper._parse_tweet_date(invalid_date)
        assert isinstance(parsed_date, datetime)
        assert parsed_date.tzinfo is not None


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
