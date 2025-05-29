"""
Unit tests for polymkt_sent.core.backends module.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

from polymkt_sent.core.backends import (
    ScrapingResult,
    ScraperBackend,
    SNSScrapeBackend,
    NitterBackend,
    BackendManager
)
from polymkt_sent.core.storage import TweetModel


class TestScrapingResult:
    """Test cases for ScrapingResult."""
    
    def test_successful_result(self):
        """Test successful scraping result."""
        tweets = [
            TweetModel(
                tweet_id="123",
                user_id="testuser",
                username="testuser",
                content="Test tweet",
                timestamp=datetime.now(timezone.utc)
            )
        ]
        
        result = ScrapingResult(
            tweets=tweets,
            success=True,
            backend_used="snscrape"
        )
        
        assert result.success is True
        assert len(result.tweets) == 1
        assert result.backend_used == "snscrape"
        assert result.error_message is None
    
    def test_failed_result(self):
        """Test failed scraping result."""
        result = ScrapingResult(
            tweets=[],
            success=False,
            error_message="Network error",
            backend_used="nitter"
        )
        
        assert result.success is False
        assert len(result.tweets) == 0
        assert result.error_message == "Network error"
        assert result.backend_used == "nitter"


class TestSNSScrapeBackend:
    """Test cases for SNSScrapeBackend."""
    
    def test_backend_name(self):
        """Test backend name."""
        backend = SNSScrapeBackend()
        assert backend.get_name() == "snscrape"
    
    def test_availability_check(self):
        """Test availability check."""
        backend = SNSScrapeBackend()
        # Availability depends on snscrape import, which we can't guarantee in tests
        # Just test that it returns a boolean
        assert isinstance(backend.is_available(), bool)
    
    @pytest.mark.asyncio
    async def test_scrape_user_tweets_unavailable(self):
        """Test user scraping when snscrape is unavailable."""
        backend = SNSScrapeBackend()
        
        with patch.object(backend, 'is_available', return_value=False):
            result = await backend.scrape_user_tweets("testuser", 50)
            
            assert result.success is False
            assert "snscrape not available" in result.error_message
            assert result.backend_used == "snscrape"
            assert len(result.tweets) == 0
    
    @pytest.mark.asyncio
    async def test_scrape_search_tweets_unavailable(self):
        """Test search scraping when snscrape is unavailable."""
        backend = SNSScrapeBackend()
        
        with patch.object(backend, 'is_available', return_value=False):
            result = await backend.scrape_search_tweets("test query", 50)
            
            assert result.success is False
            assert "snscrape not available" in result.error_message
            assert result.backend_used == "snscrape"
            assert len(result.tweets) == 0


class TestNitterBackend:
    """Test cases for NitterBackend."""
    
    def test_backend_name(self):
        """Test backend name."""
        mirrors = ["http://test1.com", "http://test2.com"]
        backend = NitterBackend(mirrors)
        assert backend.get_name() == "nitter"
    
    def test_availability_with_mirrors(self):
        """Test availability when mirrors are provided."""
        mirrors = ["http://test1.com", "http://test2.com"]
        backend = NitterBackend(mirrors)
        assert backend.is_available() is True
    
    def test_availability_without_mirrors(self):
        """Test availability when no mirrors are provided."""
        backend = NitterBackend([])
        assert backend.is_available() is False
    
    @pytest.mark.asyncio
    async def test_scrape_user_tweets_no_mirrors(self):
        """Test user scraping with no mirrors."""
        backend = NitterBackend([])
        
        result = await backend.scrape_user_tweets("testuser", 50)
        
        assert result.success is False
        assert "No Nitter mirrors available" in result.error_message
        assert result.backend_used == "nitter"
        assert len(result.tweets) == 0
    
    @pytest.mark.asyncio
    async def test_scrape_search_tweets_no_mirrors(self):
        """Test search scraping with no mirrors."""
        backend = NitterBackend([])
        
        result = await backend.scrape_search_tweets("test query", 50)
        
        assert result.success is False
        assert "No Nitter mirrors available" in result.error_message
        assert result.backend_used == "nitter"
        assert len(result.tweets) == 0
    
    @pytest.mark.asyncio
    async def test_scrape_user_tweets_success(self):
        """Test successful user scraping with Nitter."""
        mirrors = ["http://test.com"]
        backend = NitterBackend(mirrors)
        
        # Mock Nitter response
        mock_tweets = {
            "tweets": [
                {
                    "link": "https://twitter.com/user/status/123456789",
                    "text": "Test tweet content",
                    "date": "2024-01-01T12:00:00Z",
                    "user": {"username": "testuser"},
                    "stats": {"likes": 5, "retweets": 2, "comments": 1}
                }
            ]
        }
        
        with patch.object(backend, '_get_nitter_for_mirror') as mock_get_nitter:
            mock_nitter = Mock()
            mock_nitter.get_tweets.return_value = mock_tweets
            mock_get_nitter.return_value = mock_nitter
            
            result = await backend.scrape_user_tweets("testuser", 50)
            
            assert result.success is True
            assert len(result.tweets) == 1
            assert result.tweets[0].tweet_id == "123456789"
            assert result.tweets[0].content == "Test tweet content"
            assert result.tweets[0].username == "testuser"
            assert result.backend_used == "nitter:http://test.com"


class TestBackendManager:
    """Test cases for BackendManager."""
    
    def test_initialization_with_available_backends(self):
        """Test manager initialization with available backends."""
        mirrors = ["http://test1.com", "http://test2.com"]
        
        with patch('polymkt_sent.core.backends.SNSScrapeBackend') as mock_sns:
            with patch('polymkt_sent.core.backends.NitterBackend') as mock_nitter:
                # Mock both backends as available
                mock_sns_instance = Mock()
                mock_sns_instance.is_available.return_value = True
                mock_sns_instance.get_name.return_value = "snscrape"
                mock_sns.return_value = mock_sns_instance
                
                mock_nitter_instance = Mock()
                mock_nitter_instance.is_available.return_value = True
                mock_nitter_instance.get_name.return_value = "nitter"
                mock_nitter.return_value = mock_nitter_instance
                
                manager = BackendManager(mirrors)
                
                assert len(manager.backends) == 2
                assert manager.get_available_backends() == ["snscrape", "nitter"]
    
    def test_initialization_no_available_backends(self):
        """Test manager initialization with no available backends."""
        mirrors = []
        
        with patch('polymkt_sent.core.backends.SNSScrapeBackend') as mock_sns:
            with patch('polymkt_sent.core.backends.NitterBackend') as mock_nitter:
                # Mock both backends as unavailable
                mock_sns_instance = Mock()
                mock_sns_instance.is_available.return_value = False
                mock_sns.return_value = mock_sns_instance
                
                mock_nitter_instance = Mock()
                mock_nitter_instance.is_available.return_value = False
                mock_nitter.return_value = mock_nitter_instance
                
                with pytest.raises(ValueError, match="No scraper backends available"):
                    BackendManager(mirrors)
    
    @pytest.mark.asyncio
    async def test_scrape_user_tweets_primary_success(self):
        """Test user scraping with primary backend success."""
        mirrors = ["http://test.com"]
        manager = BackendManager(mirrors)
        
        # Mock successful result from first backend
        mock_result = ScrapingResult(
            tweets=[Mock()],
            success=True,
            backend_used="snscrape"
        )
        
        with patch.object(manager.backends[0], 'scrape_user_tweets', return_value=mock_result):
            result = await manager.scrape_user_tweets("testuser", 50)
            
            assert result.success is True
            assert len(result.tweets) == 1
            assert result.backend_used == "snscrape"
    
    @pytest.mark.asyncio
    async def test_scrape_user_tweets_fallback_success(self):
        """Test user scraping with fallback backend success."""
        mirrors = ["http://test.com"]
        manager = BackendManager(mirrors)
        
        # Mock first backend failure, second backend success
        failed_result = ScrapingResult(
            tweets=[],
            success=False,
            error_message="Primary backend failed"
        )
        
        success_result = ScrapingResult(
            tweets=[Mock()],
            success=True,
            backend_used="nitter"
        )
        
        if len(manager.backends) >= 2:
            with patch.object(manager.backends[0], 'scrape_user_tweets', return_value=failed_result):
                with patch.object(manager.backends[1], 'scrape_user_tweets', return_value=success_result):
                    result = await manager.scrape_user_tweets("testuser", 50)
                    
                    assert result.success is True
                    assert len(result.tweets) == 1
                    assert result.backend_used == "nitter"
    
    @pytest.mark.asyncio
    async def test_scrape_user_tweets_all_fail(self):
        """Test user scraping when all backends fail."""
        mirrors = ["http://test.com"]
        manager = BackendManager(mirrors)
        
        # Mock all backends to fail
        failed_result = ScrapingResult(
            tweets=[],
            success=False,
            error_message="Backend failed"
        )
        
        with patch.object(manager.backends[0], 'scrape_user_tweets', return_value=failed_result):
            if len(manager.backends) >= 2:
                with patch.object(manager.backends[1], 'scrape_user_tweets', return_value=failed_result):
                    result = await manager.scrape_user_tweets("testuser", 50)
            else:
                result = await manager.scrape_user_tweets("testuser", 50)
            
            assert result.success is False
            assert "All backends failed" in result.error_message
            assert result.backend_used == "none"
