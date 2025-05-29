"""
Unit tests for polymkt_sent.api module.

Tests FastAPI endpoints, dependency injection, error handling,
and response models for the REST API.
"""

import pytest
import tempfile
import shutil
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from fastapi.testclient import TestClient
from fastapi import HTTPException

from polymkt_sent.api import (
    app,
    get_storage, 
    get_sentiment_processor,
    api_config,
    HealthResponse,
    SentimentResponse,
    SentimentSummaryResponse,
    TradingSignalResponse,
    APIConfig
)
from polymkt_sent.core.storage import TweetModel, TweetStorage, StorageConfig
from polymkt_sent.core.sentiment import SentimentProcessor


class TestAPIResponseModels:
    """Test cases for API response models."""
    
    def test_health_response_model(self):
        """Test HealthResponse model validation."""
        response = HealthResponse(
            status="healthy",
            timestamp=datetime.now(timezone.utc),
            version="1.0.0",
            components={"storage": "healthy", "processor": "healthy"}
        )
        
        assert response.status == "healthy"
        assert response.version == "1.0.0"
        assert response.components["storage"] == "healthy"
    
    def test_sentiment_response_model(self):
        """Test SentimentResponse model validation."""
        response = SentimentResponse(
            tweet_id="123",
            content="Test tweet content",
            sentiment_score=0.8,
            sentiment_label="positive",
            confidence=0.9,
            timestamp=datetime.now(timezone.utc),
            username="testuser"
        )
        
        assert response.tweet_id == "123"
        assert response.sentiment_score == 0.8
        assert response.sentiment_label == "positive"
        assert response.confidence == 0.9
    
    def test_sentiment_response_validation(self):
        """Test SentimentResponse validation constraints."""
        # Test invalid sentiment score
        with pytest.raises(ValueError):
            SentimentResponse(
                tweet_id="123",
                content="Test",
                sentiment_score=2.0,  # Invalid: > 1.0
                sentiment_label="positive",
                confidence=0.9,
                timestamp=datetime.now(timezone.utc),
                username="testuser"
            )
        
        # Test invalid confidence
        with pytest.raises(ValueError):
            SentimentResponse(
                tweet_id="123",
                content="Test",
                sentiment_score=0.8,
                sentiment_label="positive",
                confidence=1.5,  # Invalid: > 1.0
                timestamp=datetime.now(timezone.utc),
                username="testuser"
            )
    
    def test_sentiment_summary_response_model(self):
        """Test SentimentSummaryResponse model validation."""
        response = SentimentSummaryResponse(
            total_tweets=100,
            positive_tweets=60,
            negative_tweets=30,
            neutral_tweets=10,
            average_sentiment=0.25,
            last_updated=datetime.now(timezone.utc)
        )
        
        assert response.total_tweets == 100
        assert response.positive_tweets == 60
        assert response.average_sentiment == 0.25
    
    def test_trading_signal_response_model(self):
        """Test TradingSignalResponse model validation."""
        response = TradingSignalResponse(
            signal="BUY",
            confidence=0.85,
            reasoning="Strong positive sentiment trend",
            sentiment_score=0.75,
            tweet_count=150,
            timestamp=datetime.now(timezone.utc)
        )
        
        assert response.signal == "BUY"
        assert response.confidence == 0.85
        assert response.sentiment_score == 0.75


class TestAPIConfig:
    """Test cases for APIConfig."""
    
    def test_api_config_defaults(self):
        """Test API configuration defaults."""
        config = APIConfig()
        
        assert config.data_dir == "data"
        assert config.storage_config is not None
        assert config.sentiment_config is not None


class TestAPIEndpoints:
    """Test cases for API endpoints."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.client = TestClient(app)
        self.temp_dir = tempfile.mkdtemp()
        
        # Override dependencies for testing
        def override_get_storage():
            return Mock(spec=TweetStorage)
        
        def override_get_sentiment_processor():
            return Mock(spec=SentimentProcessor)
        
        app.dependency_overrides[get_storage] = override_get_storage
        app.dependency_overrides[get_sentiment_processor] = override_get_sentiment_processor
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        app.dependency_overrides.clear()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('polymkt_sent.api.Path')
    def test_health_endpoint_healthy(self, mock_path):
        """Test health endpoint with healthy components."""
        # Mock path exists
        mock_path.return_value.exists.return_value = True
        
        # Mock storage dependency
        def override_get_storage():
            mock_storage = Mock()
            mock_storage.get_tweet_count.return_value = 100
            return mock_storage
        
        # Mock sentiment processor dependency
        def override_get_sentiment_processor():
            mock_processor = Mock()
            mock_processor.analyzer = Mock()  # Has analyzer
            return mock_processor
        
        app.dependency_overrides[get_storage] = override_get_storage
        app.dependency_overrides[get_sentiment_processor] = override_get_sentiment_processor
        
        response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert "components" in data
        assert data["components"]["storage"] == "healthy"
        assert data["components"]["data_directory"] == "healthy"
        assert data["components"]["sentiment_processor"] == "healthy"
    
    @patch('polymkt_sent.api.Path')
    def test_health_endpoint_degraded(self, mock_path):
        """Test health endpoint with degraded components."""
        # Mock path does not exist
        mock_path.return_value.exists.return_value = False
        
        # Mock storage with error
        def override_get_storage():
            mock_storage = Mock()
            mock_storage.get_tweet_count.side_effect = Exception("DB connection failed")
            return mock_storage
        
        app.dependency_overrides[get_storage] = override_get_storage
        
        response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert "unhealthy" in data["components"]["storage"]
        assert data["components"]["data_directory"] == "missing"
    
    def test_sentiment_latest_endpoint(self):
        """Test latest sentiment endpoint."""
        # Mock storage dependency
        def override_get_storage():
            mock_storage = Mock()
            mock_storage.get_tweets.return_value = [
                {
                    'tweet_id': '123',
                    'content': 'Great news!',
                    'sentiment_score': 0.8,
                    'sentiment_label': 'positive',
                    'sentiment_confidence': 0.9,
                    'timestamp': datetime.now(timezone.utc),
                    'username': 'testuser'
                },
                {
                    'tweet_id': '456',
                    'content': 'Bad news',
                    'sentiment_score': -0.6,
                    'sentiment_label': 'negative',
                    'sentiment_confidence': 0.7,
                    'timestamp': datetime.now(timezone.utc),
                    'username': 'testuser2'
                }
            ]
            return mock_storage
        
        app.dependency_overrides[get_storage] = override_get_storage
        
        response = self.client.get("/sentiment/latest?limit=2")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]['tweet_id'] == '123'
        assert data[0]['sentiment_score'] == 0.8
        assert data[1]['tweet_id'] == '456'
        assert data[1]['sentiment_score'] == -0.6
    
    def test_sentiment_latest_invalid_limit(self):
        """Test latest sentiment endpoint with invalid limit."""
        response = self.client.get("/sentiment/latest?limit=150")  # > 100
        
        assert response.status_code == 422  # Validation error
    
    def test_sentiment_latest_storage_error(self):
        """Test latest sentiment endpoint with storage error."""
        def override_get_storage():
            mock_storage = Mock()
            mock_storage.get_tweets.side_effect = Exception("Storage error")
            return mock_storage
        
        app.dependency_overrides[get_storage] = override_get_storage
        
        response = self.client.get("/sentiment/latest")
        
        assert response.status_code == 500
        data = response.json()
        assert "Failed to retrieve sentiment data" in data["detail"]
    
    def test_sentiment_summary_endpoint(self):
        """Test sentiment summary endpoint."""
        def override_get_sentiment_processor():
            mock_processor = Mock()
            mock_processor.get_sentiment_summary.return_value = {
                'total_tweets': 100,
                'positive_tweets': 60,
                'negative_tweets': 30,
                'neutral_tweets': 10,
                'avg_sentiment': 0.25  # Note: API expects 'avg_sentiment', not 'average_sentiment'
            }
            return mock_processor
        
        app.dependency_overrides[get_sentiment_processor] = override_get_sentiment_processor
        
        response = self.client.get("/sentiment/summary")
        
        assert response.status_code == 200
        data = response.json()
        assert data['total_tweets'] == 100
        assert data['positive_tweets'] == 60
        assert data['average_sentiment'] == 0.25
        assert 'last_updated' in data
    
    def test_trading_signals_endpoint(self):
        """Test trading signals endpoint."""
        def override_get_sentiment_processor():
            mock_processor = Mock()
            mock_processor.get_sentiment_summary.return_value = {
                'total_tweets': 100,
                'positive_tweets': 70,
                'negative_tweets': 20,
                'neutral_tweets': 10,
                'avg_sentiment': 0.6
            }
            return mock_processor
        
        app.dependency_overrides[get_sentiment_processor] = override_get_sentiment_processor
        
        response = self.client.get("/signals/trading")
        
        assert response.status_code == 200
        data = response.json()
        assert data['signal'] in ['BUY', 'SELL', 'HOLD']
        assert 0.0 <= data['confidence'] <= 1.0
        assert 'reasoning' in data
        assert 'sentiment_score' in data
        assert 'tweet_count' in data
        assert 'timestamp' in data
    
    def test_trading_signals_no_data(self):
        """Test trading signals endpoint with no data."""
        def override_get_sentiment_processor():
            mock_processor = Mock()
            mock_processor.get_sentiment_summary.return_value = {
                'total_tweets': 0,
                'positive_tweets': 0,
                'negative_tweets': 0,
                'neutral_tweets': 0,
                'avg_sentiment': 0.0
            }
            return mock_processor
        
        app.dependency_overrides[get_sentiment_processor] = override_get_sentiment_processor
        
        response = self.client.get("/signals/trading")
        
        assert response.status_code == 200
        data = response.json()
        assert data['signal'] == 'HOLD'
        assert 'insufficient data' in data['reasoning'].lower()
    
    def test_stats_endpoint(self):
        """Test system stats endpoint."""
        def override_get_storage():
            mock_storage = Mock()
            mock_storage.get_tweet_count.return_value = 250
            return mock_storage
        
        app.dependency_overrides[get_storage] = override_get_storage
        
        response = self.client.get("/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert data['total_tweets'] == 250
        assert data['api_version'] == '1.0.0'
        assert 'timestamp' in data
        assert 'uptime' in data
        assert 'last_scrape' in data
    
    def test_stats_endpoint_error(self):
        """Test stats endpoint with storage error."""
        def override_get_storage():
            mock_storage = Mock()
            mock_storage.get_tweet_count.side_effect = Exception("DB error")
            return mock_storage
        
        app.dependency_overrides[get_storage] = override_get_storage
        
        response = self.client.get("/stats")
        
        assert response.status_code == 500
        data = response.json()
        assert "Failed to get system statistics" in data["detail"]


class TestErrorHandlers:
    """Test cases for API error handlers."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.client = TestClient(app)
    
    def test_404_handler(self):
        """Test 404 error handler."""
        response = self.client.get("/nonexistent-endpoint")
        
        assert response.status_code == 404
        data = response.json()
        assert data["error"] == "Endpoint not found"
        assert "/nonexistent-endpoint" in data["path"]
    
    def test_500_handler_simulation(self):
        """Test 500 error handler through endpoint failure."""
        # Test an endpoint that will actually fail with 500
        # by patching the method that's called within the endpoint logic
        with patch('polymkt_sent.core.sentiment.SentimentProcessor.get_sentiment_summary', side_effect=Exception("Critical error")):
            response = self.client.get("/sentiment/summary")
            
            assert response.status_code == 500
            data = response.json()
            assert "Critical error" in data["detail"] or "Failed to get sentiment summary" in data["detail"]


class TestDependencyInjection:
    """Test cases for dependency injection."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_get_storage_dependency(self):
        """Test storage dependency injection returns correct instance."""
        # Test that the dependency returns a TweetStorage instance
        storage = get_storage()
        
        assert isinstance(storage, TweetStorage)
        assert storage.config.data_dir == api_config.storage_config.data_dir
    
    def test_get_sentiment_processor_dependency(self):
        """Test sentiment processor dependency injection returns correct instance."""
        # Test that the dependency returns a SentimentProcessor instance  
        processor = get_sentiment_processor()
        
        assert isinstance(processor, SentimentProcessor)
        assert processor.config is not None


class TestAPIIntegration:
    """Integration tests for API with real components."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.client = TestClient(app)
        
        # Create real storage for integration tests
        storage_config = StorageConfig(data_dir=self.temp_dir)
        self.storage = TweetStorage(storage_config)
        
        # Override dependency with real storage
        def override_get_storage():
            return self.storage
        
        app.dependency_overrides[get_storage] = override_get_storage
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        app.dependency_overrides.clear()
        if hasattr(self, 'storage'):
            self.storage.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_sentiment_latest_with_real_storage(self):
        """Test latest sentiment endpoint with real storage."""
        # Insert test data
        tweet = TweetModel(
            tweet_id="test123",
            user_id="testuser",
            username="testuser",
            content="Integration test tweet",
            timestamp=datetime.now(timezone.utc),
            sentiment_score=0.5,
            sentiment_label="neutral",
            sentiment_confidence=0.8
        )
        self.storage.insert_tweet(tweet)
        
        response = self.client.get("/sentiment/latest?limit=1")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]['tweet_id'] == 'test123'
        assert data[0]['sentiment_score'] == 0.5
    
    def test_stats_with_real_storage(self):
        """Test stats endpoint with real storage."""
        # Insert test data
        tweets = [
            TweetModel(
                tweet_id=f"test{i}",
                user_id="testuser",
                username="testuser",
                content=f"Test tweet {i}",
                timestamp=datetime.now(timezone.utc)
            )
            for i in range(5)
        ]
        self.storage.insert_tweets_batch(tweets)
        
        response = self.client.get("/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert data['total_tweets'] == 5


if __name__ == "__main__":
    pytest.main([__file__])
