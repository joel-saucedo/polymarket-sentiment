"""
Unit tests for sentiment analysis module.

Tests cover:
- SentimentResult data class validation
- SentimentConfig parameter validation
- VADERAnalyzer functionality (if available)
- KeywordAnalyzer text processing and scoring
- EnsembleSentimentAnalyzer combination logic
- SentimentProcessor batch processing
- Text preprocessing utilities
- Error handling and edge cases
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
from typing import List, Dict, Any

from polymkt_sent.core.sentiment import (
    SentimentResult,
    SentimentLabel,
    SentimentConfig,
    SentimentAnalyzer,
    VADERAnalyzer,
    KeywordAnalyzer,
    EnsembleSentimentAnalyzer,
    SentimentProcessor,
    create_sentiment_processor,
    VADER_AVAILABLE
)
from polymkt_sent.core.storage import TweetModel


class TestSentimentResult:
    """Test SentimentResult data class."""
    
    def test_valid_sentiment_result(self):
        """Test creating valid sentiment result."""
        result = SentimentResult(
            score=0.5,
            label=SentimentLabel.POSITIVE,
            confidence=0.8,
            details={"test": "data"}
        )
        
        assert result.score == 0.5
        assert result.label == SentimentLabel.POSITIVE
        assert result.confidence == 0.8
        assert result.details == {"test": "data"}
    
    def test_sentiment_result_default_details(self):
        """Test sentiment result with default empty details."""
        result = SentimentResult(
            score=0.0,
            label=SentimentLabel.NEUTRAL,
            confidence=0.5
        )
        
        assert result.details == {}
    
    def test_sentiment_result_invalid_score(self):
        """Test validation of sentiment score range."""
        # Score too high
        with pytest.raises(ValueError, match="Sentiment score must be between -1.0 and 1.0"):
            SentimentResult(
                score=1.5,
                label=SentimentLabel.POSITIVE,
                confidence=0.8
            )
        
        # Score too low
        with pytest.raises(ValueError, match="Sentiment score must be between -1.0 and 1.0"):
            SentimentResult(
                score=-1.5,
                label=SentimentLabel.NEGATIVE,
                confidence=0.8
            )
    
    def test_sentiment_result_invalid_confidence(self):
        """Test validation of confidence range."""
        # Confidence too high
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            SentimentResult(
                score=0.5,
                label=SentimentLabel.POSITIVE,
                confidence=1.5
            )
        
        # Confidence too low
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            SentimentResult(
                score=0.5,
                label=SentimentLabel.POSITIVE,
                confidence=-0.1
            )
    
    def test_sentiment_result_boundary_values(self):
        """Test boundary values for score and confidence."""
        # Test boundary values
        result = SentimentResult(
            score=-1.0,
            label=SentimentLabel.NEGATIVE,
            confidence=0.0
        )
        assert result.score == -1.0
        assert result.confidence == 0.0
        
        result = SentimentResult(
            score=1.0,
            label=SentimentLabel.POSITIVE,
            confidence=1.0
        )
        assert result.score == 1.0
        assert result.confidence == 1.0


class TestSentimentConfig:
    """Test SentimentConfig data class."""
    
    def test_default_sentiment_config(self):
        """Test default configuration values."""
        config = SentimentConfig()
        
        assert config.vader_weight == 0.6
        assert config.keyword_weight == 0.4
        assert config.neutral_threshold == 0.1
        assert config.confidence_threshold == 0.3
        assert len(config.positive_keywords) > 0
        assert len(config.negative_keywords) > 0
        assert "bullish" in config.positive_keywords
        assert "bearish" in config.negative_keywords
    
    def test_custom_sentiment_config(self):
        """Test custom configuration values."""
        config = SentimentConfig(
            vader_weight=0.8,
            keyword_weight=0.2,
            neutral_threshold=0.05,
            positive_keywords=["good", "great"],
            negative_keywords=["bad", "awful"]
        )
        
        assert config.vader_weight == 0.8
        assert config.keyword_weight == 0.2
        assert config.neutral_threshold == 0.05
        assert config.positive_keywords == ["good", "great"]
        assert config.negative_keywords == ["bad", "awful"]


class MockSentimentAnalyzer(SentimentAnalyzer):
    """Mock sentiment analyzer for testing."""
    
    def __init__(self, name: str, score: float, confidence: float):
        self.name = name
        self.score = score
        self.confidence = confidence
    
    def analyze(self, text: str) -> SentimentResult:
        # Determine label based on score
        if self.score > 0.1:
            label = SentimentLabel.POSITIVE
        elif self.score < -0.1:
            label = SentimentLabel.NEGATIVE
        else:
            label = SentimentLabel.NEUTRAL
        
        return SentimentResult(
            score=self.score,
            label=label,
            confidence=self.confidence,
            details={"mock_analyzer": self.name}
        )
    
    def get_name(self) -> str:
        return self.name


class TestSentimentAnalyzer:
    """Test abstract SentimentAnalyzer base class."""
    
    def test_preprocess_text_default(self):
        """Test text preprocessing with default settings."""
        analyzer = MockSentimentAnalyzer("test", 0.5, 0.8)
        config = SentimentConfig()
        
        text = "Check out this AWESOME tweet! https://example.com @user #crypto"
        processed = analyzer.preprocess_text(text, config)
        
        # URLs should be removed by default
        assert "https://example.com" not in processed
        assert "awesome" in processed  # Should be lowercase
        assert "@user" in processed  # Mentions kept by default
        assert "#crypto" in processed  # Hashtags kept by default
    
    def test_preprocess_text_remove_mentions(self):
        """Test text preprocessing with mention removal."""
        analyzer = MockSentimentAnalyzer("test", 0.5, 0.8)
        config = SentimentConfig(remove_mentions=True)
        
        text = "Hey @user, this is great! @another_user"
        processed = analyzer.preprocess_text(text, config)
        
        assert "@user" not in processed
        assert "@another_user" not in processed
        assert "hey" in processed
        assert "great" in processed
    
    def test_preprocess_text_remove_hashtags(self):
        """Test text preprocessing with hashtag removal."""
        analyzer = MockSentimentAnalyzer("test", 0.5, 0.8)
        config = SentimentConfig(remove_hashtags=True)
        
        text = "This is #awesome and #great #crypto"
        processed = analyzer.preprocess_text(text, config)
        
        assert "#awesome" not in processed
        assert "#great" not in processed
        assert "#crypto" not in processed
        assert "this" in processed
        assert "and" in processed
    
    def test_preprocess_text_no_lowercase(self):
        """Test text preprocessing without lowercasing."""
        analyzer = MockSentimentAnalyzer("test", 0.5, 0.8)
        config = SentimentConfig(lowercase=False)
        
        text = "This is AWESOME!"
        processed = analyzer.preprocess_text(text, config)
        
        assert "AWESOME" in processed
        assert "awesome" not in processed


@pytest.mark.skipif(not VADER_AVAILABLE, reason="VADER not available")
class TestVADERAnalyzer:
    """Test VADER sentiment analyzer."""
    
    def test_vader_analyzer_init(self):
        """Test VADER analyzer initialization."""
        analyzer = VADERAnalyzer()
        assert analyzer.get_name() == "vader"
        assert analyzer.analyzer is not None
    
    def test_vader_positive_sentiment(self):
        """Test VADER analyzer with positive text."""
        analyzer = VADERAnalyzer()
        
        result = analyzer.analyze("This is absolutely amazing and wonderful!")
        
        assert result.score > 0
        assert result.label == SentimentLabel.POSITIVE
        assert result.confidence > 0
        assert "vader_compound" in result.details
        assert "vader_positive" in result.details
    
    def test_vader_negative_sentiment(self):
        """Test VADER analyzer with negative text."""
        analyzer = VADERAnalyzer()
        
        result = analyzer.analyze("This is terrible and awful!")
        
        assert result.score < 0
        assert result.label == SentimentLabel.NEGATIVE
        assert result.confidence > 0
        assert "vader_compound" in result.details
        assert "vader_negative" in result.details
    
    def test_vader_neutral_sentiment(self):
        """Test VADER analyzer with neutral text."""
        analyzer = VADERAnalyzer()
        
        result = analyzer.analyze("The weather is normal today.")
        
        assert abs(result.score) < 0.5  # Should be relatively neutral
        assert result.label == SentimentLabel.NEUTRAL
        assert "vader_neutral" in result.details


class TestKeywordAnalyzer:
    """Test keyword-based sentiment analyzer."""
    
    def test_keyword_analyzer_init(self):
        """Test keyword analyzer initialization."""
        config = SentimentConfig()
        analyzer = KeywordAnalyzer(config)
        
        assert analyzer.get_name() == "keyword"
        assert len(analyzer.positive_words) > 0
        assert len(analyzer.negative_words) > 0
    
    def test_keyword_positive_sentiment(self):
        """Test keyword analyzer with positive keywords."""
        config = SentimentConfig(
            positive_keywords=["good", "great", "awesome"],
            negative_keywords=["bad", "terrible"]
        )
        analyzer = KeywordAnalyzer(config)
        
        result = analyzer.analyze("This is really good and awesome!")
        
        assert result.score > 0
        assert result.label == SentimentLabel.POSITIVE
        assert result.confidence > 0
        assert result.details["positive_keywords"] > 0
        assert "matched_keywords" in result.details
    
    def test_keyword_negative_sentiment(self):
        """Test keyword analyzer with negative keywords."""
        config = SentimentConfig(
            positive_keywords=["good", "great"],
            negative_keywords=["bad", "terrible", "awful"]
        )
        analyzer = KeywordAnalyzer(config)
        
        result = analyzer.analyze("This is bad and terrible!")
        
        assert result.score < 0
        assert result.label == SentimentLabel.NEGATIVE
        assert result.confidence > 0
        assert result.details["negative_keywords"] > 0
    
    def test_keyword_neutral_sentiment(self):
        """Test keyword analyzer with no keywords."""
        config = SentimentConfig(
            positive_keywords=["good"],
            negative_keywords=["bad"],
            polymarket_positive=[],  # Override default keywords
            polymarket_negative=[]   # Override default keywords
        )
        analyzer = KeywordAnalyzer(config)
        
        result = analyzer.analyze("The weather is fine today.")
        
        assert result.score == 0.0
        assert result.label == SentimentLabel.NEUTRAL
        assert result.confidence == 0.0
        assert result.details["positive_keywords"] == 0
        assert result.details["negative_keywords"] == 0
    
    def test_keyword_mixed_sentiment(self):
        """Test keyword analyzer with mixed sentiment."""
        config = SentimentConfig(
            positive_keywords=["good"],
            negative_keywords=["bad"],
            neutral_threshold=0.05  # Lower threshold for testing
        )
        analyzer = KeywordAnalyzer(config)
        
        result = analyzer.analyze("This has good and bad aspects.")
        
        # Should be close to neutral due to balance
        assert abs(result.score) < 0.5
        assert result.details["positive_keywords"] > 0
        assert result.details["negative_keywords"] > 0


class TestEnsembleSentimentAnalyzer:
    """Test ensemble sentiment analyzer."""
    
    def test_ensemble_initialization(self):
        """Test ensemble analyzer initialization."""
        config = SentimentConfig(vader_weight=0.6, keyword_weight=0.4)
        
        with patch('polymkt_sent.core.sentiment.VADER_AVAILABLE', True):
            with patch('polymkt_sent.core.sentiment.VADERAnalyzer'):
                analyzer = EnsembleSentimentAnalyzer(config)
                assert len(analyzer.analyzers) > 0
    
    def test_ensemble_no_analyzers(self):
        """Test ensemble with no available analyzers."""
        config = SentimentConfig(vader_weight=0.0, keyword_weight=0.0)
        
        with pytest.raises(ValueError, match="No sentiment analyzers available"):
            EnsembleSentimentAnalyzer(config)
    
    def test_ensemble_analyze_empty_text(self):
        """Test ensemble analyzer with empty text."""
        config = SentimentConfig(vader_weight=0.0, keyword_weight=1.0)
        analyzer = EnsembleSentimentAnalyzer(config)
        
        result = analyzer.analyze("")
        
        assert result.score == 0.0
        assert result.label == SentimentLabel.NEUTRAL
        assert result.confidence == 0.0
        assert "error" in result.details
    
    def test_ensemble_analyze_with_mock_analyzers(self):
        """Test ensemble analyzer with mock analyzers."""
        config = SentimentConfig()
        
        # Mock the analyzer initialization
        with patch.object(EnsembleSentimentAnalyzer, '__init__', lambda x, y: None):
            analyzer = EnsembleSentimentAnalyzer(config)
            
            # Manually set analyzers
            mock_analyzer1 = MockSentimentAnalyzer("mock1", 0.8, 0.9)
            mock_analyzer2 = MockSentimentAnalyzer("mock2", 0.2, 0.7)
            analyzer.analyzers = [(mock_analyzer1, 0.6), (mock_analyzer2, 0.4)]
            analyzer.config = config
            
            result = analyzer.analyze("Test text")
            
            # Weighted average: 0.8 * 0.6 + 0.2 * 0.4 = 0.56
            expected_score = 0.8 * 0.6 + 0.2 * 0.4
            assert abs(result.score - expected_score) < 0.01
            assert result.label == SentimentLabel.POSITIVE
            assert "analyzer_results" in result.details
    
    def test_ensemble_analyzer_failure(self):
        """Test ensemble when one analyzer fails."""
        config = SentimentConfig()
        
        # Create a mock analyzer that raises exception
        failing_analyzer = Mock()
        failing_analyzer.analyze.side_effect = Exception("Mock failure")
        failing_analyzer.get_name.return_value = "failing"
        
        working_analyzer = MockSentimentAnalyzer("working", 0.5, 0.8)
        
        with patch.object(EnsembleSentimentAnalyzer, '__init__', lambda x, y: None):
            analyzer = EnsembleSentimentAnalyzer(config)
            analyzer.analyzers = [(failing_analyzer, 0.5), (working_analyzer, 0.5)]
            analyzer.config = config
            
            result = analyzer.analyze("Test text")
            
            # Should still work with one analyzer, but weight is normalized to 1.0
            assert result.score == 0.5  # Only working analyzer contributes
            assert result.label == SentimentLabel.POSITIVE
            assert len(result.details["analyzer_results"]) == 1


class TestSentimentProcessor:
    """Test sentiment processor for batch processing."""
    
    @pytest.fixture
    def mock_storage(self):
        """Create mock storage for testing."""
        storage = Mock()
        storage.update_sentiment.return_value = True
        storage.get_tweets.return_value = []
        storage.get_sentiment_summary.return_value = {"total_tweets": 0}
        return storage
    
    @pytest.fixture
    def mock_config(self):
        """Create mock config for testing."""
        return SentimentConfig(vader_weight=0.0, keyword_weight=1.0)
    
    def test_processor_initialization(self, mock_storage, mock_config):
        """Test sentiment processor initialization."""
        processor = SentimentProcessor(mock_storage, mock_config)
        
        assert processor.storage == mock_storage
        assert processor.config == mock_config
        assert processor.stats["tweets_processed"] == 0
        assert processor.stats["tweets_updated"] == 0
    
    def test_process_tweet_success(self, mock_storage, mock_config):
        """Test processing a single tweet successfully."""
        processor = SentimentProcessor(mock_storage, mock_config)
        
        tweet = TweetModel(
            tweet_id="123",
            author="test_user",
            user_id="user123",
            username="test_user",
            content="This is good news!",
            timestamp=datetime.now(timezone.utc),
            url="https://example.com",
            raw_data={}
        )
        
        result = processor.process_tweet(tweet)
        
        assert result is True
        assert processor.stats["tweets_processed"] == 1
        assert processor.stats["tweets_updated"] == 1
        mock_storage.update_sentiment.assert_called_once()
    
    def test_process_tweet_dict_input(self, mock_storage, mock_config):
        """Test processing tweet from dictionary."""
        processor = SentimentProcessor(mock_storage, mock_config)
        
        tweet_dict = {
            "tweet_id": "123",
            "content": "This is good news!"
        }
        
        result = processor.process_tweet(tweet_dict)
        
        assert result is True
        mock_storage.update_sentiment.assert_called_once()
    
    def test_process_tweet_empty_content(self, mock_storage, mock_config):
        """Test processing tweet with empty content."""
        processor = SentimentProcessor(mock_storage, mock_config)
        
        tweet_dict = {
            "tweet_id": "123",
            "content": ""
        }
        
        result = processor.process_tweet(tweet_dict)
        
        assert result is False
        # Empty content warning is logged but doesn't increment error count
        assert processor.stats["tweets_processed"] == 1
        assert processor.stats["tweets_updated"] == 0
    
    def test_process_tweet_storage_failure(self, mock_storage, mock_config):
        """Test processing tweet when storage update fails."""
        mock_storage.update_sentiment.return_value = False
        processor = SentimentProcessor(mock_storage, mock_config)
        
        tweet_dict = {
            "tweet_id": "123",
            "content": "Good news!"
        }
        
        result = processor.process_tweet(tweet_dict)
        
        assert result is False
        assert processor.stats["tweets_processed"] == 1
        assert processor.stats["tweets_updated"] == 0
    
    def test_process_batch_success(self, mock_storage, mock_config):
        """Test batch processing success."""
        # Mock tweets to return
        mock_tweets = [
            {"tweet_id": "1", "content": "Good news!"},
            {"tweet_id": "2", "content": "Bad news!"},
            {"tweet_id": "3", "content": "Neutral news."}
        ]
        mock_storage.get_tweets.return_value = mock_tweets
        
        processor = SentimentProcessor(mock_storage, mock_config)
        
        result = processor.process_batch(limit=10, only_missing=True)
        
        assert "tweets_processed" in result
        assert "tweets_updated" in result
        assert "duration_seconds" in result
        assert result["tweets_processed"] == 3
        mock_storage.get_tweets.assert_called_once_with(with_sentiment=False, limit=10)
    
    def test_process_batch_no_tweets(self, mock_storage, mock_config):
        """Test batch processing with no tweets."""
        mock_storage.get_tweets.return_value = []
        processor = SentimentProcessor(mock_storage, mock_config)
        
        result = processor.process_batch()
        
        assert result["tweets_processed"] == 0
        assert result["tweets_updated"] == 0
    
    @pytest.mark.asyncio
    async def test_process_batch_async_success(self, mock_storage, mock_config):
        """Test async batch processing."""
        mock_tweets = [
            {"tweet_id": "1", "content": "Good news!"},
            {"tweet_id": "2", "content": "Bad news!"}
        ]
        mock_storage.get_tweets.return_value = mock_tweets
        
        processor = SentimentProcessor(mock_storage, mock_config)
        
        result = await processor.process_batch_async(limit=10, batch_size=1)
        
        assert result["tweets_processed"] == 2
        assert "duration_seconds" in result
        assert "tweets_per_second" in result
    
    @pytest.mark.asyncio
    async def test_process_batch_async_no_tweets(self, mock_storage, mock_config):
        """Test async batch processing with no tweets."""
        mock_storage.get_tweets.return_value = []
        processor = SentimentProcessor(mock_storage, mock_config)
        
        result = await processor.process_batch_async()
        
        assert result["tweets_processed"] == 0
        assert result["tweets_updated"] == 0
        assert result["errors"] == 0
    
    def test_get_sentiment_summary(self, mock_storage, mock_config):
        """Test getting sentiment summary."""
        mock_summary = {
            "total_tweets": 100,
            "positive_tweets": 40,
            "negative_tweets": 30,
            "neutral_tweets": 30
        }
        mock_storage.get_sentiment_summary.return_value = mock_summary
        
        processor = SentimentProcessor(mock_storage, mock_config)
        processor.stats["tweets_processed"] = 50
        
        result = processor.get_sentiment_summary(time_window_hours=24)
        
        assert result["total_tweets"] == 100
        assert result["positive_tweets"] == 40
        assert "processor_stats" in result
        assert result["processor_stats"]["tweets_processed"] == 50
        mock_storage.get_sentiment_summary.assert_called_once_with(hours=24)


class TestCreateSentimentProcessor:
    """Test sentiment processor factory function."""
    
    def test_create_processor_default_config(self):
        """Test creating processor with default config."""
        mock_storage = Mock()
        
        processor = create_sentiment_processor(mock_storage)
        
        assert processor.storage == mock_storage
        assert isinstance(processor.config, SentimentConfig)
        assert processor.config.vader_weight == 0.6  # Default value
    
    def test_create_processor_custom_config(self):
        """Test creating processor with custom config."""
        mock_storage = Mock()
        config_dict = {
            "vader_weight": 0.8,
            "keyword_weight": 0.2,
            "neutral_threshold": 0.05
        }
        
        processor = create_sentiment_processor(mock_storage, config_dict)
        
        assert processor.config.vader_weight == 0.8
        assert processor.config.keyword_weight == 0.2
        assert processor.config.neutral_threshold == 0.05


class TestErrorHandling:
    """Test error handling in sentiment analysis."""
    
    def test_sentiment_result_validation_edge_cases(self):
        """Test edge cases in sentiment result validation."""
        # Exactly on boundaries should work
        result = SentimentResult(
            score=0.0,
            label=SentimentLabel.NEUTRAL,
            confidence=0.0
        )
        assert result.score == 0.0
        assert result.confidence == 0.0
        
        # Just outside boundaries should fail
        with pytest.raises(ValueError):
            SentimentResult(
                score=1.000001,
                label=SentimentLabel.POSITIVE,
                confidence=0.5
            )
    
    def test_keyword_analyzer_empty_keywords(self):
        """Test keyword analyzer with empty keyword lists."""
        config = SentimentConfig(
            positive_keywords=[],
            negative_keywords=[],
            polymarket_positive=[],
            polymarket_negative=[]
        )
        analyzer = KeywordAnalyzer(config)
        
        result = analyzer.analyze("This is some text")
        
        assert result.score == 0.0
        assert result.label == SentimentLabel.NEUTRAL
        assert result.confidence == 0.0
    
    @pytest.mark.skipif(VADER_AVAILABLE, reason="Test for when VADER not available")
    def test_vader_analyzer_not_available(self):
        """Test VADER analyzer when library not available."""
        with pytest.raises(ImportError, match="VADER sentiment not available"):
            VADERAnalyzer()
    
    def test_ensemble_all_analyzers_fail(self):
        """Test ensemble when all analyzers fail."""
        config = SentimentConfig()
        
        failing_analyzer1 = Mock()
        failing_analyzer1.analyze.side_effect = Exception("Fail 1")
        failing_analyzer1.get_name.return_value = "fail1"
        
        failing_analyzer2 = Mock()
        failing_analyzer2.analyze.side_effect = Exception("Fail 2")
        failing_analyzer2.get_name.return_value = "fail2"
        
        with patch.object(EnsembleSentimentAnalyzer, '__init__', lambda x, y: None):
            analyzer = EnsembleSentimentAnalyzer(config)
            analyzer.analyzers = [(failing_analyzer1, 0.5), (failing_analyzer2, 0.5)]
            analyzer.config = config
            
            result = analyzer.analyze("Test text")
            
            assert result.score == 0.0
            assert result.label == SentimentLabel.NEUTRAL
            assert result.confidence == 0.0
            assert "error" in result.details
            assert result.details["error"] == "All analyzers failed"


if __name__ == "__main__":
    pytest.main([__file__])
