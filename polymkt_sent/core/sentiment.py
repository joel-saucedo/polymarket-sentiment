"""
Modular sentiment analysis framework for polymarket-sentiment.

This module provides:
- Abstract base classes for sentiment analyzers
- VADER sentiment analyzer implementation
- Keyword-based sentiment scoring
- Ensemble sentiment analysis combining multiple methods
- Configurable sentiment processing pipeline
- Integration with storage layer for batch processing
"""

import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio

# Sentiment analysis libraries
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logging.warning("VADER sentiment not available. Install with: pip install vaderSentiment")

from polymkt_sent.core.storage import TweetStorage, TweetModel


logger = logging.getLogger(__name__)


class SentimentLabel(Enum):
    """Sentiment classification labels."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    
    score: float  # Range: -1.0 (very negative) to +1.0 (very positive)
    label: SentimentLabel
    confidence: float  # Range: 0.0 to 1.0
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate sentiment result values."""
        if not -1.0 <= self.score <= 1.0:
            raise ValueError(f"Sentiment score must be between -1.0 and 1.0, got {self.score}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")


@dataclass
class SentimentConfig:
    """Configuration for sentiment analysis."""
    
    # Analyzer weights (sum should be 1.0)
    vader_weight: float = 0.6
    keyword_weight: float = 0.4
    
    # Thresholds
    neutral_threshold: float = 0.1  # |score| < threshold = neutral
    confidence_threshold: float = 0.3  # Minimum confidence for non-neutral
    
    # Keyword dictionaries
    positive_keywords: List[str] = field(default_factory=lambda: [
        "bullish", "moon", "pump", "gain", "profit", "win", "success", "good",
        "great", "excellent", "amazing", "awesome", "love", "like", "buy",
        "long", "bull", "rocket", "surge", "boom", "rally", "breakthrough"
    ])
    
    negative_keywords: List[str] = field(default_factory=lambda: [
        "bearish", "dump", "crash", "loss", "fail", "bad", "terrible", 
        "awful", "hate", "sell", "short", "bear", "collapse", "panic",
        "fear", "drop", "fall", "decline", "risk", "concern", "worry"
    ])
    
    # Polymarket-specific keywords
    polymarket_positive: List[str] = field(default_factory=lambda: [
        "prediction market", "forecast", "odds improving", "likely", "probable",
        "confident", "accurate", "correct", "winning", "resolved"
    ])
    
    polymarket_negative: List[str] = field(default_factory=lambda: [
        "unlikely", "doubtful", "wrong", "incorrect", "losing", "risky bet",
        "manipulation", "insider", "scam", "fake"
    ])
    
    # Text preprocessing
    remove_urls: bool = True
    remove_mentions: bool = False
    remove_hashtags: bool = False
    lowercase: bool = True


class SentimentAnalyzer(ABC):
    """Abstract base class for sentiment analyzers."""
    
    @abstractmethod
    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of text and return result."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return analyzer name for identification."""
        pass
    
    def preprocess_text(self, text: str, config: SentimentConfig) -> str:
        """Preprocess text before analysis."""
        processed = text
        
        if config.remove_urls:
            # Remove URLs
            processed = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', processed)
        
        if config.remove_mentions:
            # Remove @mentions
            processed = re.sub(r'@\w+', '', processed)
        
        if config.remove_hashtags:
            # Remove #hashtags
            processed = re.sub(r'#\w+', '', processed)
        
        if config.lowercase:
            processed = processed.lower()
        
        # Clean up extra whitespace
        processed = re.sub(r'\s+', ' ', processed).strip()
        
        return processed


class VADERAnalyzer(SentimentAnalyzer):
    """VADER sentiment analyzer implementation."""
    
    def __init__(self):
        """Initialize VADER analyzer."""
        if not VADER_AVAILABLE:
            raise ImportError("VADER sentiment not available. Install with: pip install vaderSentiment")
        
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze(self, text: str) -> SentimentResult:
        """Analyze text using VADER sentiment."""
        scores = self.analyzer.polarity_scores(text)
        
        # VADER returns compound score (-1 to 1)
        compound_score = scores['compound']
        
        # Determine label based on compound score
        if compound_score >= 0.05:
            label = SentimentLabel.POSITIVE
        elif compound_score <= -0.05:
            label = SentimentLabel.NEGATIVE
        else:
            label = SentimentLabel.NEUTRAL
        
        # Calculate confidence based on the strength of the compound score
        confidence = min(abs(compound_score) + 0.1, 1.0)
        
        return SentimentResult(
            score=compound_score,
            label=label,
            confidence=confidence,
            details={
                "vader_positive": scores['pos'],
                "vader_negative": scores['neg'],
                "vader_neutral": scores['neu'],
                "vader_compound": compound_score
            }
        )
    
    def get_name(self) -> str:
        """Return analyzer name."""
        return "vader"


class KeywordAnalyzer(SentimentAnalyzer):
    """Keyword-based sentiment analyzer."""
    
    def __init__(self, config: SentimentConfig):
        """Initialize keyword analyzer with configuration."""
        self.config = config
        
        # Combine all positive and negative keywords
        self.positive_words = set(
            config.positive_keywords + config.polymarket_positive
        )
        self.negative_words = set(
            config.negative_keywords + config.polymarket_negative
        )
    
    def analyze(self, text: str) -> SentimentResult:
        """Analyze text using keyword matching."""
        processed_text = self.preprocess_text(text, self.config)
        words = processed_text.split()
        
        positive_count = 0
        negative_count = 0
        matched_keywords = []
        
        for word in words:
            # Check for positive keywords
            for pos_word in self.positive_words:
                if pos_word in word or word in pos_word:
                    positive_count += 1
                    matched_keywords.append(f"+{pos_word}")
                    break
            
            # Check for negative keywords
            for neg_word in self.negative_words:
                if neg_word in word or word in neg_word:
                    negative_count += 1
                    matched_keywords.append(f"-{neg_word}")
                    break
        
        # Calculate score based on keyword balance
        total_keywords = positive_count + negative_count
        if total_keywords == 0:
            score = 0.0
            confidence = 0.0
        else:
            score = (positive_count - negative_count) / max(len(words), 1)
            # Normalize score to [-1, 1] range
            score = max(-1.0, min(1.0, score * 2))
            confidence = min(total_keywords / max(len(words), 1) + 0.1, 1.0)
        
        # Determine label
        if abs(score) < self.config.neutral_threshold:
            label = SentimentLabel.NEUTRAL
        elif score > 0:
            label = SentimentLabel.POSITIVE
        else:
            label = SentimentLabel.NEGATIVE
        
        return SentimentResult(
            score=score,
            label=label,
            confidence=confidence,
            details={
                "positive_keywords": positive_count,
                "negative_keywords": negative_count,
                "matched_keywords": matched_keywords,
                "total_words": len(words)
            }
        )
    
    def get_name(self) -> str:
        """Return analyzer name."""
        return "keyword"


class EnsembleSentimentAnalyzer:
    """Ensemble analyzer combining multiple sentiment analysis methods."""
    
    def __init__(self, config: SentimentConfig):
        """Initialize ensemble analyzer."""
        self.config = config
        self.analyzers: List[Tuple[SentimentAnalyzer, float]] = []
        
        # Initialize available analyzers
        if VADER_AVAILABLE and config.vader_weight > 0:
            self.analyzers.append((VADERAnalyzer(), config.vader_weight))
        
        if config.keyword_weight > 0:
            self.analyzers.append((KeywordAnalyzer(config), config.keyword_weight))
        
        if not self.analyzers:
            raise ValueError("No sentiment analyzers available or configured")
        
        # Normalize weights
        total_weight = sum(weight for _, weight in self.analyzers)
        self.analyzers = [(analyzer, weight / total_weight) for analyzer, weight in self.analyzers]
        
        logger.info(f"Initialized ensemble with {len(self.analyzers)} analyzers")
    
    def analyze(self, text: str) -> SentimentResult:
        """Analyze text using ensemble of methods."""
        if not text or not text.strip():
            return SentimentResult(
                score=0.0,
                label=SentimentLabel.NEUTRAL,
                confidence=0.0,
                details={"error": "Empty text"}
            )
        
        # Get results from all analyzers
        results = []
        for analyzer, weight in self.analyzers:
            try:
                result = analyzer.analyze(text)
                results.append((result, weight, analyzer.get_name()))
            except Exception as e:
                logger.warning(f"Analyzer {analyzer.get_name()} failed: {e}")
                continue
        
        if not results:
            return SentimentResult(
                score=0.0,
                label=SentimentLabel.NEUTRAL,
                confidence=0.0,
                details={"error": "All analyzers failed"}
            )
        
        # Normalize weights of successful analyzers
        total_weight = sum(weight for _, weight, _ in results)
        normalized_results = []
        for result, weight, name in results:
            normalized_weight = weight / total_weight if total_weight > 0 else 0
            normalized_results.append((result, normalized_weight, name))
        
        # Calculate weighted ensemble score
        weighted_score = sum(result.score * weight for result, weight, _ in normalized_results)
        weighted_confidence = sum(result.confidence * weight for result, weight, _ in normalized_results)
        
        # Determine final label
        if abs(weighted_score) < self.config.neutral_threshold:
            label = SentimentLabel.NEUTRAL
        elif weighted_score > 0:
            label = SentimentLabel.POSITIVE
        else:
            label = SentimentLabel.NEGATIVE
        
        # If confidence is too low, default to neutral
        if weighted_confidence < self.config.confidence_threshold:
            label = SentimentLabel.NEUTRAL
        
        # Combine details from all analyzers
        combined_details = {
            "ensemble_score": weighted_score,
            "ensemble_confidence": weighted_confidence,
            "analyzer_results": {}
        }
        
        for result, weight, name in normalized_results:
            combined_details["analyzer_results"][name] = {
                "score": result.score,
                "label": result.label.value,
                "confidence": result.confidence,
                "weight": weight,  # Use normalized weight
                "details": result.details
            }
        
        return SentimentResult(
            score=weighted_score,
            label=label,
            confidence=weighted_confidence,
            details=combined_details
        )


class SentimentProcessor:
    """Batch sentiment processor for tweets."""
    
    def __init__(self, storage: TweetStorage, config: SentimentConfig):
        """Initialize sentiment processor."""
        self.storage = storage
        self.config = config
        self.analyzer = EnsembleSentimentAnalyzer(config)
        
        # Statistics
        self.stats = {
            "tweets_processed": 0,
            "tweets_updated": 0,
            "errors": 0,
            "last_run": None
        }
    
    def process_tweet(self, tweet: Union[TweetModel, Dict[str, Any]]) -> bool:
        """Process sentiment for a single tweet."""
        try:
            # Extract text content
            if isinstance(tweet, TweetModel):
                tweet_id = tweet.tweet_id
                content = tweet.content
            else:
                tweet_id = tweet.get("tweet_id")
                content = tweet.get("content", "")
            
            if not content:
                logger.warning(f"Empty content for tweet {tweet_id}")
                self.stats["tweets_processed"] += 1
                return False
            
            # Analyze sentiment
            result = self.analyzer.analyze(content)
            
            # Update tweet in storage
            success = self.storage.update_sentiment(
                tweet_id=tweet_id,
                sentiment_score=result.score,
                sentiment_label=result.label.value,
                confidence=result.confidence
            )
            
            if success:
                self.stats["tweets_updated"] += 1
            
            self.stats["tweets_processed"] += 1
            return success
            
        except Exception as e:
            logger.error(f"Failed to process sentiment for tweet {tweet_id}: {e}")
            self.stats["errors"] += 1
            return False
    
    def process_batch(
        self,
        limit: Optional[int] = None,
        only_missing: bool = True
    ) -> Dict[str, Any]:
        """Process sentiment for a batch of tweets."""
        start_time = datetime.now(timezone.utc)
        logger.info(f"Starting sentiment processing batch (limit: {limit})")
        
        # Reset batch stats
        batch_stats = {
            "tweets_processed": 0,
            "tweets_updated": 0,
            "errors": 0
        }
        
        try:
            # Get tweets that need sentiment analysis
            if only_missing:
                # Get tweets without sentiment scores
                tweets = self.storage.get_tweets(
                    with_sentiment=False,
                    limit=limit
                )
            else:
                # Get all tweets
                tweets = self.storage.get_tweets(limit=limit)
            
            logger.info(f"Processing sentiment for {len(tweets)} tweets")
            
            # Process each tweet
            for tweet in tweets:
                success = self.process_tweet(tweet)
                if success:
                    batch_stats["tweets_updated"] += 1
                else:
                    batch_stats["errors"] += 1
                batch_stats["tweets_processed"] += 1
            
            # Update global stats
            self.stats["tweets_processed"] += batch_stats["tweets_processed"]
            self.stats["tweets_updated"] += batch_stats["tweets_updated"]
            self.stats["errors"] += batch_stats["errors"]
            self.stats["last_run"] = start_time
            
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            summary = {
                "start_time": start_time,
                "end_time": end_time,
                "duration_seconds": duration,
                "tweets_processed": batch_stats["tweets_processed"],
                "tweets_updated": batch_stats["tweets_updated"],
                "errors": batch_stats["errors"],
                "success_rate": (
                    batch_stats["tweets_updated"] / max(batch_stats["tweets_processed"], 1)
                ),
                "tweets_per_second": batch_stats["tweets_processed"] / max(duration, 1)
            }
            
            logger.info(
                f"Sentiment processing completed: {batch_stats['tweets_updated']} tweets updated "
                f"in {duration:.1f}s ({summary['tweets_per_second']:.1f} tweets/sec)"
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Batch sentiment processing failed: {e}")
            return {
                "error": str(e),
                "tweets_processed": batch_stats["tweets_processed"],
                "tweets_updated": batch_stats["tweets_updated"]
            }
    
    async def process_batch_async(
        self,
        limit: Optional[int] = None,
        only_missing: bool = True,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """Process sentiment for tweets asynchronously in batches."""
        
        async def process_chunk(tweets_chunk):
            """Process a chunk of tweets."""
            chunk_stats = {"updated": 0, "errors": 0}
            
            for tweet in tweets_chunk:
                success = self.process_tweet(tweet)
                if success:
                    chunk_stats["updated"] += 1
                else:
                    chunk_stats["errors"] += 1
            
            return chunk_stats
        
        start_time = datetime.now(timezone.utc)
        logger.info(f"Starting async sentiment processing (limit: {limit})")
        
        try:
            # Get tweets to process
            if only_missing:
                tweets = self.storage.get_tweets(with_sentiment=False, limit=limit)
            else:
                tweets = self.storage.get_tweets(limit=limit)
            
            if not tweets:
                logger.info("No tweets to process")
                return {"tweets_processed": 0, "tweets_updated": 0, "errors": 0}
            
            # Split into chunks for parallel processing
            chunks = [tweets[i:i + batch_size] for i in range(0, len(tweets), batch_size)]
            logger.info(f"Processing {len(tweets)} tweets in {len(chunks)} chunks")
            
            # Process chunks concurrently
            tasks = [process_chunk(chunk) for chunk in chunks]
            chunk_results = await asyncio.gather(*tasks)
            
            # Aggregate results
            total_updated = sum(result["updated"] for result in chunk_results)
            total_errors = sum(result["errors"] for result in chunk_results)
            
            # Update global stats
            self.stats["tweets_processed"] += len(tweets)
            self.stats["tweets_updated"] += total_updated
            self.stats["errors"] += total_errors
            self.stats["last_run"] = start_time
            
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            summary = {
                "start_time": start_time,
                "end_time": end_time,
                "duration_seconds": duration,
                "tweets_processed": len(tweets),
                "tweets_updated": total_updated,
                "errors": total_errors,
                "success_rate": total_updated / max(len(tweets), 1),
                "tweets_per_second": len(tweets) / max(duration, 1)
            }
            
            logger.info(
                f"Async sentiment processing completed: {total_updated} tweets updated "
                f"in {duration:.1f}s ({summary['tweets_per_second']:.1f} tweets/sec)"
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Async sentiment processing failed: {e}")
            return {"error": str(e), "tweets_processed": 0, "tweets_updated": 0}
    
    def get_sentiment_summary(
        self,
        time_window_hours: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get sentiment summary statistics."""
        try:
            # Get summary from storage, pass hours parameter correctly
            hours = time_window_hours if time_window_hours is not None else 24
            summary = self.storage.get_sentiment_summary(hours=hours)
            
            # Add processor stats
            summary["processor_stats"] = self.stats.copy()
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get sentiment summary: {e}")
            return {"error": str(e)}


def create_sentiment_processor(
    storage: TweetStorage,
    config_dict: Optional[Dict[str, Any]] = None
) -> SentimentProcessor:
    """Create sentiment processor with configuration."""
    
    # Create config from dict or use defaults
    if config_dict:
        config = SentimentConfig(**config_dict)
    else:
        config = SentimentConfig()
    
    return SentimentProcessor(storage, config)
