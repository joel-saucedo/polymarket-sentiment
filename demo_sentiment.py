#!/usr/bin/env python3
"""
Demo script for sentiment analysis functionality.

This script demonstrates:
- Individual sentiment analyzers (VADER, Keyword)
- Ensemble sentiment analysis
- Batch processing of tweets
- Configuration options
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import List

from polymkt_sent.core.sentiment import (
    SentimentConfig,
    VADERAnalyzer, 
    KeywordAnalyzer,
    EnsembleSentimentAnalyzer,
    SentimentProcessor,
    create_sentiment_processor,
    VADER_AVAILABLE
)
from polymkt_sent.core.storage import TweetStorage, TweetModel, StorageConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_individual_analyzers():
    """Demonstrate individual sentiment analyzers."""
    print("\n" + "="*60)
    print("INDIVIDUAL SENTIMENT ANALYZERS DEMO")
    print("="*60)
    
    # Test texts with different sentiments
    test_texts = [
        "This is absolutely amazing! I love this crypto project! 🚀",
        "This is terrible. I hate everything about this market crash.",
        "The weather is okay today. Nothing special happening.",
        "Bullish on Bitcoin! Moon time! 📈 Great investment opportunity!",
        "Bearish sentiment. Market is crashing. Panic selling everywhere.",
        "Polymarket predictions look accurate. Confident in the forecast."
    ]
    
    config = SentimentConfig()
    
    # VADER Analyzer (if available)
    if VADER_AVAILABLE:
        print("\n--- VADER Sentiment Analyzer ---")
        vader = VADERAnalyzer()
        
        for text in test_texts:
            result = vader.analyze(text)
            print(f"Text: {text[:50]}...")
            print(f"  Score: {result.score:.3f}, Label: {result.label.value}, Confidence: {result.confidence:.3f}")
            print(f"  Details: {result.details}")
            print()
    else:
        print("\n--- VADER Sentiment Analyzer ---")
        print("VADER not available. Install with: pip install vaderSentiment")
    
    # Keyword Analyzer
    print("\n--- Keyword Sentiment Analyzer ---")
    keyword = KeywordAnalyzer(config)
    
    for text in test_texts:
        result = keyword.analyze(text)
        print(f"Text: {text[:50]}...")
        print(f"  Score: {result.score:.3f}, Label: {result.label.value}, Confidence: {result.confidence:.3f}")
        print(f"  Matched keywords: {result.details.get('matched_keywords', [])}")
        print()


def demo_ensemble_analyzer():
    """Demonstrate ensemble sentiment analysis."""
    print("\n" + "="*60)
    print("ENSEMBLE SENTIMENT ANALYZER DEMO")
    print("="*60)
    
    # Create ensemble with custom configuration
    config = SentimentConfig(
        vader_weight=0.7,
        keyword_weight=0.3,
        neutral_threshold=0.05,
        confidence_threshold=0.2
    )
    
    try:
        ensemble = EnsembleSentimentAnalyzer(config)
        
        test_texts = [
            "Absolutely bullish! This is amazing news for crypto! 🚀📈",
            "Market crash incoming. Bearish signals everywhere. Panic selling.",
            "The meeting was scheduled for today at 3 PM.",
            "Polymarket odds are improving. Confident prediction results.",
            "Risky bet with high uncertainty. Doubtful about outcome."
        ]
        
        print(f"Ensemble configuration:")
        print(f"  Analyzers: {len(ensemble.analyzers)}")
        for analyzer, weight in ensemble.analyzers:
            print(f"    {analyzer.get_name()}: {weight:.2f}")
        print()
        
        for text in test_texts:
            result = ensemble.analyze(text)
            print(f"Text: {text[:60]}...")
            print(f"  Final Score: {result.score:.3f}")
            print(f"  Final Label: {result.label.value}")
            print(f"  Final Confidence: {result.confidence:.3f}")
            
            # Show individual analyzer results
            analyzer_results = result.details.get("analyzer_results", {})
            for name, details in analyzer_results.items():
                print(f"    {name}: score={details['score']:.3f}, weight={details['weight']:.2f}")
            print()
            
    except Exception as e:
        print(f"Error creating ensemble: {e}")


def demo_sentiment_processor():
    """Demonstrate sentiment processor with mock data."""
    print("\n" + "="*60)
    print("SENTIMENT PROCESSOR DEMO")
    print("="*60)
    
    # Create in-memory storage for demo
    storage_config = StorageConfig(db_file=":memory:")
    storage = TweetStorage(storage_config)
    
    # Create sample tweets
    sample_tweets = [
        TweetModel(
            tweet_id="1",
            user_id="user1",
            username="crypto_bull",
            content="Bitcoin is going to the moon! 🚀 Bullish AF!",
            timestamp=datetime.now(timezone.utc)
        ),
        TweetModel(
            tweet_id="2", 
            user_id="user2",
            username="bear_market",
            content="Market crash incoming. Sell everything. Bearish signals.",
            timestamp=datetime.now(timezone.utc)
        ),
        TweetModel(
            tweet_id="3",
            user_id="user3", 
            username="neutral_news",
            content="The Federal Reserve will meet next week to discuss rates.",
            timestamp=datetime.now(timezone.utc)
        ),
        TweetModel(
            tweet_id="4",
            user_id="user4",
            username="polymarket_user", 
            content="Polymarket prediction looks accurate. Confident in the forecast results.",
            timestamp=datetime.now(timezone.utc)
        )
    ]
    
    # Insert tweets into storage
    storage.insert_tweets_batch(sample_tweets)
    print(f"Inserted {len(sample_tweets)} sample tweets")
    
    # Create sentiment processor
    config = SentimentConfig(
        vader_weight=0.6,
        keyword_weight=0.4
    )
    processor = SentimentProcessor(storage, config)
    
    # Process sentiment for all tweets
    print("\nProcessing sentiment...")
    summary = processor.process_batch()
    
    print(f"Processing Summary:")
    print(f"  Tweets processed: {summary['tweets_processed']}")
    print(f"  Tweets updated: {summary['tweets_updated']}")
    print(f"  Success rate: {summary['success_rate']:.2%}")
    print(f"  Processing speed: {summary['tweets_per_second']:.1f} tweets/sec")
    
    # Show results
    print("\nSentiment Results:")
    tweets_with_sentiment = storage.get_tweets(with_sentiment=True)
    
    for tweet in tweets_with_sentiment:
        print(f"\nTweet ID: {tweet['tweet_id']}")
        print(f"Author: @{tweet['username']}")
        print(f"Content: {tweet['content'][:60]}...")
        print(f"Sentiment: {tweet['sentiment_label']} (score: {tweet['sentiment_score']:.3f}, confidence: {tweet['sentiment_confidence']:.3f})")
    
    # Get sentiment summary
    print("\nSentiment Summary:")
    sentiment_summary = processor.get_sentiment_summary()
    if "error" not in sentiment_summary:
        print(f"  Total tweets: {sentiment_summary.get('total_tweets', 0)}")
        print(f"  Positive: {sentiment_summary.get('positive_tweets', 0)}")
        print(f"  Negative: {sentiment_summary.get('negative_tweets', 0)}")
        print(f"  Neutral: {sentiment_summary.get('neutral_tweets', 0)}")


async def demo_async_processing():
    """Demonstrate async sentiment processing."""
    print("\n" + "="*60)
    print("ASYNC SENTIMENT PROCESSING DEMO") 
    print("="*60)
    
    # Create storage and sample tweets
    storage_config = StorageConfig(db_file=":memory:")
    storage = TweetStorage(storage_config)
    
    # Create larger batch of sample tweets
    sample_tweets = []
    sentiments = [
        ("bullish", "Amazing gains today! Moon mission activated! 🚀"),
        ("bearish", "Terrible crash. Market is doomed. Sell everything!"),
        ("neutral", "The meeting is scheduled for tomorrow at 2 PM."),
        ("positive", "Great news for crypto adoption. Bullish outlook!"),
        ("negative", "Panic selling continues. Bear market confirmed."),
        ("neutral", "Weather forecast shows rain for the weekend."),
        ("positive", "Polymarket predictions are very accurate lately."),
        ("negative", "High risk investment. Very doubtful about returns.")
    ]
    
    for i, (sentiment_type, content) in enumerate(sentiments):
        tweet = TweetModel(
            tweet_id=str(i + 1),
            user_id=f"user{i+1}",
            username=f"user_{sentiment_type}",
            content=content,
            timestamp=datetime.now(timezone.utc)
        )
        sample_tweets.append(tweet)
    
    storage.insert_tweets_batch(sample_tweets)
    print(f"Inserted {len(sample_tweets)} sample tweets for async processing")
    
    # Create processor and run async batch
    processor = create_sentiment_processor(storage, {
        "vader_weight": 0.7,
        "keyword_weight": 0.3
    })
    
    print("\nRunning async sentiment processing...")
    summary = await processor.process_batch_async(batch_size=3)
    
    print(f"Async Processing Summary:")
    print(f"  Tweets processed: {summary['tweets_processed']}")
    print(f"  Tweets updated: {summary['tweets_updated']}")
    print(f"  Success rate: {summary['success_rate']:.2%}")
    print(f"  Processing speed: {summary['tweets_per_second']:.1f} tweets/sec")
    print(f"  Duration: {summary['duration_seconds']:.2f} seconds")


def demo_custom_configuration():
    """Demonstrate custom sentiment configuration."""
    print("\n" + "="*60)
    print("CUSTOM CONFIGURATION DEMO")
    print("="*60)
    
    # Custom configuration with crypto-specific keywords
    custom_config = SentimentConfig(
        vader_weight=0.5,
        keyword_weight=0.5,
        neutral_threshold=0.02,  # More sensitive
        confidence_threshold=0.1,  # Lower threshold
        positive_keywords=[
            "pump", "moon", "bullish", "gains", "profit", "buy", "long",
            "hodl", "diamond hands", "to the moon", "rocket", "surge"
        ],
        negative_keywords=[
            "dump", "crash", "bearish", "loss", "sell", "short", "panic",
            "rekt", "paper hands", "rug pull", "collapse", "fud"
        ],
        polymarket_positive=[
            "accurate prediction", "high confidence", "good odds", 
            "winning bet", "correct forecast"
        ],
        polymarket_negative=[
            "risky bet", "low confidence", "bad odds", "wrong prediction",
            "losing position"
        ]
    )
    
    print("Custom Configuration:")
    print(f"  VADER weight: {custom_config.vader_weight}")
    print(f"  Keyword weight: {custom_config.keyword_weight}")
    print(f"  Neutral threshold: {custom_config.neutral_threshold}")
    print(f"  Positive keywords: {len(custom_config.positive_keywords)}")
    print(f"  Negative keywords: {len(custom_config.negative_keywords)}")
    
    try:
        ensemble = EnsembleSentimentAnalyzer(custom_config)
        
        crypto_texts = [
            "HODL! Diamond hands! To the moon! 🚀💎",
            "Rug pull incoming. Panic sell. Paper hands activated.",
            "Polymarket shows high confidence in accurate prediction.",
            "This is a very risky bet with bad odds. Wrong prediction likely."
        ]
        
        print("\nCustom Configuration Results:")
        for text in crypto_texts:
            result = ensemble.analyze(text)
            print(f"Text: {text}")
            print(f"  Score: {result.score:.3f}, Label: {result.label.value}, Confidence: {result.confidence:.3f}")
            print()
            
    except Exception as e:
        print(f"Error with custom configuration: {e}")


def main():
    """Run all sentiment analysis demos."""
    print("POLYMARKET SENTIMENT ANALYSIS DEMO")
    print("="*60)
    print("This demo showcases the sentiment analysis capabilities")
    print("of the polymarket-sentiment pipeline.")
    
    try:
        # Individual analyzers
        demo_individual_analyzers()
        
        # Ensemble analyzer  
        demo_ensemble_analyzer()
        
        # Sentiment processor
        demo_sentiment_processor()
        
        # Async processing
        asyncio.run(demo_async_processing())
        
        # Custom configuration
        demo_custom_configuration()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("The sentiment analysis framework is working correctly.")
        print("You can now integrate it with the scraping pipeline.")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
