#!/usr/bin/env python3
"""
Demo script showcasing the new backend system with snscrape primary and Nitter fallback.

This script demonstrates:
1. Automatic backend detection and prioritization
2. snscrape as primary scraper (faster, more reliable)
3. Nitter as fallback when snscrape fails
4. Seamless backend switching
"""

import asyncio
import logging
from pathlib import Path

from polymkt_sent.core.backends import BackendManager, SNSScrapeBackend, NitterBackend
from polymkt_sent.core.storage import TweetStorage, StorageConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_backend_system():
    """Demonstrate the new backend system functionality."""
    
    print("🚀 Polymarket-Sentiment Backend System Demo")
    print("=" * 50)
    
    # Configure mirrors for Nitter fallback
    mirrors = [
        "https://nitter.net",
        "https://nitter.pufe.org", 
        "https://nitter.hu"
    ]
    
    # Initialize backend manager
    print("\n1. Initializing Backend Manager...")
    backend_manager = BackendManager(mirrors)
    
    available_backends = backend_manager.get_available_backends()
    print(f"   Available backends: {available_backends}")
    
    # Test individual backends
    print("\n2. Testing Individual Backends...")
    
    # Test SNSScrape availability
    sns_backend = SNSScrapeBackend()
    print(f"   SNSScrape available: {sns_backend.is_available()}")
    
    # Test Nitter availability
    nitter_backend = NitterBackend(mirrors)
    print(f"   Nitter available: {nitter_backend.is_available()}")
    
    # Demo user scraping
    test_users = ["elonmusk", "VitalikButerin"]
    
    print(f"\n3. Demo: Scraping User Tweets (Primary: snscrape, Fallback: Nitter)")
    print("-" * 60)
    
    for username in test_users:
        print(f"\n   Scraping @{username}...")
        
        try:
            result = await backend_manager.scrape_user_tweets(
                username=username,
                max_tweets=5  # Small number for demo
            )
            
            if result.success:
                print(f"   ✅ Success! Got {len(result.tweets)} tweets using {result.backend_used}")
                
                # Show sample tweet
                if result.tweets:
                    sample_tweet = result.tweets[0]
                    content_preview = sample_tweet.content[:100] + "..." if len(sample_tweet.content) > 100 else sample_tweet.content
                    print(f"   📝 Sample: {content_preview}")
                    print(f"   📊 Engagement: {sample_tweet.likes} likes, {sample_tweet.retweets} retweets")
            else:
                print(f"   ❌ Failed: {result.error_message}")
                
        except Exception as e:
            print(f"   💥 Exception: {e}")
    
    # Demo search scraping
    search_queries = ["bitcoin", "ethereum"]
    
    print(f"\n4. Demo: Scraping Search Results")
    print("-" * 40)
    
    for query in search_queries:
        print(f"\n   Searching for '{query}'...")
        
        try:
            result = await backend_manager.scrape_search_tweets(
                query=query,
                max_tweets=3  # Small number for demo
            )
            
            if result.success:
                print(f"   ✅ Success! Got {len(result.tweets)} tweets using {result.backend_used}")
                
                # Show sample tweets
                for i, tweet in enumerate(result.tweets[:2], 1):
                    content_preview = tweet.content[:80] + "..." if len(tweet.content) > 80 else tweet.content
                    print(f"   {i}. @{tweet.username}: {content_preview}")
            else:
                print(f"   ❌ Failed: {result.error_message}")
                
        except Exception as e:
            print(f"   💥 Exception: {e}")
    
    # Demo with storage integration
    print(f"\n5. Demo: Integration with Storage System")
    print("-" * 45)
    
    # Create temporary storage
    storage_config = StorageConfig(data_dir="./demo_data")
    storage = TweetStorage(storage_config)
    
    print("   Scraping and storing tweets...")
    
    # Scrape some tweets and store them
    result = await backend_manager.scrape_user_tweets("polymarket", max_tweets=3)
    
    if result.success and result.tweets:
        # Store tweets
        storage.insert_tweets(result.tweets)
        
        # Get stats
        stats = storage.get_stats()
        print(f"   📊 Stored {len(result.tweets)} tweets")
        print(f"   📈 Database stats: {stats['total_tweets']} total tweets")
        print(f"   🔧 Backend used: {result.backend_used}")
    
    print(f"\n6. Backend Performance Summary")
    print("-" * 35)
    print("   ✨ SNSScrape Benefits:")
    print("      • Faster scraping (no HTTP overhead)")
    print("      • More reliable (direct API access)")
    print("      • Better rate limits")
    print("      • Consistent data format")
    print()
    print("   🔄 Nitter Fallback:")
    print("      • Works when SNSScrape fails")
    print("      • Multiple mirror support")
    print("      • No API dependencies")
    print("      • Proven reliability")
    
    print(f"\n🎉 Demo completed! Backend system is ready for production.")
    print("   Run 'polymkt-sent scrape run' to start using the new system.")


if __name__ == "__main__":
    asyncio.run(demo_backend_system())
