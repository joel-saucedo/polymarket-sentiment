#!/usr/bin/env python3
"""
Quick integration test to verify snscrape backend is working.
"""

import asyncio
import logging
from polymkt_sent.core.backends import BackendManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_backend_integration():
    """Test that snscrape backend is working as primary."""
    
    # Initialize backend manager with some Nitter mirrors as fallback
    mirrors = [
        "https://nitter.net",
        "https://nitter.pufe.org"
    ]
    
    try:
        backend_manager = BackendManager(mirrors)
        available_backends = backend_manager.get_available_backends()
        
        logger.info(f"Available backends: {available_backends}")
        
        # Test user scraping (small number of tweets)
        logger.info("Testing user tweet scraping...")
        result = await backend_manager.scrape_user_tweets("elonmusk", max_tweets=3)
        
        if result.success:
            logger.info(f"✅ Successfully scraped {len(result.tweets)} tweets using {result.backend_used}")
            for i, tweet in enumerate(result.tweets[:2]):  # Show first 2 tweets
                logger.info(f"  Tweet {i+1}: {tweet.content[:100]}...")
        else:
            logger.error(f"❌ Failed to scrape tweets: {result.error_message}")
            
        # Test search scraping
        logger.info("Testing search tweet scraping...")
        result = await backend_manager.scrape_search_tweets("bitcoin", max_tweets=3)
        
        if result.success:
            logger.info(f"✅ Successfully scraped {len(result.tweets)} tweets using {result.backend_used}")
            for i, tweet in enumerate(result.tweets[:2]):  # Show first 2 tweets
                logger.info(f"  Tweet {i+1}: {tweet.content[:100]}...")
        else:
            logger.error(f"❌ Failed to scrape search tweets: {result.error_message}")
            
    except Exception as e:
        logger.error(f"❌ Backend integration test failed: {e}")
        return False
        
    return True

if __name__ == "__main__":
    print("🔍 Testing backend integration with snscrape...")
    success = asyncio.run(test_backend_integration())
    
    if success:
        print("✅ Backend integration test completed!")
    else:
        print("❌ Backend integration test failed!")
