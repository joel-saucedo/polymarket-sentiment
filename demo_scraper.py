"""
Demo script to test the async batch scraper.
"""

import asyncio
import tempfile
from datetime import datetime

from polymkt_sent.core.scraper import (
    AsyncBatchScraper,
    ScrapingConfig,
    ScrapingTarget,
)
from polymkt_sent.core.storage import TweetStorage, StorageConfig


async def main():
    """Demo the async batch scraper."""
    
    print("🚀 Starting async batch scraper demo...")
    
    # Create temporary storage
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Using temporary storage: {temp_dir}")
        
        # Setup storage
        storage_config = StorageConfig(data_dir=temp_dir)
        storage = TweetStorage(storage_config)
        
        # Setup scraper config (conservative for demo)
        scraper_config = ScrapingConfig(
            mirrors=["https://nitter.net"],
            per_mirror_per_minute=3,
            max_concurrent_requests=1,
            tweets_per_target=5,
            max_retry=1
        )
        
        # Setup test targets
        targets = [
            ScrapingTarget(
                target_type="search",
                value="polymarket",
                weight=1.0,
                description="Polymarket mentions"
            ),
            ScrapingTarget(
                target_type="search", 
                value="crypto",
                weight=0.8,
                description="Crypto sentiment"
            )
        ]
        
        print(f"🎯 Configured {len(targets)} targets")
        for target in targets:
            print(f"  - {target.target_type}: '{target.value}' (weight: {target.weight})")
        
        # Create and run scraper
        scraper = AsyncBatchScraper(storage, scraper_config, targets)
        
        print(f"\n⏳ Running scraper with {len(scraper_config.mirrors)} mirrors...")
        start_time = datetime.now()
        
        try:
            summary = await scraper.scrape_all_targets()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print(f"\n✅ Scraping completed in {duration:.1f}s")
            print(f"📊 Summary:")
            print(f"  - Targets processed: {summary['targets_processed']}/{summary['targets_total']}")
            print(f"  - Tweets scraped: {summary['tweets_scraped']}")
            print(f"  - Success rate: {summary['success_rate']:.1%}")
            print(f"  - Tweets/second: {summary['tweets_per_second']:.2f}")
            
            # Check storage
            total_stored = storage.get_stats()["total_tweets"]
            print(f"  - Total tweets in storage: {total_stored}")
            
            if total_stored > 0:
                print(f"\n📱 Sample tweets:")
                sample_tweets = storage.get_tweets(limit=3)
                for i, tweet in enumerate(sample_tweets, 1):
                    print(f"  {i}. @{tweet['username']}: {tweet['content'][:100]}...")
            
        except Exception as e:
            print(f"❌ Scraping failed: {e}")
            print("💡 This is expected if network is unavailable or Nitter instances are down")
        
        print(f"\n🎉 Demo completed!")


if __name__ == "__main__":
    asyncio.run(main())
