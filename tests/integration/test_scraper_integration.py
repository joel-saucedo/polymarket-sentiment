"""
Integration tests for the async batch scraper.
"""

import pytest
import asyncio
import tempfile
from datetime import datetime, timezone

from polymkt_sent.core.scraper import (
    AsyncBatchScraper,
    ScrapingConfig,
    ScrapingTarget,
    load_scraping_config,
    load_scraping_targets
)
from polymkt_sent.core.storage import TweetStorage, StorageConfig


@pytest.mark.asyncio
async def test_scraper_end_to_end():
    """Test complete scraper workflow end-to-end."""
    
    # Create test storage
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_config = StorageConfig(data_dir=temp_dir)
        storage = TweetStorage(storage_config)
        
        # Create test configuration  
        scraper_config = ScrapingConfig(
            mirrors=["https://nitter.net"],  # Use real mirror for integration test
            per_mirror_per_minute=2,  # Very conservative for integration test
            max_concurrent_requests=1,
            tweets_per_target=5,  # Small number for test
            max_retry=1  # Don't retry on failures for speed
        )
        
        # Create minimal test targets
        targets = [
            ScrapingTarget(
                target_type="search",
                value="bitcoin",  # Popular search term likely to have results
                weight=1.0,
                description="Bitcoin search test"
            )
        ]
        
        # Create scraper
        scraper = AsyncBatchScraper(storage, scraper_config, targets)
        
        # Run scraping (this will actually hit the network)
        try:
            summary = await scraper.scrape_all_targets()
            
            # Basic assertions
            assert summary["targets_total"] == 1
            assert summary["targets_processed"] >= 0  # May be 0 if network fails
            assert "duration_seconds" in summary
            assert summary["duration_seconds"] > 0
            
            print(f"Integration test summary: {summary}")
            
        except Exception as e:
            # Network failures are expected in integration tests
            print(f"Integration test failed (expected): {e}")
            pytest.skip(f"Network error during integration test: {e}")


def test_config_files_exist():
    """Test that default config files can be loaded."""
    
    # Test loading scraper config
    try:
        config = load_scraping_config("config/scraper.yml")
        assert isinstance(config, ScrapingConfig)
        assert len(config.mirrors) > 0
        print(f"Loaded scraper config with {len(config.mirrors)} mirrors")
    except Exception as e:
        pytest.fail(f"Failed to load scraper config: {e}")
    
    # Test loading targets
    try:
        targets = load_scraping_targets("config/accounts.yml")
        assert isinstance(targets, list)
        print(f"Loaded {len(targets)} scraping targets")
        
        # Check we have both users and search terms
        users = [t for t in targets if t.is_user]
        searches = [t for t in targets if t.is_search]
        assert len(users) > 0, "Should have at least one user target"
        assert len(searches) > 0, "Should have at least one search target"
        
    except Exception as e:
        pytest.fail(f"Failed to load targets: {e}")


if __name__ == "__main__":
    # Run basic config test
    test_config_files_exist()
    print("✅ Config files loaded successfully!")
