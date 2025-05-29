"""
Integration tests for HTTP request handling against real Nitter instances.
These tests require a running Nitter instance and network connectivity.
"""

import pytest
import os
import yaml
from pathlib import Path

from polymkt_sent.core.request import EnhancedRequestSession, create_session_from_config


def load_test_config():
    """Load test configuration."""
    config_path = Path(__file__).parent.parent.parent / "config" / "scraper.yml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


@pytest.mark.integration
class TestRealNitterRequests:
    """Integration tests against real Nitter instances."""
    
    @classmethod
    def setup_class(cls):
        """Set up test class with configuration."""
        cls.config = load_test_config()
        cls.session = create_session_from_config(cls.config)
        
        # Test Nitter instances (public instances for testing)
        cls.test_instances = [
            "https://nitter.net",
            "https://nitter.1d4.us", 
            "https://nitter.kavin.rocks"
        ]
        
        # Skip integration tests if disabled
        if os.getenv("SKIP_INTEGRATION_TESTS") == "1":
            pytest.skip("Integration tests disabled")
    
    def test_public_nitter_instance_access(self):
        """Test access to public Nitter instances."""
        success_count = 0
        
        for instance in self.test_instances:
            try:
                # Test basic access to instance
                result = self.session.get_with_validation(f"{instance}/")
                if result is not None:
                    success_count += 1
                    print(f"✓ Successfully accessed {instance}")
                else:
                    print(f"✗ Failed to access {instance}")
            except Exception as e:
                print(f"✗ Exception accessing {instance}: {e}")
        
        # At least one instance should be accessible
        assert success_count > 0, "No Nitter instances were accessible"
    
    def test_user_timeline_access(self):
        """Test access to user timelines on public instances."""
        # Use a well-known public account for testing
        test_user = "twitter"  # Official Twitter account
        success_count = 0
        
        for instance in self.test_instances:
            try:
                url = f"{instance}/{test_user}"
                result = self.session.get_with_validation(url)
                if result is not None:
                    # Check for timeline content
                    timeline_items = result.find_all("div", class_="timeline-item")
                    if timeline_items:
                        success_count += 1
                        print(f"✓ Found {len(timeline_items)} timeline items on {instance}")
                    else:
                        print(f"✗ No timeline items found on {instance}")
                else:
                    print(f"✗ Failed to get user timeline from {instance}")
            except Exception as e:
                print(f"✗ Exception getting timeline from {instance}: {e}")
        
        # At least one instance should return timeline data
        assert success_count > 0, "No instances returned timeline data"
    
    def test_search_functionality(self):
        """Test search functionality on public instances."""
        search_query = "python"
        success_count = 0
        
        for instance in self.test_instances:
            try:
                url = f"{instance}/search?f=tweets&q={search_query}"
                result = self.session.get_with_validation(url)
                if result is not None:
                    success_count += 1
                    print(f"✓ Search worked on {instance}")
                else:
                    print(f"✗ Search failed on {instance}")
            except Exception as e:
                print(f"✗ Exception during search on {instance}: {e}")
        
        # At least one instance should support search
        assert success_count > 0, "No instances supported search"
    
    def test_error_handling(self):
        """Test error handling with invalid URLs."""
        invalid_urls = [
            "https://nitter.net/nonexistentuser123456789",
            "https://nitter.net/search?q=",
            "https://nonexistent.nitter.instance/"
        ]
        
        for url in invalid_urls:
            try:
                result = self.session.get_with_validation(url)
                # Should either return None or handle gracefully
                print(f"Result for {url}: {'Success' if result else 'Failed (expected)'}")
            except Exception as e:
                print(f"Exception for {url}: {e}")
        
        # Test should not crash - just checking graceful handling


@pytest.mark.integration
@pytest.mark.local
class TestLocalNitterInstance:
    """Integration tests against local Nitter instance."""
    
    @classmethod
    def setup_class(cls):
        """Set up test class for local testing."""
        cls.config = load_test_config()
        cls.session = create_session_from_config(cls.config)
        cls.local_instance = "http://localhost:8080"
        
        # Skip if local Nitter not requested
        if os.getenv("TEST_LOCAL_NITTER") != "1":
            pytest.skip("Local Nitter tests not enabled")
    
    def test_local_instance_connectivity(self):
        """Test connectivity to local Nitter instance."""
        try:
            result = self.session.get_with_validation(f"{self.local_instance}/")
            assert result is not None, "Could not connect to local Nitter instance"
            print("✓ Successfully connected to local Nitter")
        except Exception as e:
            pytest.fail(f"Failed to connect to local Nitter: {e}")
    
    def test_local_user_timeline(self):
        """Test user timeline on local instance."""
        test_user = "twitter"
        url = f"{self.local_instance}/{test_user}"
        
        try:
            result = self.session.get_with_validation(url)
            if result is not None:
                timeline_items = result.find_all("div", class_="timeline-item")
                print(f"Found {len(timeline_items)} timeline items")
                assert len(timeline_items) > 0, "No timeline items found"
            else:
                pytest.fail("Failed to get timeline from local instance")
        except Exception as e:
            pytest.fail(f"Error testing local timeline: {e}")
    
    def test_local_search(self):
        """Test search on local instance."""
        search_query = "test"
        url = f"{self.local_instance}/search?f=tweets&q={search_query}"
        
        try:
            result = self.session.get_with_validation(url)
            assert result is not None, "Search failed on local instance"
            print("✓ Search functionality working on local instance")
        except Exception as e:
            pytest.fail(f"Error testing local search: {e}")


def test_session_configuration():
    """Test session configuration loading."""
    config = load_test_config()
    session = create_session_from_config(config)
    
    # Basic assertions about session setup
    assert session.timeout > 0
    assert session.max_retries >= 0
    assert session.min_response_size > 0
    assert len(session.retry_delays) > 0
    
    print(f"Session config: timeout={session.timeout}s, retries={session.max_retries}")


if __name__ == "__main__":
    # Run with: python -m pytest tests/http/test_request_integration.py -v -m integration
    pytest.main([__file__, "-v", "-m", "integration"])
