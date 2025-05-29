"""
Unit tests for polymkt_sent.core.request module.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
import requests
from bs4 import BeautifulSoup

from polymkt_sent.core.request import (
    EnhancedRequestSession,
    DEFAULT_HEADERS,
    create_session_from_config
)


class TestEnhancedRequestSession:
    """Test cases for EnhancedRequestSession."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.session = EnhancedRequestSession(
            timeout=10,
            max_retries=3,
            min_response_size=100
        )
    
    def test_initialization(self):
        """Test session initialization."""
        assert self.session.timeout == 10
        assert self.session.max_retries == 3
        assert self.session.min_response_size == 100
        
        # Check default headers are set
        for key, value in DEFAULT_HEADERS.items():
            assert self.session.session.headers[key] == value
    
    def test_custom_retry_delays(self):
        """Test custom retry delays."""
        custom_delays = [0.5, 1.0, 2.0]
        session = EnhancedRequestSession(retry_delays=custom_delays)
        assert session.retry_delays == custom_delays
    
    def test_update_headers(self):
        """Test header updates."""
        new_headers = {"Custom-Header": "test-value"}
        self.session.update_headers(new_headers)
        assert self.session.session.headers["Custom-Header"] == "test-value"
    
    @patch('polymkt_sent.core.request.time.sleep')
    @patch('requests.Session.get')
    def test_successful_request(self, mock_get, mock_sleep):
        """Test successful request with valid response."""
        # Mock successful response
        mock_response = Mock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.content = b"x" * 200  # Above min_response_size
        mock_response.text = """
        <html>
            <div class="timeline-item">
                <div class="tweet-content">Test tweet</div>
            </div>
        </html>
        """
        mock_get.return_value = mock_response
        
        result = self.session.get_with_validation("http://example.com")
        
        assert result is not None
        assert isinstance(result, BeautifulSoup)
        mock_get.assert_called_once()
        mock_sleep.assert_not_called()
    
    @patch('polymkt_sent.core.request.time.sleep')
    @patch('requests.Session.get')
    def test_retry_on_http_error(self, mock_get, mock_sleep):
        """Test retry logic on HTTP errors."""
        # First two attempts fail, third succeeds
        mock_response_fail = Mock()
        mock_response_fail.ok = False
        mock_response_fail.status_code = 500
        
        mock_response_success = Mock()
        mock_response_success.ok = True
        mock_response_success.status_code = 200
        mock_response_success.content = b"x" * 200
        mock_response_success.text = """
        <html>
            <div class="timeline-item">
                <div class="tweet-content">Test tweet</div>
            </div>
        </html>
        """
        
        mock_get.side_effect = [
            mock_response_fail,
            mock_response_fail,
            mock_response_success
        ]
        
        result = self.session.get_with_validation("http://example.com")
        
        assert result is not None
        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 2  # 2 retries
    
    @patch('polymkt_sent.core.request.time.sleep')
    @patch('requests.Session.get')
    def test_retry_exhaustion(self, mock_get, mock_sleep):
        """Test behavior when all retries are exhausted."""
        # All attempts fail
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        
        result = self.session.get_with_validation("http://example.com")
        
        assert result is None
        assert mock_get.call_count == 4  # initial + 3 retries
        assert mock_sleep.call_count == 3
    
    @patch('polymkt_sent.core.request.time.sleep')
    @patch('requests.Session.get')
    def test_request_exception_retry(self, mock_get, mock_sleep):
        """Test retry on request exceptions."""
        # First attempt raises exception, second succeeds
        mock_response = Mock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.content = b"x" * 200
        mock_response.text = """
        <html>
            <div class="timeline-item">
                <div class="tweet-content">Test tweet</div>
            </div>
        </html>
        """
        
        mock_get.side_effect = [
            requests.exceptions.ConnectionError("Connection failed"),
            mock_response
        ]
        
        result = self.session.get_with_validation("http://example.com")
        
        assert result is not None
        assert mock_get.call_count == 2
        assert mock_sleep.call_count == 1
    
    @patch('requests.Session.get')
    def test_response_too_small(self, mock_get):
        """Test validation failure for small responses."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.content = b"small"  # Below min_response_size
        mock_get.return_value = mock_response
        
        result = self.session.get_with_validation("http://example.com")
        
        assert result is None
    
    @patch('requests.Session.get')
    def test_nitter_error_page(self, mock_get):
        """Test detection of Nitter error pages."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.content = b"x" * 200
        mock_response.text = """
        <html>
            <div class="error-panel">
                <span>User not found</span>
            </div>
        </html>
        """
        mock_get.return_value = mock_response
        
        result = self.session.get_with_validation("http://example.com")
        
        assert result is None
    
    @patch('requests.Session.get')
    def test_protected_account(self, mock_get):
        """Test detection of protected accounts."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.content = b"x" * 200
        mock_response.text = """
        <html>
            <div class="timeline-header timeline-protected">
                This account is protected
            </div>
        </html>
        """
        mock_get.return_value = mock_response
        
        result = self.session.get_with_validation("http://example.com")
        
        assert result is None
    
    @patch('requests.Session.get')
    def test_empty_timeline(self, mock_get):
        """Test detection of empty timeline."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.content = b"x" * 200
        mock_response.text = """
        <html>
            <div class="timeline">
                <!-- No timeline-item elements -->
            </div>
        </html>
        """
        mock_get.return_value = mock_response
        
        result = self.session.get_with_validation("http://example.com/user")
        
        assert result is None
    
    @patch('requests.Session.get')
    def test_search_url_no_timeline_required(self, mock_get):
        """Test that search URLs don't require timeline content."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.content = b"x" * 200
        mock_response.text = """
        <html>
            <div class="search-results">
                <!-- Search page without timeline-item -->
            </div>
        </html>
        """
        mock_get.return_value = mock_response
        
        result = self.session.get_with_validation("http://example.com/search?q=test")
        
        assert result is not None
    
    def test_get_retry_delay(self):
        """Test retry delay calculation."""
        # Test configured delays
        for i in range(len(self.session.retry_delays)):
            delay = self.session._get_retry_delay(i)
            expected = self.session.retry_delays[i]
            # Should be within jitter range (±20%)
            assert expected * 0.8 <= delay <= expected * 1.2
        
        # Test exponential backoff beyond configured delays
        delay = self.session._get_retry_delay(10)
        assert delay > self.session.retry_delays[-1]
    
    @patch('requests.Session.get')
    def test_instance_host_header(self, mock_get):
        """Test Host header setting for Nitter instances."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.content = b"x" * 200
        mock_response.text = """
        <html>
            <div class="timeline-item">
                <div class="tweet-content">Test tweet</div>
            </div>
        </html>
        """
        mock_get.return_value = mock_response
        
        self.session.get_with_validation(
            "http://localhost:8080/user",
            instance_host="nitter.example.com"
        )
        
        assert self.session.session.headers["Host"] == "nitter.example.com"
    
    @patch('requests.Session.get')
    def test_custom_cookies(self, mock_get):
        """Test custom cookies handling."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.content = b"x" * 200
        mock_response.text = """
        <html>
            <div class="timeline-item">
                <div class="tweet-content">Test tweet</div>
            </div>
        </html>
        """
        mock_get.return_value = mock_response
        
        custom_cookies = {"customCookie": "value"}
        self.session.get_with_validation(
            "http://example.com",
            cookies=custom_cookies
        )
        
        # Check that cookies were passed
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args[1]
        cookies = call_kwargs["cookies"]
        assert cookies["customCookie"] == "value"
        assert "hlsPlayback" in cookies  # Default cookie should be preserved


class TestCreateSessionFromConfig:
    """Test cases for create_session_from_config function."""
    
    def test_default_config(self):
        """Test session creation with default config."""
        config = {}
        session = create_session_from_config(config)
        
        assert session.timeout == 30
        assert session.max_retries == 5
        assert session.min_response_size == 2000
    
    def test_custom_config(self):
        """Test session creation with custom config."""
        config = {
            "scraping": {
                "timeout_sec": 45,
                "max_retry": 7
            },
            "rate_limit": {
                "retry_delays": [0.5, 1.0, 2.0, 4.0]
            }
        }
        
        session = create_session_from_config(config)
        
        assert session.timeout == 45
        assert session.max_retries == 7
        assert session.retry_delays == [0.5, 1.0, 2.0, 4.0]
    
    def test_partial_config(self):
        """Test session creation with partial config."""
        config = {
            "scraping": {
                "timeout_sec": 20
            }
        }
        
        session = create_session_from_config(config)
        
        assert session.timeout == 20
        assert session.max_retries == 5  # default
        assert session.retry_delays == [1, 2, 4, 8, 16]  # default


class TestDefaultHeaders:
    """Test cases for DEFAULT_HEADERS."""
    
    def test_headers_present(self):
        """Test that all required headers are present."""
        required_headers = [
            "User-Agent",
            "Accept",
            "Accept-Language",
            "Accept-Encoding",
            "Connection"
        ]
        
        for header in required_headers:
            assert header in DEFAULT_HEADERS
    
    def test_user_agent_realistic(self):
        """Test that User-Agent looks realistic."""
        ua = DEFAULT_HEADERS["User-Agent"]
        assert "Mozilla" in ua
        assert "Chrome" in ua
        assert "Safari" in ua
    
    def test_accept_headers(self):
        """Test Accept headers are properly formatted."""
        assert "text/html" in DEFAULT_HEADERS["Accept"]
        assert "application/xhtml+xml" in DEFAULT_HEADERS["Accept"]
        assert "en-US" in DEFAULT_HEADERS["Accept-Language"]


if __name__ == "__main__":
    pytest.main([__file__])
