"""
Unit tests for the rate-limit scheduler module.

Tests cover:
- BackendQuota rate limiting and performance tracking
- RequestQueue priority-based queuing
- LoadBalancer backend selection
- RateLimitScheduler request processing
- ContinuousCollector high-frequency data collection
"""

import asyncio
import pytest
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch

from polymkt_sent.core.scheduler import (
    BackendQuota, ScheduledRequest, RequestQueue, LoadBalancer,
    RateLimitScheduler, ContinuousCollector, RequestPriority, BackendStatus,
    create_rate_limit_scheduler, create_continuous_collector
)
from polymkt_sent.core.scraper import ScrapingTarget
from polymkt_sent.core.backends import BackendManager, ScrapingResult
from polymkt_sent.core.storage import TweetModel, TweetStorage


@pytest.fixture
def mock_backend_manager():
    """Create a mock backend manager."""
    manager = Mock(spec=BackendManager)
    manager.get_available_backends.return_value = ["snscrape", "nitter"]
    
    # Mock successful scraping result
    mock_result = ScrapingResult(
        tweets=[
            TweetModel(
                tweet_id="123",
                user_id="test_user",
                username="test_user",
                content="Test tweet",
                timestamp=datetime.now(timezone.utc),
                likes=10,
                retweets=5,
                replies=2,
                source_instance="test"
            )
        ],
        success=True,
        backend_used="snscrape"
    )
    
    manager.scrape_user_tweets = AsyncMock(return_value=mock_result)
    manager.scrape_search_tweets = AsyncMock(return_value=mock_result)
    
    return manager


@pytest.fixture
def mock_storage():
    """Create a mock storage."""
    storage = Mock(spec=TweetStorage)
    storage.insert_batch = Mock()
    return storage


@pytest.fixture
def sample_targets():
    """Create sample scraping targets."""
    return [
        ScrapingTarget(target_type="user", value="elonmusk", weight=1.0),
        ScrapingTarget(target_type="user", value="VitalikButerin", weight=0.8),
        ScrapingTarget(target_type="search", value="bitcoin", weight=0.6),
    ]


class TestBackendQuota:
    """Test the BackendQuota class."""
    
    def test_quota_initialization(self):
        """Test quota initialization."""
        quota = BackendQuota(
            backend_name="test",
            requests_per_minute=60,
            requests_per_hour=1000,
            concurrent_requests=5
        )
        
        assert quota.backend_name == "test"
        assert quota.requests_per_minute == 60
        assert quota.requests_per_hour == 1000
        assert quota.concurrent_requests == 5
        assert quota.active_requests == 0
        assert quota.success_rate == 1.0
        assert quota.can_make_request()
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        quota = BackendQuota(
            backend_name="test",
            requests_per_minute=2,
            requests_per_hour=10,
            concurrent_requests=1
        )
        
        # Should be able to make initial requests
        assert quota.can_make_request()
        
        # Record requests up to the limit
        quota.record_request_start()
        quota.record_request_start()
        
        # Should be rate limited
        assert not quota.can_make_request()
        
        # Wait time should be calculated correctly
        wait_time = quota.get_wait_time()
        assert wait_time > 0
    
    def test_performance_tracking(self):
        """Test performance metrics tracking."""
        quota = BackendQuota(
            backend_name="test",
            requests_per_minute=60,
            requests_per_hour=1000,
            concurrent_requests=5
        )
        
        # Record successful request
        quota.record_request_start()
        quota.record_request_end(success=True, response_time=1.5)
        
        assert quota.avg_response_time == 1.5
        assert quota.success_rate == 1.0
        assert quota.consecutive_failures == 0
        assert quota.last_success is not None
        
        # Record failed request
        quota.record_request_start()
        quota.record_request_end(success=False, response_time=5.0)
        
        assert quota.consecutive_failures == 1
        assert quota.last_failure is not None
        assert quota.success_rate < 1.0
    
    def test_status_detection(self):
        """Test backend status detection."""
        quota = BackendQuota(
            backend_name="test",
            requests_per_minute=60,
            requests_per_hour=1000,
            concurrent_requests=5
        )
        
        # Initially unknown
        assert quota.get_status() == BackendStatus.UNKNOWN
        
        # After success, should be healthy
        quota.record_request_end(success=True, response_time=1.0)
        assert quota.get_status() == BackendStatus.HEALTHY
        
        # After many failures, should be failed
        for _ in range(6):
            quota.record_request_end(success=False, response_time=1.0)
        assert quota.get_status() == BackendStatus.FAILED


class TestScheduledRequest:
    """Test the ScheduledRequest class."""
    
    def test_request_creation(self):
        """Test request creation."""
        target = ScrapingTarget(target_type="user", value="test_user")
        request = ScheduledRequest(
            target=target,
            priority=RequestPriority.HIGH,
            created_at=datetime.now(timezone.utc)
        )
        
        assert request.target == target
        assert request.priority == RequestPriority.HIGH
        assert request.retry_count == 0
        assert request.can_retry
        assert not request.is_expired
    
    def test_priority_ordering(self):
        """Test priority-based ordering."""
        target = ScrapingTarget(target_type="user", value="test_user")
        
        high_req = ScheduledRequest(
            target=target,
            priority=RequestPriority.HIGH,
            created_at=datetime.now(timezone.utc)
        )
        
        low_req = ScheduledRequest(
            target=target,
            priority=RequestPriority.LOW,
            created_at=datetime.now(timezone.utc)
        )
        
        # High priority should be "less than" low priority for heap ordering
        assert high_req < low_req
    
    def test_expiration(self):
        """Test request expiration."""
        target = ScrapingTarget(target_type="user", value="test_user")
        old_time = datetime.now(timezone.utc) - timedelta(minutes=10)
        
        request = ScheduledRequest(
            target=target,
            priority=RequestPriority.NORMAL,
            created_at=old_time
        )
        
        assert request.is_expired


class TestRequestQueue:
    """Test the RequestQueue class."""
    
    @pytest.mark.asyncio
    async def test_queue_operations(self):
        """Test basic queue operations."""
        queue = RequestQueue(max_size=10)
        target = ScrapingTarget(target_type="user", value="test_user")
        
        request = ScheduledRequest(
            target=target,
            priority=RequestPriority.NORMAL,
            created_at=datetime.now(timezone.utc)
        )
        
        # Queue should be empty initially
        assert await queue.is_empty()
        assert await queue.size() == 0
        
        # Enqueue request
        success = await queue.enqueue(request)
        assert success
        assert await queue.size() == 1
        assert not await queue.is_empty()
        
        # Dequeue request
        dequeued = await queue.dequeue()
        assert dequeued == request
        assert await queue.is_empty()
    
    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        """Test priority-based dequeuing."""
        queue = RequestQueue(max_size=10)
        target = ScrapingTarget(target_type="user", value="test_user")
        
        # Enqueue requests with different priorities
        low_req = ScheduledRequest(
            target=target,
            priority=RequestPriority.LOW,
            created_at=datetime.now(timezone.utc)
        )
        
        high_req = ScheduledRequest(
            target=target,
            priority=RequestPriority.HIGH,
            created_at=datetime.now(timezone.utc)
        )
        
        normal_req = ScheduledRequest(
            target=target,
            priority=RequestPriority.NORMAL,
            created_at=datetime.now(timezone.utc)
        )
        
        # Enqueue in reverse priority order
        await queue.enqueue(low_req)
        await queue.enqueue(normal_req)
        await queue.enqueue(high_req)
        
        # Should dequeue in priority order
        first = await queue.dequeue()
        assert first.priority == RequestPriority.HIGH
        
        second = await queue.dequeue()
        assert second.priority == RequestPriority.NORMAL
        
        third = await queue.dequeue()
        assert third.priority == RequestPriority.LOW
    
    @pytest.mark.asyncio
    async def test_queue_overflow(self):
        """Test queue overflow handling."""
        queue = RequestQueue(max_size=2)
        target = ScrapingTarget(target_type="user", value="test_user")
        
        # Fill queue to capacity
        for i in range(2):
            request = ScheduledRequest(
                target=target,
                priority=RequestPriority.NORMAL,
                created_at=datetime.now(timezone.utc)
            )
            success = await queue.enqueue(request)
            assert success
        
        # Add low priority request
        low_req = ScheduledRequest(
            target=target,
            priority=RequestPriority.LOW,
            created_at=datetime.now(timezone.utc)
        )
        await queue.enqueue(low_req)
        
        # Try to add high priority request (should succeed by dropping low priority)
        high_req = ScheduledRequest(
            target=target,
            priority=RequestPriority.HIGH,
            created_at=datetime.now(timezone.utc)
        )
        success = await queue.enqueue(high_req)
        assert success


class TestLoadBalancer:
    """Test the LoadBalancer class."""
    
    def test_backend_selection(self):
        """Test backend selection logic."""
        quotas = {
            "healthy": BackendQuota("healthy", 60, 1000, 5),
            "degraded": BackendQuota("degraded", 60, 1000, 5),
            "failed": BackendQuota("failed", 60, 1000, 5)
        }
        
        # Set up different statuses
        quotas["healthy"].record_request_end(True, 1.0)  # Healthy
        quotas["degraded"].success_rate = 0.5  # Degraded
        quotas["failed"].consecutive_failures = 10  # Failed
        
        balancer = LoadBalancer(quotas)
        target = ScrapingTarget(target_type="user", value="test_user")
        request = ScheduledRequest(
            target=target,
            priority=RequestPriority.NORMAL,
            created_at=datetime.now(timezone.utc)
        )
        
        # Should select healthy backend
        selected = balancer.select_backend(request)
        assert selected == "healthy"
    
    def test_backend_preference(self):
        """Test backend preference handling."""
        quotas = {
            "preferred": BackendQuota("preferred", 60, 1000, 5),
            "other": BackendQuota("other", 60, 1000, 5)
        }
        
        # Both backends are healthy
        quotas["preferred"].record_request_end(True, 1.0)
        quotas["other"].record_request_end(True, 1.0)
        
        balancer = LoadBalancer(quotas)
        target = ScrapingTarget(target_type="user", value="test_user")
        request = ScheduledRequest(
            target=target,
            priority=RequestPriority.NORMAL,
            created_at=datetime.now(timezone.utc),
            backend_preference="preferred"
        )
        
        # Should honor preference
        selected = balancer.select_backend(request)
        assert selected == "preferred"


class TestRateLimitScheduler:
    """Test the RateLimitScheduler class."""
    
    @pytest.mark.asyncio
    async def test_scheduler_initialization(self, mock_backend_manager, mock_storage):
        """Test scheduler initialization."""
        scheduler = RateLimitScheduler(mock_backend_manager, mock_storage)
        
        assert scheduler.backend_manager == mock_backend_manager
        assert scheduler.storage == mock_storage
        assert not scheduler._running
        assert len(scheduler.quotas) == 2  # snscrape and nitter
    
    @pytest.mark.asyncio
    async def test_request_scheduling(self, mock_backend_manager, mock_storage):
        """Test request scheduling."""
        scheduler = RateLimitScheduler(mock_backend_manager, mock_storage)
        target = ScrapingTarget(target_type="user", value="test_user")
        
        success = await scheduler.schedule_request(target, RequestPriority.HIGH)
        assert success
        
        queue_size = await scheduler.request_queue.size()
        assert queue_size == 1
    
    @pytest.mark.asyncio
    async def test_scheduler_start_stop(self, mock_backend_manager, mock_storage):
        """Test scheduler start and stop."""
        scheduler = RateLimitScheduler(mock_backend_manager, mock_storage)
        
        # Start scheduler
        await scheduler.start(num_workers=2)
        assert scheduler._running
        assert len(scheduler._worker_tasks) == 2
        
        # Stop scheduler
        await scheduler.stop()
        assert not scheduler._running
        assert len(scheduler._worker_tasks) == 0
    
    @pytest.mark.asyncio
    async def test_request_processing(self, mock_backend_manager, mock_storage):
        """Test request processing."""
        scheduler = RateLimitScheduler(mock_backend_manager, mock_storage)
        target = ScrapingTarget(target_type="user", value="test_user")
        
        # Schedule a request
        await scheduler.schedule_request(target, RequestPriority.HIGH)
        
        # Start scheduler briefly to process request
        await scheduler.start(num_workers=1)
        await asyncio.sleep(0.5)  # Let it process
        await scheduler.stop()
        
        # Check that backend was called
        mock_backend_manager.scrape_user_tweets.assert_called_once()
        
        # Check that storage was called
        mock_storage.insert_batch.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_scheduler_status(self, mock_backend_manager, mock_storage):
        """Test scheduler status reporting."""
        scheduler = RateLimitScheduler(mock_backend_manager, mock_storage)
        
        status = await scheduler.get_status()
        
        assert "running" in status
        assert "queue_size" in status
        assert "backend_status" in status
        assert "stats" in status
        
        # Should have status for both backends
        assert "snscrape" in status["backend_status"]
        assert "nitter" in status["backend_status"]


class TestContinuousCollector:
    """Test the ContinuousCollector class."""
    
    @pytest.mark.asyncio
    async def test_collector_initialization(self, mock_backend_manager, mock_storage, sample_targets):
        """Test collector initialization."""
        scheduler = RateLimitScheduler(mock_backend_manager, mock_storage)
        collector = ContinuousCollector(scheduler, sample_targets, collection_interval_sec=30)
        
        assert collector.scheduler == scheduler
        assert collector.targets == sample_targets
        assert collector.collection_interval_sec == 30
        assert not collector._running
    
    @pytest.mark.asyncio
    async def test_collector_start_stop(self, mock_backend_manager, mock_storage, sample_targets):
        """Test collector start and stop."""
        scheduler = RateLimitScheduler(mock_backend_manager, mock_storage)
        collector = ContinuousCollector(scheduler, sample_targets, collection_interval_sec=1)
        
        # Start collector
        await collector.start()
        assert collector._running
        assert collector._collection_task is not None
        
        # Let it run briefly
        await asyncio.sleep(0.1)
        
        # Stop collector
        await collector.stop()
        assert not collector._running
        assert collector._collection_task is None
    
    @pytest.mark.asyncio
    async def test_target_collection_logic(self, mock_backend_manager, mock_storage, sample_targets):
        """Test target collection logic."""
        scheduler = RateLimitScheduler(mock_backend_manager, mock_storage)
        collector = ContinuousCollector(scheduler, sample_targets, collection_interval_sec=1)
        
        # Fresh targets should be collected
        fresh_target = sample_targets[0]
        assert collector._should_collect_target(fresh_target)
        
        # Recently scraped targets should be skipped
        fresh_target.last_scraped = datetime.now(timezone.utc)
        assert not collector._should_collect_target(fresh_target)
        
        # Old targets should be collected again
        old_target = sample_targets[1]
        old_target.last_scraped = datetime.now(timezone.utc) - timedelta(seconds=30)
        assert collector._should_collect_target(old_target)
    
    @pytest.mark.asyncio
    async def test_collector_stats(self, mock_backend_manager, mock_storage, sample_targets):
        """Test collector statistics."""
        scheduler = RateLimitScheduler(mock_backend_manager, mock_storage)
        collector = ContinuousCollector(scheduler, sample_targets, collection_interval_sec=30)
        
        stats = collector.get_stats()
        
        assert "running" in stats
        assert "targets_count" in stats
        assert "collection_interval_sec" in stats
        assert "stats" in stats
        
        assert stats["targets_count"] == len(sample_targets)
        assert stats["collection_interval_sec"] == 30


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_create_rate_limit_scheduler(self, mock_backend_manager, mock_storage):
        """Test scheduler creation helper."""
        custom_quotas = {
            "snscrape": {
                "requests_per_minute": 120,
                "requests_per_hour": 5000,
                "concurrent_requests": 10
            }
        }
        
        scheduler = create_rate_limit_scheduler(
            mock_backend_manager,
            mock_storage,
            custom_quotas
        )
        
        assert scheduler.backend_manager == mock_backend_manager
        assert scheduler.storage == mock_storage
        assert "snscrape" in scheduler.quotas
        assert scheduler.quotas["snscrape"].requests_per_minute == 120
    
    def test_create_continuous_collector(self, mock_backend_manager, mock_storage, sample_targets):
        """Test collector creation helper."""
        scheduler = RateLimitScheduler(mock_backend_manager, mock_storage)
        collector = create_continuous_collector(scheduler, sample_targets, interval_sec=60)
        
        assert collector.scheduler == scheduler
        assert collector.targets == sample_targets
        assert collector.collection_interval_sec == 60


class TestIntegration:
    """Integration tests for the scheduler system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_collection(self, mock_backend_manager, mock_storage, sample_targets):
        """Test end-to-end collection workflow."""
        # Create scheduler and collector
        scheduler = RateLimitScheduler(mock_backend_manager, mock_storage)
        collector = ContinuousCollector(scheduler, sample_targets, collection_interval_sec=1)
        
        # Start both
        await scheduler.start(num_workers=2)
        await collector.start()
        
        # Let them run briefly
        await asyncio.sleep(1.0)
        
        # Stop both
        await collector.stop()
        await scheduler.stop()
        
        # Verify some requests were processed
        assert mock_backend_manager.scrape_user_tweets.called or mock_backend_manager.scrape_search_tweets.called
        
        # Check stats
        scheduler_status = await scheduler.get_status()
        collector_stats = collector.get_stats()
        
        assert scheduler_status["stats"]["requests_processed"] >= 0
        assert collector_stats["stats"]["cycles_completed"] >= 0
