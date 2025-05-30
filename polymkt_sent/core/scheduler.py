"""
Rate-limit scheduler for polymarket-sentiment.

This module provides intelligent rate limiting and request scheduling for
high-frequency data collection across multiple scraping backends.

Key Components:
- RateLimitScheduler: Core scheduling and rate limiting logic
- BackendQuota: Per-backend rate limit tracking
- RequestQueue: Priority-based request queuing
- ContinuousCollector: High-frequency collection support (30s intervals)
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import List, Dict, Any, Optional, Callable, Tuple
import heapq
from statistics import mean, median

from polymkt_sent.core.backends import BackendManager, ScrapingResult
from polymkt_sent.core.scraper import ScrapingTarget
from polymkt_sent.core.storage import TweetStorage


logger = logging.getLogger(__name__)


class RequestPriority(Enum):
    """Priority levels for scraping requests."""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0


class BackendStatus(Enum):
    """Status of a scraping backend."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    RATE_LIMITED = "rate_limited"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class BackendQuota:
    """Rate limit quota tracking for a backend."""
    backend_name: str
    requests_per_minute: int
    requests_per_hour: int
    concurrent_requests: int
    
    # Current usage tracking
    minute_requests: deque = field(default_factory=deque)
    hour_requests: deque = field(default_factory=deque)
    active_requests: int = 0
    
    # Performance tracking
    avg_response_time: float = 0.0
    success_rate: float = 1.0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    
    def __post_init__(self):
        """Initialize deques with proper setup."""
        if not isinstance(self.minute_requests, deque):
            self.minute_requests = deque()
        if not isinstance(self.hour_requests, deque):
            self.hour_requests = deque()
    
    def can_make_request(self) -> bool:
        """Check if we can make a request within rate limits."""
        now = time.time()
        
        # Clean old entries
        self._clean_old_requests(now)
        
        # Check all limits
        return (
            len(self.minute_requests) < self.requests_per_minute and
            len(self.hour_requests) < self.requests_per_hour and
            self.active_requests < self.concurrent_requests
        )
    
    def record_request_start(self) -> None:
        """Record the start of a request."""
        now = time.time()
        self.minute_requests.append(now)
        self.hour_requests.append(now)
        self.active_requests += 1
    
    def record_request_end(self, success: bool, response_time: float) -> None:
        """Record the completion of a request."""
        now = datetime.now(timezone.utc)
        logger.debug(f"Recording request end for {self.backend_name}. Before active_requests: {self.active_requests}")
        self.active_requests = max(0, self.active_requests - 1)
        logger.debug(f"After active_requests: {self.active_requests}")
        
        # Update performance metrics
        if self.avg_response_time == 0:
            self.avg_response_time = response_time
        else:
            # Exponential moving average
            self.avg_response_time = 0.8 * self.avg_response_time + 0.2 * response_time
        
        if success:
            self.last_success = now
            self.consecutive_failures = 0
            # Update success rate (exponential moving average)
            self.success_rate = 0.9 * self.success_rate + 0.1 * 1.0
        else:
            self.last_failure = now
            self.consecutive_failures += 1
            # Update success rate (exponential moving average)
            self.success_rate = 0.9 * self.success_rate + 0.1 * 0.0
    
    def get_wait_time(self) -> float:
        """Get the time to wait before next request."""
        if self.can_make_request():
            return 0.0
        
        now = time.time()
        self._clean_old_requests(now)
        
        # Calculate wait time based on minute limit (most restrictive)
        if len(self.minute_requests) >= self.requests_per_minute:
            oldest_request = self.minute_requests[0]
            wait_time = 60 - (now - oldest_request)
            return max(0, wait_time)
        
        return 0.0
    
    def get_status(self) -> BackendStatus:
        """Get the current status of this backend."""
        now = datetime.now(timezone.utc)
        logger.debug(f"Getting status for {self.backend_name}. Consecutive failures: {self.consecutive_failures}, Can make request: {self.can_make_request()}, Success rate: {self.success_rate}, Avg response time: {self.avg_response_time}, Last success: {self.last_success}")

        # Check if we have recent failures
        if self.consecutive_failures >= 5:
            return BackendStatus.FAILED
        
        # Check if we're rate limited
        if not self.can_make_request():
            return BackendStatus.RATE_LIMITED
        
        # Check if performance is degraded
        if self.success_rate < 0.7 or self.avg_response_time > 30.0:
            return BackendStatus.DEGRADED
        
        # Check if we've had recent activity
        if self.last_success is None:
            return BackendStatus.UNKNOWN
        
        return BackendStatus.HEALTHY
    
    def _clean_old_requests(self, now: float) -> None:
        """Remove old request timestamps."""
        logger.debug(f"Cleaning old requests for {self.backend_name} at time {now}. Minute queue size before: {len(self.minute_requests)}, Hour queue size before: {len(self.hour_requests)}")
        # Clean minute requests (older than 60 seconds)
        while self.minute_requests and now - self.minute_requests[0] > 60:
            self.minute_requests.popleft()
        
        # Clean hour requests (older than 3600 seconds)
        while self.hour_requests and now - self.hour_requests[0] > 3600:
            self.hour_requests.popleft()
        logger.debug(f"Minute queue size after: {len(self.minute_requests)}, Hour queue size after: {len(self.hour_requests)}")


@dataclass
class ScheduledRequest:
    """A request scheduled for execution."""
    target: ScrapingTarget
    priority: RequestPriority
    created_at: datetime
    retry_count: int = 0
    max_retries: int = 3
    backend_preference: Optional[str] = None
    
    def __lt__(self, other):
        """Support priority queue ordering."""
        # Lower priority value = higher priority
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        # Then by creation time (FIFO for same priority)
        return self.created_at < other.created_at
    
    @property
    def is_expired(self) -> bool:
        """Check if this request has expired."""
        age = datetime.now(timezone.utc) - self.created_at
        return age.total_seconds() > 300  # 5 minute expiry
    
    @property
    def can_retry(self) -> bool:
        """Check if this request can be retried."""
        return self.retry_count < self.max_retries


class RequestQueue:
    """Priority-based request queue with rate limiting support."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize the request queue."""
        self.max_size = max_size
        self._queue: List[ScheduledRequest] = []
        self._lock = asyncio.Lock()
        
        # Statistics
        self.stats = {
            "total_queued": 0,
            "total_processed": 0,
            "total_dropped": 0,
            "avg_wait_time": 0.0
        }
    
    async def enqueue(self, request: ScheduledRequest) -> bool:
        """Add a request to the queue."""
        async with self._lock:
            # Check if queue is full
            if len(self._queue) >= self.max_size:
                # Drop oldest low-priority request
                if not self._drop_low_priority_request():
                    logger.warning("Request queue full, dropping new request")
                    self.stats["total_dropped"] += 1
                    return False
            
            heapq.heappush(self._queue, request)
            self.stats["total_queued"] += 1
            logger.debug(f"Queued {request.target.target_type} '{request.target.value}' with priority {request.priority.name}")
            return True
    
    async def dequeue(self) -> Optional[ScheduledRequest]:
        """Get the next request from the queue."""
        async with self._lock:
            # Remove expired requests
            self._clean_expired_requests()
            
            if not self._queue:
                return None
            
            request = heapq.heappop(self._queue)
            self.stats["total_processed"] += 1
            
            # Calculate wait time
            wait_time = (datetime.now(timezone.utc) - request.created_at).total_seconds()
            if self.stats["avg_wait_time"] == 0:
                self.stats["avg_wait_time"] = wait_time
            else:
                self.stats["avg_wait_time"] = 0.9 * self.stats["avg_wait_time"] + 0.1 * wait_time
            
            return request
    
    async def peek(self) -> Optional[ScheduledRequest]:
        """Peek at the next request without removing it."""
        async with self._lock:
            self._clean_expired_requests()
            return self._queue[0] if self._queue else None
    
    async def size(self) -> int:
        """Get the current queue size."""
        async with self._lock:
            return len(self._queue)
    
    async def is_empty(self) -> bool:
        """Check if the queue is empty."""
        return await self.size() == 0
    
    def _drop_low_priority_request(self) -> bool:
        logger.debug(f"Attempting to drop low priority request. Queue size: {len(self._queue)}, Queue state for drop decision: {[(r.target.value, r.priority.name, r.created_at.isoformat()) for r in self._queue]}")
        if not self._queue:
            logger.debug("Queue is empty, cannot drop.")
            return False
        
        worst_request = None
        worst_request_idx = -1

        # Find the request with the numerically largest priority value (lowest actual priority)
        # If priorities are tied, pick the one created earliest (smallest created_at timestamp)
        for i, req in enumerate(self._queue):
            current_worst_details = f"(Idx: {worst_request_idx}, Prio: {worst_request.priority.name if worst_request else 'N/A'}, Created: {worst_request.created_at.isoformat() if worst_request else 'N/A'})"
            logger.debug(f"Evaluating req at index {i}: {(req.target.value, req.priority.name, req.created_at.isoformat())}. Current worst: {current_worst_details}")
            if worst_request is None:
                worst_request = req
                worst_request_idx = i
                logger.debug(f"Set initial worst_request to index {i}: {(worst_request.target.value, worst_request.priority.name)}")
            else:
                # Higher priority.value means lower actual priority (e.g., LOW.value > NORMAL.value)
                if req.priority.value > worst_request.priority.value:
                    worst_request = req
                    worst_request_idx = i
                    logger.debug(f"New worst_request (lower prio) at index {i}: {(worst_request.target.value, worst_request.priority.name)}")
                elif req.priority.value == worst_request.priority.value:
                    # Same priority, pick older one
                    if req.created_at < worst_request.created_at:
                        worst_request = req
                        worst_request_idx = i
                        logger.debug(f"New worst_request (same prio, older) at index {i}: {(worst_request.target.value, worst_request.priority.name)}")
        
        if worst_request_idx != -1:
            # Ensure the found index is valid before popping
            if worst_request_idx < len(self._queue):
                request_to_drop = self._queue[worst_request_idx]
                dropped_req_details = f"{request_to_drop.target.value} (Priority: {request_to_drop.priority.name}, Created: {request_to_drop.created_at.isoformat()})"
                logger.info(f"Queue full, identified to drop {dropped_req_details} from index {worst_request_idx}.")
                
                self._queue.pop(worst_request_idx)
                heapq.heapify(self._queue) # Re-maintain heap property
                self.stats["total_dropped"] += 1 # Increment for the dropped request
                
                logger.debug(f"Successfully dropped. Queue after drop: {[(r.target.value, r.priority.name) for r in self._queue]}")
                return True
            else:
                logger.error(f"Calculated worst_request_idx {worst_request_idx} is out of bounds for queue size {len(self._queue)}. This is a bug.")
                return False
        else:
            # This case should ideally not be reached if the queue was not empty at the start.
            logger.warning("Could not identify a request to drop (worst_request_idx is -1), though queue was not empty. This might indicate an issue.")
            return False
    
    def _clean_expired_requests(self) -> None:
        """Remove expired requests from the queue."""
        original_size = len(self._queue)
        self._queue = [req for req in self._queue if not req.is_expired]
        
        if len(self._queue) != original_size:
            heapq.heapify(self._queue)  # Restore heap property
            dropped = original_size - len(self._queue)
            self.stats["total_dropped"] += dropped
            logger.debug(f"Dropped {dropped} expired requests")


class LoadBalancer:
    """Load balancer for distributing requests across backends."""
    
    def __init__(self, quotas: Dict[str, BackendQuota]):
        """Initialize the load balancer."""
        self.quotas = quotas
        self._backend_weights = {}
        self._update_weights()
    
    def select_backend(self, request: ScheduledRequest) -> Optional[str]:
        """Select the best backend for a request."""
        # Honor backend preference if specified and available
        if request.backend_preference and request.backend_preference in self.quotas:
            quota = self.quotas[request.backend_preference]
            if quota.can_make_request() and quota.get_status() in [BackendStatus.HEALTHY, BackendStatus.DEGRADED, BackendStatus.UNKNOWN]: # MODIFIED: Added UNKNOWN
                return request.backend_preference
        
        # Find best available backend
        available_backends = []
        for backend_name, quota in self.quotas.items():
            if quota.can_make_request():
                status = quota.get_status()
                if status in [BackendStatus.HEALTHY, BackendStatus.DEGRADED, BackendStatus.UNKNOWN]: # MODIFIED: Added UNKNOWN
                    weight = self._backend_weights.get(backend_name, 0)
                    available_backends.append((backend_name, weight, status))
        
        if not available_backends:
            return None
        
        # Sort by status (healthy first) then by weight
        available_backends.sort(key=lambda x: (x[2] != BackendStatus.HEALTHY, -x[1]))
        return available_backends[0][0]
    
    def get_wait_time(self, backend: str) -> float:
        """Get wait time for a specific backend."""
        if backend not in self.quotas:
            return float('inf')
        return self.quotas[backend].get_wait_time()
    
    def get_next_available_time(self) -> Tuple[Optional[str], float]:
        """Get the backend and time for the next available slot."""
        min_wait = float('inf')
        best_backend = None
        
        for backend_name, quota in self.quotas.items():
            wait_time = quota.get_wait_time()
            if wait_time < min_wait:
                min_wait = wait_time
                best_backend = backend_name
        
        return best_backend, min_wait
    
    def _update_weights(self) -> None:
        """Update backend weights based on performance."""
        for backend_name, quota in self.quotas.items():
            # Weight based on success rate and response time
            success_weight = quota.success_rate
            time_weight = max(0, 1 - (quota.avg_response_time / 30.0))  # Normalize to 30s
            overall_weight = (success_weight + time_weight) / 2
            self._backend_weights[backend_name] = overall_weight


class RateLimitScheduler:
    """Main scheduler coordinating rate-limited requests across backends."""
    
    def __init__(
        self, 
        backend_manager: BackendManager,
        storage: TweetStorage,
        quotas: Optional[Dict[str, BackendQuota]] = None
    ):
        """Initialize the rate limit scheduler."""
        self.backend_manager = backend_manager
        self.storage = storage
        self.request_queue = RequestQueue()
        
        # Initialize backend quotas
        if quotas is None:
            quotas = self._create_default_quotas()
        self.quotas = quotas
        self.load_balancer = LoadBalancer(quotas)
        
        # Control flags
        self._running = False
        self._shutdown = False
        self._worker_tasks: List[asyncio.Task] = []
        
        # Statistics
        self.stats = {
            "requests_processed": 0,
            "requests_failed": 0,
            "backends_used": defaultdict(int),
            "avg_processing_time": 0.0,
            "start_time": None
        }
        
        logger.info(f"RateLimitScheduler initialized with {len(quotas)} backends")
    
    def _create_default_quotas(self) -> Dict[str, BackendQuota]:
        """Create default quotas for available backends."""
        quotas = {}
        available_backends = self.backend_manager.get_available_backends()
        
        for backend in available_backends:
            if backend == "snscrape":
                # SNSScrape can handle more requests
                quotas[backend] = BackendQuota(
                    backend_name=backend,
                    requests_per_minute=100,
                    requests_per_hour=3000,
                    concurrent_requests=10
                )
            elif backend == "nitter":
                # Nitter is more conservative
                quotas[backend] = BackendQuota(
                    backend_name=backend,
                    requests_per_minute=30,
                    requests_per_hour=1000,
                    concurrent_requests=5
                )
            else:
                # Default conservative limits
                quotas[backend] = BackendQuota(
                    backend_name=backend,
                    requests_per_minute=20,
                    requests_per_hour=500,
                    concurrent_requests=3
                )
        
        return quotas
    
    async def schedule_request(
        self,
        target: ScrapingTarget,
        priority: RequestPriority = RequestPriority.NORMAL,
        backend_preference: Optional[str] = None
    ) -> bool:
        """Schedule a scraping request."""
        request = ScheduledRequest(
            target=target,
            priority=priority,
            created_at=datetime.now(timezone.utc),
            backend_preference=backend_preference
        )
        
        success = await self.request_queue.enqueue(request)
        if success:
            logger.debug(f"Scheduled {target.target_type} '{target.value}' with priority {priority.name}")
        else:
            logger.warning(f"Failed to schedule request for {target.target_type} '{target.value}'")
        
        return success
    
    async def start(self, num_workers: int = 3) -> None:
        """Start the scheduler with worker tasks."""
        if self._running:
            logger.warning("Scheduler already running")
            return
        
        self._running = True
        self._shutdown = False
        self.stats["start_time"] = datetime.now(timezone.utc)
        
        # Start worker tasks
        for i in range(num_workers):
            task = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self._worker_tasks.append(task)
        
        logger.info(f"RateLimitScheduler started with {num_workers} workers")
    
    async def stop(self) -> None:
        """Stop the scheduler and wait for workers to finish."""
        if not self._running:
            return
        
        logger.info("Stopping RateLimitScheduler...")
        self._shutdown = True
        
        # Wait for workers to finish
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
            self._worker_tasks.clear()
        
        self._running = False
        logger.info("RateLimitScheduler stopped")
    
    async def _worker_loop(self, worker_name: str) -> None:
        """Main worker loop for processing requests."""
        logger.debug(f"Worker {worker_name} started")
        
        while not self._shutdown:
            try:
                # Get next request
                request = await self.request_queue.dequeue()
                if request is None:
                    # No requests available, wait briefly
                    await asyncio.sleep(0.1)
                    continue
                
                # Process the request
                await self._process_request(request, worker_name)
                
            except asyncio.CancelledError:
                logger.debug(f"Worker {worker_name} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                await asyncio.sleep(1)  # Brief pause on error
        
        logger.debug(f"Worker {worker_name} stopped")
    
    async def _process_request(self, request: ScheduledRequest, worker_name: str) -> None:
        """Process a single scraping request."""
        start_time = time.time()
        
        # Select backend
        backend = self.load_balancer.select_backend(request)
        if backend is None:
            # No backend available, requeue with retry
            if request.can_retry:
                request.retry_count += 1
                await self.request_queue.enqueue(request)
                logger.warning(f"No backend available for {request.target.value}, requeued (retry {request.retry_count})")
            else:
                logger.error(f"No backend available for {request.target.value}, dropping request")
                self.stats["requests_failed"] += 1
            return
        
        # Get quota and wait if needed
        quota = self.quotas[backend]
        wait_time = quota.get_wait_time()
        if wait_time > 0:
            logger.debug(f"Worker {worker_name} waiting {wait_time:.1f}s for {backend}")
            await asyncio.sleep(wait_time)
        
        # Record request start
        quota.record_request_start()
        
        try:
            # Execute the request
            if request.target.is_user:
                result = await self.backend_manager.scrape_user_tweets(
                    request.target.value,
                    max_tweets=100
                )
            else:
                result = await self.backend_manager.scrape_search_tweets(
                    request.target.value,
                    max_tweets=100
                )
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Record result
            success = result.success
            quota.record_request_end(success, response_time)
            
            if success:
                # Store tweets
                if result.tweets:
                    await asyncio.to_thread(self.storage.insert_batch, result.tweets)
                    logger.info(f"Processed {len(result.tweets)} tweets from {request.target.value} using {backend}")
                
                # Update stats
                self.stats["requests_processed"] += 1
                self.stats["backends_used"][backend] += 1
                
                # Update average processing time
                if self.stats["avg_processing_time"] == 0:
                    self.stats["avg_processing_time"] = response_time
                else:
                    self.stats["avg_processing_time"] = (
                        0.9 * self.stats["avg_processing_time"] + 0.1 * response_time
                    )
                
                # Update target timestamp
                request.target.last_scraped = datetime.now(timezone.utc)
                
            else:
                # Handle failure
                if request.can_retry:
                    request.retry_count += 1
                    await self.request_queue.enqueue(request)
                    logger.warning(f"Request failed for {request.target.value}, requeued (retry {request.retry_count})")
                else:
                    logger.error(f"Request failed for {request.target.value}, dropping after {request.retry_count} retries")
                    self.stats["requests_failed"] += 1
        
        except Exception as e:
            response_time = time.time() - start_time
            quota.record_request_end(False, response_time)
            
            # Handle retry
            if request.can_retry:
                request.retry_count += 1
                await self.request_queue.enqueue(request)
                logger.warning(f"Exception processing {request.target.value}: {e}, requeued (retry {request.retry_count})")
            else:
                logger.error(f"Exception processing {request.target.value}: {e}, dropping after {request.retry_count} retries")
                self.stats["requests_failed"] += 1
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        queue_size = await self.request_queue.size()
        
        backend_status = {}
        for backend, quota in self.quotas.items():
            backend_status[backend] = {
                "status": quota.get_status().value,
                "success_rate": quota.success_rate,
                "avg_response_time": quota.avg_response_time,
                "active_requests": quota.active_requests,
                "minute_requests": len(quota.minute_requests),
                "hour_requests": len(quota.hour_requests),
                "can_make_request": quota.can_make_request(),
                "wait_time": quota.get_wait_time()
            }
        
        return {
            "running": self._running,
            "queue_size": queue_size,
            "workers_active": len(self._worker_tasks),
            "stats": self.stats,
            "backend_status": backend_status,
            "queue_stats": self.request_queue.stats
        }


class ContinuousCollector:
    """Continuous high-frequency data collector (30-second intervals)."""
    
    def __init__(
        self,
        scheduler: RateLimitScheduler,
        targets: List[ScrapingTarget],
        collection_interval_sec: int = 30
    ):
        """Initialize continuous collector."""
        self.scheduler = scheduler
        self.targets = targets
        self.collection_interval_sec = collection_interval_sec
        
        # Control flags
        self._running = False
        self._collection_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            "cycles_completed": 0,
            "total_requests_scheduled": 0,
            "last_cycle_time": None,
            "avg_cycle_duration": 0.0
        }
        
        logger.info(f"ContinuousCollector initialized with {len(targets)} targets, {collection_interval_sec}s interval")
    
    async def start(self) -> None:
        """Start continuous collection."""
        if self._running:
            logger.warning("ContinuousCollector already running")
            return
        
        self._running = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        logger.info("ContinuousCollector started")
    
    async def stop(self) -> None:
        """Stop continuous collection."""
        if not self._running:
            return
        
        logger.info("Stopping ContinuousCollector...")
        self._running = False
        
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
            self._collection_task = None
        
        logger.info("ContinuousCollector stopped")
    
    async def _collection_loop(self) -> None:
        """Main collection loop."""
        logger.info("Starting continuous collection loop")
        
        while self._running:
            cycle_start = time.time()
            
            try:
                # Schedule all targets for collection
                scheduled_count = 0
                for target in self.targets:
                    # Check if target needs fresh data
                    if self._should_collect_target(target):
                        success = await self.scheduler.schedule_request(
                            target,
                            priority=RequestPriority.HIGH  # High priority for continuous collection
                        )
                        if success:
                            scheduled_count += 1
                
                # Update statistics
                cycle_duration = time.time() - cycle_start
                self.stats["cycles_completed"] += 1
                self.stats["total_requests_scheduled"] += scheduled_count
                self.stats["last_cycle_time"] = datetime.now(timezone.utc)
                
                if self.stats["avg_cycle_duration"] == 0:
                    self.stats["avg_cycle_duration"] = cycle_duration
                else:
                    self.stats["avg_cycle_duration"] = (
                        0.9 * self.stats["avg_cycle_duration"] + 0.1 * cycle_duration
                    )
                
                logger.debug(f"Collection cycle completed: {scheduled_count} requests scheduled in {cycle_duration:.1f}s")
                
                # Wait for next cycle
                sleep_time = max(0, self.collection_interval_sec - cycle_duration)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                logger.debug("Collection loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                await asyncio.sleep(5)  # Brief pause on error
        
        logger.info("Collection loop stopped")
    
    def _should_collect_target(self, target: ScrapingTarget) -> bool:
        """Check if a target should be collected based on freshness."""
        if target.last_scraped is None:
            return True
        
        age = datetime.now(timezone.utc) - target.last_scraped
        return age.total_seconds() >= self.collection_interval_sec
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics."""
        return {
            "running": self._running,
            "targets_count": len(self.targets),
            "collection_interval_sec": self.collection_interval_sec,
            "stats": self.stats
        }


# Helper functions for creating and configuring schedulers

def create_rate_limit_scheduler(
    backend_manager: BackendManager,
    storage: TweetStorage,
    custom_quotas: Optional[Dict[str, Dict[str, int]]] = None
) -> RateLimitScheduler:
    """Create a pre-configured rate limit scheduler."""
    quotas = {}
    
    if custom_quotas:
        for backend, limits in custom_quotas.items():
            quotas[backend] = BackendQuota(
                backend_name=backend,
                requests_per_minute=limits.get("requests_per_minute", 30),
                requests_per_hour=limits.get("requests_per_hour", 1000),
                concurrent_requests=limits.get("concurrent_requests", 5)
            )
    
    scheduler = RateLimitScheduler(backend_manager, storage, quotas)
    return scheduler


def create_continuous_collector(
    scheduler: RateLimitScheduler,
    targets: List[ScrapingTarget],
    interval_sec: int = 30
) -> ContinuousCollector:
    """Create a continuous collector for high-frequency data collection."""
    return ContinuousCollector(scheduler, targets, interval_sec)
