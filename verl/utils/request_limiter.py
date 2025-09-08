# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from typing import Optional

logger = logging.getLogger(__name__)

MAX_CONCURRENT_REQUESTS = 128

class GlobalRequestLimiter:
    """Global semaphore to limit concurrent HTTP requests across the entire system."""
    
    _instances = {}  # Map event loop to instance
    _thread_lock = threading.Lock()  # Use thread lock instead of asyncio lock
    
    def __init__(self, max_concurrent_requests: int = MAX_CONCURRENT_REQUESTS):
        self.max_concurrent = max_concurrent_requests
        self.active_requests = 0
        self.total_requests = 0
        self.waiting_requests = 0
        
        # Create semaphore in the current event loop
        try:
            self.semaphore = asyncio.Semaphore(max_concurrent_requests)
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running, create a dummy one for now
            self.semaphore = None
            self.loop = None
            
        logger.info(f"Initialized GlobalRequestLimiter with max_concurrent_requests={max_concurrent_requests}")
    
    def _ensure_semaphore(self):
        """Ensure semaphore is created in the current event loop."""
        try:
            current_loop = asyncio.get_running_loop()
            if self.semaphore is None or self.loop != current_loop:
                self.semaphore = asyncio.Semaphore(self.max_concurrent)
                self.loop = current_loop
        except RuntimeError:
            # No event loop running
            if self.semaphore is None:
                # Create a temporary semaphore
                self.semaphore = asyncio.Semaphore(self.max_concurrent)
                self.loop = None
    
    @classmethod
    async def get_instance(cls, max_concurrent_requests: int = MAX_CONCURRENT_REQUESTS) -> 'GlobalRequestLimiter':
        """Get or create the global request limiter instance for the current event loop."""
        try:
            current_loop = asyncio.get_running_loop()
            loop_id = id(current_loop)
        except RuntimeError:
            # No event loop running, use a default key
            loop_id = 'no_loop'
            
        with cls._thread_lock:
            if loop_id not in cls._instances:
                cls._instances[loop_id] = cls(max_concurrent_requests)
            return cls._instances[loop_id]
    
    async def acquire(self):
        """Acquire a request slot."""
        self._ensure_semaphore()
        self.waiting_requests += 1
        try:
            await self.semaphore.acquire()
            self.active_requests += 1
            self.total_requests += 1
        finally:
            self.waiting_requests -= 1
    
    def release(self):
        """Release a request slot."""
        self._ensure_semaphore()
        self.active_requests -= 1
        self.semaphore.release()
    
    def get_stats(self) -> dict:
        """Get current statistics."""
        self._ensure_semaphore()
        available = self.semaphore._value
        return {
            "max_concurrent": self.max_concurrent,
            "active_requests": self.active_requests,
            "waiting_requests": self.waiting_requests,
            "available_slots": available,
            "total_requests": self.total_requests,
            "utilization": f"{((self.max_concurrent - available) / self.max_concurrent * 100):.1f}%"
        }


@asynccontextmanager
async def limit_concurrent_requests(max_concurrent: int = 64):
    """Context manager for limiting concurrent HTTP requests.
    
    Usage:
        async with limit_concurrent_requests() as limiter:
            # Make HTTP request here
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    return await response.json()
    
    Args:
        max_concurrent: Maximum number of concurrent requests allowed
    """
    limiter = await GlobalRequestLimiter.get_instance(max_concurrent)
    
    # Log when requests are being queued
    if limiter.waiting_requests > 0:
        logger.debug(f"Request queued - {limiter.waiting_requests} waiting, {limiter.active_requests} active")
    
    await limiter.acquire()
    
    try:
        yield limiter
    finally:
        limiter.release()


async def get_request_stats() -> dict:
    """Get current request limiter statistics."""
    limiter = await GlobalRequestLimiter.get_instance()
    return limiter.get_stats()


async def log_request_stats_periodically(interval: int = 30):
    """Periodically log request statistics for monitoring.
    
    Args:
        interval: Logging interval in seconds
    """
    while True:
        try:
            stats = await get_request_stats()
            if stats["active_requests"] > 0 or stats["waiting_requests"] > 0:
                logger.info(f"HTTP Request Stats: {stats}")
            
            # Also log HTTP connection pool stats
            try:
                from verl.utils.http_client import log_http_stats
                await log_http_stats()
            except Exception as e:
                logger.debug(f"Failed to log HTTP connection stats: {e}")
                
        except Exception as e:
            logger.error(f"Error logging request stats: {e}")
        
        await asyncio.sleep(interval) 