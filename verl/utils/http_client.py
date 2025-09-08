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
import time
from contextlib import asynccontextmanager
from typing import Dict, Optional, Any
import aiohttp

logger = logging.getLogger(__name__)


class SharedHTTPClient:
    """Shared HTTP client with optimized connection pooling for high concurrency."""
    
    _instance: Optional['SharedHTTPClient'] = None
    _lock = asyncio.Lock()
    
    def __init__(self, 
                 max_connections: int = 1000,
                 max_connections_per_host: int = 200,
                 keepalive_timeout: int = 30,
                 timeout_total: int = 300):
        """Initialize shared HTTP client with optimized settings.
        
        Args:
            max_connections: Total connection pool size (increased from default 100)
            max_connections_per_host: Per-host connection limit (increased from default 30)  
            keepalive_timeout: Connection keepalive timeout in seconds
            timeout_total: Default request timeout in seconds
        """
        # Configure custom DNS resolver to use 8.8.8.8
        resolver = aiohttp.resolver.AsyncResolver(nameservers=['8.8.8.8'])
        
        # Configure connector with optimized settings for high concurrency and custom DNS
        connector = aiohttp.TCPConnector(
            limit=max_connections,                    # Total connection pool size
            limit_per_host=max_connections_per_host,  # Per-host connection limit
            keepalive_timeout=keepalive_timeout,      # Keep connections alive
            enable_cleanup_closed=True,               # Clean up closed connections
            ttl_dns_cache=300,                        # DNS cache TTL (5 minutes)
            use_dns_cache=True,                       # Enable DNS caching
            resolver=resolver,                        # Use custom DNS resolver with 8.8.8.8
        )
        
        # Configure default timeout
        timeout = aiohttp.ClientTimeout(total=timeout_total)
        
        # Create the session
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            raise_for_status=False,  # Don't auto-raise on HTTP errors
        )
        
        self.max_connections = max_connections
        self.max_connections_per_host = max_connections_per_host
        
        logger.info(f"SharedHTTPClient initialized: max_connections={max_connections}, "
                   f"max_connections_per_host={max_connections_per_host}, DNS=8.8.8.8")
    
    @classmethod
    async def get_instance(cls, **kwargs) -> 'SharedHTTPClient':
        """Get or create the shared HTTP client instance."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(**kwargs)
        return cls._instance
    
    @classmethod
    async def close_instance(cls):
        """Close the shared HTTP client instance."""
        if cls._instance is not None:
            async with cls._lock:
                if cls._instance is not None:
                    await cls._instance.session.close()
                    cls._instance = None
                    logger.info("SharedHTTPClient instance closed")
    
    async def post(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make a POST request using the shared session."""
        return await self.session.post(url, **kwargs)
    
    async def get(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make a GET request using the shared session."""
        return await self.session.get(url, **kwargs)
    
    async def delete(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make a DELETE request using the shared session."""
        return await self.session.delete(url, **kwargs)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        connector = self.session.connector
        
        # Count idle connections (in _conns)
        idle_connections = sum(len(conns) for conns in connector._conns.values())
        
        # Count connections being acquired (in _acquired_per_host)
        acquiring_connections = sum(len(conns) for conns in getattr(connector, '_acquired_per_host', {}).values())
        
        # Total connections includes both idle and acquiring
        total_connections = idle_connections + acquiring_connections
        
        stats = {
            "idle_connections": idle_connections,
            "acquiring_connections": acquiring_connections,
            "total_connections": total_connections,
            "max_connections": self.max_connections,
            "max_connections_per_host": self.max_connections_per_host,
        }
        
        # Add per-host connection counts (idle)
        host_connections = {}
        for key, conns in connector._conns.items():
            host_key = f"{key.host}:{key.port}"
            host_connections[host_key] = len(conns)
        stats["idle_per_host"] = host_connections
        
        # Add per-host acquiring counts
        host_acquiring = {}
        acquired_per_host = getattr(connector, '_acquired_per_host', {})
        for key, conns in acquired_per_host.items():
            host_key = f"{key.host}:{key.port}"
            host_acquiring[host_key] = len(conns)
        stats["acquiring_per_host"] = host_acquiring
        
        return stats


@asynccontextmanager
async def http_request(method: str, url: str, timeout: Optional[float] = None, **kwargs):
    """Context manager for making HTTP requests with the shared client.
    
    Usage:
        async with http_request('POST', url, json=data) as response:
            result = await response.json()
    
    Args:
        method: HTTP method (GET, POST, DELETE, etc.)
        url: Request URL
        timeout: Override timeout for this request
        **kwargs: Additional arguments passed to aiohttp
    """
    client = await SharedHTTPClient.get_instance()
    
    # Override timeout if specified
    if timeout is not None:
        kwargs['timeout'] = aiohttp.ClientTimeout(total=timeout)
    
    # Use the session directly instead of the wrapper methods
    method_lower = method.lower()
    if method_lower == 'post':
        async with client.session.post(url, **kwargs) as response:
            yield response
    elif method_lower == 'get':
        async with client.session.get(url, **kwargs) as response:
            yield response
    elif method_lower == 'delete':
        async with client.session.delete(url, **kwargs) as response:
            yield response
    elif method_lower == 'put':
        async with client.session.put(url, **kwargs) as response:
            yield response
    elif method_lower == 'patch':
        async with client.session.patch(url, **kwargs) as response:
            yield response
    else:
        # Fallback to request method for other HTTP methods
        async with client.session.request(method, url, **kwargs) as response:
            yield response


async def log_http_stats():
    """Log HTTP client connection statistics."""
    try:
        client = await SharedHTTPClient.get_instance()
        stats = client.get_connection_stats()
        
        total_conns = stats["total_connections"]
        max_conns = stats["max_connections"]
        utilization = (total_conns / max_conns) * 100 if max_conns > 0 else 0
        
        logger.info(f"HTTP Connection Pool: {total_conns}/{max_conns} ({utilization:.1f}% utilized)")
        
        # Log per-host stats if there are many connections
        if total_conns > 10:
            host_stats = stats["connections_per_host"]
            top_hosts = sorted(host_stats.items(), key=lambda x: x[1], reverse=True)[:3]
            if top_hosts:
                host_summary = ", ".join([f"{host}: {count}" for host, count in top_hosts])
                logger.info(f"Top connection hosts: {host_summary}")
                
    except Exception as e:
        logger.warning(f"Failed to log HTTP stats: {e}")


# Convenience functions for common HTTP methods
async def post_json(url: str, data: Dict[str, Any], headers: Optional[Dict[str, str]] = None, 
                   timeout: Optional[float] = None) -> str:
    """Make a POST request with JSON data and return response text."""
    async with http_request('POST', url, json=data, headers=headers, timeout=timeout) as response:
        return await response.text()


async def get_json(url: str, headers: Optional[Dict[str, str]] = None, 
                  timeout: Optional[float] = None) -> Dict[str, Any]:
    """Make a GET request and return JSON response."""
    async with http_request('GET', url, headers=headers, timeout=timeout) as response:
        return await response.json() 