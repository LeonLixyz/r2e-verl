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
from typing import Dict, Optional
import aiohttp
import requests

from verl.utils.request_limiter import limit_concurrent_requests, log_request_stats_periodically
from verl.utils.http_client import http_request

logger = logging.getLogger(__name__)
TIMEOUT = 120


def _can_use_async_http() -> bool:
    """Check if we can safely use async HTTP context managers."""
    try:
        # Check if there's a running event loop
        loop = asyncio.get_running_loop()
        
        # Check if we're in a proper task context
        current_task = asyncio.current_task(loop)
        return current_task is not None
    except RuntimeError:
        # No event loop running
        return False


class ToolboxSessionManager:
    """Manages toolbox API sessions with automatic lifecycle management."""
    
    def __init__(self, base_url: str = "https://toolbox.modal-origin.relace.run", 
                 api_key: str = "rlc-rtISTILowpUiVcLJjFHUmaYUJ3Y3C9ftGhO-wg",
                 enable_request_monitoring: bool = False,
                 default_image_tag: str = None):
        self.base_url = base_url
        # self.base_url = "https://toolbox-preston.modal-origin.relace.run"
        self.api_key = api_key
        self.default_image_tag = default_image_tag
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Maps conversation_id -> session_id
        self._conversation_sessions: Dict[str, str] = {}
        # Track active sessions for cleanup
        self._active_sessions: Dict[str, bool] = {}
        
        # Start request monitoring if enabled
        if enable_request_monitoring:
            self._start_request_monitoring()
    
    def _start_request_monitoring(self):
        """Start background task to monitor request statistics."""
        try:
            # Start monitoring in the background
            asyncio.create_task(log_request_stats_periodically(interval=30))
            logger.info("Started HTTP request monitoring (logging every 30 seconds)")
        except Exception as e:
            logger.warning(f"Failed to start request monitoring: {e}")
        
    async def get_session_for_conversation(self, conversation_id: str, image_tag: str = None) -> str:
        """Get or create a session for a conversation."""
        # print(f"[SESSION_MANAGER_DEBUG] get_session_for_conversation: conversation_id={conversation_id}, image_tag={image_tag}")
        if conversation_id in self._conversation_sessions:
            session_id = self._conversation_sessions[conversation_id]
            # print(f"Using existing session {session_id} for conversation {conversation_id}")
            return session_id
            
        # Create new session with specified or default image tag
        effective_image_tag = image_tag or self.default_image_tag
        print(f"[SESSION_MANAGER_DEBUG] creating conversation {conversation_id} with image_tag {effective_image_tag}")
        session_id = await self._create_session(image_tag=effective_image_tag)
        self._conversation_sessions[conversation_id] = session_id
        self._active_sessions[session_id] = True
        
        # print(f"Created new session {session_id} for conversation {conversation_id}")
        return session_id
        
    async def _create_session(self, image_tag: str = None) -> str:
        """Create a new toolbox session."""
        start_time = time.time()
        
        # Prepare request body
        request_body = {}
        # print(f"[SESSION_MANAGER_DEBUG] _create_session called with image_tag: {image_tag}")
        if image_tag:
            request_body["image_tag"] = image_tag
            print(f"Creating toolbox session with image_tag: {image_tag}")
        else:
            print(f"[SESSION_MANAGER_DEBUG] No image_tag provided, creating session with default parameters")
        
        try:
            # Check if we can use async HTTP safely
            if _can_use_async_http():
                # Use async HTTP client
                async with limit_concurrent_requests():
                    async with http_request('POST', f"{self.base_url}/session/", 
                                           headers=self.headers, 
                                           json=request_body if request_body else None,
                                           timeout=TIMEOUT) as response:
                        if response.status == 200:
                            data = await response.json()
                            session_id = data["session_id"]
                            duration = time.time() - start_time
                            print(f"Created toolbox session: {session_id} in {duration:.2f} seconds")
                            print(f"[SESSION_MANAGER_DEBUG] Session created successfully: {session_id} with image_tag: {image_tag}")
                            return session_id
                        else:
                            error_text = await response.text()
                            raise Exception(f"Failed to create session (status {response.status}): {error_text}")
            else:
                # Fall back to sync requests
                logger.debug("Using sync HTTP client for session creation due to async context issues")
                response = requests.post(
                    f"{self.base_url}/session/",
                    headers=self.headers,
                    json=request_body if request_body else None,
                    timeout=TIMEOUT
                )
                if response.status_code == 200:
                    data = response.json()
                    session_id = data["session_id"]
                    duration = time.time() - start_time
                    print(f"Created toolbox session: {session_id} in {duration:.2f} seconds (sync)")
                    # print(f"[SESSION_MANAGER_DEBUG] Session created successfully (sync): {session_id}")
                    return session_id
                else:
                    raise Exception(f"Failed to create session (status {response.status_code}): {response.text}")
                    
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error creating toolbox session: {type(e).__name__}: {e} after {duration:.2f} seconds")
            raise
            
    def cleanup_conversation_sync(self, conversation_id: str) -> None:
        """Synchronous cleanup for a conversation that doesn't attempt HTTP requests."""
        if conversation_id not in self._conversation_sessions:
            return
            
        session_id = self._conversation_sessions[conversation_id]
        
        # Call the sync deletion method
        try:
            self._delete_session_sync(session_id)
        except Exception as e:
            print(f"[SESSION DEBUG] Error in _delete_session_sync for session {session_id}: {e}")
        
        # Always remove from tracking, even if deletion failed
        logger.debug(f"Clearing session {session_id} for conversation {conversation_id}")
        
        del self._conversation_sessions[conversation_id]
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]
            
    async def cleanup_conversation(self, conversation_id: str) -> None:
        """Clean up session for a conversation."""
        if conversation_id not in self._conversation_sessions:
            return
            
        session_id = self._conversation_sessions[conversation_id]

        # Always use sync cleanup to avoid async context issues during cleanup
        logger.debug(f"Using sync cleanup for session {session_id} for conversation {conversation_id}")
        
        try:
            self._delete_session_sync(session_id)
        except Exception as e:
            logger.error(f"Error cleaning up session {session_id} for conversation {conversation_id}: {e}")
        finally:
            # Always remove from tracking, even if deletion failed
            if conversation_id in self._conversation_sessions:
                del self._conversation_sessions[conversation_id]
            if session_id in self._active_sessions:
                del self._active_sessions[session_id]
                
    def _delete_session_sync(self, session_id: str) -> None:
        """Delete a toolbox session using sync requests with retry logic."""
        start_time = time.time()
        max_retries = 3
        timeout = TIMEOUT  # Reduced from 60s to fail faster
        
        for attempt in range(max_retries):
            try:
                response = requests.delete(
                    f"{self.base_url}/session/{session_id}",
                    headers=self.headers,
                    timeout=timeout
                )
                duration = time.time() - start_time
                
                if response.status_code == 204:
                    print(f"[SESSION] Successfully deleted session: {session_id} in {duration:.2f}s")
                    logger.debug(f"Deleted toolbox session: {session_id} in {duration:.2f}s")
                    return  # Success, exit retry loop
                else:
                    print(f"[SESSION DEBUG] Failed to delete session {session_id} - Status: {response.status_code} (attempt {attempt + 1}/{max_retries})")
                    if attempt == max_retries - 1:  # Last attempt
                        print(f"[SESSION DEBUG] Response text: {response.text}")
                    continue  # Try again
                    
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                duration = time.time() - start_time
                if attempt < max_retries - 1:
                    print(f"[SESSION DEBUG] Network error deleting session {session_id} after {duration:.3f}s (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(1)  # Brief pause before retry
                    continue
                else:
                    print(f"[SESSION DEBUG] Final network error deleting session {session_id} after {duration:.3f}s: {e}")
                    return
                    
            except Exception as e:
                duration = time.time() - start_time
                print(f"[SESSION DEBUG] Unexpected error deleting session {session_id} after {duration:.3f}s: {e}")
                return  # Don't retry for unexpected errors
            
    async def _delete_session(self, session_id: str) -> None:
        """Delete a toolbox session using async requests."""
        start_time = time.time()
        try:
            # Check if we can use async HTTP safely
            if _can_use_async_http():
                async with limit_concurrent_requests():
                    async with http_request('DELETE', f"{self.base_url}/session/{session_id}", 
                                           headers=self.headers, timeout=TIMEOUT) as response:
                        if response.status == 204:
                            logger.debug(f"Deleted toolbox session: {session_id}")
                        else:
                            error_text = await response.text()
                            logger.warning(f"Failed to delete session {session_id} (status {response.status}): {error_text}")
            else:
                # Fall back to sync
                self._delete_session_sync(session_id)
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error deleting toolbox session {session_id}: {type(e).__name__}: {e} after {duration:.2f} seconds")
            
    def cleanup_sessions_sync(self) -> None:
        """Synchronous cleanup that doesn't attempt HTTP requests."""
        logger.debug(f"Clearing {len(self._active_sessions)} sessions without HTTP cleanup")
        self._conversation_sessions.clear()
        self._active_sessions.clear()
            
    async def cleanup_all_sessions(self) -> None:
        """Clean up all active sessions using sync approach to avoid async context issues."""
        logger.debug(f"Cleaning up {len(self._active_sessions)} sessions using sync approach")
        
        cleanup_errors = []
        for session_id in list(self._active_sessions.keys()):
            try:
                self._delete_session_sync(session_id)
            except Exception as e:
                cleanup_errors.append(f"Error cleaning up session {session_id}: {e}")
                logger.error(f"Error cleaning up session {session_id}: {e}")
                
        self._conversation_sessions.clear()
        self._active_sessions.clear()
        
        if cleanup_errors:
            logger.warning(f"Session cleanup completed with {len(cleanup_errors)} errors")
        else:
            logger.info("Cleaned up all active toolbox sessions")


# Global session manager instance
_session_manager: Optional[ToolboxSessionManager] = None


def get_session_manager(default_image_tag: str = None) -> ToolboxSessionManager:
    """Get the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = ToolboxSessionManager(default_image_tag=default_image_tag)
    return _session_manager


async def cleanup_session_manager():
    """Clean up the global session manager."""
    global _session_manager
    if _session_manager:
        try:
            # Always use sync cleanup to avoid async context issues
            logger.debug("Using sync cleanup for session manager to avoid async context issues")
            _session_manager.cleanup_sessions_sync()
        except Exception as e:
            logger.error(f"Error during session manager cleanup: {e}")
        finally:
            _session_manager = None