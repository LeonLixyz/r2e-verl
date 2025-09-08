import json
import os
import time
import subprocess
import asyncio
import aiohttp
from typing import Any, Optional, Tuple, Dict
from uuid import uuid4

from .base_tool import BaseTool, OpenAIFunctionToolSchema
from .session_manager import get_session_manager
from verl.utils.request_limiter import limit_concurrent_requests

TIMEOUT = 120
BASH_TIMEOUT = 120  # 3 minutes for bash commands
MAX_RESULT_LENGTH = 5000  # Maximum length of results returned by tools


class APIBasedTool(BaseTool):
    """Base class for all API-based tools."""
    
    tool_name: str = None  # Must be set by subclasses
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self._http_session = None
    
    async def _get_http_session(self) -> aiohttp.ClientSession:
        """Get or create the shared HTTP session like session manager."""
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()
        return self._http_session

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {}
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        """Execute tool via API call"""
        start_time = time.time()
        try:
            # Get conversation ID from kwargs and determine session ID
            conversation_id = kwargs.get('conversation_id', 'default')
            image_tag = kwargs.get('image_tag', None)
            session_manager = get_session_manager()
            session_id = await session_manager.get_session_for_conversation(conversation_id, image_tag=image_tag)
            
            # Format as your endpoint expects: {"name": "tool_name", "arguments": {...}}
            api_payload = {
                "name": self.tool_name,
                "arguments": parameters
            }
            
            endpoint = f"https://toolbox.modal-origin.relace.run/session/{session_id}/tool/"
            
            print(f"[DEBUG] {self.__class__.__name__} SENDING TO {endpoint} at time {time.time()}")
            
            headers = {"Authorization": "Bearer rlc-rtISTILowpUiVcLJjFHUmaYUJ3Y3C9ftGhO-wg", "Content-Type": "application/json"}
            
            # Use the same pattern as session manager
            http_session = await self._get_http_session()
            
            timeout_val = BASH_TIMEOUT if self.tool_name == "bash" else TIMEOUT
            
            async with limit_concurrent_requests():
                async with http_session.post(
                    endpoint, 
                    json=api_payload, 
                    headers=headers, 
                    timeout=aiohttp.ClientTimeout(total=timeout_val)
                ) as response:
                    result = await response.text()
                    status_code = response.status
                    
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    
                    if status_code == 200:
                        print(f"[DEBUG] {self.__class__.__name__} API call successful for session {session_id} - Elapsed time: {elapsed_time:.3f}s - status code: {status_code} - result: {result[:100]}")
                        # Truncate result to maximum length
                        truncated_result = result[:MAX_RESULT_LENGTH]
                        if len(result) > MAX_RESULT_LENGTH:
                            truncated_result += f"\n... [Result truncated. Total length: {len(result)} chars, showing first {MAX_RESULT_LENGTH} chars]"
                        return truncated_result, 0.0, {"status_code": status_code}
                    else:
                        print(f"[DEBUG] {self.__class__.__name__} API call failed for session {session_id} - Elapsed time: {elapsed_time:.3f}s - status code: {status_code} - result: {result[:100]}")
                        # Truncate error result to maximum length
                        error_result = f"API Error (status {status_code}): {result}"
                        truncated_error = error_result[:MAX_RESULT_LENGTH]
                        if len(error_result) > MAX_RESULT_LENGTH:
                            truncated_error += f"\n... [Error truncated. Total length: {len(error_result)} chars, showing first {MAX_RESULT_LENGTH} chars]"
                        return truncated_error, 0.0, {"status_code": status_code}
                        
        except Exception as e:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"[DEBUG] {self.__class__.__name__} API call failed - Elapsed time: {elapsed_time:.3f}s: {e}")
            return f"Error: {str(e)}", 0.0, {}

    async def release(self, instance_id: str, **kwargs):
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


class ViewFileTool(APIBasedTool):
    """Tool for viewing/exploring the contents of existing files and directories."""
    tool_name = "view_file"


class EditFileTool(APIBasedTool):
    """Tool for editing existing files or creating new files."""
    tool_name = "edit_file"


class DeleteFilesTool(APIBasedTool):
    """Tool for deleting multiple files or directories."""
    tool_name = "delete_files"


class BashTool(APIBasedTool):
    """Tool for running bash commands."""
    tool_name = "bash"
    
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"cwd": os.getcwd()}
        return instance_id


class GrepSearchTool(APIBasedTool):
    """Tool for fast text-based regex search using ripgrep."""
    tool_name = "grep_search"


class UndoTool(APIBasedTool):
    """Tool for undoing the most recent change to tracked files."""
    tool_name = "undo"


class SearchAndReplaceTool(APIBasedTool):
    """Tool for searching and replacing text in files."""
    tool_name = "search_and_replace"


class VerifierTool(BaseTool):
    """Tool for verifying solutions by running test cases. This tool is NOT exposed to the LLM."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        # Shared aiohttp session like session manager
        self._http_session = None
    
    async def _get_http_session(self) -> aiohttp.ClientSession:
        """Get or create the shared HTTP session like session manager."""
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()
        return self._http_session

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {}
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        """Execute verification by running test cases on the generated solution file.
        
        Expected parameters:
        - file_path: str - Path to the Python file containing the solution
        - test_input: str or dict - Test input data to verify the solution
        - timeout: int - Timeout in seconds (optional, default 30)
        - session_id: str - Session ID to use for verification (optional, will be extracted from conversation_id if not provided)
        """
        start_time = time.time()
        try:
            # Get session ID - either from parameters or from conversation_id
            session_id = parameters.get('session_id')
            if not session_id:
                conversation_id = kwargs.get('conversation_id', 'default')
                image_tag = kwargs.get('image_tag', None)
                session_manager = get_session_manager()
                session_id = await session_manager.get_session_for_conversation(conversation_id, image_tag=image_tag)
            
            # Get timeout from parameters (only for HTTP timeout, not API payload)
            exec_timeout = parameters.get("timeout", TIMEOUT)
            
            # Format verification request - this will run the solution against test cases
            api_payload = {
                "path": parameters.get("file_path"),
                "input": parameters.get("test_input")
            }
            
            endpoint = f"https://toolbox.modal-origin.relace.run/session/{session_id}/exec/"
            # HTTP timeout should be longer than execution timeout
            http_timeout = exec_timeout + TIMEOUT
            
            print(f"[DEBUG] VerifierTool SENDING TO {endpoint} at time {time.time()}")
            
            headers = {"Authorization": "Bearer rlc-rtISTILowpUiVcLJjFHUmaYUJ3Y3C9ftGhO-wg", "Content-Type": "application/json"}
            
            # Use the same pattern as session manager
            http_session = await self._get_http_session()
            
            async with limit_concurrent_requests():
                async with http_session.post(
                    endpoint, 
                    json=api_payload, 
                    headers=headers, 
                    timeout=aiohttp.ClientTimeout(total=http_timeout)
                ) as response:
                    result = await response.text()
                    status_code = response.status
                    
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    
                    print(f"[DEBUG] VerifierTool API response (status {status_code}): {result}")
                    
                    if status_code == 200:
                        print(f"[DEBUG] VerifierTool API call successful for session {session_id} - Elapsed time: {elapsed_time:.3f}s - status code: {status_code} - result: {result[:100]}")
                        # Truncate result to maximum length
                        truncated_result = result[:MAX_RESULT_LENGTH]
                        if len(result) > MAX_RESULT_LENGTH:
                            truncated_result += f"\n... [Result truncated. Total length: {len(result)} chars, showing first {MAX_RESULT_LENGTH} chars]"
                        return truncated_result, 0.0, {"status_code": status_code}
                    else:
                        print(f"[DEBUG] VerifierTool API call failed for session {session_id} - Elapsed time: {elapsed_time:.3f}s - status code: {status_code} - result: {result[:100]}")
                        # Truncate error result to maximum length
                        error_result = f"API Error (status {status_code}): {result}"
                        truncated_error = error_result[:MAX_RESULT_LENGTH]
                        if len(error_result) > MAX_RESULT_LENGTH:
                            truncated_error += f"\n... [Error truncated. Total length: {len(error_result)} chars, showing first {MAX_RESULT_LENGTH} chars]"
                        return truncated_error, 0.0, {"status_code": status_code}
                        
        except Exception as e:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"[DEBUG] VerifierTool API call failed - Elapsed time: {elapsed_time:.3f}s: {e}")
            return f'{{"error": "Verification failed: {str(e)}", "exit_code": 1}}', 0.0, {}

    async def release(self, instance_id: str, **kwargs):
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]