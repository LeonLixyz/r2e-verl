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

import json
import aiohttp
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema


class APITestTool(BaseTool):
    """API Test Tool for making HTTP requests.
    
    This tool demonstrates the API calling functionality requested by the user.
    It can send POST requests to external APIs and return the responses.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema = None):
        if tool_schema is None:
            tool_schema = OpenAIFunctionToolSchema.parse_obj({
                "type": "function",
                "function": {
                    "name": "api_test",
                    "description": "Send a POST request to an API endpoint with data",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "The message to send to the API"
                            },
                            "api_endpoint": {
                                "type": "string", 
                                "description": "The API endpoint URL (optional, defaults to test endpoint)"
                            },
                            "data": {
                                "type": "object",
                                "description": "Additional data to send (optional)"
                            }
                        },
                        "required": ["message"]
                    }
                }
            })
        
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.timeout = config.get("timeout", 30)
        self.default_endpoint = config.get("default_endpoint", "https://iamleonlixyz.requestcatcher.com/test")

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """Create a tool instance."""
        if instance_id is None:
            instance_id = str(uuid4())
        
        self._instance_dict[instance_id] = {
            "requests": [],
            "responses": []
        }
        return instance_id

    async def execute(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> Tuple[str, float, Dict]:
        """Execute the API test tool.
        
        Args:
            instance_id: The instance id of the tool
            parameters: Parameters containing message, api_endpoint, and optional data
            
        Returns:
            Tuple of (response_text, reward_score, metrics)
        """
        message = parameters.get("message", "")
        endpoint = parameters.get("api_endpoint", self.default_endpoint)
        data = parameters.get("data", {})
        
        if not message:
            return "Error: No message provided", -1.0, {"error": "missing_message"}
        
        try:
            # Record the request
            self._instance_dict[instance_id]["requests"].append({
                "message": message,
                "endpoint": endpoint,
                "data": data
            })
            
            # Make the API call
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                if data:
                    # Send JSON data if additional data is provided
                    payload = {"message": message, **data}
                    async with session.post(endpoint, json=payload) as response:
                        response_text = await response.text()
                        status_code = response.status
                else:
                    # Send raw message data (mimicking curl -d 'message')
                    async with session.post(endpoint, data=message) as response:
                        response_text = await response.text()
                        status_code = response.status
            
            # Record the response
            self._instance_dict[instance_id]["responses"].append({
                "status_code": status_code,
                "response": response_text
            })
            
            # Calculate reward based on successful execution
            reward = 1.0 if status_code < 400 else 0.0
            
            metrics = {
                "status_code": status_code,
                "endpoint": endpoint,
                "request_count": len(self._instance_dict[instance_id]["requests"])
            }
            
            return f"API call successful! Status: {status_code}, Response: {response_text}", reward, metrics
            
        except Exception as e:
            error_msg = f"API call failed: {str(e)}"
            self._instance_dict[instance_id]["responses"].append({
                "error": str(e)
            })
            
            return error_msg, -1.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate reward based on successful API calls."""
        if instance_id not in self._instance_dict:
            return 0.0
        
        responses = self._instance_dict[instance_id]["responses"]
        if not responses:
            return 0.0
        
        # Calculate reward based on successful responses
        successful_calls = sum(1 for resp in responses if resp.get("status_code", 0) < 400)
        total_calls = len(responses)
        
        return successful_calls / total_calls if total_calls > 0 else 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id] 