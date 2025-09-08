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

from collections import defaultdict
import asyncio
import torch
import time
import concurrent.futures
import json
import aiohttp
from verl import DataProto
from verl.workers.reward_manager import register
from verl.tools.session_manager import get_session_manager

NUM_CONCURRENT_WORKERS = 32
TIMEOUT = 120

@register("r2e_gym")
class R2EGymRewardManager:
    """Reward manager that uses R2E-Gym's test endpoint for simple pass/fail rewards."""

    def __init__(self, tokenizer, num_examine, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        # Use the same API key as the session manager (it will add "Bearer " prefix)
        self.api_key = "rlc-rtISTILowpUiVcLJjFHUmaYUJ3Y3C9ftGhO-wg"
        self.base_url = "https://toolbox.modal-origin.relace.run"
        
        # Shared aiohttp session like session manager
        self._http_session = None
    
    async def _get_http_session(self) -> aiohttp.ClientSession:
        """Get or create the shared HTTP session like session manager."""
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()
        return self._http_session

    def __call__(self, data: DataProto, return_dict=False):
        """Synchronous wrapper for backward compatibility."""
        try:
            # Check if we're already in an async context
            loop = asyncio.get_running_loop()
            # If we're in an async context, we can't use run_until_complete
            # Create a new thread to run the async code
            import threading
            
            def run_async():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(self.async_call(data, return_dict))
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_async)
                return future.result()
                
        except RuntimeError:
            # No event loop running, we can use run_until_complete
            return asyncio.run(self.async_call(data, return_dict))
    
    async def async_call(self, data: DataProto, return_dict=False):
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        # Extract session information
        conversation_ids = data.non_tensor_batch.get("__conversation_ids__", [])
        session_ids = data.non_tensor_batch.get("__session_ids__", [])
        
        print(f"[R2E-Gym Reward Manager] Processing {len(data)} items with {len(session_ids)} sessions")
        print(f"[R2E-Gym Reward Manager] Session IDs to test: {session_ids[:3]}..." if len(session_ids) > 3 else f"[R2E-Gym Reward Manager] Session IDs to test: {session_ids}")
        print(f"[R2E-Gym Reward Manager] Conversation IDs: {conversation_ids[:3]}..." if len(conversation_ids) > 3 else f"[R2E-Gym Reward Manager] Conversation IDs: {conversation_ids}")
        
        # Check for None session IDs
        none_count = sum(1 for sid in session_ids if sid is None)
        if none_count > 0:
            print(f"[R2E-Gym Reward Manager] WARNING: {none_count}/{len(session_ids)} session IDs are None!")

        # Prepare all test tasks
        batch_info = []
        
        for i in range(len(data)):
            data_item = data[i]
            
            # Decode response for logging
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][data_item.batch["prompts"].shape[-1]:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            # Get session ID
            session_id = session_ids[i] if i < len(session_ids) else None
            data_source = data_item.non_tensor_batch.get("data_source", "unknown")
            
            # Store batch info for processing results later
            batch_info.append({
                'index': i,
                'session_id': session_id,
                'response_str': response_str,
                'data_source': data_source,
                'valid_response_length': valid_response_length
            })
        
        # Execute all test tasks concurrently using async
        test_start_time = time.time()
        
        print(f"[R2E-Gym Reward Manager] Starting {len([info for info in batch_info if info['session_id'] is not None])} concurrent tests")
        
        # Create test tasks
        test_tasks = []
        for info in batch_info:
            if info['session_id'] is not None:
                test_tasks.append(self._test_session(info['session_id']))
            else:
                # Create a task that returns failed result
                test_tasks.append(asyncio.create_task(self._create_failed_result()))
        
        # Execute all tests concurrently
        test_results = await asyncio.gather(*test_tasks, return_exceptions=True)
        
        # Handle any exceptions
        for i, result in enumerate(test_results):
            if isinstance(result, Exception):
                print(f"[R2E-Gym DEBUG] Test failed with exception for index {i}: {result}")
                test_results[i] = {"PASSED": 0, "FAILED": 1}
        
        test_end_time = time.time()
        test_duration = test_end_time - test_start_time
        active_tests = len([info for info in batch_info if info['session_id'] is not None])
        
        print(f"[R2E-Gym Reward Manager] Completed {active_tests} concurrent tests in {test_duration:.2f}s")
        if active_tests > 0:
            print(f"[R2E-Gym Reward Manager] Average test time: {test_duration/active_tests:.2f}s")

        # Process results
        success_count = 0
        total_passed = 0
        total_failed = 0
        
        for i, (info, test_result) in enumerate(zip(batch_info, test_results)):
            # Extract counts from the result
            passed_count = test_result.get("PASSED", 0)
            failed_count = test_result.get("FAILED", 0)

            if passed_count == 0 and failed_count == 0:
                print(f"[R2E-Gym Reward Manager] Test result for index {i}: PASSED=0, FAILED=0 (Both zero)")
            
            # Binary reward: 1.0 only if ALL tests pass (no failures)
            reward = 1.0 if failed_count == 0 and passed_count > 0 else 0.0
            
            if reward > 0:
                success_count += 1
                
            total_passed += passed_count
            total_failed += failed_count
                
            # Store metrics
            reward_extra_info["success"].append(reward)
            reward_extra_info["passed_count"].append(float(passed_count))
            reward_extra_info["failed_count"].append(float(failed_count))

            # Set reward at the last token of the response
            reward_tensor[i, info['valid_response_length'] - 1] = reward
            
            # Debug logging
            data_source = info['data_source']
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"\n[R2E-Gym Reward Debug {i}]")
                print(f"  Data source: {data_source}")
                print(f"  Session ID: {info['session_id']}")
                print(f"  Response: {info['response_str'][:200]}...")
                print(f"  Test counts: PASSED={passed_count}, FAILED={failed_count}")
                print(f"  All tests passed: {failed_count == 0 and passed_count > 0}")
                print(f"  Final reward: {reward}\n")

        success_rate = success_count / len(data) if len(data) > 0 else 0.0
        print(f"[R2E-Gym Reward Manager] Overall test results: PASSED={total_passed}, FAILED={total_failed}")
        print(f"[R2E-Gym Reward Manager] Perfect sessions: {success_count}/{len(data)} ({success_rate:.1%})")

        # Clean up sessions after reward calculation
        await self._cleanup_sessions(conversation_ids, session_ids)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
    
    async def _create_failed_result(self) -> dict:
        """Create a failed test result for sessions that don't exist."""
        return {"PASSED": 0, "FAILED": 1}
    
    async def _test_session(self, session_id: str) -> dict:
        """Test a session using the R2E-Gym test endpoint."""
        try:
            # Use the same pattern as session manager
            timeout = aiohttp.ClientTimeout(total=TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.base_url}/session/{session_id}/test/", 
                    headers={"Authorization": f"Bearer {self.api_key}"}
                ) as response:
                    status_code = response.status
                    response_text = await response.text()

                    # print(f"[R2E-Gym DEBUG] Test response for session {session_id}: {response_text}")
                    # print(f"[R2E-Gym DEBUG] Test status code for session {session_id}: {status_code}")
                    
                    if status_code == 200:
                        try:
                            # Try to parse the response as JSON to extract counts
                            response_data = json.loads(response_text)
                            
                            # Look for counts in the response
                            if "counts" in response_data:
                                counts = response_data["counts"]
                                passed = counts.get("PASSED", 0)
                                failed = counts.get("FAILED", 0)
                                print(f"[R2E-Gym DEBUG] Extracted counts for session {session_id}: PASSED={passed}, FAILED={failed}")
                                if passed == 0 and failed == 0:
                                    print(f"[R2E-Gym DEBUG] Both passed and failed are 0 for session {session_id} with response: {response_text}")
                                return {"PASSED": passed, "FAILED": failed}
                            else:
                                # If no counts in response, try to extract from the response text
                                # Look for patterns like "PASSED":26,"FAILED":1
                                import re
                                passed_match = re.search(r'"PASSED":\s*(\d+)', response_text)
                                failed_match = re.search(r'"FAILED":\s*(\d+)', response_text)
                                
                                if passed_match and failed_match:
                                    passed = int(passed_match.group(1))
                                    failed = int(failed_match.group(1))
                                    print(f"[R2E-Gym DEBUG] Extracted counts from text for session {session_id}: PASSED={passed}, FAILED={failed}")
                                    return {"PASSED": passed, "FAILED": failed}
                                else:
                                    # If we can't extract counts, assume success means some tests passed
                                    print(f"[R2E-Gym DEBUG] No counts found, assuming 1 passed test for session {session_id}")
                                    return {"PASSED": 1, "FAILED": 0}
                        except json.JSONDecodeError:
                            # If response is not JSON, assume success means some tests passed
                            print(f"[R2E-Gym DEBUG] Non-JSON response, assuming 1 passed test for session {session_id}")
                            return {"PASSED": 1, "FAILED": 0}
                    else:
                        print(f"[R2E-Gym DEBUG] Test failed for session {session_id}: status {status_code}, response: {response_text[:200]}")
                        return {"PASSED": 0, "FAILED": 1}
                        
        except Exception as e:
            print(f"[R2E-Gym DEBUG] Test error for session {session_id}: {e}")
            return {"PASSED": 0, "FAILED": 1}
    
    async def _cleanup_sessions(self, conversation_ids, session_ids):
        """Clean up all sessions after reward calculation."""
        if len(conversation_ids) == 0:
            return
            
        print(f"[R2E-Gym Reward Manager] Cleaning up {len(conversation_ids)} sessions after reward calculation")
        
        session_manager = get_session_manager()
        
        # Use async cleanup for each session
        print(f"[R2E-Gym Reward Manager] Using async cleanup for {len(conversation_ids)} sessions")
        
        start_time = time.time()
        
        # Execute all cleanups concurrently
        cleanup_tasks = []
        for conversation_id in conversation_ids:
            if conversation_id:  # Only cleanup if conversation_id is valid
                cleanup_tasks.append(session_manager.cleanup_conversation(conversation_id))
        
        # Wait for all cleanup tasks to complete
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        print(f"[R2E-Gym Reward Manager] Completed async cleanup in {total_duration:.2f}s for {len(conversation_ids)} sessions")
        print(f"[R2E-Gym Reward Manager] Average time per session: {total_duration/len(conversation_ids):.2f}s") 