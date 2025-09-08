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
import re
import torch
import time
import concurrent.futures
import requests
import json
from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.rllm_rewards.rl_reward import rllm_reward_fn
from verl.tools.session_manager import get_session_manager

NUM_CONCURRENT_WORKERS = 64

@register("rllm")
class RLLMRewardManager:
    """Reward manager that uses RLLM's reward system."""

    def __init__(self, tokenizer, num_examine, reward_fn_key="data_source", **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        
        # We'll use direct HTTP calls instead of the VerifierTool to avoid async issues
        self.verification_headers = {
            "Authorization": "Bearer rlc-rtISTILowpUiVcLJjFHUmaYUJ3Y3C9ftGhO-wg", 
            "Content-Type": "application/json"
        }

    def __call__(self, data: DataProto, return_dict=False):
        """Synchronous call - now completely synchronous without async complexity."""
        return self.sync_call(data, return_dict)
    
    def sync_call(self, data: DataProto, return_dict=False):
        """Completely synchronous implementation."""
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        # Extract session information for verification
        conversation_ids = data.non_tensor_batch.get("__conversation_ids__", [])
        session_ids = data.non_tensor_batch.get("__session_ids__", [])
        
        print(f"[RLLM Reward Manager] Processing {len(data)} items with {len(session_ids)} sessions")

        # Prepare all verification tasks
        verification_tasks = []
        batch_info = []
        
        for i in range(len(data)):
            data_item = data[i]
            
            # Decode prompt and response
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            # Get metadata
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            
            # Handle ground_truth parsing with proper error handling
            test_input = ""
            expected_output = ""
            try:
                # Try to parse it as JSON first
                if isinstance(ground_truth, (list, tuple)) and len(ground_truth) > 0:
                    gt_str = ground_truth[0]
                elif isinstance(ground_truth, str):
                    gt_str = ground_truth
                else:
                    print(f"[REWARD DEBUG] Item {i}: Unexpected ground_truth type: {type(ground_truth)}, value: {ground_truth}")
                    gt_str = str(ground_truth)
                
                # Skip empty or whitespace-only strings
                if not gt_str or not gt_str.strip():
                    print(f"[REWARD DEBUG] Item {i}: Empty or whitespace-only ground_truth: '{gt_str}'")
                else:
                    parsed_gt = json.loads(gt_str)
                    
                    # Handle both single and multiple test cases: [{"input": "...", "output": "..."}] or [..., ...]
                    if isinstance(parsed_gt, list) and len(parsed_gt) >= 1 and all(isinstance(tc, dict) for tc in parsed_gt):
                        # Concatenate all test cases
                        test_inputs = []
                        expected_outputs = []
                        
                        for test_case in parsed_gt:
                            if "input" in test_case and "output" in test_case:
                                test_inputs.append(test_case["input"])
                                expected_outputs.append(test_case["output"])
                        
                        if test_inputs and expected_outputs:
                            # Combine multiple test cases - each test case input gets processed separately
                            # but we combine the expected outputs for comparison
                            test_input = "\n".join(test_inputs)
                            expected_output = "\n".join(expected_outputs)
                            print(f"[REWARD DEBUG] Item {i}: Successfully parsed {len(test_inputs)} test case(s) from ground_truth")
                        else:
                            print(f"[REWARD DEBUG] Item {i}: Test cases found but missing input/output keys")
                            test_input = ""
                            expected_output = ""
                    else:
                        print(f"[REWARD DEBUG] Item {i}: Unexpected ground_truth format: {type(parsed_gt)}, value: {parsed_gt}")
                        # Fallback - leave empty for original reward function
                        test_input = ""
                        expected_output = ""
                    
            except (json.JSONDecodeError, IndexError, KeyError, TypeError) as e:
                print(f"[REWARD DEBUG] Item {i}: Failed to parse ground_truth as JSON: {e}")
                print(f"[REWARD DEBUG] Item {i}: Raw ground_truth: {ground_truth}")
                print(f"[REWARD DEBUG] Item {i}: Type: {type(ground_truth)}")
                # Leave test_input and expected_output as empty strings for fallback
            
            # Extract file path from the model's response
            file_path = self._extract_file_path(response_str)
            session_id = session_ids[i]
            
            # Store batch info for processing results later
            batch_info.append({
                'index': i,
                'file_path': file_path,
                'session_id': session_id,
                'test_input': test_input,
                'expected_output': expected_output,
                'prompt_str': prompt_str,
                'response_str': response_str,
                'ground_truth': ground_truth,
                'data_source': data_source,
                'valid_response_length': valid_response_length
            })
            
            # Create verification task if we have both file_path and session_id
            if file_path and session_id:
                verification_tasks.append((file_path, test_input, session_id))
            else:
                verification_tasks.append(None)  # Placeholder for failed items
        
        # Execute all verification tasks concurrently using ThreadPoolExecutor with sync requests
        verification_start_time = time.time()
        
        def verify_single_sync(file_path: str, test_input: str, session_id: str) -> str:
            """Verify a single solution synchronously using requests."""
            try:
                # Prepare verification request payload
                api_payload = {
                    "path": file_path,
                    "input": test_input
                }
                
                endpoint = f"https://toolbox.modal-origin.relace.run/session/{session_id}/exec/"
                
                print(f"[DEBUG] VerifierTool SENDING TO {endpoint} at time {time.time()}")
                
                # Use synchronous requests instead of async aiohttp
                response = requests.post(
                    endpoint,
                    json=api_payload,
                    headers=self.verification_headers,
                    timeout=120  # 2 minute timeout
                )
                
                result = response.text
                status_code = response.status_code
                
                print(f"[DEBUG] VerifierTool API response (status {status_code}): {result}")
                
                if status_code == 200:
                    print(f"[DEBUG] VerifierTool API call successful for session {session_id} - status code: {status_code} - result: {result[:100]}")
                    return result
                else:
                    print(f"[DEBUG] VerifierTool API call failed for session {session_id} - status code: {status_code} - result: {result[:100]}")
                    return f'{{"error": "API Error (status {status_code}): {result}", "exit_code": 1}}'
                    
            except Exception as e:
                print(f"[REWARD DEBUG] Verification failed for {file_path}: {e}")
                return f'{{"error": "Verification failed: {str(e)}", "exit_code": 1}}'
        
        print(f"[RLLM Reward Manager] Starting {len([t for t in verification_tasks if t is not None])} concurrent verifications")
        
        # Execute verifications in parallel using ThreadPoolExecutor with sync requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(NUM_CONCURRENT_WORKERS, len(verification_tasks))) as executor:
            # Submit verification tasks
            future_to_index = {}
            for i, task in enumerate(verification_tasks):
                if task is not None:
                    file_path, test_input, session_id = task
                    future = executor.submit(verify_single_sync, file_path, test_input, session_id)
                    future_to_index[future] = i
                else:
                    # Handle items that don't need verification
                    future = executor.submit(lambda: '{"error": "No verification needed", "exit_code": 1}')
                    future_to_index[future] = i
            
            # Collect results
            verification_results = [None] * len(verification_tasks)
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    verification_results[index] = result
                except Exception as e:
                    print(f"[REWARD DEBUG] Verification future failed for index {index}: {e}")
                    verification_results[index] = f'{{"error": "Verification failed: {str(e)}", "exit_code": 1}}'
        
        verification_end_time = time.time()
        verification_duration = verification_end_time - verification_start_time
        active_verifications = len([t for t in verification_tasks if t is not None])
        
        print(f"[RLLM Reward Manager] Completed {active_verifications} concurrent verifications in {verification_duration:.2f}s")
        if active_verifications > 0:
            print(f"[RLLM Reward Manager] Average verification time: {verification_duration/active_verifications:.2f}s")

        # Process results
        for i, (info, result) in enumerate(zip(batch_info, verification_results)):
            reward = 0.0
            
            if isinstance(result, Exception):
                print(f"[REWARD DEBUG] Item {i}: Verification failed with exception: {result}")
                reward = 0.0
            elif info['file_path'] and info['session_id'] and result is not None:
                # Parse verification result and compare with expected output
                reward = self._parse_verification_result_simple(result, info['expected_output'])
            elif info['file_path'] is None:
                print(f"[REWARD DEBUG] Item {i}: No file path found in response, giving reward 0")
            else:
                print(f"[REWARD DEBUG] Item {i}: No session id found, giving reward 0")
                
            # Store only supplementary metrics - main rewards come from reward_tensor
            reward_extra_info["success"].append(1.0 if reward > 0 else 0.0)

            reward_tensor[i, info['valid_response_length'] - 1] = reward
            
            # Debug logging
            data_source = info['data_source']
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"\n[RLLM Reward Debug {i}]")
                print(f"  Data source: {data_source}")
                print(f"  Prompt: {info['prompt_str'][:200]}...")
                print(f"  Response: {info['response_str'][:200]}...")
                print(f"  Ground truth: {str(info['ground_truth'])[:200]}...")
                print(f"  File path: {info['file_path']}")
                print(f"  Final reward: {reward}\n")

        # Clean up sessions after reward calculation - use sync cleanup
        self._cleanup_sessions_sync(conversation_ids, session_ids)

        if return_dict:
            print(f"[REWARD DEBUG] Returning Both reward_tensor and reward_extra_info")
            print(f"[REWARD DEBUG] Returning reward_tensor: {reward_tensor}")
            print(f"[REWARD DEBUG] Returning reward_extra_info: {reward_extra_info}")
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            print(f"[REWARD DEBUG] Returning reward_tensor")
            print(f"[REWARD DEBUG] Returning reward_tensor: {reward_tensor}")
            return reward_tensor

    def _extract_file_path(self, response_str: str) -> str:
        """Extract file path from the model's response.
        
        Look for the specific pattern:
        <Final Code Path>: path/to/your/code.py
        """
        # Primary pattern: <Final Code Path>: path/to/file
        pattern = r'<Final\s+Code\s+Path>\s*:\s*([^\s\n]+)'
        
        match = re.search(pattern, response_str, re.IGNORECASE | re.MULTILINE)
        if match:
            file_path = match.group(1).strip()
            return file_path
        
        return None
    
    def _parse_verification_result_simple(self, verification_result: str, expected_output: str) -> float:
        """Parse verification result and compare with expected output."""
        if not verification_result:
            return 0.0
        
        try:
            # Parse JSON response
            result_data = json.loads(verification_result)
            
            exit_code = result_data.get("exit_code", 1)
            stdout = result_data.get("stdout", "")
            stderr = result_data.get("stderr", "")
            
            print(f"[REWARD DEBUG] Verification result parsed:")
            print(f"  Exit code: {exit_code}")
            print(f"  Stdout: {stdout[:200]}{'...' if len(stdout) > 200 else ''}")
            print(f"  Stderr: {stderr[:200]}{'...' if len(stderr) > 200 else ''}")
            
            # Check for execution errors first
            if exit_code != 0:
                print(f"[REWARD DEBUG] Non-zero exit code ({exit_code}), returning failure")
                return 0.0
            
            # Check for runtime errors in stderr
            if stderr and any(error_word in stderr.lower() for error_word in ['error', 'exception', 'traceback']):
                print(f"[REWARD DEBUG] Runtime error detected in stderr, returning failure")
                return 0.0
            
            # Simple string matching between stdout and expected output
            actual_output = stdout.strip()
            expected_output_clean = expected_output.strip()
            
            if actual_output == expected_output_clean:
                print(f"[REWARD DEBUG] Output matches, return reward 1.0")
                return 1.0
            else:
                print(f"[REWARD DEBUG] Output mismatch, return reward 0.0")
                
                # Enhanced debugging for multiple test cases
                actual_lines = actual_output.split('\n')
                expected_lines = expected_output_clean.split('\n')
                
                print(f"[REWARD DEBUG] Expected {len(expected_lines)} line(s), got {len(actual_lines)} line(s)")
                
                # Show first few lines that differ for debugging
                max_debug_lines = 3
                for i in range(min(max_debug_lines, max(len(actual_lines), len(expected_lines)))):
                    actual_line = actual_lines[i] if i < len(actual_lines) else "<missing>"
                    expected_line = expected_lines[i] if i < len(expected_lines) else "<missing>"
                    
                    if actual_line != expected_line:
                        print(f"[REWARD DEBUG] Line {i+1} differs:")
                        print(f"  Actual  : '{actual_line}'")
                        print(f"  Expected: '{expected_line}'")
                
                return 0.0
                
        except json.JSONDecodeError as e:
            print(f"[REWARD DEBUG] Failed to parse JSON verification result: {e}")
            print(f"[REWARD DEBUG] Raw result: {verification_result[:200]}...")
            return 0.0
        except Exception as e:
            print(f"[REWARD DEBUG] Unexpected error in verification parsing: {e}")
            return 0.0

    def _cleanup_sessions_sync(self, conversation_ids, session_ids):
        """Clean up all sessions synchronously after reward calculation."""
        if len(conversation_ids) == 0:
            return
            
        print(f"[RLLM Reward Manager] Cleaning up {len(conversation_ids)} sessions after reward calculation")
        
        session_manager = get_session_manager()
        
        # Use parallel cleanup for much faster performance
        print(f"[RLLM Reward Manager] Using parallel sync cleanup for {len(conversation_ids)} sessions")
        
        start_time = time.time()
        
        def cleanup_single_conversation(conversation_id):
            """Cleanup a single conversation in a thread."""
            try:
                if conversation_id:  # Only cleanup if conversation_id is valid
                    session_manager.cleanup_conversation_sync(conversation_id)
                    return f"SUCCESS: {conversation_id}"
                else:
                    return f"SKIPPED: empty conversation_id"
            except Exception as e:
                return f"ERROR: {conversation_id} - {e}"
        
        # Execute all cleanups in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(NUM_CONCURRENT_WORKERS, len(conversation_ids))) as executor:
            # Submit all cleanup tasks
            future_to_conversation = {
                executor.submit(cleanup_single_conversation, conv_id): conv_id 
                for conv_id in conversation_ids
            }
            
            # Wait for all to complete and collect results
            cleanup_results = []
            for future in concurrent.futures.as_completed(future_to_conversation):
                conversation_id = future_to_conversation[future]
                try:
                    result = future.result()
                    cleanup_results.append(result)
                    if result.startswith("ERROR"):
                        print(f"[RLLM Reward Manager] Error during parallel cleanup: {result}")
                except Exception as e:
                    error_result = f"ERROR: {conversation_id} - Exception: {e}"
                    cleanup_results.append(error_result)
                    print(f"[RLLM Reward Manager] Exception during parallel cleanup: {error_result}")
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        success_count = len([r for r in cleanup_results if r.startswith("SUCCESS")])
        error_count = len([r for r in cleanup_results if r.startswith("ERROR")])
        
        print(f"[RLLM Reward Manager] Completed parallel cleanup: {success_count} success, {error_count} errors")
        print(f"[RLLM Reward Manager] Total cleanup time: {total_duration:.2f}s for {len(conversation_ids)} sessions")
        print(f"[RLLM Reward Manager] Average time per session: {total_duration/len(conversation_ids):.2f}s")

    def analyze_and_plot_conversation_completion(self, conversation_logs_dir="./conversation_logs"):
        """Analyze conversation logs to plot finished vs truncated responses."""
        import os
        import matplotlib.pyplot as plt
        from pathlib import Path
        
        finished_count = 0
        truncated_count = 0
        
        # Find all conversation log directories
        logs_path = Path(conversation_logs_dir)
        if not logs_path.exists():
            print(f"[RLLM Analysis] Conversation logs directory not found: {conversation_logs_dir}")
            return
            
        # Process all batch directories
        for batch_dir in logs_path.iterdir():
            if batch_dir.is_dir() and batch_dir.name.startswith("batch_"):
                json_dir = batch_dir / "json"
                if json_dir.exists():
                    print(f"[RLLM Analysis] Processing {batch_dir.name}...")
                    
                    # Process all conversation JSON files
                    for json_file in json_dir.glob("*.json"):
                        try:
                            with open(json_file, 'r', encoding='utf-8') as f:
                                conversation_data = json.load(f)
                            
                            # Get the last assistant message (the model's response)
                            conversation = conversation_data.get("conversation", [])
                            assistant_messages = [msg for msg in conversation if msg.get("role") == "assistant"]
                            
                            if assistant_messages:
                                last_response = assistant_messages[-1].get("content", "")
                                
                                # Use the existing _extract_file_path method to check for completion
                                file_path = self._extract_file_path(last_response)
                                
                                if file_path:
                                    finished_count += 1
                                    print(f"[RLLM Analysis] Finished: {json_file.name} -> {file_path}")
                                else:
                                    truncated_count += 1
                                    print(f"[RLLM Analysis] Truncated: {json_file.name}")
                            else:
                                truncated_count += 1
                                print(f"[RLLM Analysis] No assistant response: {json_file.name}")
                                
                        except Exception as e:
                            print(f"[RLLM Analysis] Error processing {json_file}: {e}")
                            truncated_count += 1
        
        # Create the plot
        categories = ['Finished\n(Has Final Code Path)', 'Truncated\n(No Final Code Path)']
        counts = [finished_count, truncated_count]
        colors = ['#2E8B57', '#DC143C']  # Sea green for finished, crimson for truncated
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                    f'{count}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Calculate percentages
        total = finished_count + truncated_count
        if total > 0:
            finished_pct = (finished_count / total) * 100
            truncated_pct = (truncated_count / total) * 100
            
            plt.text(0, finished_count/2, f'{finished_pct:.1f}%', 
                    ha='center', va='center', fontsize=14, fontweight='bold', color='white')
            plt.text(1, truncated_count/2, f'{truncated_pct:.1f}%', 
                    ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        
        plt.title('Model Response Completion Analysis\nFinished vs Truncated Responses', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Number of Responses', fontsize=12, fontweight='bold')
        plt.xlabel('Response Type', fontsize=12, fontweight='bold')
        
        # Add grid for better readability
        plt.grid(True, axis='y', alpha=0.3, linestyle='--')
        
        # Add summary text
        plt.figtext(0.5, 0.02, 
                   f'Total Responses: {total} | Finished: {finished_count} | Truncated: {truncated_count}',
                   ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        
        # Save the plot
        output_file = "conversation_completion_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"[RLLM Analysis] Plot saved to {output_file}")
        
        # Show summary
        print(f"\n[RLLM Analysis] Summary:")
        print(f"  Total responses analyzed: {total}")
        print(f"  Finished (with Final Code Path): {finished_count} ({finished_pct:.1f}%)")
        print(f"  Truncated (no Final Code Path): {truncated_count} ({truncated_pct:.1f}%)")
        
        plt.show()
        
        return {
            'finished': finished_count,
            'truncated': truncated_count,
            'total': total,
            'finished_percentage': finished_pct if total > 0 else 0,
            'truncated_percentage': truncated_pct if total > 0 else 0
        }