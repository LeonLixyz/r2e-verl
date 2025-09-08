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
import heapq
import importlib
import itertools
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from uuid import uuid4

import aiohttp
import numpy as np
import torch
from cachetools import LRUCache
from omegaconf import DictConfig
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.tools.base_tool import initialize_tools_from_config
from verl.tools.session_manager import get_session_manager
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.debug import simple_timer
from verl.utils.http_client import http_request

logger = logging.getLogger(__file__)


def create_thinking_limit_regex(max_chars: int = 1000 * 3.5) -> str:
    """Create a guided regex pattern that limits thinking content to max_chars."""

    # Simplified regex pattern that doesn't use lookahead/lookbehind:
    # - ^[^<]* : Start with any non-< characters
    # - (?:<think>.{0,max_chars}</think>[^<]*)* : Zero or more thinking sections
    #   where inside <think> we allow 0-max_chars of any character, then </think>,
    #   followed by any non-< characters
    # - $ : End of string

    regex_pattern = f"^[^<]*(?:<think>.{{0,{max_chars}}}Considering the limited time by the user, I have to give the tool call based on the thinking directly now.\n</think>.\n\n<tool_call>[^<]*)*$"

    return regex_pattern


class CompletionCallback(ABC):
    def __init__(self, config: DictConfig, scheduler: "ChatCompletionScheduler"):
        self.config = config
        self.scheduler = scheduler

        # Initialize tools from config file
        self.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self._tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        print(f"Initialized tools: {self.tools}", flush=True)

        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)

    @property
    def tool_schemas(self):
        """OpenAI JSON tool schemas."""
        return self._tool_schemas

    @property
    def extra_body(self) -> Dict[str, Any]:
        """Extra body pass to OpenAI API."""
        return None

    @abstractmethod
    async def __call__(self, messages: List[Dict[str, str]], completions: ChatCompletion, info: Dict[str, Any]):
        """Call back function to process completions.

        Args:
            messages: List of messages including raw prompt and assistant, tool response generated so far.
            completions: Chat completions from OpenAI compatible server.
            info: Any other auxiliary information pass across multi-turn.
        """
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, batch: DataProto, batch_conversations: List[List[Dict[str, str]]], n: int) -> DataProto:
        """Post process batch data.

        Args:
            batch: Batch input messages from RLHFDataset.
            batch_conversations: List of messages including raw prompt, assistant response, tool response.
                Note that `len(batch_conversations) == len(batch) * n`, e.g n=2,
                batch_conversations=[messages_0_0, messages_0_1, messages_1_0, messages_1_1, ...]
            n: How many chat completion choices to generate for each input message.

        Returns:
            Batch data, should include ["prompts", "responses", "response_mask", "input_ids", "attention_mask", "position_ids"].
        """
        raise NotImplementedError


class ToolCompletionCallback(CompletionCallback):
    def __init__(self, config: DictConfig, scheduler: "ChatCompletionScheduler"):
        super().__init__(config, scheduler)

        # TODO: add reward manager to calculate reward score once a sample finish

    @property
    def extra_body(self) -> Dict[str, Any]:
        """Extra body pass to OpenAI API to disable thinking mode."""
        return {
            "chat_template_kwargs": {"enable_thinking": True},
            # "guided_regex": create_thinking_limit_regex(max_chars=4000)
        }

    async def __call__(self, messages: List[Dict[str, str]], completions: ChatCompletion, info: Dict[str, Any]):
        message = completions.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)
        # print(f"[DEBUG] I am using ToolCompletionCallback __call__ with generated message: {message}")
        if "content" not in message:
            message["content"] = ""
        messages.append(message)
        finish_reason = completions.choices[0].finish_reason

        # Get conversation ID for session management
        conversation_id = info.get("__conversation_id__")
        if conversation_id is None:
            logger.error("No conversation ID found in info - this should not happen!")
            return

        # STEP 0: check if we reach max turns
        if self.max_assistant_turns and len(messages) >= self.max_assistant_turns:
            print(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason},conv_id={conversation_id}] Reach max turns, done!")
            return

        # STEP 1: check if the model called tools
        if finish_reason != "tool_calls":
            # Check if this might be a tool parsing error from VLLM
            if message.get("content") and any(keyword in message["content"].lower() for keyword in ["<tool_call>", "function_name", "arguments"]):
                print(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason},conv_id={conversation_id}] Detected potential tool call content but finish_reason is not 'tool_calls' - likely VLLM tool parser error!")
                print(f"[id={completions.id},conv_id={conversation_id}] Message content preview: {message['content'][:500]}...")
                # Add error message to help model recover instead of terminating
                error_message = {
                    "role": "tool", 
                    "content": "Error: Tool call format was wrong.",
                    "tool_call_id": "parsing_error"
                }
                messages.append(error_message)
                # Continue the conversation instead of terminating
                self.scheduler.submit_chat_completions(messages=messages, request_id=completions.id, info=info)
                print(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason},conv_id={conversation_id}] Restarting the conversation.")
            else:
                print(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason},conv_id={conversation_id}] No tool called, done!")
            return

        # STEP 2: call tools - with error handling for malformed tool calls
        try:
            tool_calls = completions.choices[0].message.tool_calls
            if not tool_calls:
                print(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason},conv_id={conversation_id}] tool_calls is None or empty, treating as parsing error")
                error_message = {
                    "role": "tool", 
                    "content": "Error: Tool call was expected but not found. Please try again or continue without tools.",
                    "tool_call_id": "unknown"
                }
                messages.append(error_message)
                # Continue the conversation instead of terminating
                self.scheduler.submit_chat_completions(messages=messages, request_id=completions.id, info=info)
                return
                
            # print(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason},conv_id={conversation_id}] Call {len(tool_calls)} tools")
            
            # Pre-validate tool calls for parsing errors
            for tool_call in tool_calls:
                if not tool_call.function.arguments:
                    raise ValueError(f"Tool call {tool_call.function.name} has empty arguments")
                # Try to parse arguments to catch malformed JSON early
                json.loads(tool_call.function.arguments)
                
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            print(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason},conv_id={conversation_id}] Tool parsing/validation error: {e}")
            # Add error message and continue conversation instead of terminating
            error_message = {
                "role": "tool", 
                "content": f"Error: Tool call arguments are malformed ({str(e)}). Please fix the format and try again, or continue without tools.",
                "tool_call_id": getattr(tool_calls[0], 'id', 'unknown') if tool_calls and len(tool_calls) > 0 else "unknown"
            }
            messages.append(error_message)
            # Continue the conversation instead of terminating
            self.scheduler.submit_chat_completions(messages=messages, request_id=completions.id, info=info)
            return
        
        tasks = []
        for tool_call in tool_calls:
            tasks.append(self._call_tool(tool_call, conversation_id))
        tool_responses = await asyncio.gather(*tasks)
        if any(isinstance(item, Exception) for item in tool_responses):
            print(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason},conv_id={conversation_id}] Error when calling tools, done!")
            return
        messages.extend(tool_responses)

        # STEP 3: resubmit completion request with tool responses
        self.scheduler.submit_chat_completions(messages=messages, request_id=completions.id, info=info)

    async def _call_tool(self, tool_call, conversation_id: str) -> Dict[str, str]:
        """Call tool and return tool response."""
        try:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            tool = self.tools[tool_name]
            
            # print(f"[DEBUG] Executing tool: {tool_name}")
            # print(f"[DEBUG] Tool arguments: {tool_args}")

            instance_id = await tool.create()
            try:
                tool_response, tool_reward_score, tool_metrics = await tool.execute(
                    instance_id, 
                    tool_args, 
                    conversation_id=conversation_id
                )
                # print(f"[DEBUG] Tool response: {tool_response[:200]}{'...' if len(str(tool_response)) > 200 else ''}")
                # print(f"[DEBUG] Tool reward score: {tool_reward_score}")
                # print(f"[DEBUG] Tool metrics: {tool_metrics}")
            except Exception as e:
                logger.exception(f"Error when executing tool: {e}")
                print(f"[DEBUG] Tool execution failed: {e}")
                # Don't cleanup session here - let it be handled by the caller
                return e
            finally:
                await tool.release(instance_id)

            return {
                "role": "tool",
                "content": tool_response,
                "tool_call_id": tool_call.id,
            }
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Handle tool parsing errors gracefully
            error_msg = f"Tool parsing error: {str(e)}"
            print(f"[DEBUG] {error_msg}")
            return {
                "role": "tool",
                "content": f"Error: {error_msg}",
                "tool_call_id": tool_call.id if hasattr(tool_call, 'id') else "unknown",
            }



    def postprocess(self, batch: DataProto, batch_conversations: List[List[Dict[str, str]]], n: int) -> DataProto:
        # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

        # Get PPO token limit from config
        ppo_max_token_len = getattr(self.config, 'actor_rollout_ref', {}).get('actor', {}).get('ppo_max_token_len_per_gpu', 32768)
        if hasattr(self.config, 'actor_rollout_ref') and hasattr(self.config.actor_rollout_ref, 'actor'):
            ppo_max_token_len = getattr(self.config.actor_rollout_ref.actor, 'ppo_max_token_len_per_gpu', 32768)
        else:
            ppo_max_token_len = 32768
        ppo_max_token_len = ppo_max_token_len - 1000
        print(f"[TRUNCATION] Using PPO max token length: {ppo_max_token_len}")
        
        # prompts: [prompt] from input dataset
        prompts = [self.tokenizer.apply_chat_template(prompt, tools=self.tool_schemas, add_generation_prompt=True, tokenize=False, enable_thinking=True) for prompt in batch.non_tensor_batch["raw_prompt"]]
        
        assert len(batch_conversations) == len(prompts) * n

        # sequences: [prompt + response] - Apply response truncation here
        raw_sequences = [self.tokenizer.apply_chat_template(conversation, tools=self.tool_schemas, add_generation_prompt=False, tokenize=False, enable_thinking=True) for conversation in batch_conversations]
        
        sequences = []
        truncation_info = []  # Track truncation details for each sequence
        
        for i, sequence in enumerate(raw_sequences):
            # Get the corresponding prompt (same prompt for every n responses)
            prompt_idx = i // n  # 0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2, ...
            prompt = prompts[prompt_idx]
            
            # Calculate token lengths
            prompt_tokens = self.tokenizer.encode(prompt)
            prompt_len = len(prompt_tokens)
            
            # Calculate max response length: ppo_max_token - prompt - 500 buffer
            max_response_len = ppo_max_token_len - prompt_len - 200
            
            if max_response_len <= 0:
                print(f"[WARNING] Prompt {prompt_idx} too long ({prompt_len} tokens), no room for response")
                # Use original sequence, will be handled by downstream error handling
                sequences.append(sequence)
                truncation_info.append({
                    "was_truncated": False,
                    "original_length": len(self.tokenizer.encode(sequence)),
                    "truncated_length": len(self.tokenizer.encode(sequence)),
                    "original_text": sequence,
                    "truncated_text": sequence,
                    "truncation_reason": "prompt_too_long"
                })
                continue
            
            # ROUND 1: Always drop tool response if it's the last message
            conversation = batch_conversations[i]
            last_message = conversation[-1] if conversation else None
            
            if last_message and last_message.get('role') == 'tool':
                # Drop the tool response entirely
                conversation_without_tool = conversation[:-1]
                batch_conversations[i] = conversation_without_tool  # Update conversation array
                
                # Regenerate sequence after dropping tool response
                sequence = self.tokenizer.apply_chat_template(
                    conversation_without_tool, 
                    tools=self.tool_schemas, 
                    add_generation_prompt=False, 
                    tokenize=False, 
                    enable_thinking=True
                )
                print(f"[TRUNCATION] Dropped tool response for sequence {i} (always drop tool at end)")
            
            # ROUND 2: Check if sequence is still too long and apply hard truncation
            full_sequence_tokens = self.tokenizer.encode(sequence)
            if len(full_sequence_tokens) > ppo_max_token_len:
                # Hard truncate to exact token limit
                truncated_tokens = full_sequence_tokens[:ppo_max_token_len-1] + [self.tokenizer.eos_token_id]
                truncated_sequence = self.tokenizer.decode(truncated_tokens, skip_special_tokens=False)
                sequences.append(truncated_sequence)
                print(f"[TRUNCATION] Hard truncated sequence {i} from {len(full_sequence_tokens)} to {len(truncated_tokens)} tokens")
                
                # Save truncation info
                truncation_info.append({
                    "was_truncated": True,
                    "original_length": len(full_sequence_tokens),
                    "truncated_length": len(truncated_tokens),
                    "original_text": sequence,  # Always include this
                    "truncated_text": truncated_sequence,  # Always include this
                    "truncation_reason": "hard_truncate_after_tool_drop"
                })
            else:
                sequences.append(sequence)
                truncation_info.append({
                    "was_truncated": False,
                    "original_length": len(full_sequence_tokens),
                    "truncated_length": len(full_sequence_tokens),
                    "original_text": sequence,
                    "truncated_text": sequence,
                    "truncation_reason": "no_truncation_needed"
                })
        
        # Verify sequences are within PPO token limit
        for i, seq in enumerate(sequences):
            seq_tokens = self.tokenizer.encode(seq)
            if len(seq_tokens) > ppo_max_token_len:
                print(f"[WARNING] Sequence {i} still exceeds PPO token limit: {len(seq_tokens)} > {ppo_max_token_len}")
        
        # 📊 TOKEN COUNTING: Count tokens in prompts and full conversations
        prompt_token_counts = []
        sequence_token_counts = []
        
        for i, sequence in enumerate(sequences):  # Use truncated sequences
            # Get the corresponding prompt (same prompt for every n responses)
            prompt_idx = i // n  # 0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2, ...
            prompt = prompts[prompt_idx]
            
            # Count tokens for this specific prompt and sequence
            prompt_tokens = len(self.tokenizer.encode(prompt))
            sequence_tokens = len(self.tokenizer.encode(sequence))
            
            prompt_token_counts.append(prompt_tokens)
            sequence_token_counts.append(sequence_tokens)
            
            # Optional: Log token usage for monitoring
            if i < 3:  # Only log first 3 to avoid spam
                print(f"[TOKEN_COUNT] Conv {i}: Prompt={prompt_tokens} tokens, Full={sequence_tokens} tokens, Growth={sequence_tokens-prompt_tokens}")
        
        # Calculate statistics
        avg_prompt_tokens = sum(prompt_token_counts) / len(prompt_token_counts) if prompt_token_counts else 0
        avg_sequence_tokens = sum(sequence_token_counts) / len(sequence_token_counts) if sequence_token_counts else 0
        max_prompt_tokens = max(prompt_token_counts) if prompt_token_counts else 0
        max_sequence_tokens = max(sequence_token_counts) if sequence_token_counts else 0
        
        print(f"[TOKEN_STATS] Prompts: avg={avg_prompt_tokens:.1f}, max={max_prompt_tokens}")
        print(f"[TOKEN_STATS] Sequences: avg={avg_sequence_tokens:.1f}, max={max_sequence_tokens}")
        
        # Verify no sequence exceeds the limit
        if max_sequence_tokens > ppo_max_token_len:
            print(f"[TOKEN_ERROR] Max sequence tokens {max_sequence_tokens} still exceeds limit {ppo_max_token_len}")
        
        # 🗂️ SAVE CONVERSATION HISTORY WITH SESSION IDs (include token counts and truncation info)
        self._save_conversation_history(batch, batch_conversations, sequences, prompts, n, 
                                        prompt_token_counts, sequence_token_counts, truncation_info)

        # responses: [response]
        responses = [sequence[len(prompts[i // n]) :] for i, sequence in enumerate(sequences)]

        prompts = self.tokenizer(prompts, return_tensors="pt", padding="longest", padding_side="left")
        responses = self.tokenizer(responses, return_tensors="pt", padding="longest", padding_side="right")
        if n > 1:
            prompts["input_ids"] = prompts["input_ids"].repeat_interleave(n, dim=0)
            prompts["attention_mask"] = prompts["attention_mask"].repeat_interleave(n, dim=0)

        # response_mask: mask with tools calling tokens masked out (for multi-turn training)
        response_mask = self._mask_out_tools_calling_tokens(batch.non_tensor_batch["raw_prompt"].repeat(n, axis=0), batch_conversations, responses["input_ids"], responses["attention_mask"])

        input_ids = torch.cat([prompts["input_ids"], responses["input_ids"]], dim=1)
        attention_mask = torch.cat([prompts["attention_mask"], responses["attention_mask"]], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        # Create full loss_mask for entire sequence (prompt + response)
        prompt_loss_mask = torch.zeros_like(prompts["attention_mask"])  # Don't train on prompts
        loss_mask = torch.cat([prompt_loss_mask, response_mask], dim=1)

        tensor_batch = TensorDict(
            {
                "prompts": prompts["input_ids"],  # [bsz, prompt_length]
                "responses": responses["input_ids"],  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length] - for backward compatibility
                "loss_mask": loss_mask,  # [bsz, prompt_length + response_length] - for multi-turn
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                "position_ids": position_ids,  # [bsz, prompt_length + response_length]
            },
            batch_size=len(input_ids),
        )

        num_turns = np.array([len(conversation) for conversation in batch_conversations], dtype=np.int32)
        
        # Preserve existing non_tensor_batch data and add metrics
        preserved_non_tensor_batch = {}
        if hasattr(batch, 'non_tensor_batch') and batch.non_tensor_batch:
            for key, val in batch.non_tensor_batch.items():
                print(f"[DEBUG] Processing {key}: {val.shape if hasattr(val, 'shape') else type(val)}")
                if key in ['__conversation_ids__', '__session_ids__']:
                    print(f"[DEBUG] Keeping {key} as-is")
                    preserved_non_tensor_batch[key] = val  
                else:
                    print(f"[DEBUG] Expanding {key}")
                    # Expand original dataset fields to match new batch size (n responses per prompt)
                    if isinstance(val, np.ndarray):
                        preserved_non_tensor_batch[key] = np.repeat(val, n, axis=0)
                    else:
                        # For non-array types, convert to array and repeat
                        preserved_non_tensor_batch[key] = np.repeat(np.array(val, dtype=object), n, axis=0)

        preserved_non_tensor_batch["__num_turns__"] = num_turns
        preserved_non_tensor_batch["__prompt_token_counts__"] = np.array(prompt_token_counts, dtype=np.int32)
        preserved_non_tensor_batch["__sequence_token_counts__"] = np.array(sequence_token_counts, dtype=np.int32)
        
        print(f"[POSTPROCESS_DEBUG] Preserved keys: {list(preserved_non_tensor_batch.keys())}")
        print(f"[POSTPROCESS_DEBUG] Has __session_ids__: {'__session_ids__' in preserved_non_tensor_batch}")
        print(f"[POSTPROCESS_DEBUG] Has __conversation_ids__: {'__conversation_ids__' in preserved_non_tensor_batch}")
        
        # DEBUG: Check what's in non_tensor_batch and their sizes
        print(f"[POSTPROCESS_DEBUG] Input batch size: {len(batch)}")
        print(f"[POSTPROCESS_DEBUG] Target batch size: {len(batch) * n} (n={n})")
        if hasattr(batch, 'non_tensor_batch') and batch.non_tensor_batch:
            for key, val in batch.non_tensor_batch.items():
                if isinstance(val, np.ndarray):
                    print(f"[POSTPROCESS_DEBUG] {key}: {val.shape} (type: {val.dtype})")
                else:
                    print(f"[POSTPROCESS_DEBUG] {key}: {len(val) if hasattr(val, '__len__') else 'no len'} (type: {type(val)})")

        #DEBUG: Check what's in preserved_non_tensor_batch and their sizes
        print(f"[POSTPROCESS_DEBUG] Preserved non-tensor batch keys: {list(preserved_non_tensor_batch.keys())}")
        for key, val in preserved_non_tensor_batch.items():
            if isinstance(val, np.ndarray):
                print(f"[POSTPROCESS_DEBUG] {key}: {val.shape} (type: {val.dtype})")
            else:
                print(f"[POSTPROCESS_DEBUG] {key}: {len(val) if hasattr(val, '__len__') else 'no len'} (type: {type(val)})")
        
        return DataProto(batch=tensor_batch, non_tensor_batch=preserved_non_tensor_batch)

    def _save_conversation_history(self, batch: DataProto, batch_conversations: List[List[Dict[str, str]]], sequences: List[str], prompts: List[str], n: int, prompt_token_counts: List[int] = None, sequence_token_counts: List[int] = None, truncation_info: List[Dict] = None):
        """Save conversation history in both JSON and text formats using session IDs."""
        import json
        import os
        from datetime import datetime
        from verl.tools.session_manager import get_session_manager
        
        # Create base directory for conversation logs
        base_dir = os.getenv("CONVERSATION_LOG_DIR", "./conversation_logs")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(base_dir, f"batch_{timestamp}")
        
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, "json"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "text"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "metadata"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "truncation"), exist_ok=True)
        
        session_manager = get_session_manager()
        
        print(f"[CONVERSATION_SAVE] Saving {len(batch_conversations)} conversations to {log_dir}")
        print(f"[CONVERSATION_SAVE] DEBUG: Batch details - conversations: {len(batch_conversations)}, sequences: {len(sequences)}, prompts: {len(prompts)}, n: {n}")
        
        # Get conversation IDs from batch metadata
        conversation_ids = batch.non_tensor_batch.get("__conversation_ids__", [])
        # print(f"[CONVERSATION_SAVE] DEBUG: Raw conversation_ids from batch: {conversation_ids}")
        print(f"[CONVERSATION_SAVE] DEBUG: Type: {type(conversation_ids)}, Length: {len(conversation_ids) if hasattr(conversation_ids, '__len__') else 'no length'}")
        
        if len(conversation_ids) == 0:
            print(f"[CONVERSATION_SAVE] WARNING: No conversation IDs found in batch metadata")
            conversation_ids = [f"unknown_{i}" for i in range(len(batch_conversations))]
        else:
            # Convert numpy array to list for easier handling
            conversation_ids = conversation_ids.tolist() if hasattr(conversation_ids, 'tolist') else list(conversation_ids)
            # print(f"[CONVERSATION_SAVE] DEBUG: After conversion to list: {conversation_ids[:5]}...") # Show first 5
            print(f"[CONVERSATION_SAVE] DEBUG: Unique conversation IDs: {len(set(conversation_ids))} out of {len(conversation_ids)} total")
            if len(set(conversation_ids)) != len(conversation_ids):
                print(f"[CONVERSATION_SAVE] WARNING: Found duplicate conversation IDs! This will cause file overwrites!")
        
        for i, (conversation, sequence) in enumerate(zip(batch_conversations, sequences)):
            try:
                conversation_id = conversation_ids[i] if i < len(conversation_ids) else f"missing_{i}"
                
                # Get session ID from session manager
                session_id = session_manager._conversation_sessions.get(conversation_id, f"no_session_{conversation_id}")
                # print(f"[CONVERSATION_SAVE] DEBUG: Session ID: {session_id}")
                
                # Create filenames with session ID and unique index to prevent collisions
                # Use session ID as primary identifier since it's guaranteed unique by toolbox API
                if session_id.startswith("no_session_"):
                    # If no real session, use conversation ID
                    base_name = f"conv_{conversation_id}_idx_{i:03d}"
                else:
                    # Use real session ID
                    base_name = f"session_{session_id}_idx_{i:03d}"
                
                json_filename = f"{base_name}.json"
                text_filename = f"{base_name}.txt"
                metadata_filename = f"{base_name}_metadata.json"
                
                # print(f"[CONVERSATION_SAVE] Conversation {i}: ID={conversation_id}, Session={session_id}, Filename={json_filename}")
                
                # 📝 SAVE JSON FORMAT: Raw conversation with interleaved messages
                json_data = {
                    "session_id": session_id,
                    "conversation_id": conversation_id,
                    "timestamp": datetime.now().isoformat(),
                    "trajectory_index": i,
                    "total_messages": len(conversation),
                    "conversation": conversation,
                    # "message_analysis": self._analyze_conversation(conversation)
                }
                
                # Add token counts if available
                if prompt_token_counts and i < len(prompt_token_counts):
                    json_data["prompt_tokens"] = prompt_token_counts[i]
                if sequence_token_counts and i < len(sequence_token_counts):
                    json_data["sequence_tokens"] = sequence_token_counts[i]
                    if prompt_token_counts and i < len(prompt_token_counts):
                        json_data["token_growth"] = sequence_token_counts[i] - prompt_token_counts[i]
                
                json_path = os.path.join(log_dir, "json", json_filename)
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                
                # 📄 SAVE TEXT FORMAT: Chat template applied
                truncation_status = ""
                if truncation_info and i < len(truncation_info):
                    trunc_info = truncation_info[i]
                    if trunc_info["was_truncated"]:
                        truncation_status = f"\nTRUNCATION: YES ({trunc_info['truncation_reason']})\nORIGINAL LENGTH: {trunc_info['original_length']} tokens\nTRUNCATED LENGTH: {trunc_info['truncated_length']} tokens\nTOKENS SAVED: {trunc_info['original_length'] - trunc_info['truncated_length']}"
                        if 'original_response_tokens' in trunc_info:
                            truncation_status += f"\nRESPONSE TOKENS: {trunc_info['original_response_tokens']} → {trunc_info['truncated_response_tokens']}"
                    else:
                        truncation_status = f"\nTRUNCATION: NO ({trunc_info['truncation_reason']})"
                
                text_data = f"""SESSION ID: {session_id}\nCONVERSATION ID: {conversation_id}\nTIMESTAMP: {datetime.now().isoformat()}\nTRAJECTORY INDEX: {i}\nTOTAL MESSAGES: {len(conversation)}\nMODEL: {getattr(self, 'model_name', 'unknown')}{truncation_status}\n\n{'='*80}\n\nTEMPLATED CONVERSATION (FINAL - AFTER TRUNCATION):\n\n{'='*80}\n\n{sequence}\n\n"""
                
                # Add original sequence if truncated
                if truncation_info and i < len(truncation_info) and truncation_info[i]["was_truncated"]:
                    # Check if original_text exists, use fallback if not
                    original_text = truncation_info[i].get('original_text', 'Original text not available')
                    text_data += f"""{'='*80}\n\nORIGINAL CONVERSATION (BEFORE TRUNCATION):\n\n{'='*80}\n\n{original_text}\n\n"""
                
                text_data += f"""{'='*80}\n\nRAW MESSAGE BREAKDOWN:\n\n{'='*80}\n\n"""
                
                for j, msg in enumerate(conversation):
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    tool_calls = msg.get('tool_calls', [])
                    tool_call_id = msg.get('tool_call_id', None)
                    
                    text_data += f"[Message {j}] ROLE: {role}\n"
                    if tool_call_id:
                        text_data += f"[Message {j}] TOOL_CALL_ID: {tool_call_id}\n"
                    text_data += f"[Message {j}] CONTENT: {content}\n"
                    
                    if tool_calls:
                        text_data += f"[Message {j}] TOOL_CALLS ({len(tool_calls)}):\n"
                        for k, tc in enumerate(tool_calls):
                            # Handle both ToolCall objects and dictionaries
                            if hasattr(tc, 'function'):
                                function_name = tc.function.name
                                arguments = tc.function.arguments
                            elif isinstance(tc, dict) and 'function' in tc:
                                function_info = tc.get('function', {})
                                function_name = function_info.get('name', 'unknown')
                                arguments = function_info.get('arguments', '')
                            else:
                                function_name = 'unknown'
                                arguments = ''
                            text_data += f"  [{k}] Function: {function_name}\n"
                            text_data += f"  [{k}] Arguments: {arguments}\n"
                    text_data += "\n" + "-"*50 + "\n"
                
                text_path = os.path.join(log_dir, "text", text_filename)
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(text_data)
                
                    
            except Exception as e:
                print(f"[CONVERSATION_SAVE] ERROR saving conversation {i}: {e}")
                import traceback
                traceback.print_exc()
        
        # 📋 SAVE BATCH SUMMARY
        batch_summary = {
            "timestamp": datetime.now().isoformat(),
            "num_conversations": len(batch_conversations),
            "num_original_prompts": len(prompts),
            "responses_per_prompt": n,
            "session_ids_used": list(set(session_manager._conversation_sessions.get(cid, "unknown") 
                                        for cid in conversation_ids[:len(batch_conversations)])),
            "log_directory": log_dir,
            "files_created": {
                "json_files": len(batch_conversations),
                "text_files": len(batch_conversations),
                "truncation_files": sum(1 for info in truncation_info if info.get("was_truncated", False)) if truncation_info else 0
            }
        }
        
        # Add truncation statistics
        if truncation_info:
            truncated_count = sum(1 for info in truncation_info if info["was_truncated"])
            total_tokens_saved = sum(info["original_length"] - info["truncated_length"] 
                                   for info in truncation_info if info["was_truncated"])
            batch_summary["truncation_stats"] = {
                "total_conversations": len(truncation_info),
                "conversations_truncated": truncated_count,
                "conversations_not_truncated": len(truncation_info) - truncated_count,
                "truncation_rate": truncated_count / len(truncation_info) if truncation_info else 0,
                "total_tokens_saved": total_tokens_saved,
                "reasons": {
                    reason: sum(1 for info in truncation_info if info["truncation_reason"] == reason)
                    for reason in set(info["truncation_reason"] for info in truncation_info)
                }
            }
        
        # Add token statistics to summary if available
        if prompt_token_counts:
            batch_summary["token_stats"] = {
                "prompt_tokens": {
                    "avg": sum(prompt_token_counts) / len(prompt_token_counts),
                    "min": min(prompt_token_counts),
                    "max": max(prompt_token_counts),
                    "total": sum(prompt_token_counts)
                }
            }
            
        if sequence_token_counts:
            if "token_stats" not in batch_summary:
                batch_summary["token_stats"] = {}
            batch_summary["token_stats"]["sequence_tokens"] = {
                "avg": sum(sequence_token_counts) / len(sequence_token_counts),
                "min": min(sequence_token_counts),
                "max": max(sequence_token_counts),
                "total": sum(sequence_token_counts)
            }
            
            if prompt_token_counts:
                growth_counts = [seq - prompt for seq, prompt in zip(sequence_token_counts, prompt_token_counts)]
                batch_summary["token_stats"]["token_growth"] = {
                    "avg": sum(growth_counts) / len(growth_counts),
                    "min": min(growth_counts),
                    "max": max(growth_counts)
                }
        
        summary_path = os.path.join(log_dir, "batch_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(batch_summary, f, indent=2, ensure_ascii=False)
        
        print(f"[CONVERSATION_SAVE] Batch complete. Summary saved to {summary_path}")

    def _analyze_conversation(self, conversation: List[Dict[str, str]]) -> Dict:
        """Analyze conversation structure and patterns."""
        analysis = {
            "message_roles": [msg.get('role', 'unknown') for msg in conversation],
            "role_transitions": [],
            "tool_call_patterns": [],
            "conversation_flow": []
        }
        
        # Analyze role transitions
        for i in range(1, len(conversation)):
            prev_role = conversation[i-1].get('role', 'unknown')
            curr_role = conversation[i].get('role', 'unknown')
            analysis["role_transitions"].append(f"{prev_role} -> {curr_role}")
        
        # Analyze tool usage patterns
        for i, msg in enumerate(conversation):
            if msg.get('tool_calls'):
                tool_names = []
                for tc in msg.get('tool_calls', []):
                    # Handle both ToolCall objects and dictionaries
                    if hasattr(tc, 'function'):
                        tool_names.append(tc.function.name)
                    elif isinstance(tc, dict) and 'function' in tc:
                        tool_names.append(tc['function'].get('name', 'unknown'))
                    else:
                        tool_names.append('unknown')
                analysis["tool_call_patterns"].append({
                    "message_index": i,
                    "tool_names": tool_names,
                    "num_tools": len(tool_names)
                })
        
        # Analyze conversation flow
        for i, msg in enumerate(conversation):
            role = msg.get('role', 'unknown')
            content_length = len(msg.get('content', ''))
            has_tools = bool(msg.get('tool_calls'))
            is_tool_response = bool(msg.get('tool_call_id'))
            
            analysis["conversation_flow"].append({
                "index": i,
                "role": role,
                "content_length": content_length,
                "has_tool_calls": has_tools,
                "is_tool_response": is_tool_response
            })
        
        return analysis

    def _analyze_tool_usage(self, conversation: List[Dict[str, str]]) -> Dict:
        """Analyze tool usage patterns in the conversation."""
        tool_analysis = {
            "tools_used": [],
            "tool_success_patterns": [],
            "tool_call_sequences": [],
            "tool_response_quality": []
        }
        
        tool_call_to_response = {}
        
        for i, msg in enumerate(conversation):
            role = msg.get('role', 'unknown')
            
            if role == 'assistant' and msg.get('tool_calls'):
                # Assistant making tool calls
                for tc in msg.get('tool_calls', []):
                    # Handle both ToolCall objects and dictionaries
                    if hasattr(tc, 'function'):
                        # ToolCall object
                        tool_call_id = tc.id
                        function_name = tc.function.name
                        arguments = tc.function.arguments
                    elif isinstance(tc, dict):
                        # Dictionary representation
                        tool_call_id = tc.get('id', 'unknown')
                        function_info = tc.get('function', {})
                        function_name = function_info.get('name', 'unknown')
                        arguments = function_info.get('arguments', '')
                    else:
                        # Fallback
                        tool_call_id = 'unknown'
                        function_name = 'unknown'
                        arguments = ''
                    
                    tool_info = {
                        "message_index": i,
                        "tool_call_id": tool_call_id,
                        "function_name": function_name,
                        "arguments": arguments,
                        "arguments_length": len(arguments) if arguments else 0
                    }
                    tool_analysis["tools_used"].append(tool_info)
                    tool_call_to_response[tool_call_id] = {"call": tool_info, "response": None}
            
            elif role == 'tool':
                # Tool response
                tool_call_id = msg.get('tool_call_id')
                content = msg.get('content', '')
                
                response_info = {
                    "message_index": i,
                    "tool_call_id": tool_call_id,
                    "response_length": len(content),
                    "response_preview": content[:100] + "..." if len(content) > 100 else content,
                    "appears_successful": not any(error_word in content.lower() 
                                                for error_word in ['error', 'failed', 'exception', 'invalid'])
                }
                tool_analysis["tool_response_quality"].append(response_info)
                
                if tool_call_id in tool_call_to_response:
                    tool_call_to_response[tool_call_id]["response"] = response_info
        
        # Analyze success patterns
        for tool_call_id, data in tool_call_to_response.items():
            if data["response"]:
                success_pattern = {
                    "tool_call_id": tool_call_id,
                    "function_name": data["call"]["function_name"],
                    "call_index": data["call"]["message_index"],
                    "response_index": data["response"]["message_index"],
                    "appears_successful": data["response"]["appears_successful"],
                    "response_delay": data["response"]["message_index"] - data["call"]["message_index"]
                }
                tool_analysis["tool_success_patterns"].append(success_pattern)
        
        return tool_analysis

    def _mask_out_tools_calling_tokens(
        self,
        raw_prompts: List[List[Dict[str, str]]],
        batch_conversations: List[List[Dict[str, str]]],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mask out tools calling tokens in the responses.

        Args:
            raw_prompts: [prompt] from input dataset
            batch_conversations: [prompt + response]
            input_ids: responses tokens
            attention_mask: responses attention mask

        Returns:
            mask: (batch_size, response_length)
        """
        batch_size = input_ids.size(0)
        assert len(raw_prompts) == batch_size, f"{len(raw_prompts)} != {batch_size}"
        assert len(batch_conversations) == batch_size, f"{len(batch_conversations)} != {batch_size}"

        # Deduplicate adjacent tool calls, since they're merged into one turn.
        # [user, assistant, tool, tool, assistant] -> [user, assistant, tool, assistant]
        # TODO: it's chat_template specific, find a more generic way to do this.
        def deduplicate_adjacent_tool_calls(roles):
            result = []
            for role, group in itertools.groupby(roles):
                if role == "tool":
                    result.append(role)
                else:
                    result.extend(group)
            return result

        loss_mask = attention_mask.clone()
        for i in range(batch_size):
            responses = batch_conversations[i][len(raw_prompts[i]) :]
            assert len(responses) > 0, f"responses is empty: {responses}"

            roles = deduplicate_adjacent_tool_calls([response["role"] for response in responses])
            # Each turn should be: [BOS]...[EOS]
            all_eos_indices = input_ids[i].eq(self.tokenizer.eos_token_id).nonzero().squeeze(1)
            
            # Handle edge case: fewer EOS tokens than expected roles (safety check)
            if len(all_eos_indices) < len(roles):
                print(f"[MASK_WARNING] Sequence {i} has {len(all_eos_indices)} EOS tokens but {len(roles)} roles. Using available EOS tokens.")
                eos_indices = all_eos_indices
                effective_roles = len(all_eos_indices)
            else:
                eos_indices = all_eos_indices[:len(roles)]
                effective_roles = len(roles)
            
            for j in range(effective_roles):
                if j < len(roles) and roles[j] == "tool":
                    bos = eos_indices[j - 1] + 1 if j > 0 else 0
                    eos = eos_indices[j]
                    loss_mask[i, bos : eos + 1] = 0

        return loss_mask


class ChatCompletionScheduler:
    def __init__(
        self,
        config: DictConfig,
        server_addresses: List[str],
        max_cache_size: int = 10000,
    ):
        """
        Args:
            config: DictConfig.
            server_addresses: List[str], OpenAI compatible server addresses.
            max_cache_size: int, max cache size of request_id to address mapping.
        """
        self.config = config.actor_rollout_ref.rollout
        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])

        # Least requests load balancing
        self.weighted_addresses = [[0, address] for address in server_addresses]
        heapq.heapify(self.weighted_addresses)

        # LRU cache to map request_id to address
        self.request_id_to_address = LRUCache(maxsize=max_cache_size)

        self.background_tasks = set()
        if self.config.multi_turn.completion_callback is None:
            self.completion_callback = ToolCompletionCallback(config, self)
            logger.warning("completion_callback is None, use ToolCompletionCallback")
        else:
            module_path, class_name = self.config.multi_turn.completion_callback.rsplit(".", 1)
            module = importlib.import_module(module_path)
            self.completion_callback = getattr(module, class_name)(config, self)

    def submit_chat_completions(self, *, messages: List[Dict[str, str]], request_id: str, info: Dict[str, Any]):
        """Submit chat completion request without wait, completion_callback will be called when the request is done.

        Args:
            messages: List of messages.
            request_id: Request id.
            info: Any other auxiliary information pass across multi-turn.
        """
        info["__depth__"] += 1
        task = asyncio.create_task(self._submit_chat_completions_and_callback(messages, request_id, info))

        # "fire-and-forget" background tasks
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    async def _submit_chat_completions_and_callback(
        self,
        messages: List[Dict[str, str]],
        request_id: str,
        info: Dict[str, Any],
    ):
        """Submit chat completion request, wait request finish and do callback."""
        if request_id:
            request_id = request_id.removeprefix("chatcmpl-")
            assert request_id in self.request_id_to_address
            address = self.request_id_to_address.pop(request_id)
        else:
            address = self.weighted_addresses[0][1]
            self.weighted_addresses[0][0] += 1
            heapq.heapreplace(self.weighted_addresses, self.weighted_addresses[0])

        # use new request_id to avoid duplicate request_id problem
        request_id = uuid4().hex
        self.request_id_to_address[request_id] = address

        completions, exception = None, None
        try:
            # NOTE: OpenAI client uses httpx, seems to have performance issue in high concurrency requests.
            completions = await self._chat_completions_aiohttp(
                address,
                messages=messages,
                tools=self.completion_callback.tool_schemas,
                extra_body=self.completion_callback.extra_body,
                extra_headers={"x-request-id": request_id},
                **info["__sampling_params__"],
            )
        except Exception as e:
            # Let user handle the exception
            exception = e

        info["__depth__"] -= 1

        if exception is not None:
            logger.exception(f"chat completion failed with exception: {exception}")
        else:
            try:
                await self.completion_callback(messages, completions, info)
            except Exception as e:
                logger.exception(f"completion callback failed with exception: {e}")

        # No more ongoing completion requests
        if info["__depth__"] == 0:
            info["__done__"].set()

    async def _chat_completions_openai(self, address: str, **chat_complete_request) -> ChatCompletion:
        client = AsyncOpenAI(base_url=f"http://{address}/v1", api_key="token-abc123", timeout=None, max_retries=0)
        return await client.chat.completions.create(**chat_complete_request)

    async def _chat_completions_aiohttp(self, address: str, **chat_complete_request) -> ChatCompletion:
        # print(f"[DEBUG] Chat completion request: {chat_complete_request}")
        t_request_start = time.time()
        
        extra_body = chat_complete_request.pop("extra_body", {})
        chat_complete_request.update(extra_body or {})
        extra_headers = chat_complete_request.pop("extra_headers")
        
        # Extract request_id for debug logging
        request_id = extra_headers.get("x-request-id", "unknown")
        
        # Debug logging - print the exact request being sent
        # print(f"[DEBUG] Chat completion request {request_id} to {address}:")
        # print(f"[DEBUG] Messages ({len(chat_complete_request.get('messages', []))} total):")
        # for i, msg in enumerate(chat_complete_request.get('messages', [])):
        #     content_preview = str(msg.get('content', ''))[:200] + ('...' if len(str(msg.get('content', ''))) > 200 else '')
        #     print(f"  [{i}] {msg.get('role', 'unknown')}: {content_preview}")
        
        # if 'tools' in chat_complete_request and chat_complete_request['tools']:
        #     print(f"[DEBUG] Tools available ({len(chat_complete_request['tools'])} total):")
        #     for i, tool in enumerate(chat_complete_request['tools']):
        #         tool_name = tool.get('function', {}).get('name', 'unknown')
        #         print(f"  [{i}] {tool_name}")
        
        # print(f"[DEBUG] Sampling params: temperature={chat_complete_request.get('temperature')}, top_p={chat_complete_request.get('top_p')}")
        
        # Use shared HTTP client with no timeout limit for LLM inference
        # t_http_start = time.time()
        async with http_request('POST', f"http://{address}/v1/chat/completions",
                               headers={"Authorization": "Bearer token-abc123", **extra_headers},
                               json=chat_complete_request,
                               timeout=1200) as resp:
            data = await resp.json()
            # t_http_end = time.time()
            
            # Debug logging - print the response
            completion = ChatCompletion(**data)
            if completion.choices and len(completion.choices) > 0:
                choice = completion.choices[0]
                if choice.message:
                    content_preview = str(choice.message.content or '')[:200] + ('...' if len(str(choice.message.content or '')) > 200 else '')
                    # print(f"[DEBUG] Response {request_id}: {choice.message.role}: {content_preview}")
                    
                    # if choice.message.tool_calls:
                    #     print(f"[DEBUG] Tool calls in response ({len(choice.message.tool_calls)} total):")
                    #     for i, tool_call in enumerate(choice.message.tool_calls):
                    #         print(f"  [{i}] {tool_call.function.name}: {tool_call.function.arguments}")
            
            # t_request_end = time.time()
            # print(f"[ASYNC_VLLM_HTTP_TIMING] Request {request_id} - HTTP: {t_http_end - t_http_start:.2f}s, Total: {t_request_end - t_request_start:.2f}s")
            
            return completion

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        timing_detailed = {}
        t_total_start = time.time()
        
        print(f"[ASYNC_VLLM_TIMING] Starting async VLLM data collection for {len(batch)} prompts")
        
        with simple_timer("setup_sampling_params", timing_detailed):
            kwargs = dict(
                model=self.model_name,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
            )

            # override sampling params for validation
            if batch.meta_info.get("validate", False):
                kwargs["top_p"] = self.config.val_kwargs.top_p
                kwargs["temperature"] = self.config.val_kwargs.temperature
                kwargs["repetition_penalty"] = self.config.val_kwargs.repetition_penalty

        # print(f"[ChatCompletionScheduler] generate_sequences sampling params: {kwargs}")

        # NOTE: For multi-turn rollout, repeat raw_prompt n times and process each prompt independently,
        # validation dataset has already been repeated in `PPOTrainer._validate`.
        n = 1 if batch.meta_info.get("validate", False) else self.config.n
        
        with simple_timer("prepare_conversations", timing_detailed):
            tasks, batch_conversations = [], [None] * len(batch) * n
            conversation_ids = []
            
            for batch_index, conversation in enumerate(batch.non_tensor_batch["raw_prompt"].repeat(n, axis=0)):
                # raw_prompt: [{"role": "user", "content": ""}, ["role": "assistant", "content"], ...]
                batch_conversations[batch_index] = conversation.tolist()
                
                # Create a unique conversation ID for this conversation
                conversation_id = str(uuid4())
                conversation_ids.append(conversation_id)
        
        print(f"[ASYNC_VLLM_TIMING] Prepared {len(batch_conversations)} conversations in {timing_detailed['prepare_conversations']:.2f}s")
        
        with simple_timer("session_creation", timing_detailed):
            # Pre-create sessions for all conversations
            session_manager = get_session_manager()
            session_creation_tasks = []

            # Extract image tags from batch if available and expand them n times to match conversations
            original_image_tags = batch.non_tensor_batch.get("docker_image", [])
            original_prompts = batch.non_tensor_batch["raw_prompt"]
            image_tags = np.repeat(original_image_tags, n, axis=0)  # Repeat each image n times
            
            # 🔍 VERIFICATION CHECK: Track (prompt, image) pairs
            prompt_image_pairs = {}
            
            for i in range(len(conversation_ids)):
                conversation = batch_conversations[i]
                raw_image_tag = image_tags[i]
                image_tag = str(raw_image_tag.item() if hasattr(raw_image_tag, 'item') else raw_image_tag)
                
                # Extract the actual prompt content
                prompt_content = "No user message found"
                if isinstance(conversation, list) and len(conversation) > 0:
                    for msg in conversation:
                        if isinstance(msg, dict) and msg.get('role') == 'user':
                            prompt_content = msg.get('content', '')
                            break
                
                # Track (prompt, image) pair
                pair = (prompt_content, image_tag)
                if pair not in prompt_image_pairs:
                    prompt_image_pairs[pair] = 0
                prompt_image_pairs[pair] += 1
            
            # Validate: should have exactly len(original_prompts) unique pairs, each appearing n times
            assert len(prompt_image_pairs) == len(original_prompts), f"Expected {len(original_prompts)} unique pairs, got {len(prompt_image_pairs)}"
            
            for pair, count in prompt_image_pairs.items():
                assert count == n, f"Pair ('{pair[0][:50]}...', '{pair[1]}') has {count} replicas, expected {n}"
            
            print(f"[PROMPT_IMAGE_MATCHING] Validation passed: {len(prompt_image_pairs)} unique pairs, each appearing {n} times")
            
            for i, conversation_id in enumerate(conversation_ids):
                try:
                    if i < len(image_tags):
                        raw_image_tag = image_tags[i]
                        
                        # Simple conversion - handle numpy arrays/scalars properly
                        if hasattr(raw_image_tag, 'item'):
                            image_tag = str(raw_image_tag.item())
                        else:
                            image_tag = str(raw_image_tag)
                    else:
                        raise ValueError(f"conversation index {i} >= len(image_tags) {len(image_tags)}")
                    
                    session_creation_tasks.append(
                        session_manager.get_session_for_conversation(conversation_id, image_tag=image_tag)
                    )
                except Exception as e:
                    logger.error(f"Failed to pre-create session for conversation {conversation_id} with image {image_tag}: {e}")
            
            if session_creation_tasks:
                await asyncio.gather(*session_creation_tasks, return_exceptions=True)
        
        # with simple_timer("session_creation", timing_detailed):
        #     # Pre-create sessions for all conversations
        #     session_manager = get_session_manager()
        #     session_creation_tasks = []
        #     for conversation_id in conversation_ids:
        #         try:
        #             session_creation_tasks.append(session_manager.get_session_for_conversation(conversation_id))
        #         except Exception as e:
        #             logger.error(f"Failed to pre-create session for conversation {conversation_id}: {e}")
            
        #     if session_creation_tasks:
        #         await asyncio.gather(*session_creation_tasks, return_exceptions=True)
                
        # print(f"[ASYNC_VLLM_TIMING] Created sessions in {timing_detailed['session_creation']:.2f}s")
        
        with simple_timer("task_creation", timing_detailed):
            for batch_index, conversation_id in enumerate(conversation_ids):
                # print(f"[DEBUG] Batch conversation {batch_index}: {batch_conversations[batch_index]}")
                tasks.append(
                    asyncio.create_task(
                        self._submit_chat_completions_semaphore(
                            messages=batch_conversations[batch_index],
                            request_id=None,
                            sampling_params=kwargs,
                            conversation_id=conversation_id,
                        )
                    )
                )
        
        # print(f"[ASYNC_VLLM_TIMING] Created {len(tasks)} async tasks in {timing_detailed['task_creation']:.2f}s")

        try:
            with simple_timer("async_execution", timing_detailed):
                # print(f"[ASYNC_VLLM_TIMING] Starting async execution of {len(tasks)} tasks...")
                t_async_start = time.time()
                await asyncio.gather(*tasks)
                t_async_end = time.time()
                # print(f"[ASYNC_VLLM_TIMING] Completed async execution in {t_async_end - t_async_start:.2f}s")
                
            with simple_timer("postprocess_batch", timing_detailed):
                # Add conversation IDs to batch metadata for conversation saving
                batch.non_tensor_batch["__conversation_ids__"] = np.array(conversation_ids, dtype=object)
                
                # Store session information for reward calculation
                session_manager = get_session_manager()
                session_ids = []
                for conversation_id in conversation_ids:
                    session_id = session_manager._conversation_sessions.get(conversation_id, None)
                    session_ids.append(session_id)
                    print(f"[SESSION_DEBUG] Conversation {conversation_id} -> Session {session_id}")
                batch.non_tensor_batch["__session_ids__"] = np.array(session_ids, dtype=object)
                
                output_batch = self.completion_callback.postprocess(batch, batch_conversations, n=n)
                
            timing_detailed["total_time"] = time.time() - t_total_start
            output_batch.meta_info["timing"] = timing_detailed
            output_batch.meta_info["timing"]["generate_sequences"] = timing_detailed["total_time"]  # Keep backward compatibility
            
            print(f"[ASYNC_VLLM_TIMING] Data collection complete! Total time: {timing_detailed['total_time']:.2f}s")
            print(f"[ASYNC_VLLM_TIMING] Breakdown:")
            for phase, duration in timing_detailed.items():
                percentage = (duration / timing_detailed['total_time']) * 100 if timing_detailed['total_time'] > 0 else 0
                print(f"  - {phase}: {duration:.2f}s ({percentage:.1f}%)")
            
            print("[ChatCompletionScheduler] generate_sequences done")
            return output_batch
        finally:
            # DON'T clean up sessions here - they will be cleaned up after reward calculation
            print(f"[DEBUG] Keeping {len(conversation_ids)} sessions alive for reward calculation")
            # Sessions will be cleaned up by the reward manager after verification

    async def _submit_chat_completions_semaphore(self, messages: List[Dict[str, str]], request_id: str, sampling_params: Dict[str, Any], conversation_id: str):
        t_start = time.time()
        
        done = asyncio.Event()

        info = {
            "__done__": done,
            "__depth__": 0,  # indicate how many ongoing completion requests
            "__sampling_params__": sampling_params,
            "__conversation_id__": conversation_id,  # Pass conversation ID through
            "__start_time__": t_start,  # Track start time for this request
        }

        self.submit_chat_completions(messages=messages, request_id=request_id, info=info)

        # Wait until all completion requests are done
        await done.wait()
        
        t_end = time.time()
        # print(f"[ASYNC_VLLM_REQUEST_TIMING] Request {conversation_id} completed in {t_end - t_start:.2f}s")
