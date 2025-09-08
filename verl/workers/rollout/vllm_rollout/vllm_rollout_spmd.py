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
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import logging
import os
import json
import re
import asyncio
import aiohttp
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.lora.request import LoRARequest
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics

# Tool calling patterns
TOOL_CALL_PATTERN = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
FUNCTION_CALL_PATTERN = r'```json\s*(\{.*?\})\s*```'

class ToolCallExecutor:
    """Handles tool calling and API requests"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.max_iterations = config.get("max_tool_iterations", 100)
        self.api_timeout = config.get("api_timeout", 30)
        
    async def execute_api_call(self, tool_call: Dict[str, Any]) -> str:
        """Execute API call based on tool call"""

        try:
            # Always send to request catcher as a string
            endpoint = "https://toolbox.modal-origin.relace.run/session/001/tool/"
            
            # Convert the entire tool call to a JSON string
            data = json.dumps(tool_call)
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.api_timeout)) as session:
                async with session.post(endpoint, json=tool_call) as response:
                    result = await response.text()
            
            return f"API Response: {result}"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Extract tool calls from generated text"""
        tool_calls = []
        
        # Look for <tool_call> pattern
        matches = re.findall(TOOL_CALL_PATTERN, text, re.DOTALL)
        for match in matches:
            try:
                tool_call = json.loads(match.strip())
                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool call: {match}")
        
        # Look for function call pattern
        if not tool_calls:
            matches = re.findall(FUNCTION_CALL_PATTERN, text, re.DOTALL)
            for match in matches:
                try:
                    tool_call = json.loads(match.strip())
                    tool_calls.append(tool_call)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse function call: {match}")
        
        return tool_calls
    
    def has_tool_calls(self, text: str) -> bool:
        """Check if text contains tool calls"""
        return bool(re.search(TOOL_CALL_PATTERN, text, re.DOTALL) or 
                   re.search(FUNCTION_CALL_PATTERN, text, re.DOTALL))

# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), "disable CUDA graph (enforce_eager = False) if free cache engine"
        
        # Initialize tool calling capabilities
        self.enable_tool_calling = config.get("enable_tool_calling", True)
        if self.enable_tool_calling:
            self.tool_executor = ToolCallExecutor(config)
            self.debug_tool_calling = config.get("debug_tool_calling", False)  # Enable by default for debugging
            logger.info("Tool calling enabled")
        else:
            self.tool_executor = None
            self.debug_tool_calling = False

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            if vllm_version in (
                "0.5.4",
                "0.6.3",
            ):
                train_tp = kwargs.get("train_tp")
                num_tp_per_train_tp = train_tp // tensor_parallel_size
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size, num_tp_per_train_tp=num_tp_per_train_tp)
            else:
                vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(model_hf_config.llm_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(model_hf_config.text_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")

            assert max_position_embeddings >= config.prompt_length + config.response_length, "model context length should be greater than total sequence length"

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = {} if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **lora_kwargs,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != "0.3.1":
            kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        # print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    async def execute_tool_calls_and_continue_generation(self, vllm_inputs: List[Dict], batch_size: int, non_tensor_batch: Dict, **kwargs) -> List[List[int]]:
        """Execute tool calls and continue generation iteratively"""
        all_responses = []
        
        # Create debug directory (only if debugging is enabled)
        debug_dir = None
        if self.debug_tool_calling:
            import os
            import datetime
            debug_dir = f"tool_calling_debug_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(debug_dir, exist_ok=True)
            print(f"[DEBUG] Tool calling debug mode enabled, saving to: {debug_dir}")
        
        for batch_idx in range(batch_size):
            current_prompt = vllm_inputs[batch_idx].copy()
            
            # Extract original conversation structure from raw_prompt
            conversation_history = []
            try:
                if "raw_prompt" in non_tensor_batch and batch_idx < len(non_tensor_batch["raw_prompt"]):
                    # Get the original conversation structure
                    raw_prompt = non_tensor_batch["raw_prompt"][batch_idx]
                    if isinstance(raw_prompt, np.ndarray):
                        raw_prompt = raw_prompt.tolist()
                    
                    # Copy the original conversation messages (system + user messages)
                    conversation_history = raw_prompt.copy()
                    print(f"[DEBUG] Extracted original conversation with {len(conversation_history)} messages:")
                    for i, msg in enumerate(conversation_history):
                        print(f"  [{i}] {msg['role']}: {msg['content'][:150]}{'...' if len(msg['content']) > 150 else ''}")
                else:
                    # Fallback: try to parse the formatted prompt text
                    original_prompt_ids = current_prompt.get("prompt_token_ids", [])
                    original_prompt_text = self.tokenizer.decode(original_prompt_ids, skip_special_tokens=True)
                    
                    # Basic parsing to extract system and user parts
                    # This is a rough fallback - ideally we'd always have raw_prompt
                    if "<|im_start|>system" in original_prompt_text:
                        # Try to parse system and user messages
                        parts = original_prompt_text.split("<|im_start|>")
                        for part in parts[1:]:  # Skip first empty part
                            if part.startswith("system"):
                                content = part.replace("system\n", "").split("<|im_end|>")[0]
                                conversation_history.append({"role": "system", "content": content})
                            elif part.startswith("user"):
                                content = part.replace("user\n", "").split("<|im_end|>")[0]
                                conversation_history.append({"role": "user", "content": content})
                    else:
                        # Last resort: treat entire prompt as user message
                        conversation_history.append({
                            "role": "user", 
                            "content": original_prompt_text
                        })
                    
                    print(f"[DEBUG] Parsed conversation from formatted prompt: {len(conversation_history)} messages")
                    for i, msg in enumerate(conversation_history):
                        print(f"  [{i}] {msg['role']}: {msg['content'][:150]}{'...' if len(msg['content']) > 150 else ''}")
            except Exception as e:
                print(f"[DEBUG] Warning: Could not extract original conversation: {e}")
                # Continue with empty conversation history - this will likely fail downstream
            
            iteration = 0
            
            # Initialize trajectory debug data
            trajectory_debug = {
                "batch_idx": batch_idx,
                "initial_prompt": current_prompt,
                "iterations": [],
                "final_response": None,
                "total_iterations": 0,
                "tool_calls_made": 0,
                "errors": []
            } if self.debug_tool_calling else None
            
            while iteration < self.tool_executor.max_iterations:
                iteration += 1
                print(f"[DEBUG] Tool calling iteration {iteration} for batch {batch_idx}")
                
                # Initialize iteration debug data
                iteration_debug = {
                    "iteration": iteration,
                    "input_prompt_length": len(current_prompt.get("prompt_token_ids", [])),
                    "conversation_before": conversation_history.copy(),
                    "generated_text": None,
                    "tool_calls_found": [],
                    "tool_responses": [],
                    "conversation_after": None,
                    "complete_conversation_history": None,
                    "new_prompt_length": None,
                    "has_tool_calls": False,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
                # Generate response
                with self.update_sampling_params(**kwargs):
                    outputs = self.inference_engine.generate(
                        prompts=[current_prompt],
                        sampling_params=self.sampling_params,
                        use_tqdm=False,
                    )
                
                # Get the generated text
                generated_tokens = outputs[0].outputs[0].token_ids
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
                
                # Save iteration data
                iteration_debug["generated_text"] = generated_text
                
                print(f"[DEBUG] === ITERATION {iteration} for BATCH {batch_idx} ===")
                print(f"[DEBUG] Conversation before generation:")
                for i, msg in enumerate(conversation_history):
                    print(f"  [{i}] {msg['role']}: {msg['content'][:150]}{'...' if len(msg['content']) > 150 else ''}")
                print(f"[DEBUG] Generated text: {generated_text}")
                print(f"[DEBUG] Generated text length: {len(generated_text)} chars")
                
                # Check for tool calls
                if self.tool_executor.has_tool_calls(generated_text):
                    tool_calls = self.tool_executor.extract_tool_calls(generated_text)
                    print(f"[DEBUG] Found {len(tool_calls)} tool calls: {tool_calls}")
                    
                    iteration_debug["has_tool_calls"] = True
                    iteration_debug["tool_calls_found"] = tool_calls
                    if trajectory_debug:
                        trajectory_debug["tool_calls_made"] += len(tool_calls)
                    
                    # Execute all tool calls
                    tool_responses = []
                    for tool_call in tool_calls:
                        try:
                            response = await self.tool_executor.execute_api_call(tool_call)
                            tool_responses.append({
                                "tool_call": tool_call,
                                "response": response,
                                "success": True
                            })
                            logger.info(f"Tool response: {response}")
                        except Exception as e:
                            error_msg = f"Error: {str(e)}"
                            tool_responses.append({
                                "tool_call": tool_call,
                                "response": error_msg,
                                "success": False,
                                "error": str(e)
                            })
                            if trajectory_debug:
                                trajectory_debug["errors"].append({
                                    "iteration": iteration,
                                    "tool_call": tool_call,
                                    "error": str(e)
                                })
                            logger.error(f"Error executing tool call: {e}")
                    
                    iteration_debug["tool_responses"] = tool_responses
                    
                    # Add tool responses to conversation and update prompt
                    response_contents = [tr["response"] for tr in tool_responses]
                    conversation_history.extend([
                        {"role": "assistant", "content": generated_text},
                        {"role": "user", "content": "<tool_responses>\n" + "\n".join(response_contents) + "\n</tool_responses>"}
                    ])
                    
                    iteration_debug["conversation_after"] = conversation_history.copy()
                    iteration_debug["complete_conversation_history"] = conversation_history.copy()
                    
                    print(f"[DEBUG] Complete conversation after tool calls:")
                    for i, msg in enumerate(conversation_history):
                        print(f"  [{i}] {msg['role']}: {msg['content'][:150]}{'...' if len(msg['content']) > 150 else ''}")
                    print(f"[DEBUG] Total conversation length: {len(conversation_history)} messages")
                    
                    # Create new prompt with conversation history
                    if hasattr(self.tokenizer, 'apply_chat_template'):
                        new_prompt_text = self.tokenizer.apply_chat_template(
                            conversation_history, 
                            tokenize=False, 
                            add_generation_prompt=True,
                            enable_thinking=True,
                        )
                        new_prompt_ids = self.tokenizer.encode(new_prompt_text, add_special_tokens=False)
                        current_prompt = {"prompt_token_ids": new_prompt_ids}
                        
                        print(f"[DEBUG] New prompt text created from conversation:")
                        print(f"[DEBUG] Prompt text (first 500 chars): {new_prompt_text[:500]}{'...' if len(new_prompt_text) > 500 else ''}")
                        print(f"[DEBUG] New prompt token length: {len(new_prompt_ids)}")
                    else:
                        # Fallback: concatenate responses
                        all_text = generated_text + "\n" + "\n".join(response_contents) + "\n"
                        new_prompt_ids = self.tokenizer.encode(all_text, add_special_tokens=False)
                        current_prompt = {"prompt_token_ids": new_prompt_ids}
                        
                        print(f"[DEBUG] Fallback prompt text: {all_text[:500]}{'...' if len(all_text) > 500 else ''}")
                        print(f"[DEBUG] New prompt token length: {len(new_prompt_ids)}")
                    
                    iteration_debug["new_prompt_length"] = len(current_prompt["prompt_token_ids"])
                    
                else:
                    # No tool calls, we're done
                    conversation_history.append({"role": "assistant", "content": generated_text})
                    iteration_debug["conversation_after"] = conversation_history.copy()
                    iteration_debug["complete_conversation_history"] = conversation_history.copy()
                    if trajectory_debug:
                        trajectory_debug["final_response"] = generated_text
                    all_responses.append(generated_tokens)
                    
                    print(f"[DEBUG] NO TOOL CALLS - Final conversation:")
                    for i, msg in enumerate(conversation_history):
                        print(f"  [{i}] {msg['role']}: {msg['content'][:150]}{'...' if len(msg['content']) > 150 else ''}")
                    print(f"[DEBUG] Trajectory completed with {len(conversation_history)} messages")
                    
                # Save iteration debug data
                if self.debug_tool_calling and trajectory_debug:
                    trajectory_debug["iterations"].append(iteration_debug)
                    
                    # Save iteration to individual file for real-time debugging
                    iteration_file = os.path.join(debug_dir, f"batch_{batch_idx}_iteration_{iteration}.json")
                    with open(iteration_file, 'w') as f:
                        json.dump(iteration_debug, f, indent=2, default=str)
                
                if not iteration_debug["has_tool_calls"]:
                    break
            
            # If we've exhausted iterations, use the last generated response
            if iteration >= self.tool_executor.max_iterations:
                logger.warning(f"Reached max tool calling iterations ({self.tool_executor.max_iterations})")
                if trajectory_debug:
                    trajectory_debug["errors"].append({
                        "type": "max_iterations_reached",
                        "max_iterations": self.tool_executor.max_iterations
                    })
                if not all_responses or len(all_responses) <= batch_idx:
                    all_responses.append(generated_tokens)
            
            # Finalize trajectory debug data
            if self.debug_tool_calling and trajectory_debug:
                trajectory_debug["total_iterations"] = iteration
                trajectory_debug["completed"] = len(all_responses) > batch_idx
                
                # Save complete trajectory data
                trajectory_file = os.path.join(debug_dir, f"trajectory_batch_{batch_idx}_complete.json")
                with open(trajectory_file, 'w') as f:
                    json.dump(trajectory_debug, f, indent=2, default=str)
                
                # Save human-readable summary
                summary_file = os.path.join(debug_dir, f"trajectory_batch_{batch_idx}_summary.txt")
                with open(summary_file, 'w') as f:
                    f.write(f"TRAJECTORY SUMMARY - Batch {batch_idx}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Total iterations: {trajectory_debug['total_iterations']}\n")
                    f.write(f"Tool calls made: {trajectory_debug['tool_calls_made']}\n")
                    f.write(f"Errors: {len(trajectory_debug['errors'])}\n")
                    f.write(f"Completed: {trajectory_debug['completed']}\n\n")
                    
                    for i, iter_data in enumerate(trajectory_debug["iterations"], 1):
                        f.write(f"ITERATION {i}:\n")
                        f.write(f"  Generated text: {iter_data['generated_text'][:200]}...\n")
                        f.write(f"  Tool calls found: {len(iter_data['tool_calls_found'])}\n")
                        if iter_data['tool_calls_found']:
                            for j, tc in enumerate(iter_data['tool_calls_found']):
                                f.write(f"    Tool call {j+1}: {tc}\n")
                        f.write(f"  Tool responses: {len(iter_data['tool_responses'])}\n")
                        
                        # Write complete conversation history
                        if 'complete_conversation_history' in iter_data and iter_data['complete_conversation_history']:
                            f.write(f"  Complete conversation ({len(iter_data['complete_conversation_history'])} messages):\n")
                            for j, msg in enumerate(iter_data['complete_conversation_history']):
                                f.write(f"    [{j}] {msg['role']}: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}\n")
                        f.write("\n")
        
        # Save overall debug summary
        if self.debug_tool_calling and debug_dir:
            summary_file = os.path.join(debug_dir, "debug_summary.txt")
            with open(summary_file, 'w') as f:
                f.write(f"TOOL CALLING DEBUG SESSION\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total batches processed: {batch_size}\n")
                f.write(f"Debug files saved to: {debug_dir}\n\n")
                f.write("Files created:\n")
                for batch_idx in range(batch_size):
                    f.write(f"  - trajectory_batch_{batch_idx}_complete.json\n")
                    f.write(f"  - trajectory_batch_{batch_idx}_summary.txt\n")
                    f.write(f"  - batch_{batch_idx}_iteration_*.json (per iteration)\n")
            
            print(f"[DEBUG] Tool calling debug data saved to: {debug_dir}")
        return all_responses

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        print(f"[DEBUG] Hey inside vllm_rollout_spmd.generate_sequences")
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.init_cache_engine()

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array([_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")] * batch_size

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            # Check if tool calling is enabled
            if self.enable_tool_calling and self.tool_executor:
                logger.info("Using tool calling generation mode")
                import asyncio
                # Use async tool calling generation - now pass non_tensor_batch
                if asyncio.iscoroutinefunction(self.execute_tool_calls_and_continue_generation):
                    response_tokens = asyncio.run(self.execute_tool_calls_and_continue_generation(vllm_inputs, batch_size, non_tensor_batch, **kwargs))
                else:
                    # Fallback to sync version
                    response_tokens = []
                    for i in range(batch_size):
                        outputs = self.inference_engine.generate(
                            prompts=[vllm_inputs[i]],
                            sampling_params=self.sampling_params,
                            lora_request=[lora_requests[i]] if lora_requests else None,
                            use_tqdm=False,
                        )
                        response_tokens.append(outputs[0].outputs[0].token_ids)
                
                response = response_tokens
                rollout_log_probs = []  # TODO: Implement log probs for tool calling mode
            else:
                # Standard generation without tool calling
                outputs = self.inference_engine.generate(
                    prompts=vllm_inputs,  # because we have already convert it to prompt token id
                    sampling_params=self.sampling_params,
                    lora_request=lora_requests,
                    use_tqdm=False,
                )

                # TODO(sgm): disable logprob when recompute_log_prob is enable
                # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

                response = []
                rollout_log_probs = []
                for output in outputs:
                    for sample_id in range(len(output.outputs)):
                        response_ids = output.outputs[sample_id].token_ids
                        response.append(response_ids)
                        if self.config.calculate_log_probs:
                            curr_log_prob = []
                            for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                                curr_log_prob.append(logprob[response_ids[i]].logprob)
                            rollout_log_probs.append(curr_log_prob)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_2d_list_to_length(rollout_log_probs, -1, max_length=self.config.response_length).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)

            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                # NOTE(linjunrong): for multi-turn https://github.com/volcengine/verl/pull/1037
                if "tools_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["tools_kwargs"] = _repeat_interleave(non_tensor_batch["tools_kwargs"], self.sampling_params.n)
                if "interaction_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["interaction_kwargs"] = _repeat_interleave(non_tensor_batch["interaction_kwargs"], self.sampling_params.n)
                if "raw_prompt" in non_tensor_batch.keys():
                    non_tensor_batch["raw_prompt"] = _repeat_interleave(non_tensor_batch["raw_prompt"], self.sampling_params.n)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        print(f"[DEBUG] Hey finished vllm_rollout_spmd.generate_sequences, batch: {batch}")

        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs

        # free vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(self, *args, **kwargs):
        # Engine is deferred to be initialized in init_worker
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False

    def init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is initialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)
