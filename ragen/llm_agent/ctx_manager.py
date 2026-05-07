"""
This is the context manager for the LLM agent.
author: Kangrui Wang, Zihan Wang
date: 2025-03-30
"""
from itertools import zip_longest

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import re
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from transformers import AutoTokenizer
import hydra
from ragen.utils import register_resolvers
from ragen.env import REGISTERED_ENV_CONFIGS
from tensordict import TensorDict

from dataclasses import asdict
register_resolvers()

import pdb


def _find_last_subsequence(sequence: List[int], pattern: List[int]) -> int:
    if not pattern or len(pattern) > len(sequence):
        return -1
    for start in range(len(sequence) - len(pattern), -1, -1):
        if sequence[start:start + len(pattern)] == pattern:
            return start
    return -1

def get_special_tokens(tokenizer: AutoTokenizer):
    # Prefer checking actual special tokens over name-based heuristics
    qwen_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    qwen_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if qwen_start_id is not None and qwen_end_id is not None:
        unk_id = tokenizer.unk_token_id
        if (unk_id is None) or (qwen_start_id != unk_id and qwen_end_id != unk_id):
            special_token = qwen_start_id
            reward_token = qwen_end_id
            return special_token, reward_token

    if "qwen" in tokenizer.name_or_path.lower():
        special_token = tokenizer.encode("<|im_start|>")[0]
        reward_token = tokenizer.encode("<|im_end|>")[0]
    elif "llama-3" in tokenizer.name_or_path.lower():
        special_token = 128006
        reward_token = 128009
    else:
        raise ValueError(f"Unsupported model: {tokenizer.name_or_path}")
    return special_token, reward_token

def get_masks_and_scores(input_ids: torch.Tensor, tokenizer: AutoTokenizer, messages_list: List[List[Dict]], all_scores: List[List[float]] = None, use_turn_scores: bool = False, enable_response_mask: bool = False, filter_single_turn: bool = False):
    """
    input_ids: shape (bsz, seq_len)
    Get loss mask that only learns between <|im_start|>assistant and <|im_end|>. Currently only supports qwen.
    NOTE: important! This assumes that the input_ids starts with system and then user & assistant in alternative ways
    """
    special_token, reward_token = get_special_tokens(tokenizer)
    
    turn_starts = torch.where(input_ids == special_token, 1, 0)
    turn_indicators = torch.cumsum(turn_starts, dim=-1)
    if enable_response_mask:
        loss_mask = (turn_indicators % 2 == 1) & (turn_indicators > 1) # only learns all assistant turns
    else:
        loss_mask = (turn_indicators > 1) # learns everything after system prompt
    response_mask = (turn_indicators % 2 == 1) & (turn_indicators > 1)
    
    score_tensor = torch.zeros_like(input_ids, dtype=torch.float32)
    if use_turn_scores: 
        # NOTE: Never enter this branch
        # # Build a per-sample list indicating which assistant turns have empty responses
        # empty_assistant_turns = []  # List[List[bool]] with shape: (batch_size, num_assistant_turns_for_sample)
        # for messages in messages_list:
        #     assistant_contents = [m.get("content", "") for m in messages if m.get("role") == "assistant"]
        #     empty_assistant_turns.append([c.strip() == "" for c in assistant_contents])

        for idx, scores in enumerate(zip_longest(*all_scores, fillvalue=0)):
            scores = torch.tensor(scores, dtype=torch.float32)
            # # Zero-out scores where the corresponding assistant response is empty for this turn
            # empty_mask = torch.tensor(
            #     [(idx < len(flags) and flags[idx]) for flags in empty_assistant_turns],
            #     dtype=torch.bool,
            # )
            # scores = torch.where(empty_mask, torch.zeros_like(scores), scores)

            turn_indicator = idx * 2 + 3 # 0: pad. 1: system. 2+2n: user. 3+2n: assistant
            reward_position = (input_ids == reward_token) & (turn_indicators == turn_indicator)
            # Set the last token of the rows where all positions are False to True
            reward_position[~reward_position.any(dim=-1), -1] = True
            score_tensor[reward_position] = scores
        if "qwen" in tokenizer.name_or_path.lower():
            # for Qwen, there is a "\n" between special token and reward token, so we shift this to make sure reward is assigned to the last token of a turn
            score_tensor = score_tensor.roll(shifts=1, dims=-1)
    else:
        # pdb.set_trace()
        if filter_single_turn:
            scores = [0 if len(i) == 1 else sum(i) for i in all_scores]
        else:
            scores = [sum(i) for i in all_scores]
        score_tensor[:, -1] = torch.tensor(scores, dtype=torch.float32)
    score_tensor = score_tensor[:, 1:] # remove the first token
    loss_mask = loss_mask[:, :-1] # remove the last token
    response_mask = response_mask[:, :-1] # remove the last token

    return score_tensor, loss_mask, response_mask



class ContextManager:
    """
    Manages the context for LLM interactions with environments.
    Translates between environment outputs and LLM inputs, and vice versa.
    """

    def __init__(self, 
                 config,
                 tokenizer,
                 processor = None,
                 mode: str = "train",
                 ):
        """
        Initialize the ContextManager.
        Processor is used to process the image data.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.mode = mode
        self.action_sep = self.config.agent_proxy.action_sep
        self.special_token_list = ["<think>", "</think>", "<answer>", "</answer>", "<|im_start|>", "<|im_end|>"]

        self.es_cfg = self.config.es_manager[mode]
        self.env_nums = {
                env_tag: n_group * self.es_cfg.group_size
                for n_group, env_tag in zip(self.es_cfg.env_configs.n_groups, self.es_cfg.env_configs.tags)
        }
        self._init_prefix_lookup()

    def refresh_from_config(self):
        # Refresh cached env config when es_manager settings change at runtime
        self.es_cfg = self.config.es_manager[self.mode]
        self.env_nums = {
            env_tag: n_group * self.es_cfg.group_size
            for n_group, env_tag in zip(self.es_cfg.env_configs.n_groups, self.es_cfg.env_configs.tags)
        }
        self._init_prefix_lookup()
    
    def _check_env_installed(self, env_type: str):
        if env_type not in REGISTERED_ENV_CONFIGS:
            raise ValueError(f"Environment {env_type} is not installed. Please install it using the scripts/setup_{env_type}.sh script.")

    def _init_prefix_lookup(self):
        prefix_lookup = {}
        prefixes = {}
        env_config_lookup = {}
        env_config = {}
        for env_tag, env_config in self.config.custom_envs.items():
            if env_tag not in self.es_cfg.env_configs.tags:
                continue

            self._check_env_installed(env_config.env_type)
            env_config_new = asdict(REGISTERED_ENV_CONFIGS[env_config.env_type]())
            for k,v in env_config.items():
                env_config_new[k] = v
            env_instruction = env_config_new.get("env_instruction", "")
            if env_config_new.get("grid_vocab", False):
                grid_vocab_str = "\nThe meaning of each symbol in the state is:\n" + ", ".join([f"{k}: {v}" for k, v in env_config_new["grid_vocab"].items()])
                env_instruction += grid_vocab_str
            if env_config_new.get("action_lookup", False):
                action_lookup_str = "\nYour available actions are:\n" + ", ".join([f"{v}" for k, v in env_config_new["action_lookup"].items()])
                action_lookup_str += f"\nYou can make up to {env_config_new['max_actions_per_traj']} actions, separated by the action separator \" " + self.action_sep + " \"\n"
                env_instruction += action_lookup_str
            prefixes[env_tag] = env_instruction
            env_config_lookup[env_tag] = {'max_tokens': env_config.get("max_tokens", self.config.actor_rollout_ref.rollout.response_length)}

        tags = self.es_cfg.env_configs.tags
        n_groups = self.es_cfg.env_configs.n_groups
        group_size = self.es_cfg.group_size

        cur_group = 0
        for env_tag, n_group in zip(tags, n_groups):
            env_instruction = prefixes[env_tag]
            start_idx = cur_group * group_size
            end_idx = (cur_group + n_group) * group_size
            for i in range(start_idx, end_idx):
                prefix_lookup[i] = env_instruction
                env_config_lookup[i] = env_config_lookup[env_tag]
            cur_group += n_group
            
        self.prefix_lookup = prefix_lookup
        self.env_config_lookup = env_config_lookup

    def _parse_response(self, response: str) -> List:
        pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>' if self.config.agent_proxy.enable_think else r'<answer>(.*?)</answer>'
        match = re.search(pattern, response, re.DOTALL)
        if not match:
            # think_content, action_content, actions = "", "", [] # do not remove this kind of invalid string
            llm_response, actions = response, []
        else:
            if self.config.agent_proxy.enable_think:
                think_content, action_content = match.group(1), match.group(2)
            else:
                think_content, action_content = "", match.group(1)

                
            for special_token in self.special_token_list:
                action_content = action_content.replace(special_token, "").strip()
                think_content = think_content.replace(special_token, "").strip()
            
            actions = [action.strip() for action in action_content.split(self.action_sep) if action.strip()]
            max_actions = self.config.agent_proxy.max_actions_per_turn

            if len(actions) > max_actions:
                actions = actions[:max_actions] #Only the first MAX_ACTIONS actions are kept in the rollout.
                action_content = (" " + self.action_sep + " ").join(actions)

            llm_response = f"<think>{think_content}</think><answer>{action_content}</answer>" if self.config.agent_proxy.enable_think else f"<answer>{action_content}</answer>"
        return llm_response, actions

    def _get_attacker_format(self) -> str:
        return getattr(self.config.agent_proxy, "attacker_format", "legacy_think_answer")

    def _is_qwen3_attacker(self) -> bool:
        return self._get_attacker_format() == "qwen3_native"

    def _serialize_qwen3_messages(self, messages: List[Dict], add_generation_prompt: bool) -> str:
        parts = []
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "") or ""
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
            if not getattr(self.config.agent_proxy, "qwen3_enable_thinking", True):
                parts.append("<think>\n\n</think>\n\n")
        return "".join(parts)

    def _apply_attacker_chat_template(self, messages: List[Dict], add_generation_prompt: bool) -> str:
        sanitized_messages = []
        for message in messages:
            sanitized = dict(message)
            content = sanitized.get("content", "")
            if content is None:
                content = ""
            elif not isinstance(content, str):
                content = str(content)
            sanitized["content"] = content
            sanitized_messages.append(sanitized)
        if self._is_qwen3_attacker():
            return self._serialize_qwen3_messages(sanitized_messages, add_generation_prompt=add_generation_prompt)
        return self.tokenizer.apply_chat_template(sanitized_messages, add_generation_prompt=add_generation_prompt, tokenize=False)

    def _split_qwen3_response_from_token_ids(self, response_ids: List[int]):
        full_text = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        endthink_ids = self.tokenizer.encode("</think>", add_special_tokens=False)
        cut_idx = _find_last_subsequence(response_ids, endthink_ids)
        if cut_idx >= 0:
            end_pos = cut_idx + len(endthink_ids)
            final_ids = response_ids[end_pos:]
            final_content = self.tokenizer.decode(final_ids, skip_special_tokens=True).strip()
            return full_text, final_content, "ok"

        if "<think>" in full_text and "</think>" not in full_text:
            return full_text, "", "missing_endthink_with_open_think"
        return full_text, full_text, "no_think_block_found"

    def _split_qwen3_response_from_text(self, response: str):
        response = (response or "").strip()
        if "</think>" in response:
            final_content = response.rsplit("</think>", 1)[1].strip()
            return response, final_content, "ok_text_fallback"
        if "<think>" in response:
            return response, "", "missing_endthink_with_open_think_text_fallback"
        return response, response, "no_think_block_found_text_fallback"

    def _build_qwen3_env_response(self, response_text: str, response_ids: Optional[List[int]] = None):
        if response_ids is not None:
            llm_response, final_content, parse_status = self._split_qwen3_response_from_token_ids(response_ids)
        else:
            llm_response, final_content, parse_status = self._split_qwen3_response_from_text(response_text)

        action_content = final_content.strip()
        actions = [action_content] if action_content else [""]
        return llm_response, actions, parse_status

    def _compute_qwen3_loss_think_stats(self, input_ids: torch.Tensor, loss_mask: torch.Tensor):
        if not self._is_qwen3_attacker():
            return {}
        think_open_ids = self.tokenizer.encode("<think>", add_special_tokens=False)
        think_close_ids = self.tokenizer.encode("</think>", add_special_tokens=False)
        if not think_open_ids or not think_close_ids:
            return {}

        think_mask = torch.zeros_like(loss_mask, dtype=torch.bool)
        seqs = input_ids[:, 1:].tolist()
        for row_idx, seq in enumerate(seqs):
            search_pos = 0
            while search_pos < len(seq):
                open_start = -1
                for candidate in range(search_pos, len(seq) - len(think_open_ids) + 1):
                    if seq[candidate:candidate + len(think_open_ids)] == think_open_ids:
                        open_start = candidate
                        break
                if open_start < 0:
                    break
                close_start = -1
                for candidate in range(open_start + len(think_open_ids), len(seq) - len(think_close_ids) + 1):
                    if seq[candidate:candidate + len(think_close_ids)] == think_close_ids:
                        close_start = candidate
                        break
                if close_start < 0:
                    break
                think_mask[row_idx, open_start:close_start + len(think_close_ids)] = True
                search_pos = close_start + len(think_close_ids)

        think_in_loss = think_mask & loss_mask.bool()
        think_tokens = int(think_in_loss.sum().item())
        total_loss_tokens = int(loss_mask.sum().item())
        ratio = float(think_tokens / total_loss_tokens) if total_loss_tokens > 0 else 0.0
        return {
            "qwen3_loss_think_tokens": think_tokens,
            "qwen3_loss_total_tokens": total_loss_tokens,
            "qwen3_loss_think_token_ratio": ratio,
        }
        
    def _normalize_score_tensor(self, score_tensor: torch.Tensor, env_outputs: List[Dict]) -> torch.Tensor:
        """
        Normalize the score tensor to be between 0 and 1.
        NOTE: only support score at the last token for now
        """
        assert self.config.agent_proxy.use_turn_scores == False, "Reward normalization is not supported for use_turn_scores == True"
        
        rn_cfg = self.config.agent_proxy.reward_normalization
        grouping, method = rn_cfg.grouping, rn_cfg.method
        if grouping == "state":
            group_tags = [env_output["group_id"] for env_output in env_outputs]
        elif grouping == "inductive":
            group_tags = [env_output["tag"] for env_output in env_outputs]
        elif grouping == "batch":
            group_tags = [1] * len(env_outputs)
        else:
            raise ValueError(f"Invalid grouping: {grouping}")


        if method == "mean_std":
            norm_func = lambda x: (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6) if x.std(dim=-1, keepdim=True).abs().max() > 1e-6 else torch.zeros_like(x) # stable to bf16 than x.std()
        elif method == "mean":
            norm_func = lambda x: (x - x.mean(dim=-1, keepdim=True))
        elif method == "asym_clip":
            norm_func = lambda x: ((x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6) if x.std(dim=-1, keepdim=True).abs().max() > 1e-6 else torch.zeros_like(x)).clamp(min=-1, max=3)
        elif method == "identity":
            norm_func = lambda x: x
        else:
            raise ValueError(f"Invalid normalization method: {method}")

        # apply groupwise normalization
        group2index = {}
        for i, env_tag in enumerate(group_tags):
            if env_tag not in group2index:
                group2index[env_tag] = []
            group2index[env_tag].append(i)
        group2index = {k: torch.tensor(v) for k, v in group2index.items()}

        
        # apply penalty pre-normalization
        acc_scores = score_tensor[:, -1]
        normalized_acc_scores = acc_scores.clone()
        penalty = torch.tensor([env_output.get("penalty", 0) for env_output in env_outputs], dtype=torch.float32)
        normalized_acc_scores = normalized_acc_scores + penalty

        if len(group2index) < acc_scores.shape[0]: # the group size > 1
            for group, index in group2index.items():
                normalized_acc_scores[index] = norm_func(normalized_acc_scores[index])

        score_tensor[:, -1] = normalized_acc_scores

        return score_tensor
    
    def get_lm_inputs(self, env_outputs: List[Dict], prepare_for_update: bool) -> DataProto:
        """
        env_outputs - please see below example
        [
            {"env_id": 1, "history": [{"state": "###\n#x_#", "llm_response": "Response 1", "reward": 0.5}, {"state": "###\n#x_#"}]},
            {"env_id": 2, "history": [{"state": "###\n#x_#"}]},
            ...
        ]
        prefix_lookup - from env_id to initial prompt
        """
        llm_input_texts = []
        messages_list = [] # for api calling
        visible_messages_list = [] # target-visible messages for judger/refusal
        for env_output in env_outputs:
            if 'state' in env_output['history'][-1] and prepare_for_update:
                env_output['history'] = env_output['history'][:-1] # when prepare for update, we do not add the state from the n+1 turn to the trajectory
            
            max_k = getattr(self.config.agent_proxy, "max_context_window", None)
            if max_k is not None and isinstance(max_k, int) and max_k > 0:
                env_output['history'] = env_output['history'][-max_k:]
            
            env_id = env_output["env_id"]
            system_prefix = self.prefix_lookup.get(env_id)
            if system_prefix is None:
                # Fallback to any available prefix to avoid KeyError in dynamic eval sizing
                system_prefix = next(iter(self.prefix_lookup.values()), "")
            messages = [
                {"role": "system", "content": system_prefix},
                {"role": "user", "content": env_output["init_prompt"]}
            ]
            visible_messages = [
                {"role": "system", "content": system_prefix},
                {"role": "user", "content": env_output["init_prompt"]}
            ]
            # pdb.set_trace()

            for idx, content in enumerate(env_output["history"]):
                if "state" in content and self.config.ctx_manager.add_prefix_suffix_prompt:
                    FORMAT_PROMPT = "<think> [Your thoughts] </think> <answer> [your answer] </answer>" if self.config.agent_proxy.enable_think else "<answer> [your answer] </answer>"
                    env_cfg = self.env_config_lookup.get(env_id) or next(iter(self.env_config_lookup.values()), {"max_tokens": self.config.actor_rollout_ref.rollout.response_length})
                    LENGTH_PROMPT = f"Max response length: {env_cfg['max_tokens']} words (tokens)."
                    messages[-1]["content"] += f"State:\n{content['state']}\nYou have {content['actions_left']} actions left. Always output: {FORMAT_PROMPT} with no extra text. Strictly follow this format. {LENGTH_PROMPT}\n"
                    visible_messages[-1]["content"] = messages[-1]["content"]
                if "llm_response" in content:
                    messages.append({"role": "assistant", "content": content["llm_response"]})
                    visible_messages.append({"role": "assistant", "content": content.get("visible_llm_response", content["llm_response"])})
                if "env_response" in content:
                    messages.append({"role": "user", "content": content['env_response']})
                    visible_messages.append({"role": "user", "content": content['env_response']})
            assert all(msg["role"] == "assistant" for msg in messages[2::2])
            assert all(msg["role"] == "assistant" for msg in visible_messages[2::2])

            text = self._apply_attacker_chat_template(messages, add_generation_prompt=(not prepare_for_update))
            llm_input_texts.append(text)
            messages_list.append(messages)
            visible_messages_list.append(visible_messages)

        inputs = self.tokenizer(llm_input_texts, return_tensors="pt", padding=True, padding_side="left", truncation=False) # do not truncate here. Process later at TODO
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        position_ids = attention_mask.cumsum(dim=-1)
        if prepare_for_update:
            scores = [[i.get('reward', 0.0) for i in env_output['history']] for env_output in env_outputs]
            score_tensor, loss_mask, response_mask = get_masks_and_scores(input_ids=input_ids, tokenizer=self.tokenizer, messages_list=messages_list, all_scores=scores, use_turn_scores=self.config.agent_proxy.use_turn_scores, enable_response_mask=self.config.enable_response_mask, filter_single_turn=self.config.algorithm.filter_single_turn)

            normalized_score_tensor = score_tensor.clone()
            if not self.config.agent_proxy.use_turn_scores:
                # normalized_score_tensor = self._normalize_score_tensor(score_tensor, env_outputs)
                normalized_score_tensor = self._normalize_score_tensor(normalized_score_tensor, env_outputs)
            response_length = response_mask.sum(dim=-1).float().mean().item()

        llm_inputs = DataProto()
        llm_inputs.batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": input_ids[:, 1:], # remove the first token
        }, batch_size=input_ids.shape[0])

        if prepare_for_update:
            llm_inputs.batch["loss_mask"] = loss_mask # remove the first token
            llm_inputs.batch["rm_scores"] = normalized_score_tensor # remove the first token
            llm_inputs.batch["original_rm_scores"] = score_tensor # remove the first token
            # llm_inputs.batch["judger_scores"] = judger_scores # remove the first token
        # pdb.set_trace()
        llm_inputs.non_tensor_batch = {
            "env_ids": np.array([env_output["env_id"] for env_output in env_outputs], dtype=object),
            "group_ids": np.array([env_output["group_id"] for env_output in env_outputs], dtype=object),
            "messages_list": np.array(messages_list, dtype=object),
            "visible_messages_list": np.array(visible_messages_list, dtype=object),
            "harmful_targets": np.array([env_output["harmful_target"] for env_output in env_outputs], dtype=object),
            "turn_scores": np.array([env_output.get("turn_scores", None) for env_output in env_outputs], dtype=object),
            "data_source": np.array([env_output.get("data_source", "unknown") for env_output in env_outputs], dtype=object),
            "target_model_profiles": np.array([env_output.get("target_model_profile", None) for env_output in env_outputs], dtype=object),
            "env_metrics": np.array([env_output.get("metrics", {}) for env_output in env_outputs], dtype=object),
        }
        if prepare_for_update:
            judger_scores = [[i.get('info', {}).get('score', 0.0) for i in env_output['history']] for env_output in env_outputs]
            llm_inputs.non_tensor_batch["judger_scores"] = np.array(judger_scores, dtype=object)
            env_response_tokens = [[i.get('env_response_tokens', 0) for i in env_output['history']] for env_output in env_outputs]
            llm_inputs.non_tensor_batch["env_response_tokens"] = np.array(env_response_tokens, dtype=object)

        if prepare_for_update:
            metrics = {}
            for env_output in env_outputs:
                for key, value in env_output["metrics"].items():
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
            mean_metrics = {}
            for key, value in metrics.items():
                if isinstance(value[0], list):
                    continue
                else:    
                    arr = np.array(value)
                    env_key = key.split("/")[0]
                    mean = np.sum(arr) / self.env_nums[env_key]
                mean_metrics[key] = mean

            for key, values in metrics.items():
                if not isinstance(values, list) or any(isinstance(v, list) for v in values):
                    continue
                prefix, suffix = key.split("/", 1)
                non_zero_values = [v for v in values if v != 0]
                if non_zero_values:  # Avoid division by zero
                    non_zero_key = f"{prefix}/non-zero/{suffix}"
                    mean_metrics[non_zero_key] = np.mean(non_zero_values)
            metrics = mean_metrics
            metrics["response_length"] = response_length
            metrics.update(self._compute_qwen3_loss_think_stats(input_ids, loss_mask))
            llm_inputs.meta_info = {"metrics": metrics}
        return llm_inputs

    def get_env_inputs(self, lm_outputs: DataProto) -> List[Dict]:
        response_ids_batch = None
        if lm_outputs.batch is not None and 'responses' in lm_outputs.batch.keys():
            response_ids_batch = lm_outputs.batch['responses']
            responses = self.tokenizer.batch_decode(
                response_ids_batch, 
                skip_special_tokens=True
            )
        else: # dataproto has textual responses
            responses = lm_outputs.non_tensor_batch['response_texts']
        if self.config.ctx_manager.add_prefix_suffix_prompt:
            responses = ["<think>" + response if self.config.agent_proxy.enable_think else "<answer>" + response for response in responses] # The LLM generation does not include <think> tags. Add them back here.
            
        env_ids = lm_outputs.non_tensor_batch['env_ids']
        env_inputs = []
        for idx, (env_id, response) in enumerate(zip(env_ids, responses)):
            if response is None:
                response = ""
            elif not isinstance(response, str):
                response = str(response)
            attacker_response_tokens = 0
            parse_status = None
            if self._is_qwen3_attacker():
                response_ids = None
                if response_ids_batch is not None:
                    response_ids = response_ids_batch[idx].tolist()
                    pad_token_id = self.tokenizer.pad_token_id
                    if pad_token_id is None:
                        attacker_response_tokens = len(response_ids)
                    else:
                        attacker_response_tokens = sum(1 for token_id in response_ids if token_id != pad_token_id)
                llm_response, actions, parse_status = self._build_qwen3_env_response(response, response_ids=response_ids)
            elif self.config.agent_proxy.parse_response:
                llm_response, actions = self._parse_response(response)
            else:
                llm_response, actions = response, [response]
            if llm_response is None:
                llm_response = ""
            elif not isinstance(llm_response, str):
                llm_response = str(llm_response)
            normalized_actions = []
            for action in actions:
                if action is None:
                    normalized_actions.append("")
                elif isinstance(action, str):
                    normalized_actions.append(action)
                else:
                    normalized_actions.append(str(action))
            actions = normalized_actions or [""]
            if attacker_response_tokens == 0:
                try:
                    attacker_response_tokens = len(self.tokenizer.encode(llm_response or "", add_special_tokens=False))
                except Exception:
                    attacker_response_tokens = 0

            visible_llm_response = llm_response
            if self._is_qwen3_attacker():
                visible_llm_response = actions[0] if actions else ""

            env_inputs.append({
                "env_id": env_id,
                "llm_raw_response": llm_response,
                "llm_response": llm_response,
                "visible_llm_response": visible_llm_response,
                "actions": actions,
                "parse_status": parse_status,
                "attacker_response_tokens": attacker_response_tokens,
            })
        return env_inputs

    def formulate_rollouts(self, env_outputs: List[Dict]) -> DataProto:
        llm_inputs = self.get_lm_inputs(env_outputs, prepare_for_update=True)
        return llm_inputs

    



@hydra.main(version_base = None, config_path = "../../config", config_name = "base")
def main(config):
    import json
    tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
    ctx_manager = ContextManager(config=config, tokenizer=tokenizer)
    print("ctx_manager prefix", ctx_manager.prefix_lookup)
    


    env_outputs = [
        {
            "env_id": 1,
            "history": [
                {"state": "###\n#x_#<image>", "llm_response": "Response 1", "reward": 0.5, "actions_left": 2},
                {"state": "###\n#x_#<image>", "llm_response": "Response 2", "reward": 0.8, "actions_left": 1},
                {"state": "###\n#x_#<image>", "actions_left": 0}
            ],
            "group_id": 0,
            "metrics": {}
        },
        {
            "env_id": 2,
            "history": [
                {"state": "###\n#x_#<image>", "llm_response": "Response 3", "reward": 0.3, "actions_left": 1},
                {"state": "###\n#x_#<image>", "actions_left": 0}
            ],
            "group_id": 1,
            "metrics": {}
        }
    ]
    
    prefix_lookup = {1: "Initial prompt", 2: "Initial prompt 2"}
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    env_prompt = ctx_manager.get_lm_inputs(env_outputs, prepare_for_update=False)
    print(env_prompt)
    formulate_rollouts_rst= ctx_manager.formulate_rollouts(env_outputs)
    print(formulate_rollouts_rst)

if __name__ == "__main__":
    main()
    
