"""
This is the environment state manager for the LLM agent.
author: Pingyue Zhang
date: 2025-03-30
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import PIL.Image
import hydra
import random
import numpy as np
import math
from openai import OpenAI
import re
import time
import os
import socket
from urllib.parse import urlparse
from ragen.llm_agent.vllm_local_client import VllmLocalClient

from tqdm import tqdm
import pdb

from ragen.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS
from ragen.utils import register_resolvers
register_resolvers()

from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer
from ragen.env.jailbreak.env import LLAMA2_CLS_PROMPT


def _coerce_str_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            inner = text[1:-1].strip()
            if not inner:
                return []
            parts = [item.strip().strip("\'\"") for item in inner.split(",")]
            return [item for item in parts if item]
        return [item.strip() for item in text.split("||") if item.strip()]
    if isinstance(value, dict):
        return [str(item).strip() for item in value.values() if str(item).strip()]
    if hasattr(value, "__iter__"):
        try:
            return [str(item).strip() for item in list(value) if str(item).strip()]
        except Exception:
            pass
    text = str(value).strip()
    return [text] if text else []


def _truncate_text_by_tokens(text: str, tokenizer, max_tokens: Optional[int]) -> str:
    if text is None:
        return ""
    if tokenizer is None or max_tokens is None:
        return text
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except Exception:
        return text
    if len(token_ids) <= max_tokens:
        return text
    try:
        return tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=False)
    except Exception:
        return text


def _truncate_messages_by_tokens(messages, tokenizer, max_tokens: Optional[int]):
    if tokenizer is None or max_tokens is None:
        return messages
    if not messages:
        return messages
    if isinstance(messages, str):
        return _truncate_text_by_tokens(messages, tokenizer, max_tokens)
    try:
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    except Exception:
        return messages
    if len(token_ids) <= max_tokens:
        return messages
    # Truncate the last message content to fit the max token budget
    try:
        truncated = [dict(m) for m in messages]
        last_idx = len(truncated) - 1
        content = truncated[last_idx].get("content", "")
        content_ids = tokenizer.encode(content, add_special_tokens=False)
        overflow = len(token_ids) - max_tokens
        if overflow >= len(content_ids):
            new_content_ids = content_ids[:1]
        else:
            new_content_ids = content_ids[:-overflow]
        new_content = tokenizer.decode(new_content_ids, skip_special_tokens=False)
        truncated[last_idx]["content"] = new_content
        # Ensure final prompt length <= max_tokens
        for _ in range(3):
            prompt = tokenizer.apply_chat_template(truncated, add_generation_prompt=False, tokenize=False)
            if len(tokenizer.encode(prompt, add_special_tokens=False)) <= max_tokens:
                break
            if len(new_content_ids) <= 1:
                break
            new_content_ids = new_content_ids[:-max(1, len(new_content_ids) // 10)]
            new_content = tokenizer.decode(new_content_ids, skip_special_tokens=False)
            truncated[last_idx]["content"] = new_content
        return truncated
    except Exception:
        return messages


def _count_prompt_tokens(messages, tokenizer) -> Optional[int]:
    if tokenizer is None or messages is None:
        return None
    try:
        if isinstance(messages, str):
            return len(tokenizer.encode(messages, add_special_tokens=False))
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        return len(tokenizer.encode(prompt, add_special_tokens=False))
    except Exception:
        return None


def _coerce_config_bool(value, key_name: str, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off", ""}:
            return False
        print(
            f"[WARN][config] invalid boolean value for {key_name}: {value!r}. "
            f"Falling back to default={default}.",
            flush=True,
        )
        return default
    return bool(value)


def _cfg_get(cfg_obj, key: str, default=None):
    if cfg_obj is None:
        return default
    if hasattr(cfg_obj, "get"):
        try:
            return cfg_obj.get(key, default)
        except Exception:
            pass
    return getattr(cfg_obj, key, default)


class OpenAIVLLMClient:
    def __init__(
        self,
        base_url,
        api_model,
        tokenizer_path=None,
        api_key="EMPTY",
        enable_logit_bias=False,
        client_name="llm",
        max_retries=3,
        timeout_sec=20,
        use_responses_api=False,
        no_retry_error_markers=None,
    ):
        self.base_url = base_url
        self.api_model = api_model
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.tokenizer = None
        self.juder_logit_bias = None
        self.enable_logit_bias = enable_logit_bias
        self.client_name = client_name
        self.max_retries = max_retries
        self.timeout_sec = timeout_sec
        self.use_responses_api = use_responses_api
        self.no_retry_error_markers = [marker.lower() for marker in _coerce_str_list(no_retry_error_markers)]
        self.leave_out_header = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        self.juder_allowed_tokens = ["Yes", "No"] # Hardcoded for Harmbench classifier
        self._net_debug_printed = False

        if tokenizer_path:
            if tokenizer_path.endswith("tokenizer.model") or tokenizer_path.endswith("tokenizer.json") or tokenizer_path.endswith("tokenizer_config.json"):
                tokenizer_path = os.path.dirname(tokenizer_path)
            if os.path.exists(tokenizer_path):
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
                    print(f"[INFO][{self.client_name}] tokenizer loaded from: {tokenizer_path}")
                    if self.enable_logit_bias:
                        self.juder_allowed_token_ids = [
                            self.tokenizer.encode(t, add_special_tokens=False)[0] for t in self.juder_allowed_tokens
                        ]
                        self.juder_logit_bias = {token_id: 100 for token_id in self.juder_allowed_token_ids}
                except Exception as e:
                    print(f"[WARN] Failed to load tokenizer from {tokenizer_path}: {e}")
            else:
                print(f"[WARN] tokenizer_path not found: {tokenizer_path}")

    def _request_with_retries(self, request_fn, *args, **kwargs):
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if "timeout" not in kwargs:
                    kwargs["timeout"] = self.timeout_sec
                return request_fn(*args, **kwargs)
            except Exception as e:
                last_err = e
                err_text = str(e).lower()
                if self.no_retry_error_markers and any(marker in err_text for marker in self.no_retry_error_markers):
                    print(
                        f"[WARN][{self.client_name}] request failed with non-retryable marker: "
                        f"{type(e).__name__}: {e} "
                        f"(base_url={self.base_url} api_model={self.api_model} timeout={self.timeout_sec})"
                    )
                    raise
                # Exponential backoff with cap
                sleep_s = min(2 ** (attempt - 1), 30)
                err_type = type(e).__name__
                if attempt < self.max_retries:
                    print(
                        f"[WARN][{self.client_name}] request failed (attempt {attempt}/{self.max_retries}): "
                        f"{err_type}: {e}. Retrying in {sleep_s}s "
                        f"(base_url={self.base_url} api_model={self.api_model} timeout={self.timeout_sec})"
                    )
                else:
                    print(
                        f"[WARN][{self.client_name}] request failed (attempt {attempt}/{self.max_retries}): "
                        f"{err_type}: {e}. No more retries "
                        f"(base_url={self.base_url} api_model={self.api_model} timeout={self.timeout_sec})"
                    )
                if not self._net_debug_printed:
                    self._net_debug_printed = True
                    try:
                        parsed = urlparse(self.base_url)
                        host = parsed.hostname or ""
                        ip = socket.gethostbyname(host) if host else "N/A"
                    except Exception as dns_e:
                        host = parsed.hostname if "parsed" in locals() else ""
                        ip = f"DNS_ERROR:{dns_e}"
                    print(
                        f"[DEBUG][{self.client_name}] net_debug pid={os.getpid()} host={host} ip={ip} "
                        f"HTTP_PROXY={os.environ.get('HTTP_PROXY','')} HTTPS_PROXY={os.environ.get('HTTPS_PROXY','')} "
                        f"NO_PROXY={os.environ.get('NO_PROXY','')}"
                    )
                if attempt < self.max_retries:
                    time.sleep(sleep_s)
        raise last_err

    def _normalize_chat_content(self, content):
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
                        continue
                    inner_text = item.get("content")
                    if isinstance(inner_text, str) and inner_text:
                        parts.append(inner_text)
                        continue
                else:
                    text = getattr(item, "text", None)
                    if text:
                        parts.append(str(text))
                        continue
                    inner_text = getattr(item, "content", None)
                    if isinstance(inner_text, str) and inner_text:
                        parts.append(inner_text)
                        continue
            if parts:
                return "".join(parts)
        return str(content)

    def _safe_preview(self, value, limit: int = 600) -> str:
        if value is None:
            return ""
        try:
            if isinstance(value, str):
                text = value
            elif hasattr(value, "model_dump_json"):
                text = value.model_dump_json(exclude_none=True)
            elif hasattr(value, "model_dump"):
                text = str(value.model_dump(exclude_none=True))
            else:
                text = str(value)
        except Exception:
            try:
                text = repr(value)
            except Exception:
                text = "<unprintable>"
        text = text.replace("\n", "\\n")
        return text[:limit]

    def _extract_usage_summary(self, usage) -> Dict[str, Any]:
        if usage is None:
            return {}
        if isinstance(usage, dict):
            return {
                "prompt_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
                "completion_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
                "total_tokens": usage.get("total_tokens"),
            }
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", None)),
            "completion_tokens": getattr(usage, "completion_tokens", getattr(usage, "output_tokens", None)),
            "total_tokens": getattr(usage, "total_tokens", None),
        }

    def _build_chat_debug_meta(self, response, content: str) -> Dict[str, Any]:
        choice = response.choices[0]
        message = choice.message
        return {
            "api_type": "chat.completions",
            "response_id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
            "finish_reason": getattr(choice, "finish_reason", None),
            "refusal": getattr(message, "refusal", None),
            "usage": self._extract_usage_summary(getattr(response, "usage", None)),
            "content_len": len(content or ""),
            "raw_preview": self._safe_preview(response),
        }

    def _build_responses_debug_meta(self, response, content: str) -> Dict[str, Any]:
        return {
            "api_type": "responses",
            "response_id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
            "status": getattr(response, "status", None),
            "usage": self._extract_usage_summary(getattr(response, "usage", None)),
            "content_len": len(content or ""),
            "raw_preview": self._safe_preview(response),
        }

    def _log_suspicious_response(self, messages, meta: Dict[str, Any], content: str, debug_label: Optional[str] = None):
        finish_reason = str(meta.get("finish_reason", "") or "").lower()
        refusal = meta.get("refusal")
        suspicious = (content or "") == "" or finish_reason in {"length", "content_filter"} or bool(refusal)
        if not suspicious:
            return
        label = debug_label or self.client_name
        msg_tail = []
        try:
            msg_tail = messages[-2:] if isinstance(messages, list) else messages
        except Exception:
            msg_tail = messages
        print(
            f"[DEBUG][{self.client_name}] suspicious_response label={label} "
            f"base_url={self.base_url} api_model={self.api_model} "
            f"finish_reason={meta.get('finish_reason')} refusal={refusal!r} "
            f"content_len={len(content or '')} usage={meta.get('usage')} "
            f"messages_tail={self._safe_preview(msg_tail, limit=400)} "
            f"raw_preview={meta.get('raw_preview', '')}",
            flush=True,
        )

    def _extract_responses_output_text(self, response):
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str):
            return output_text
        if output_text is not None:
            return str(output_text)

        output = getattr(response, "output", None)
        if not output:
            return ""

        texts = []
        try:
            for item in output:
                item_type = getattr(item, "type", None)
                if item_type is None and isinstance(item, dict):
                    item_type = item.get("type")
                if item_type != "message":
                    continue
                content_list = getattr(item, "content", None)
                if content_list is None and isinstance(item, dict):
                    content_list = item.get("content")
                if not content_list:
                    continue
                for part in content_list:
                    part_type = getattr(part, "type", None)
                    if part_type is None and isinstance(part, dict):
                        part_type = part.get("type")
                    if part_type not in ("output_text", "text"):
                        continue
                    text = getattr(part, "text", None)
                    if text is None and isinstance(part, dict):
                        text = part.get("text")
                    if text:
                        texts.append(str(text))
            return "".join(texts)
        except Exception:
            return ""

    def chat(self, messages, **kwargs):
        # Debug: print routing info for each chat batch
        #print(f"[DEBUG][{self.client_name}.chat] base_url={self.base_url} api_model={self.api_model} kwargs={list(kwargs.keys())}")
        return_debug_meta = bool(kwargs.pop("return_debug_meta", False))
        debug_label = kwargs.pop("debug_label", None)
        try:
            if self.use_responses_api:
                response_kwargs = dict(kwargs)
                if "max_tokens" in response_kwargs and "max_output_tokens" not in response_kwargs:
                    response_kwargs["max_output_tokens"] = response_kwargs.pop("max_tokens")
                response_kwargs.setdefault("instructions", "")
                response_kwargs.setdefault("reasoning", {"effort": "low"})
                response = self._request_with_retries(
                    self.client.responses.create,
                    model=self.api_model,
                    input=messages,
                    **response_kwargs,
                )
                content = self._normalize_chat_content(self._extract_responses_output_text(response))
                debug_meta = self._build_responses_debug_meta(response, content)
                if content == "":
                    try:
                        print(
                            f"[DEBUG][{self.client_name}][responses] empty parsed output_text. "
                            f"id={getattr(response, 'id', None)} "
                            f"output_preview={str(getattr(response, 'output', None))[:400]!r}",
                            flush=True,
                        )
                    except Exception:
                        pass
                self._log_suspicious_response(messages, debug_meta, content, debug_label=debug_label)
                if return_debug_meta:
                    return {"text": content, "meta": debug_meta}
                return content

            response = self._request_with_retries(
                self.client.chat.completions.create,
                model=self.api_model,
                messages=messages,
                **kwargs,
            )
            content = self._normalize_chat_content(getattr(response.choices[0].message, "content", None))
            debug_meta = self._build_chat_debug_meta(response, content)
            self._log_suspicious_response(messages, debug_meta, content, debug_label=debug_label)
            if return_debug_meta:
                return {"text": content, "meta": debug_meta}
            return content
        except Exception as e:
            # Fallback to completions if chat template is required
            msg = str(e)
            if ("chat template" in msg.lower()) or ("default chat template" in msg.lower()):
                if self.tokenizer is None:
                    raise
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                response = self._request_with_retries(
                    self.client.completions.create,
                    model=self.api_model,
                    prompt=prompt,
                    **kwargs,
                )
                text = response.choices[0].text
                if text is None:
                    if return_debug_meta:
                        return {
                            "text": "",
                            "meta": {
                                "api_type": "completions_fallback",
                                "response_id": getattr(response, "id", None),
                                "model": getattr(response, "model", None),
                                "finish_reason": getattr(response.choices[0], "finish_reason", None),
                                "usage": self._extract_usage_summary(getattr(response, "usage", None)),
                                "content_len": 0,
                                "raw_preview": self._safe_preview(response),
                            },
                        }
                    return ""
                text = str(text).replace(self.leave_out_header, "")
                debug_meta = {
                    "api_type": "completions_fallback",
                    "response_id": getattr(response, "id", None),
                    "model": getattr(response, "model", None),
                    "finish_reason": getattr(response.choices[0], "finish_reason", None),
                    "usage": self._extract_usage_summary(getattr(response, "usage", None)),
                    "content_len": len(text),
                    "raw_preview": self._safe_preview(response),
                }
                self._log_suspicious_response(messages, debug_meta, text, debug_label=debug_label)
                if return_debug_meta:
                    return {"text": text, "meta": debug_meta}
                return text
            raise
    
    def batch_complete(self, batch_messages, **kwargs):
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized; batch_complete requires a tokenizer_path.")
        print(f"[DEBUG][{self.client_name}.batch_complete] base_url={self.base_url} api_model={self.api_model} kwargs={list(kwargs.keys())}")
        batch_converted_messages = []
        for messages in batch_messages:
            converted_messages = self.tokenizer.apply_chat_template(
                messages, tokenize=False
            )
            batch_converted_messages.append(converted_messages)

        response = self._request_with_retries(
            self.client.completions.create,
            model=self.api_model,
            prompt=batch_converted_messages,
            **kwargs,
        )

        results = []
        for choice in response.choices:
            text = choice.text
            if text is None:
                results.append("")
            else:
                results.append(str(text).replace(self.leave_out_header, ""))
        return results

    def get_logprobs_chat(self, messages, index, **kwargs):
        response = self._request_with_retries(
            self.client.chat.completions.create,
            model=self.api_model,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].logprobs.content[index].top_logprobs
    
    def batch_get_logprobs_complete(self, batch_prompts, index, **kwargs):
        # print(f"[DEBUG][{self.client_name}.batch_get_logprobs_complete] base_url={self.base_url} api_model={self.api_model} index={index} kwargs={list(kwargs.keys())}")
        request_kwargs = dict(
            model=self.api_model,
            prompt=batch_prompts,
            **kwargs
        )
        if self.juder_logit_bias is not None:
            request_kwargs["logit_bias"] = self.juder_logit_bias
        response = self._request_with_retries(self.client.completions.create, **request_kwargs)
        if index == 0:
            return [choice.logprobs.top_logprobs for choice in response.choices]


@dataclass
class EnvStatus:
    """Status of an environment"""
    truncated: bool = False # done but not success
    terminated: bool = False # done and success
    num_actions: int = 0 # current action step (single action)
    rewards: List[float] = field(default_factory=list) # rewards for each turn
    seed: Optional[int] = None # what seed is used to reset this environment



class EnvStateManager:
    """Manager for the environment state
    The class is responsible for managing multiple (kinds of) environments
    
    """
    def __init__(self, config, mode: str = "train"):
        self.sys_config = config
        self.mode = mode
        self.config = getattr(self.sys_config.es_manager, mode)
        self.env_groups = int(self.config.env_groups)
        self.group_size = self.config.group_size
        self.train_base_env_groups = self.env_groups
        self.mixed_train_targets: List[str] = []
        self.expand_mixed_train_groups = False
        self._configure_mixed_train_group_layout()
        seed_cfg = getattr(self.sys_config, "seed", None)
        if seed_cfg is not None:
            self.base_seed = seed_cfg.get(mode, None)
        else:
            self.base_seed = None
        self.seed_counter = 0
        self.harmful_target_base_idx = 0
        from ragen.env.jailbreak.env import JailbreakEnv
        from ragen.env.jailbreak.config import JailbreakEnvConfig
        env_cfg = JailbreakEnvConfig()
        self.harmful_threshold = env_cfg.harmful_threshold
        self.targets, self.target_sources = self._load_targets(env_cfg.data_paths)
        if self.mode == "train":
            # self.val_targets = self._load_targets([env_cfg.val_data_paths[1]]) if hasattr(env_cfg, 'val_data_paths') else []
            self.val_targets, self.val_target_sources = self._load_targets(env_cfg.val_data_paths) if hasattr(env_cfg, 'val_data_paths') else ([], [])
        elif self.mode == "val":
            # self.val_targets = self._load_targets([env_cfg.val_data_paths[0]]) if hasattr(env_cfg, 'val_data_paths') else []
            use_test_paths = bool(getattr(self.config, "use_test_data_paths", False))
            if use_test_paths and hasattr(env_cfg, "test_data_paths"):
                data_paths = env_cfg.test_data_paths
            else:
                data_paths = env_cfg.val_data_paths if hasattr(env_cfg, "val_data_paths") else []
            self.val_targets, self.val_target_sources = self._load_targets(data_paths) if data_paths else ([], [])
        if self.mode == "val" and getattr(self.config, "auto_full_eval", False):
            # auto use each val target once (no repetition); keep group_size for pass@k
            self.env_groups = len(self.val_targets)
            self.config.env_groups = self.env_groups
            self.config.env_configs.n_groups = [self.env_groups]
        
        self._init_envs()
        self.rollout_cache = None
        self._init_llms()

    def _configure_mixed_train_group_layout(self) -> None:
        env_llm_cfg = getattr(self.sys_config, "env_llm", None)
        env_llm_mode = str(_cfg_get(env_llm_cfg, "mode", "single")).strip().lower()
        if env_llm_mode == "fixed":
            env_llm_mode = "mixed"
        if self.mode != "train" or env_llm_mode != "mixed":
            return

        train_targets = _coerce_str_list(_cfg_get(env_llm_cfg, "train_targets", None))
        if not train_targets:
            return

        self.mixed_train_targets = train_targets
        self.expand_mixed_train_groups = True
        self.train_base_env_groups = int(self.config.env_groups)
        expanded_env_groups = self.train_base_env_groups * len(self.mixed_train_targets)
        self.env_groups = expanded_env_groups
        self.config.env_groups = expanded_env_groups
        self.config.env_configs.n_groups = [expanded_env_groups]
        print(
            f"[DEBUG][mixed_env_llm.layout] base_seed_groups={self.train_base_env_groups} "
            f"train_targets={self.mixed_train_targets} expanded_env_groups={expanded_env_groups}",
            flush=True,
        )

    def _build_env_client_bundle(self, env_llm_cfg, profile_name: Optional[str] = None) -> Dict[str, Any]:
        env_api_model = _cfg_get(env_llm_cfg, "api_model", _cfg_get(env_llm_cfg, "model_path", None))
        env_tokenizer_path = _cfg_get(env_llm_cfg, "tokenizer_path", None)
        model_path = _cfg_get(env_llm_cfg, "model_path", None)
        if env_tokenizer_path is None and model_path and os.path.exists(str(model_path)):
            env_tokenizer_path = model_path
        use_env_responses_api = _coerce_config_bool(
            _cfg_get(env_llm_cfg, "GPT_OSS", False),
            key_name=f"env_llm[{profile_name or 'single'}].GPT_OSS",
            default=False,
        )
        try:
            env_timeout_sec = int(_cfg_get(env_llm_cfg, "timeout_sec", 120))
        except Exception:
            env_timeout_sec = 120
        client = OpenAIVLLMClient(
            base_url=_cfg_get(env_llm_cfg, "base_url"),
            api_model=env_api_model,
            tokenizer_path=env_tokenizer_path,
            api_key=_cfg_get(env_llm_cfg, "api_key", "EMPTY"),
            enable_logit_bias=False,
            client_name=f"env_llm[{profile_name}]" if profile_name else "env_llm",
            max_retries=3,
            timeout_sec=env_timeout_sec,
            use_responses_api=use_env_responses_api,
        )
        if use_env_responses_api:
            llm_params = {
                "max_output_tokens": int(_cfg_get(env_llm_cfg, "responses_max_output_tokens", 2048)),
                "temperature": _cfg_get(env_llm_cfg, "temperature", 0.7),
                "reasoning": {"effort": _cfg_get(env_llm_cfg, "reasoning_effort", "low")},
            }
        else:
            llm_params = {
                "max_tokens": _cfg_get(env_llm_cfg, "max_tokens", 4096),
                "temperature": _cfg_get(env_llm_cfg, "temperature", 0.7),
                "stop": _cfg_get(env_llm_cfg, "stop", None),
            }
        try:
            cfg_max_tokens = _cfg_get(env_llm_cfg, "max_tokens", None)
            env_llm_max_tokens = int(cfg_max_tokens) if cfg_max_tokens is not None else None
        except Exception:
            env_llm_max_tokens = None
        env_default_max_workers = 48 if use_env_responses_api else 256
        try:
            env_llm_max_workers = int(_cfg_get(env_llm_cfg, "max_workers", env_default_max_workers))
        except Exception:
            env_llm_max_workers = env_default_max_workers
        env_llm_max_workers = max(1, env_llm_max_workers)
        return {
            "profile_name": profile_name or "single",
            "client": client,
            "llm_params": llm_params,
            "max_tokens": env_llm_max_tokens,
            "max_workers": env_llm_max_workers,
            "use_responses_api": use_env_responses_api,
        }

    def _apply_env_bundle_aliases(self, bundle: Dict[str, Any]) -> None:
        self.env_llm = bundle["client"]
        self.env_llm_params = bundle["llm_params"]
        self.env_llm_max_tokens = bundle["max_tokens"]
        self.env_llm_max_workers = bundle["max_workers"]

    def _normalize_env_profile_name(self, profile_name: Optional[str]) -> Optional[str]:
        if not self.is_mixed_env_llm_mode():
            return None
        if profile_name is None:
            return self.active_env_profile
        normalized = str(profile_name).strip()
        if not normalized:
            return self.active_env_profile
        if normalized not in self.env_llm_bundles:
            fallback = self.active_env_profile or (self.env_llm_profiles[0] if self.env_llm_profiles else None)
            print(f"[WARN][env_llm.mixed] unknown profile={profile_name!r}, fallback to {fallback!r}", flush=True)
            return fallback
        return normalized

    def is_mixed_env_llm_mode(self) -> bool:
        return getattr(self, "env_llm_mode", "single") == "mixed"

    def is_fixed_env_llm_mode(self) -> bool:
        return self.is_mixed_env_llm_mode()

    def get_validation_profiles(self) -> List[str]:
        if not self.is_mixed_env_llm_mode():
            return []
        return list(self.env_llm_validation_profiles)

    def get_active_env_profile(self) -> Optional[str]:
        return getattr(self, "active_env_profile", None)

    def set_active_env_profile(self, profile_name: str) -> None:
        if not self.is_mixed_env_llm_mode():
            return
        normalized = self._normalize_env_profile_name(profile_name)
        if normalized is None:
            raise ValueError("mixed env_llm requires a valid active profile")
        self.active_env_profile = normalized
        self._apply_env_bundle_aliases(self.env_llm_bundles[normalized])

    def _get_train_env_profile(self, group_id: int) -> Optional[str]:
        if not self.is_mixed_env_llm_mode():
            return None
        if self.expand_mixed_train_groups and self.mixed_train_targets:
            return self.mixed_train_targets[group_id % len(self.mixed_train_targets)]
        if not self.env_llm_train_profiles:
            return self.active_env_profile
        return self.env_llm_train_profiles[group_id % len(self.env_llm_train_profiles)]

    def _resolve_env_profile_for_group(self, group_id: int) -> Optional[str]:
        if not self.is_mixed_env_llm_mode():
            return None
        if self.mode == "val":
            return self.active_env_profile
        return self._get_train_env_profile(group_id)

    def _get_env_bundle(self, profile_name: Optional[str] = None) -> Dict[str, Any]:
        if not self.is_mixed_env_llm_mode():
            return self.env_llm_bundle
        normalized = self._normalize_env_profile_name(profile_name)
        if normalized is None:
            raise ValueError("mixed env_llm requires a profile to resolve bundle")
        return self.env_llm_bundles[normalized]

    def _init_llms(self):
        env_llm_cfg = self.sys_config.env_llm
        judger_llm_cfg = self.sys_config.judger_llm
        self.env_llm_mode = str(_cfg_get(env_llm_cfg, "mode", "single")).strip().lower()
        if self.env_llm_mode == "fixed":
            self.env_llm_mode = "mixed"
        if self.env_llm_mode not in {"single", "mixed"}:
            print(f"[WARN][env_llm] invalid mode={self.env_llm_mode!r}, fallback to 'single'", flush=True)
            self.env_llm_mode = "single"

        self.env_llm_bundles: Dict[str, Dict[str, Any]] = {}
        self.env_llm_profiles: List[str] = []
        self.env_llm_train_profiles: List[str] = []
        self.env_llm_validation_profiles: List[str] = []
        self.active_env_profile: Optional[str] = None

        if self.env_llm_mode == "mixed":
            profiles_cfg = _cfg_get(env_llm_cfg, "profiles", None)
            if profiles_cfg is None:
                raise ValueError("env_llm.mode=mixed requires env_llm.profiles")
            for profile_name, profile_cfg in profiles_cfg.items():
                normalized = str(profile_name).strip()
                if not normalized:
                    continue
                self.env_llm_bundles[normalized] = self._build_env_client_bundle(profile_cfg, profile_name=normalized)
            self.env_llm_profiles = list(self.env_llm_bundles.keys())
            if not self.env_llm_profiles:
                raise ValueError("env_llm.mode=mixed requires at least one env_llm profile")
            if self.expand_mixed_train_groups and self.mixed_train_targets:
                self.env_llm_train_profiles = [
                    self.mixed_train_targets[group_id % len(self.mixed_train_targets)]
                    for group_id in range(self.env_groups)
                ]
            else:
                self.env_llm_train_profiles = [
                    p for p in _coerce_str_list(_cfg_get(env_llm_cfg, "train_profiles", [])) if p in self.env_llm_bundles
                ]
                if not self.env_llm_train_profiles:
                    self.env_llm_train_profiles = [self.env_llm_profiles[0]] * max(1, self.env_groups)
            self.env_llm_validation_profiles = [
                p for p in _coerce_str_list(_cfg_get(env_llm_cfg, "validation_profiles", [])) if p in self.env_llm_bundles
            ]
            if not self.env_llm_validation_profiles:
                self.env_llm_validation_profiles = list(self.env_llm_profiles)
            print(
                f"[DEBUG][mixed_env_llm.init] profiles={self.env_llm_profiles} "
                f"train_profiles={self.env_llm_train_profiles} validation_profiles={self.env_llm_validation_profiles}",
                flush=True,
            )
            initial_profile = self.env_llm_validation_profiles[0] if self.mode == "val" else self.env_llm_train_profiles[0]
            self.set_active_env_profile(initial_profile)
        else:
            self.env_llm_bundle = self._build_env_client_bundle(env_llm_cfg, profile_name=None)
            self._apply_env_bundle_aliases(self.env_llm_bundle)

        judger_api_model = getattr(judger_llm_cfg, "api_model", judger_llm_cfg.model_path)
        judger_tokenizer_path = getattr(judger_llm_cfg, "tokenizer_path", None)
        if judger_tokenizer_path is None and hasattr(judger_llm_cfg, "model_path"):
            if os.path.exists(str(judger_llm_cfg.model_path)):
                judger_tokenizer_path = judger_llm_cfg.model_path
        self.judger_llm = OpenAIVLLMClient(
            base_url=judger_llm_cfg.base_url,
            api_model=judger_api_model,
            tokenizer_path=judger_tokenizer_path,
            api_key=getattr(judger_llm_cfg, "api_key", "EMPTY"),
            enable_logit_bias=True,
            client_name="judger_llm",
            max_retries=3,
            timeout_sec=20
        )
        self.judger_llm_params = {
            "max_tokens": getattr(judger_llm_cfg, "max_tokens", 1),
            "temperature": getattr(judger_llm_cfg, "temperature", 0.0),
            "logprobs": 2,
        }
        self.judger_llm_max_context = getattr(judger_llm_cfg, "max_context_tokens", 2048)
        try:
            _judger_out_tokens = int(self.judger_llm_params.get("max_tokens", 1) or 1)
        except Exception:
            _judger_out_tokens = 0
        if self.judger_llm_max_context is not None:
            self.judger_llm_max_prompt_tokens = max(self.judger_llm_max_context - _judger_out_tokens - 2, 0)
        else:
            self.judger_llm_max_prompt_tokens = None

    def _chat_batch(self, messages_batch, llm_params, profile_name: Optional[str] = None):
        bundle = self._get_env_bundle(profile_name)
        client = bundle["client"]
        max_workers = bundle["max_workers"]
        max_tokens = bundle["max_tokens"]
        print(
            f"[DEBUG][env_llm._chat_batch] profile={bundle['profile_name']} base_url={client.base_url} api_model={client.api_model} llm_params={llm_params}"
        )
        results = [None] * len(messages_batch)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(client.chat, messages, **llm_params): idx
                for idx, messages in enumerate(messages_batch)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                    if results[idx] is not None:
                        if max_tokens is not None and client.tokenizer is not None:
                            results[idx] = _truncate_text_by_tokens(results[idx], client.tokenizer, max_tokens)
                        compact = re.sub(r"[^A-Za-z]", "", results[idx])
                        if compact and re.fullmatch(r"(Yes|No)+", compact):
                            preview_msgs = messages_batch[idx][-3:] if messages_batch[idx] else []
                            print(
                                f"profile={bundle['profile_name']} base_url={client.base_url} api_model={client.api_model} "
                                f"resp_len={len(results[idx])} compact_len={len(compact)} messages_tail={preview_msgs}"
                            )
                except Exception as e:
                    print(f"Error in chat_batch for idx {idx}: {e}")
                    results[idx] = ""
        return results

    def _chat_batch_allinone(self, messages_batch, llm_params, profile_name: Optional[str] = None):
        bundle = self._get_env_bundle(profile_name)
        client = bundle["client"]
        max_tokens = bundle["max_tokens"]
        results = client.batch_complete(messages_batch, **llm_params)
        if max_tokens is not None and client.tokenizer is not None:
            results = [_truncate_text_by_tokens(r, client.tokenizer, max_tokens) for r in results]
        return results

    def _prefer_thread_chat_for_profile(self, profile_name: Optional[str] = None) -> bool:
        bundle = self._get_env_bundle(profile_name)
        return bool(bundle.get("use_responses_api", False))

    def _get_env_response_tokens(self, response: str, profile_name: Optional[str] = None) -> int:
        bundle = self._get_env_bundle(profile_name)
        client = bundle["client"]
        if client.tokenizer is None:
            return 0
        try:
            return len(client.tokenizer.encode(response or "", add_special_tokens=False))
        except Exception:
            return 0

    def batch_generate_env_llm(self, messages_batch: List[List[Dict]], env_profiles: Optional[List[Optional[str]]] = None):
        if not messages_batch:
            return []
        if not self.is_mixed_env_llm_mode():
            llm_params = dict(self.env_llm_params)
            if self._prefer_thread_chat_for_profile(None):
                return self._chat_batch(messages_batch, llm_params)
            try:
                return self._chat_batch_allinone(messages_batch, llm_params)
            except Exception as e:
                print(f"[WARN][batch_generate_env_llm] fallback to thread batch: {e}")
                return self._chat_batch(messages_batch, llm_params)

        if env_profiles is not None and len(env_profiles) != len(messages_batch):
            raise ValueError("env_profiles length must match messages_batch length in mixed env_llm mode")

        profile_by_index: List[str] = []
        for idx in range(len(messages_batch)):
            requested = env_profiles[idx] if env_profiles is not None else None
            normalized = self._normalize_env_profile_name(requested)
            if normalized is None:
                raise ValueError("mixed env_llm mode requires a resolved env profile")
            profile_by_index.append(normalized)

        results = [""] * len(messages_batch)
        grouped: Dict[str, List[tuple[int, List[Dict]]]] = {}
        for idx, (messages, profile_name) in enumerate(zip(messages_batch, profile_by_index)):
            grouped.setdefault(profile_name, []).append((idx, messages))

        for profile_name, items in grouped.items():
            sub_messages = [messages for _, messages in items]
            llm_params = dict(self._get_env_bundle(profile_name)["llm_params"])
            if self._prefer_thread_chat_for_profile(profile_name):
                print(
                    f"[DEBUG][batch_generate_env_llm] profile={profile_name} uses thread chat path "
                    f"because responses API does not support all-in-one completions batching"
                )
                sub_results = self._chat_batch(sub_messages, llm_params, profile_name=profile_name)
            else:
                try:
                    sub_results = self._chat_batch_allinone(sub_messages, llm_params, profile_name=profile_name)
                except Exception as e:
                    print(f"[WARN][batch_generate_env_llm] profile={profile_name} fallback to thread batch: {e}")
                    sub_results = self._chat_batch(sub_messages, llm_params, profile_name=profile_name)
            for (original_idx, _), result in zip(items, sub_results):
                results[original_idx] = result
        return results

    def _get_logprobs_batch_multi_thread(self, messages_batch, llm_params):
        results = [None] * len(messages_batch)
        with ThreadPoolExecutor(max_workers=256) as executor:
            futures = {
                executor.submit(self.judger_llm.get_logprobs, messages, **llm_params): idx
                for idx, messages in enumerate(messages_batch)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"Error in get_logprobs_batch for idx {idx}: {e}")
                    results[idx] = None
        return results

    def _get_logprobs_batch(self, messages_batch, llm_params):
        index = int(self.sys_config.judger_llm.max_tokens) - 1
        batch_size = getattr(self.sys_config.judger_llm, "batch_size", None)
        if not batch_size or batch_size <= 0:
            batch_size = len(messages_batch)
        print(f"[DEBUG][judger_llm._get_logprobs_batch] base_url={self.judger_llm.base_url} api_model={self.judger_llm.api_model} llm_params={llm_params} index={index} batch_size={batch_size}")
        all_results = []
        neutral_logprob = math.log(0.5)
        for start in range(0, len(messages_batch), batch_size):
            chunk = messages_batch[start:start + batch_size]
            try:
                results = self.judger_llm.batch_get_logprobs_complete(batch_prompts=chunk, index=index, **llm_params)
                all_results.extend(results)
            except Exception as e:
                # If retries fail for this chunk, return neutral logprobs for the chunk
                print(f"[WARN][judger_llm._get_logprobs_batch] fallback to neutral logprobs for chunk due to error: {e}")
                all_results.extend([[{"Yes": neutral_logprob, "No": neutral_logprob}] for _ in chunk])
        return all_results

    def batch_score_with_judger(self, x0_list: List[str], y_list: List[str]):
        if not x0_list:
            return []
        prompts = []
        for x0, y in zip(x0_list, y_list):
            prompt = LLAMA2_CLS_PROMPT["prompt"].format(behavior=x0 or "", generation=y or "")
            if self.judger_llm_max_prompt_tokens and self.judger_llm.tokenizer is not None:
                prompt = _truncate_text_by_tokens(prompt, self.judger_llm.tokenizer, self.judger_llm_max_prompt_tokens)
            prompts.append(prompt)
        logprobs_list = self._get_logprobs_batch(prompts, self.judger_llm_params)
        wanted = getattr(self.sys_config.judger_llm, "wanted_token", "Yes")
        scores = []
        for logprobs in logprobs_list:
            score = 0.0
            try:
                if logprobs and isinstance(logprobs, list):
                    first = logprobs[0]
                    if isinstance(first, dict):
                        variants = [wanted, " " + wanted, "▁" + wanted]
                        lp = None
                        for v in variants:
                            if v in first:
                                lp = first[v]
                                break
                        if lp is not None:
                            score = math.exp(lp)
                            if score > 1:
                                score = 1.0
            except Exception:
                score = 0.0
            scores.append(score)
        return scores

    def _load_targets(self, data_paths):
        import pandas as pd
        import json
        all_targets = []
        all_sources = []
        for data_path in data_paths:
            try:
                if data_path.endswith('.parquet'):
                    df = pd.read_parquet(data_path)
                    targets = df['prompt'].dropna().tolist()
                    print(data_path, "data length: ", len(targets))
                    all_targets.extend(targets)
                    source = "advbench" if "advbench" in data_path.lower() else "parquet"
                    all_sources.extend([source] * len(targets))
                elif data_path.endswith('.csv'):
                    df = pd.read_csv(data_path)
                    cols = set(df.columns)
                    # StrongREJECT
                    if {"forbidden_prompt", "source"} <= cols:
                        targets = df[df["source"] != "AdvBench"]["forbidden_prompt"].dropna().tolist()
                        print(data_path, "data length: ", len(targets))
                        all_targets.extend(targets)
                        all_sources.extend(["strongreject"] * len(targets))
                    # JailbreakBench
                    elif {"Goal", "Source"} <= cols:
                        targets = df[df["Source"] == "Original"]["Goal"].dropna().tolist()
                        print(data_path, "data length: ", len(targets))
                        all_targets.extend(targets)
                        all_sources.extend(["jailbench"] * len(targets))
                    # HarmBench
                    elif {"Behavior", "FunctionalCategory"} <= cols:
                        targets = df[df["FunctionalCategory"] == "standard"]["Behavior"].dropna().tolist()
                        print(data_path, "data length: ", len(targets))
                        all_targets.extend(targets)
                        all_sources.extend(["harmbench"] * len(targets))
                    # AdvBench
                    elif {"instruct"} <= cols:
                        targets = df["instruct"].dropna().tolist()
                        print(data_path, "data length: ", len(targets))
                        all_targets.extend(targets)
                        all_sources.extend(["advbench"] * len(targets))
                    else:
                        print(f"Warning: Unrecognized csv schema in {data_path}, skip.")
                elif data_path.endswith('.jsonl'):
                    targets = []
                    with open(data_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                            except Exception:
                                continue
                            if isinstance(obj, dict) and "vanilla" in obj:
                                val = obj.get("vanilla")
                                if isinstance(val, str) and val.strip():
                                    targets.append(val)
                    if targets:
                        print(data_path, "data length: ", len(targets))
                        all_targets.extend(targets)
                        source = "wildjailbreak" if "wildjailbreak" in data_path.lower() else "jsonl"
                        all_sources.extend([source] * len(targets))
            except Exception as e:
                print(f"Warning: Failed to load data from {data_path}: {e}")
                continue
        return all_targets, all_sources

    def _init_envs(self):
        """Initialize the environments. train_envs and val_envs are lists of envs:
        Input: tags: ["SimpleSokoban", "HarderSokoban"]; n_groups: [1, 1]; group_size: 16
        Output: envs: List[Dict], each **entry** is a dict with keys: tag, group_id, env_id, env, env_config, status
        Example: [{"tag": "SimpleSokoban", "group_id": 0, "env_id": 0, "env": env, "config": env_config, "status": EnvStatus()},
            ...
            {"tag": "SimpleSokoban", "group_id": 0, "env_id": 15 (group_size - 1), ...},
            {"tag": "HarderSokoban", "group_id": 1, "env_id": 16, ...}
            ...]
        """
        assert sum(self.config.env_configs.n_groups) == self.env_groups, f"Sum of n_groups must equal env_groups. Got sum({self.config.env_configs.n_groups}) != {self.env_groups}"
        assert len(self.config.env_configs.tags) == len(self.config.env_configs.n_groups), f"Number of tags must equal number of n_groups. Got {len(self.config.env_configs.tags)} != {len(self.config.env_configs.n_groups)}"
        self.envs = self._init_env_instances(self.config)

    def _init_env_instances(self, config):
        env_list = []
        done_groups = 0
        for tag, n_group in zip(config.env_configs.tags, config.env_configs.n_groups):
            for env_id in range(done_groups * self.group_size, (done_groups + n_group) * self.group_size):
                cfg_template = self.sys_config.custom_envs[tag]
                env_class = cfg_template.env_type
                max_actions_per_traj = cfg_template.max_actions_per_traj
                if cfg_template.env_config is None:
                    env_config = REGISTERED_ENV_CONFIGS[env_class]()
                else:
                    env_config = REGISTERED_ENV_CONFIGS[env_class](**cfg_template.env_config)
                if hasattr(env_config, "attacker_format"):
                    env_config.attacker_format = getattr(self.sys_config.agent_proxy, "attacker_format", getattr(env_config, "attacker_format", "legacy_think_answer"))
                if hasattr(env_config, "max_turns"):
                    env_config.max_turns = int(getattr(self.sys_config.agent_proxy, "max_turn", getattr(env_config, "max_turns", 5)))
                env_obj = REGISTERED_ENVS[env_class](env_config)
                entry = {'tag': tag, 'group_id': env_id // self.group_size, 'env_id': env_id, 
                        'env': env_obj, 'config': env_config, 'status': EnvStatus(), 'max_actions_per_traj': max_actions_per_traj}
                env_list.append(entry)
            done_groups += n_group
        return env_list

    def reset(self, seed: Optional[int] = None):
        """
        Reset the environments and get initial observation
        build up rollout cache like [{"env_id": int, "history": List[Dict], "group_id": int}, ...]
        """
        def _expand_seed(seed: int):
            if self.expand_mixed_train_groups and self.mixed_train_targets:
                seeds = []
                for base_group_id in range(self.train_base_env_groups):
                    current_seed = seed + base_group_id
                    for _ in self.mixed_train_targets:
                        seeds.extend([current_seed] * self.group_size)
                return seeds
            seeds = [[seed + i] * self.group_size for i in range(self.env_groups)] # [[seed, ..., seed], [seed+1, ..., seed+1], ...]
            return sum(seeds, [])

        envs = self.envs
        rollout_cache = [{"env_id": entry['env_id'], "history": [], "group_id": entry['group_id'], "tag": entry['tag'], "penalty": 0} for entry in envs]

        # reset all environments
        if seed is None:
            if self.mode == "train":
                if self.base_seed is not None:
                    seed = int(self.base_seed + self.seed_counter)
                    self.seed_counter += self.train_base_env_groups if self.expand_mixed_train_groups else self.env_groups
                else:
                    seed = int(random.randint(0, 1000000))
            else:
                seed = int(123 if self.base_seed is None else self.base_seed)
        else:
            if self.mode == "train" and self.base_seed is not None:
                self.seed_counter = int(seed - self.base_seed + 1)
            seed = int(seed)
        seeds = _expand_seed(seed) if envs else []

        if self.mode == "val":
            total_targets = len(self.val_targets)
            target_sources = self.val_target_sources
        else:
            total_targets = len(self.targets)
            target_sources = self.target_sources

        for entry, seed in zip(envs, seeds):
            group_id = entry['group_id']
            if self.mode == "val":
                idx = (self.harmful_target_base_idx + group_id) % total_targets
                harmful_target = self.val_targets[idx]
            else:
                base_group_id = group_id // len(self.mixed_train_targets) if self.expand_mixed_train_groups and self.mixed_train_targets else group_id
                idx = (self.harmful_target_base_idx + base_group_id) % total_targets
                harmful_target = self.targets[idx]
            target_model_profile = self._resolve_env_profile_for_group(group_id)
            entry['env'].reset(seed=seed, mode=self.mode, harmful_target=harmful_target)
            entry['status'] = EnvStatus(seed=seed)
            entry['data_source'] = target_sources[idx] if total_targets > 0 else "unknown"
            entry['target_model_profile'] = target_model_profile
            
        for cache in rollout_cache:
            cache['init_prompt'] = envs[cache['env_id']]['env'].init_prompt
            cache['harmful_target'] = envs[cache['env_id']]['env'].current_target
            cache['data_source'] = envs[cache['env_id']].get('data_source', "unknown")
            cache['target_model_profile'] = envs[cache['env_id']].get('target_model_profile', None)

        if self.is_mixed_env_llm_mode() and envs:
            group_profile_map = {}
            for entry in envs:
                group_profile_map.setdefault(entry['group_id'], entry.get('target_model_profile', None))
            ordered_map = {gid: group_profile_map[gid] for gid in sorted(group_profile_map)}
            print(f"[DEBUG][mixed_env_llm.reset] mode={self.mode} group_profile_map={ordered_map}", flush=True)

        # update rollout cache
        for cache, env in zip(rollout_cache, envs):
            next_state = self._handle_mm_state(env['env'].render())
            cache['history'] = self._update_cache_history(cache['history'], next_state=next_state, actions_left=env['max_actions_per_traj'], num_actions_info=None)
            
        if total_targets > 0:
            target_group_advance = self.train_base_env_groups if self.expand_mixed_train_groups else self.env_groups
            self.harmful_target_base_idx = (self.harmful_target_base_idx + target_group_advance) % total_targets
        else:
            self.harmful_target_base_idx = 0
        self.rollout_cache = rollout_cache if rollout_cache else []
        # pdb.set_trace()
        return self.rollout_cache

    def step(self, all_env_inputs: List[Dict]):
        """Step the environments.
        1. extract valid actions from the action lookup table (if exists) and execute the actions, and update rollout cache
        2. Since rollout does not need to act over done envs, whenever the environment is done, we only update rollout cache, but not output env_outputs.
        Input:
        all_env_inputs: List[Dict]
            {env_id: int, llm_response: str, actions: List[str]}
            NOTE: should use env_id as index for existing some already done envs
        env_outputs: List[Dict]
            {env_id: int, history: List[Dict][{state: str, actions: List[str], reward: float, info: Dict, llm_response: str, llm_raw_response: str, (Optional)images: List[PIL.Image.Image]}]}
        """
        def _execute_actions(env, actions):
            acc_reward, turn_info, turn_done = 0, {}, False
            executed_actions = []
            for action in actions:
                _, reward, done, info = env.step(action)
                acc_reward += reward
                turn_info.update(info) # NOTE: currently use last info for multi-action
                executed_actions.append(action)
                if done:
                    turn_done = True
                    break
            
            return acc_reward, turn_info, turn_done, executed_actions

        def _log_env_state(status, history, cur_obs, max_actions_per_traj, executed_actions, all_actions, acc_reward, turn_done, turn_info, env_input, env_response, env_response_tokens, attacker_response_tokens):
            obs = self._handle_mm_state(cur_obs)
            status.num_actions += len(executed_actions)
            status.rewards.append(acc_reward) # NOTE use turn-wise acc_reward
            actions_left = max_actions_per_traj - status.num_actions
            if turn_done:
                # status.terminated = True # TODO check terminated definition in gymnasium
                # status.truncated = not turn_info.get('success', False)
                status.truncated = not turn_info.get('turn_success', False) # TODO check truncated definition in gymnasium
                status.terminated = turn_info.get('turn_success', False)
            history = self._update_cache_history(history, next_state=obs, actions_left=actions_left, num_actions_info={
                'actions': executed_actions, 'reward': acc_reward, 'info': turn_info,
                'llm_response': env_input['llm_response'], 'llm_raw_response': env_input['llm_raw_response'],
                'visible_llm_response': env_input.get('visible_llm_response', env_input['llm_response']),
                'env_response': env_response,
                'attacker_response_tokens': attacker_response_tokens,
                'env_response_tokens': env_response_tokens,
            })
            # filter out invalid actions
            # history = [content for content in history[:-1] if content['actions']] + [history[-1]]
            return status, history

        envs = self.envs
        env_outputs = []

        env_llm_requests: Dict[str, List[tuple[int, List[Dict]]]] = {}

        for env_input in all_env_inputs:
            entry = envs[env_input['env_id']]
            env_id, env = entry['env_id'], entry['env']

            if env.done:
                continue

            env_messages = env.get_env_llm_messages(env_input['actions'][0])
            if env_messages:
                profile_name = self._resolve_env_profile_for_group(entry['group_id'])
                request_key = profile_name or "single"
                env_llm_requests.setdefault(request_key, []).append((env_id, env_messages))

        env_responses = {}
        env_response_tokens = {}
        if self.is_mixed_env_llm_mode() and env_llm_requests:
            request_sizes = {profile: len(items) for profile, items in env_llm_requests.items()}
            print(f"[DEBUG][mixed_env_llm.step] mode={self.mode} request_sizes={request_sizes}", flush=True)
        for profile_name, items in env_llm_requests.items():
            actual_profile = None if profile_name == "single" else profile_name
            env_llm_batch = [messages for _, messages in items]
            env_llm_indices = [env_id for env_id, _ in items]
            try:
                if self.is_mixed_env_llm_mode():
                    env_llm_responses = self.batch_generate_env_llm(
                        env_llm_batch,
                        env_profiles=[actual_profile] * len(env_llm_batch),
                    )
                else:
                    llm_params = dict(self.env_llm_params)
                    env_llm_responses = self._chat_batch(env_llm_batch, llm_params)
                for env_id, response in zip(env_llm_indices, env_llm_responses):
                    env_responses[env_id] = response
                    env_response_tokens[env_id] = self._get_env_response_tokens(response, actual_profile)
            except Exception as e:
                print(f"Error in batch env LLM call for profile={actual_profile}: {e}")
                for env_id in env_llm_indices:
                    env_responses[env_id] = ""
                    env_response_tokens[env_id] = 0

        judger_llm_batch = []
        judger_llm_indices = []
        
        for env_input in all_env_inputs:
            entry = envs[env_input['env_id']]
            env_id, env = entry['env_id'], entry['env']
            
            if env.done:
                continue
                
            env.dialogue_history.append({"role": "assistant", "content": env_input['actions'][0]})
            if env_id in env_responses:
                env.dialogue_history.append({"role": "user", "content": env_responses[env_id]})
            
            judger_messages = env.get_judger_llm_messages()
            if judger_messages:
                if self.judger_llm_max_prompt_tokens and self.judger_llm.tokenizer is not None:
                    token_count = _count_prompt_tokens(judger_messages, self.judger_llm.tokenizer)
                    if token_count is not None and token_count > self.judger_llm_max_prompt_tokens:
                        print(
                            f"[WARN][judger_llm] prompt too long before truncation: "
                            f"tokens={token_count}, max_prompt_tokens={self.judger_llm_max_prompt_tokens}, "
                            f"type={'str' if isinstance(judger_messages, str) else type(judger_messages)}",
                            flush=True,
                        )
                    if isinstance(judger_messages, str):
                        judger_messages = _truncate_text_by_tokens(
                            judger_messages, self.judger_llm.tokenizer, self.judger_llm_max_prompt_tokens
                        )
                    else:
                        judger_messages = _truncate_messages_by_tokens(
                            judger_messages, self.judger_llm.tokenizer, self.judger_llm_max_prompt_tokens
                        )
                    token_count_after = _count_prompt_tokens(judger_messages, self.judger_llm.tokenizer)
                    if token_count_after is not None and token_count_after > self.judger_llm_max_prompt_tokens:
                        preview = None
                        if isinstance(judger_messages, str):
                            preview = judger_messages[-200:]
                        elif isinstance(judger_messages, list) and judger_messages:
                            try:
                                preview = judger_messages[-1].get("content", "")[-200:]
                            except Exception:
                                preview = None
                        print(
                            f"[ERROR][judger_llm] prompt still too long after truncation: "
                            f"tokens={token_count_after}, max_prompt_tokens={self.judger_llm_max_prompt_tokens}, "
                            f"type={'str' if isinstance(judger_messages, str) else type(judger_messages)}, "
                            f"preview_tail={preview}",
                            flush=True,
                        )
                judger_llm_batch.append(judger_messages)
                judger_llm_indices.append(env_id)

        judger_scores = {}
        if judger_llm_batch:
            try:
                judger_responses = self._get_logprobs_batch(judger_llm_batch, self.judger_llm_params)
                for env_id, logprobs in zip(judger_llm_indices, judger_responses):
                    try:
                        if logprobs and isinstance(logprobs, list):
                            unsafe_logprob = logprobs[0][self.sys_config.judger_llm.wanted_token]
                            if unsafe_logprob is not None:
                                if math.exp(unsafe_logprob) > 1:
                                    raise ValueError(f"Unsafe prob is too large: {math.exp(unsafe_logprob)}. Unsafe logprob: {unsafe_logprob}")
                                judger_scores[env_id] = math.exp(unsafe_logprob)
                            else:
                                print(f"Warning: No unsafe logprob found for env {env_id}")
                                judger_scores[env_id] = 0.0
                        else:
                            print(f"Warning: logprobs is not a list for env {env_id}")
                            judger_scores[env_id] = 0.0
                    except Exception as e:
                        if str(e) == "'Yes'":
                            judger_scores[env_id] = 0.0 # Too small even set logit_bias
                        else:
                            print(f"Error parsing logprobs for env {env_id}: {e}")
                            judger_scores[env_id] = 0.0
            except Exception as e:
                print(f"Error in batch judger LLM call: {e}")
                for env_id in judger_llm_indices:
                    judger_scores[env_id] = 0.0

        for env_input in all_env_inputs:
            entry = envs[env_input['env_id']]
            env_id, env = entry['env_id'], entry['env']
            actions_left_before = entry['max_actions_per_traj'] - entry['status'].num_actions

            if env.done:
                continue

            env_response = env_responses.get(env_id, "")
            env_resp_tokens = env_response_tokens.get(env_id, 0)
            score = judger_scores.get(env_id, 0.0)
            
            next_obs, reward, done, info = env.step_with_llm_response(
                env_input['actions'][0], env_response, score
            )
            
            valid_actions = self._extract_map_valid_actions(entry, env_input['actions'])
            if len(valid_actions) > 1:
                raise ValueError("Multi-action not supported, a query is seen as an action")
                acc_reward, turn_info, turn_done, executed_actions = _execute_actions(env, valid_actions[1:])
                reward += acc_reward
                info.update(turn_info)
                if turn_done:
                    done = True
            else:
                executed_actions = valid_actions[:1] if valid_actions else []
                turn_info = info

            attacker_resp_tokens = int(env_input.get("attacker_response_tokens", 0) or 0)

            parse_status = env_input.get("parse_status", None)
            if parse_status is not None:
                turn_info = dict(turn_info)
                turn_info["qwen3_parse_status"] = parse_status
                turn_info["qwen3_parse_ok"] = float(str(parse_status).startswith("ok"))
                turn_info["qwen3_parse_malformed"] = float("missing_endthink" in str(parse_status))
            
            if len(valid_actions) != len(env_input['actions']) or not valid_actions:
                self.rollout_cache[env_id]["penalty"] += self.sys_config.es_manager.format_penalty
                
            status, history = _log_env_state(
                entry['status'],
                self.rollout_cache[env_id]['history'],
                next_obs,
                entry['max_actions_per_traj'],
                executed_actions,
                valid_actions,
                reward,
                done,
                turn_info,
                env_input,
                env_response,
                env_resp_tokens,
                attacker_resp_tokens,
            )
            entry['status'] = status
            if entry['status'].num_actions >= entry['max_actions_per_traj'] and not done:
                entry['status'].truncated = True
                entry['status'].terminated = False
                done = True
            self.rollout_cache[env_id]['history'] = history
            self.rollout_cache[env_id]['dialogue_history'] = env.dialogue_history
            self.rollout_cache[env_id]['full_dialogue_history'] = self._build_full_dialogue_history(self.rollout_cache[env_id])
            self.rollout_cache[env_id]['turn_scores'] = [turn['info']['score'] for turn in history[:-1]]
            self.rollout_cache[env_id]['attacker_tokens'] = [int(turn.get('attacker_response_tokens', 0)) for turn in history[:-1]]
            self.rollout_cache[env_id]['target_tokens'] = [int(turn.get('env_response_tokens', 0)) for turn in history[:-1]]
            if not done: # NOTE done environments are not sent for further llm generation (for efficiency)
                env_outputs.append(self.rollout_cache[env_id])

        return env_outputs

    def get_rollout_states(self):
        """Get the final output for all environment"""
        envs = self.envs
        rollout_cache = self.rollout_cache
        TURN_LVL_METRICS = ['action_is_effective', 'action_is_valid', 'end_of_page']

        # add metrics to rollout cache
        for entry, cache in zip(envs, rollout_cache):
            status = entry['status']
            env_metric = {
                'success': float(status.terminated and (not status.truncated)),
                'num_actions': status.num_actions,
            }
            custom_metric = {}
            for turn in cache['history']:
                for k, v in turn.get('info', {}).items():
                    if k not in custom_metric:
                        custom_metric[k] = []
                    custom_metric[k].append(v)
            for k, v in custom_metric.items():
                env_metric[k] = v
            if "qwen3_parse_ok" in custom_metric and custom_metric["qwen3_parse_ok"]:
                env_metric["qwen3_parse_ok_rate"] = float(sum(custom_metric["qwen3_parse_ok"]) / len(custom_metric["qwen3_parse_ok"]))
            if "qwen3_parse_malformed" in custom_metric and custom_metric["qwen3_parse_malformed"]:
                env_metric["qwen3_malformed_rate"] = float(sum(custom_metric["qwen3_parse_malformed"]) / len(custom_metric["qwen3_parse_malformed"]))

            cache['history'][-1]['metrics'] = custom_metric
            env_metric = {f"{entry['tag']}/{k}": v for k, v in env_metric.items()}
            cache['metrics'] = env_metric
            if entry['tag'] == "MetamathQA":
                cache['correct_answer'] = entry['env'].correct_answer

        # calculate pass@k where k is the group size
        group_success = {}
        for entry, cache in zip(envs, rollout_cache):
            key = (entry['tag'], entry['group_id'])
            success_val = cache['metrics'].get(f"{entry['tag']}/success", 0.0)
            group_success.setdefault(key, []).append(success_val)

        for (tag, gid), succ_list in group_success.items():
            pass_success = float(any(succ_list))
            for entry, cache in zip(envs, rollout_cache):
                if entry['tag'] == tag and entry['group_id'] == gid:
                    cache['metrics'][f"{tag}/pass@{self.group_size}"] = pass_success
        return rollout_cache




    def _update_cache_history(self, history: List[Dict], next_state, actions_left, num_actions_info: Optional[Dict] = None):
        """
        Update last step info and append state to history
        """
        if num_actions_info is not None: # update last step info
            assert len(history), "History should not be empty"
            history[-1].update(num_actions_info)
        
        entry = {} # append state to history
        if isinstance(next_state, str): # text state
            entry['state'] = next_state
        else: # multimodal state
            entry['state'] = "<images>" * len(next_state)
            entry['images'] = next_state
        entry['actions_left'] = actions_left
        history.append(entry)
        return history

    def _build_full_dialogue_history(self, cache: Dict) -> List[Dict]:
        dialogue = []
        init_prompt = cache.get('init_prompt', None)
        if init_prompt is not None:
            dialogue.append({"role": "user", "content": init_prompt})
        for turn in cache.get('history', []):
            if 'llm_response' in turn:
                dialogue.append({"role": "assistant", "content": turn.get('llm_response', '')})
            if 'env_response' in turn:
                dialogue.append({"role": "user", "content": turn.get('env_response', '')})
        return dialogue

    def _extract_map_valid_actions(self, entry: Dict, actions: List[str]):
        """extract valid actions from the action lookup table (if exists)"""
        mapped_actions = []
        action_lookup = getattr(entry['env'].config, 'action_lookup', None)
        if action_lookup is None:
            mapped_actions = actions
        else: # the envs have pre-defined action lookup
            rev_action_lookup = {v.lower(): k for k, v in action_lookup.items()}
            actions = [action.lower() for action in actions]
            mapped_actions = [rev_action_lookup[action] for action in actions if action in rev_action_lookup]
        return mapped_actions
    
    def _handle_mm_state(self, state: Union[str, np.ndarray, list[np.ndarray]]):
        """Handle the state from the environment
        """
        if state is None:
            # Some text-only envs can transiently produce an empty render cache.
            # Treat it as an empty text state instead of falling into the image path.
            return ""
        if isinstance(state, str): # text state
            return state
        elif isinstance(state, np.ndarray): # when env state is a single image, convert it to a list to unify output format
            state = [state]
        results = [PIL.Image.fromarray(_state, mode='RGB') for _state in state]
        return results
        
    def render(self):
        rendered_list = [entry['env'].render() for entry in self.envs]
        return rendered_list

    def close(self):
        for entry in self.envs:
            entry['env'].close()




@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config):
    """
    Unit test for EnvStateManager
    """
    es_manager = EnvStateManager(config, mode="train")
    print("Initializing environments...")
    es_manager.reset(seed=123)

    renders = es_manager.render()
    for i, render in enumerate(renders[:4]):  # Show first 2 environments
        print(f"Environment {i}:\n{render}\n")
    
    print("\nRunning step for training environments...")
    all_env_inputs = [
        {
            "env_id": 0,
            "llm_raw_response": "Go down",
            "llm_response": "Go down",
            "actions": ["down"]
        },
        {
            "env_id": 3,
            "llm_raw_response": "Go down",
            "llm_response": "Go down",
            "actions": ["down"]
        }
    ]
    env_outputs = es_manager.step(all_env_inputs)
    print(f"Active environments after step: {len(env_outputs)}")
    print(f"env_outputs[:2]: {env_outputs[:2]}")
    
    renders = es_manager.render()
    for i, render in enumerate(renders[:4]):  # Show first 2 environments
        print(f"Environment {i}:\n{render}\n")

    all_env_inputs = [
        {
            "env_id": 0,
            "llm_raw_response": "Go left, go up",
            "llm_response": "Go left, go up",
            "actions": ["left", "up"]
        },
        {
            "env_id": 3,
            "llm_raw_response": "Go up, go up",
            "llm_response": "Go up, go up",
            "actions": ["up", "up", "up", "up", "up"]
        }
    ]
    env_outputs = es_manager.step(all_env_inputs)
    print(f"Active environments after step: {len(env_outputs)}")
    print(f"env_outputs[:2]: {env_outputs[:2]}")
    
    renders = es_manager.render()
    for i, render in enumerate(renders[:4]):  # Show first 2 environments
        print(f"Environment {i}:\n{render}\n")
    
    print("\nRendering final output...")
    final_outputs = es_manager.get_rollout_states()
    print(f"final outputs[:4]: {final_outputs[:4]}")
    
    print("\nClosing environments...")
    es_manager.close()
    print("Test completed successfully!")


if __name__ == "__main__":
	main()
