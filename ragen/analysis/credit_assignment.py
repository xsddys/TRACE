# -*- coding: utf-8 -*-
"""Credit assignment pre-experiment utilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import json
import math
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class MaskConfig:
    strategy: str = "delete"  # delete | mask | omit | neutral | custom
    mask_text: str = "[MASK]"
    omit_text: str = "A round of dialogue is omitted here."
    neutral_text: str = "Let's talk about something else for a moment."


@dataclass
class ScoreConfig:
    max_y_tokens: Optional[int] = None
    trust_remote_code: bool = False
    dtype: str = "bf16"  # bf16 | fp16 | fp32
    device: Optional[str] = None  # e.g., "cuda", "cpu"


def load_rollout_jsonl(path: str, max_samples: Optional[int] = None) -> List[Dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if max_samples is not None and len(samples) >= max_samples:
                break
    return samples


def extract_turns(dialogue_history: List[Dict]) -> Tuple[Dict, List[Tuple[Dict, Dict]]]:
    if not dialogue_history:
        raise ValueError("empty dialogue_history")
    if len(dialogue_history) < 3:
        raise ValueError("dialogue_history too short to form turns")

    init_prompt = dialogue_history[0]
    turns = []
    for i in range(1, len(dialogue_history), 2):
        if i + 1 >= len(dialogue_history):
            break
        x = dialogue_history[i]     # attacker prompt (assistant)
        y = dialogue_history[i + 1] # target response (user)
        turns.append((x, y))
    return init_prompt, turns


def _apply_mask_to_turn(x: Dict, y: Dict, mask_cfg: MaskConfig) -> Optional[Tuple[Dict, Dict]]:
    if mask_cfg.strategy == "delete":
        return None
    if mask_cfg.strategy == "mask":
        x_new = dict(x)
        y_new = dict(y)
        x_new["content"] = mask_cfg.mask_text
        y_new["content"] = mask_cfg.mask_text
        return x_new, y_new
    if mask_cfg.strategy == "omit":
        x_new = dict(x)
        y_new = dict(y)
        x_new["content"] = mask_cfg.omit_text
        y_new["content"] = mask_cfg.omit_text
        return x_new, y_new
    if mask_cfg.strategy == "neutral":
        x_new = dict(x)
        y_new = dict(y)
        x_new["content"] = mask_cfg.neutral_text
        y_new["content"] = mask_cfg.neutral_text
        return x_new, y_new
    if mask_cfg.strategy == "custom":
        x_new = dict(x)
        y_new = dict(y)
        x_new["content"] = mask_cfg.mask_text
        y_new["content"] = mask_cfg.mask_text
        return x_new, y_new
    raise ValueError("Unknown mask strategy: {}".format(mask_cfg.strategy))


def build_env_llm_messages(
    dialogue_history: List[Dict],
    mask_turn_index: Optional[int],
    mask_cfg: MaskConfig,
) -> Tuple[List[Dict], str, Dict]:
    """
    Build messages for env-llm (target) conditioning, and return y_T.

    Returns:
        messages: List[Dict] - roles swapped, ending with user message = x_T
        y_T: str - final target response content
        meta: Dict - info about turns
    """
    _, turns = extract_turns(dialogue_history)
    if len(turns) < 1:
        raise ValueError("No turns extracted")

    T = len(turns)
    last_x, last_y = turns[-1]
    y_T = last_y.get("content", "")

    past_turns = turns[:-1]

    built_entries: List[Dict] = []
    for t_idx, (x, y) in enumerate(past_turns):
        if mask_turn_index is not None and t_idx == mask_turn_index:
            masked = _apply_mask_to_turn(x, y, mask_cfg)
            if masked is None:
                continue
            x, y = masked
        built_entries.append({"role": "assistant", "content": x.get("content", "")})
        built_entries.append({"role": "user", "content": y.get("content", "")})

    # Swap roles to match env_llm usage
    messages: List[Dict] = []
    for entry in built_entries:
        swapped_role = "assistant" if entry["role"] == "user" else "user"
        messages.append({"role": swapped_role, "content": entry["content"]})

    # Append current attacker query as user
    messages.append({"role": "user", "content": last_x.get("content", "")})

    meta = {
        "num_turns": T,
        "mask_turn_index": mask_turn_index,
    }
    return messages, y_T, meta


def _get_dtype(dtype: str):
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    raise ValueError("Unsupported dtype: {}".format(dtype))


def load_model_and_tokenizer(model_path: str, score_cfg: ScoreConfig):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=score_cfg.trust_remote_code)
    dtype = _get_dtype(score_cfg.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=score_cfg.trust_remote_code,
        device_map="auto" if score_cfg.device is None else None,
    )
    if score_cfg.device is not None:
        model = model.to(score_cfg.device)
    model.eval()
    return model, tokenizer


def score_yT_logprobs(
    model,
    tokenizer,
    messages: List[Dict],
    y_T: str,
    score_cfg: ScoreConfig,
) -> Tuple[List[float], float]:
    """
    Return per-token logprobs and sum logprob for y_T.
    """
    prompt_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids
    target_ids = tokenizer(y_T, add_special_tokens=False).input_ids

    if score_cfg.max_y_tokens is not None:
        target_ids = target_ids[: score_cfg.max_y_tokens]

    if len(target_ids) == 0:
        return [], 0.0

    input_ids = torch.cat([prompt_ids, torch.tensor([target_ids], dtype=prompt_ids.dtype)], dim=1)

    device = score_cfg.device
    if device is None:
        device = model.device
    input_ids = input_ids.to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits

    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    start = prompt_ids.shape[1] - 1

    token_logprobs: List[float] = []
    for i, tok_id in enumerate(target_ids):
        lp = log_probs[0, start + i, tok_id].item()
        token_logprobs.append(lp)

    sum_logprob = float(sum(token_logprobs))
    return token_logprobs, sum_logprob


def compute_delta_logprob(
    full_logprobs: List[float],
    masked_logprobs: List[float],
) -> Tuple[float, float]:
    if len(full_logprobs) != len(masked_logprobs):
        raise ValueError("logprob length mismatch between full and masked")
    delta = sum(f - m for f, m in zip(full_logprobs, masked_logprobs))
    ratio = math.exp(delta) if delta < 80 else float("inf")
    return delta, ratio


def is_harmful_from_scores(
    turn_scores: Optional[List[float]],
    outcome_score: Optional[float],
    threshold: float,
) -> Tuple[Optional[float], Optional[bool]]:
    if turn_scores and len(turn_scores) > 0:
        final_score = float(turn_scores[-1])
        return final_score, final_score >= threshold
    if outcome_score is not None:
        final_score = float(outcome_score)
        return final_score, final_score >= threshold
    return None, None


def extract_x0_from_init_prompt(text: str) -> Optional[str]:
    if text is None:
        return None
    marker1 = "Harmful objective:"
    marker2 = "Output the first query:"
    idx1 = text.find(marker1)
    if idx1 == -1:
        return None
    idx2 = text.find(marker2, idx1 + len(marker1))
    if idx2 == -1:
        return None
    chunk = text[idx1 + len(marker1):idx2]
    return chunk.strip()


def messages_list_to_dialogue_history(messages_list: List[Dict]) -> List[Dict]:
    if not messages_list:
        return []
    if messages_list[0].get("role") == "system":
        if len(messages_list) > 1 and messages_list[1].get("role") == "user":
            merged = dict(messages_list[1])
            merged["content"] = "{}\n{}".format(messages_list[0].get("content", ""), messages_list[1].get("content", ""))
            return [merged] + messages_list[2:]
        return messages_list[1:]
    return messages_list


def compute_loo_deltas_for_dialogue_history(
    dialogue_history: List[Dict],
    model,
    tokenizer,
    mask_cfg: MaskConfig,
    score_cfg: ScoreConfig,
) -> Tuple[List[Optional[float]], Dict, str]:
    """
    Compute LOO delta logprob for each turn t < T using the same y_T.
    Returns:
        deltas: List[Optional[float]] length T-1
        meta: Dict from build_env_llm_messages
        y_T: str
    """
    if not dialogue_history:
        return [], {"num_turns": 0}, ""
    full_messages, y_T, meta = build_env_llm_messages(dialogue_history, None, mask_cfg)
    num_turns = meta.get("num_turns", 0)
    if num_turns <= 1:
        return [], meta, y_T
    full_token_logprobs, _ = score_yT_logprobs(model, tokenizer, full_messages, y_T, score_cfg)
    deltas: List[Optional[float]] = []
    for t in range(num_turns - 1):
        masked_messages, y_T_masked, _ = build_env_llm_messages(dialogue_history, t, mask_cfg)
        if y_T_masked != y_T:
            deltas.append(None)
            continue
        masked_token_logprobs, _ = score_yT_logprobs(model, tokenizer, masked_messages, y_T, score_cfg)
        delta, _ = compute_delta_logprob(full_token_logprobs, masked_token_logprobs)
        deltas.append(delta)
    return deltas, meta, y_T


def compute_loo_deltas_for_messages_list(
    messages_list: List[Dict],
    model,
    tokenizer,
    mask_cfg: MaskConfig,
    score_cfg: ScoreConfig,
) -> Tuple[List[Optional[float]], Dict, str]:
    dialogue_history = messages_list_to_dialogue_history(messages_list)
    return compute_loo_deltas_for_dialogue_history(dialogue_history, model, tokenizer, mask_cfg, score_cfg)
