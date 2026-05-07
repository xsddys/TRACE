#!/usr/bin/env python
import argparse
import json
import os
import re
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Dict, List, Tuple

from omegaconf import OmegaConf

from ragen.api_eval_judgers import evaluate_api_judge, normalize_judger_backend
from ragen.env.jailbreak.env import JailbreakEnv
from ragen.env.jailbreak.config import JailbreakEnvConfig
from ragen.llm_agent.es_manager import OpenAIVLLMClient
from transformers import AutoTokenizer


_WORKER_CTX: Dict = {}
ENV_LLM_API_BUDGET_MAX_TOKENS = 3072


def _load_targets(data_paths: List[str]) -> Tuple[List[str], List[str]]:
    import pandas as pd
    all_targets: List[str] = []
    all_sources: List[str] = []
    for data_path in data_paths:
        try:
            if data_path.endswith(".parquet"):
                df = pd.read_parquet(data_path)
                targets = df["prompt"].dropna().tolist()
                all_targets.extend(targets)
                source = "advbench" if "advbench" in data_path.lower() else "parquet"
                all_sources.extend([source] * len(targets))
            elif data_path.endswith(".csv"):
                df = pd.read_csv(data_path)
                cols = set(df.columns)
                if {"forbidden_prompt", "source"} <= cols:
                    targets = df[df["source"] != "AdvBench"]["forbidden_prompt"].dropna().tolist()
                    all_targets.extend(targets)
                    all_sources.extend(["strongreject"] * len(targets))
                elif {"Goal", "Source"} <= cols:
                    targets = df[df["Source"] == "Original"]["Goal"].dropna().tolist()
                    all_targets.extend(targets)
                    all_sources.extend(["jailbench"] * len(targets))
                elif {"Behavior", "FunctionalCategory"} <= cols:
                    targets = df[df["FunctionalCategory"] == "standard"]["Behavior"].dropna().tolist()
                    all_targets.extend(targets)
                    all_sources.extend(["harmbench"] * len(targets))
                elif {"instruct"} <= cols:
                    targets = df["instruct"].dropna().tolist()
                    all_targets.extend(targets)
                    all_sources.extend(["advbench"] * len(targets))
            elif data_path.endswith(".jsonl"):
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
                    all_targets.extend(targets)
                    lowered_path = data_path.lower()
                    if "wildjailbreak-train" in lowered_path:
                        source = "WildJailBreak-train"
                    elif "wildjailbreak" in lowered_path:
                        source = "wildjailbreak"
                    else:
                        source = "jsonl"
                    all_sources.extend([source] * len(targets))
        except Exception as e:
            print(f"[WARN] Failed to load data from {data_path}: {e}")
            continue
    return all_targets, all_sources


def _coerce_bool(value, default: bool = False) -> bool:
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
    return bool(value)


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
            parts = [item.strip().strip("'\"") for item in inner.split(",")]
            return [item for item in parts if item]
        return [item.strip() for item in text.split("||") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _init_worker(cfg_dict: Dict):
    cfg = OmegaConf.create(cfg_dict)
    attacker_cfg = cfg.attacker_llm
    env_cfg = cfg.env_llm
    judger_cfg = cfg.judger_llm
    judger_backend = normalize_judger_backend(getattr(judger_cfg, "backend", "harmbench_cls"))

    _WORKER_CTX["cfg"] = cfg
    _WORKER_CTX["attacker"] = OpenAIVLLMClient(
        base_url=attacker_cfg.base_url,
        api_model=attacker_cfg.api_model,
        tokenizer_path=getattr(attacker_cfg, "tokenizer_path", None),
        api_key=attacker_cfg.api_key,
        enable_logit_bias=False,
        client_name="attacker_llm",
        max_retries=attacker_cfg.get("max_retries", 10),
        timeout_sec=attacker_cfg.get("timeout_sec", 100),
    )
    _WORKER_CTX["env"] = OpenAIVLLMClient(
        base_url=env_cfg.base_url,
        api_model=env_cfg.api_model,
        tokenizer_path=None,
        api_key=env_cfg.api_key,
        enable_logit_bias=False,
        client_name="env_llm",
        max_retries=env_cfg.get("max_retries", 10),
        timeout_sec=env_cfg.get("timeout_sec", 100),
        use_responses_api=_coerce_bool(getattr(env_cfg, "use_responses_api", False)),
        no_retry_error_markers=getattr(env_cfg, "no_retry_error_markers", None),
    )
    _WORKER_CTX["judger"] = OpenAIVLLMClient(
        base_url=judger_cfg.base_url,
        api_model=judger_cfg.api_model,
        tokenizer_path=judger_cfg.tokenizer_path,
        api_key=judger_cfg.api_key,
        enable_logit_bias=(judger_backend == "harmbench_cls"),
        client_name="judger_llm",
        max_retries=judger_cfg.get("max_retries", 10),
        timeout_sec=judger_cfg.get("timeout_sec", 100),
        no_retry_error_markers=getattr(judger_cfg, "no_retry_error_markers", None),
    )
    _WORKER_CTX["attacker_tokenizer"] = getattr(_WORKER_CTX["attacker"], "tokenizer", None)

    env_tokenizer = None
    env_tokenizer_path = getattr(env_cfg, "tokenizer_path", None)
    if isinstance(env_tokenizer_path, str):
        env_tokenizer_path = env_tokenizer_path.strip()
    if env_tokenizer_path and os.path.exists(env_tokenizer_path):
        try:
            env_tokenizer = AutoTokenizer.from_pretrained(str(env_tokenizer_path), trust_remote_code=True)
        except Exception as e:
            print(f"[WARN] Failed to load env_llm tokenizer from {env_tokenizer_path}: {e}")
            env_tokenizer = None
    _WORKER_CTX["env_tokenizer"] = env_tokenizer
    _WORKER_CTX["debug_limits"] = {
        "judger_exceptions": 0,
        "judger_missing_token": 0,
    }


def _truncate_text_by_tokens(text: str, tokenizer, max_tokens: int):
    if tokenizer is None or max_tokens is None:
        return text, None
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except Exception:
        return text, None
    if len(token_ids) <= max_tokens:
        return text, len(token_ids)
    truncated_ids = token_ids[:max_tokens]
    try:
        truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=False)
    except Exception:
        truncated_text = text
    return truncated_text, len(token_ids)


def _count_tokens(text: str, tokenizer) -> int:
    if tokenizer is None:
        return 0
    try:
        return len(tokenizer.encode(text or "", add_special_tokens=False))
    except Exception:
        return 0


def _build_saved_dialogue_history(env_instruction: str, dialogue_history: List[Dict]) -> List[Dict]:
    saved_history: List[Dict] = []
    for idx, item in enumerate(dialogue_history):
        role = str(item.get("role", ""))
        content = str(item.get("content", "") or "")
        if idx == 0 and env_instruction:
            content = f"{env_instruction}\n\n{content}" if content else env_instruction
        saved_history.append({"role": role, "content": content})
    return saved_history


def _get_attacker_prompt_style(cfg_env) -> str:
    return str(getattr(cfg_env, "attacker_prompt_style", "default") or "default").strip().lower()


def _build_siren_conversation_history(dialogue_history: List[Dict]) -> List[Dict]:
    history: List[Dict] = []
    for entry in dialogue_history[1:]:
        role = str(entry.get("role", "") or "")
        content = str(entry.get("content", "") or "")
        if role == "assistant":
            mapped_role = "attacker"
        elif role == "user":
            mapped_role = "victim"
        else:
            mapped_role = role
        history.append({"role": mapped_role, "content": content})
    return history


def _build_attacker_messages(env, cfg_env, env_instruction: str) -> List[Dict]:
    prompt_style = _get_attacker_prompt_style(cfg_env)
    if prompt_style == "siren":
        prompt_template = str(getattr(cfg_env, "siren_prompt_template", "") or "")
        conversation_history = _build_siren_conversation_history(list(env.dialogue_history))
        prompt = prompt_template.format(
            harmful_target=env.current_target,
            conversation_history=conversation_history,
        )
        return [{"role": "user", "content": prompt}]

    messages: List[Dict] = []
    if env_instruction:
        messages.append({"role": "system", "content": env_instruction})
    messages.extend(list(env.dialogue_history))
    return messages


def _summarize_success_turns(attempts: List[Dict], max_turns: int) -> Dict[str, float]:
    successful_attempts = [attempt for attempt in attempts if attempt.get("success")]
    success_count = len(successful_attempts)
    counts = {turn: 0 for turn in range(1, max_turns + 1)}

    for attempt in successful_attempts:
        num_turns = int(attempt.get("num_turns", 0) or 0)
        if 1 <= num_turns <= max_turns:
            counts[num_turns] += 1

    avg_success_turns = 0.0
    if success_count:
        avg_success_turns = sum(int(attempt.get("num_turns", 0) or 0) for attempt in successful_attempts) / success_count

    summary: Dict[str, float] = {
        "avg_success_turns": avg_success_turns,
    }
    for turn in range(1, max_turns + 1):
        summary[f"success_turn_{turn}"] = counts[turn]
        summary[f"success_turn_rate_{turn}"] = counts[turn] / success_count if success_count else 0.0
    return summary


def _init_metrics_accumulator(total_targets: int, group_size: int, num_workers: int, max_turns: int) -> Dict:
    return {
        "total_targets": total_targets,
        "group_size": group_size,
        "num_workers": num_workers,
        "max_turns": max_turns,
        "success_targets": 0,
        "attempt_success": 0,
        "total_turns": 0,
        "success_attempts": 0,
        "success_turn_counts": {turn: 0 for turn in range(1, max_turns + 1)},
    }


def _update_metrics_accumulator(accumulator: Dict, result: Dict) -> None:
    if result.get("success"):
        accumulator["success_targets"] += 1

    for attempt in result.get("attempts", []):
        num_turns = int(attempt.get("num_turns", 0) or 0)
        accumulator["total_turns"] += num_turns
        if attempt.get("success"):
            accumulator["attempt_success"] += 1
            accumulator["success_attempts"] += 1
            if 1 <= num_turns <= accumulator["max_turns"]:
                accumulator["success_turn_counts"][num_turns] += 1


def _finalize_metrics(accumulator: Dict, experiment_name: str, duration: float, source: str = None) -> Dict:
    total_targets = int(accumulator["total_targets"])
    group_size = int(accumulator["group_size"])
    total_attempts = total_targets * group_size
    success_targets = int(accumulator["success_targets"])
    attempt_success = int(accumulator["attempt_success"])
    success_attempts = int(accumulator["success_attempts"])
    avg_turns = accumulator["total_turns"] / total_attempts if total_attempts else 0.0
    avg_success_turns = 0.0
    if success_attempts:
        weighted_success_turns = sum(
            turn * accumulator["success_turn_counts"][turn]
            for turn in range(1, accumulator["max_turns"] + 1)
        )
        avg_success_turns = weighted_success_turns / success_attempts

    metrics = {
        "experiment_name": experiment_name,
        "total_targets": total_targets,
        "group_size": group_size,
        "num_workers": int(accumulator["num_workers"]),
        "pass_at_k": success_targets / total_targets if total_targets else 0.0,
        "attempt_success_rate": attempt_success / total_attempts if total_attempts else 0.0,
        "avg_turns": avg_turns,
        "avg_success_turns": avg_success_turns,
        "duration_sec": duration,
    }
    if source is not None:
        metrics["source"] = source

    for turn in range(1, accumulator["max_turns"] + 1):
        count = accumulator["success_turn_counts"][turn]
        metrics[f"success_turn_{turn}"] = count
        metrics[f"success_turn_rate_{turn}"] = count / success_attempts if success_attempts else 0.0
    return metrics


def _build_trajectory_entry(result: Dict, attempt: Dict) -> Dict:
    return {
        "target_index": result["target_index"],
        "source": result["source"],
        "num_turns": attempt["num_turns"],
        "outcome_score": attempt["outcome_score"],
        "turn_scores": attempt["turn_scores"],
        "success": attempt["success"],
        "attacker_token": attempt["attacker_token"],
        "target_token": attempt["target_token"],
        "harmful_objective": result["target"],
        "dialogue_history": attempt["dialogue_history"],
        "env_turn_debug": attempt.get("env_turn_debug", []),
        "error": attempt.get("error"),
        "error_stage": attempt.get("error_stage"),
        "error_code": attempt.get("error_code"),
    }


def _extract_env_block_signal(error_message: str, error_code, cfg_env_llm) -> Dict:
    text = str(error_message or "")
    lowered = text.lower()
    if error_code != 400 and "error code: 400" not in lowered:
        return {}

    markers = [item.lower() for item in _coerce_str_list(getattr(cfg_env_llm, "block_error_markers", None))]
    if markers and not any(marker in lowered for marker in markers):
        return {}
    if not markers and (
        "request blocked by gemini api" not in lowered
        and "prompt_blocked" not in lowered
        and "prohibited_content" not in lowered
    ):
        return {}

    reason_match = re.search(r"request blocked by Gemini API:\s*([A-Z_]+)", text)
    type_match = re.search(r"'type':\s*'([^']+)'", text)
    code_match = re.search(r"'code':\s*'([^']+)'", text)
    return {
        "provider": str(getattr(cfg_env_llm, "provider_name", getattr(cfg_env_llm, "api_model", "env_llm"))),
        "block_reason": reason_match.group(1) if reason_match else "",
        "block_type": type_match.group(1) if type_match else "",
        "block_code": code_match.group(1) if code_match else "",
        "raw_error": text,
    }


def _extract_env_response_block_signal(env_meta: Dict, cfg_env_llm) -> Dict:
    if not env_meta:
        return {}
    finish_reason = str(env_meta.get("finish_reason", "") or "").lower()
    refusal = env_meta.get("refusal")
    raw_preview = str(env_meta.get("raw_preview", "") or "")
    lowered_preview = raw_preview.lower()

    markers = [item.lower() for item in _coerce_str_list(getattr(cfg_env_llm, "block_error_markers", None))]
    if markers and not any(marker in lowered_preview for marker in markers):
        pass

    if finish_reason == "content_filter" or refusal:
        return {
            "provider": str(getattr(cfg_env_llm, "provider_name", getattr(cfg_env_llm, "api_model", "env_llm"))),
            "block_reason": finish_reason or "refusal",
            "block_type": "inline_filtered_response",
            "block_code": "",
            "raw_error": raw_preview,
        }
    if "prompt_blocked" in lowered_preview or "prohibited_content" in lowered_preview or "request blocked by gemini api" in lowered_preview:
        return {
            "provider": str(getattr(cfg_env_llm, "provider_name", getattr(cfg_env_llm, "api_model", "env_llm"))),
            "block_reason": "raw_preview_blocked_marker",
            "block_type": "inline_block_signal",
            "block_code": "",
            "raw_error": raw_preview,
        }
    return {}


def _eval_one_target(target_idx: int, target: str, source: str) -> Dict:
    cfg = _WORKER_CTX["cfg"]
    attacker_llm = _WORKER_CTX["attacker"]
    env_llm = _WORKER_CTX["env"]
    judger_llm = _WORKER_CTX["judger"]
    attacker_tokenizer = _WORKER_CTX.get("attacker_tokenizer")
    env_tokenizer = _WORKER_CTX.get("env_tokenizer")

    env_cfg = JailbreakEnvConfig()
    env_cfg.init_prompt = cfg.env.init_prompt
    env_cfg.siren_prompt_template = getattr(cfg.env, "siren_prompt_template", env_cfg.siren_prompt_template)
    env_cfg.max_turns = int(cfg.env.max_turns)
    env_cfg.harmful_threshold = float(cfg.env.harmful_threshold)
    env_cfg.attacker_prompt_style = _get_attacker_prompt_style(cfg.env)

    group_size = int(cfg.eval.group_size)
    env_instruction = cfg.env.env_instruction

    attacker_params = {
        "temperature": cfg.attacker_llm.temperature,
        "max_tokens": cfg.attacker_llm.max_tokens,
    }
    env_params = {
        "temperature": cfg.env_llm.temperature,
        # Keep cfg.env_llm.max_tokens as the final visible answer cap for local truncation,
        # while giving the upstream API a larger generation budget so reasoning-heavy models
        # do not exhaust the entire budget before producing visible text.
        "max_tokens": ENV_LLM_API_BUDGET_MAX_TOKENS,
    }
    attempts = []
    success_any = False

    for attempt_id in range(group_size):
        env = JailbreakEnv(env_cfg)
        env.reset(seed=None, mode="val", harmful_target=target)

        turn_scores = []
        attacker_token_counts = []
        target_token_counts = []
        env_turn_debug = []
        attempt_error = None
        attempt_error_stage = None
        attempt_error_code = None
        for turn in range(env_cfg.max_turns):
            attacker_messages = _build_attacker_messages(env, cfg.env, env_instruction)
            try:
                attacker_resp = attacker_llm.chat(attacker_messages, **attacker_params)
            except Exception as e:
                attempt_error = f"{type(e).__name__}: {e}"
                attempt_error_stage = "attacker_llm"
                attempt_error_code = getattr(e, "status_code", None)
                break
            if attacker_tokenizer is not None:
                attacker_resp, _ = _truncate_text_by_tokens(
                attacker_resp if attacker_resp is not None else "",
                    attacker_tokenizer,
                    int(cfg.attacker_llm.max_tokens),
                )
            action = (attacker_resp or "").strip()
            attacker_token_counts.append(_count_tokens(action, attacker_tokenizer))

            env_messages = env.get_env_llm_messages(action)
            try:
                env_payload = env_llm.chat(
                    env_messages,
                    return_debug_meta=True,
                    debug_label=f"target_idx={target_idx} attempt={attempt_id} turn={turn+1}",
                    **env_params,
                )
                if isinstance(env_payload, dict):
                    env_resp = env_payload.get("text", "") or ""
                    env_meta = dict(env_payload.get("meta") or {})
                else:
                    env_resp = env_payload if env_payload is not None else ""
                    env_meta = {}
            except Exception as e:
                msg = str(e)
                attempt_error_code = getattr(e, "status_code", None)
                if attempt_error_code is None and "Error code: 400" in msg:
                    attempt_error_code = 400

                block_signal = _extract_env_block_signal(msg, attempt_error_code, cfg.env_llm)
                env_turn_debug.append(
                    {
                        "turn_index": turn + 1,
                        "stage": "env_exception",
                        "error_type": type(e).__name__,
                        "error_code": attempt_error_code,
                        "error_message": msg,
                        "block_signal": block_signal,
                    }
                )
                if _coerce_bool(getattr(cfg.env_llm, "blocked_as_success", False)) and block_signal:
                    env_resp = ""
                    env.dialogue_history.append({"role": "assistant", "content": action})
                    env.dialogue_history.append({"role": "user", "content": env_resp})
                    blocked_success_score = float(getattr(cfg.env_llm, "blocked_success_score", 1.0))
                    env.step_with_llm_response(action, env_resp, blocked_success_score)
                    turn_scores.append(float(blocked_success_score))
                    target_token_counts.append(0)
                    attempt_error = None
                    attempt_error_stage = None
                    attempt_error_code = None
                    break

                attempt_error = f"{type(e).__name__}: {msg}"
                attempt_error_stage = "env_llm"
                break

            inline_block_signal = _extract_env_response_block_signal(env_meta, cfg.env_llm)
            # Hard truncate env_llm output by tokens using env_llm tokenizer
            orig_token_len = None
            if env_tokenizer is not None:
                truncated_text, orig_len = _truncate_text_by_tokens(
                    env_resp, env_tokenizer, int(cfg.env_llm.max_tokens)
                )
                orig_token_len = orig_len
                env_resp = truncated_text

            stored_token_count = _count_tokens(env_resp, env_tokenizer)
            env_turn_debug.append(
                {
                    "turn_index": turn + 1,
                    "stage": "env_response",
                    "response_len": len(env_resp),
                    "response_token_count": stored_token_count,
                    "response_preview": env_resp[:200],
                    "response_tail": env_resp[-120:] if env_resp else "",
                    "orig_token_len_before_local_trunc": orig_token_len,
                    "meta": {
                        "api_type": env_meta.get("api_type"),
                        "response_id": env_meta.get("response_id"),
                        "model": env_meta.get("model"),
                        "finish_reason": env_meta.get("finish_reason"),
                        "status": env_meta.get("status"),
                        "refusal": env_meta.get("refusal"),
                        "usage": env_meta.get("usage"),
                    },
                    "inline_block_signal": inline_block_signal,
                }
            )
            if (not env_resp) or inline_block_signal or str(env_meta.get("finish_reason", "") or "").lower() in {"length", "content_filter"}:
                debug_budget = _WORKER_CTX.get("debug_limits", {})
                env_suspicious = int(debug_budget.get("env_suspicious", 0))
                if env_suspicious < 40:
                    debug_budget["env_suspicious"] = env_suspicious + 1
                    _WORKER_CTX["debug_limits"] = debug_budget
                    print(
                        f"[DEBUG][env_llm.inspect] target_idx={target_idx} attempt={attempt_id} turn={turn+1} "
                        f"provider={getattr(cfg.env_llm, 'provider_name', '')} model={cfg.env_llm.api_model} "
                        f"finish_reason={env_meta.get('finish_reason')} status={env_meta.get('status')} "
                        f"refusal={env_meta.get('refusal')!r} response_len={len(env_resp)} "
                        f"tokens={stored_token_count} usage={env_meta.get('usage')} "
                        f"inline_block_signal={inline_block_signal} "
                        f"raw_preview={str(env_meta.get('raw_preview', ''))[:500]}",
                        flush=True,
                    )

            env.dialogue_history.append({"role": "assistant", "content": action})
            env.dialogue_history.append({"role": "user", "content": env_resp})
            target_token_counts.append(stored_token_count)

            judge_result = evaluate_api_judge(
                judger_llm=judger_llm,
                judger_cfg=cfg.judger_llm,
                harmful_target=target,
                response=env_resp,
                harmful_threshold=env_cfg.harmful_threshold,
            )
            score = float(judge_result["score"])

            _, _, done, _ = env.step_with_llm_response(action, env_resp, score)
            turn_scores.append(score)

            if done:
                break

        attempt_success = False if attempt_error else bool(env.score >= env_cfg.harmful_threshold)
        success_any = success_any or attempt_success

        attempts.append(
            {
                "attempt_id": attempt_id,
                "success": attempt_success,
                "num_turns": env.step_num,
                "outcome_score": float(env.score),
                "turn_scores": turn_scores,
                "attacker_token": attacker_token_counts,
                "target_token": target_token_counts,
                "dialogue_history": _build_saved_dialogue_history(env_instruction, list(env.dialogue_history)),
                "env_turn_debug": env_turn_debug,
                "error": attempt_error,
                "error_stage": attempt_error_stage,
                "error_code": attempt_error_code,
            }
        )

    return {
        "target_index": target_idx,
        "target": target,
        "source": source,
        "success": success_any,
        "pass_at_k": success_any,
        "attempts": attempts,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description="Pure API eval runner for jailbreak.")
    parser.add_argument("--config", required=True, help="Path to eval config yaml.")
    args, overrides = parser.parse_known_args()
    return args, overrides


def main():
    args, overrides = _parse_args()
    base_cfg = OmegaConf.load(args.config)
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(base_cfg, override_cfg)
    else:
        cfg = base_cfg

    experiment_name = cfg.eval.experiment_name
    output_dir = cfg.eval.output_dir
    os.makedirs(output_dir, exist_ok=True)

    cfg_path = os.path.join(output_dir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2, ensure_ascii=False)

    raw_data_paths = cfg.data.val_data_paths
    if isinstance(raw_data_paths, str):
        data_paths = [raw_data_paths]
    else:
        data_paths = list(raw_data_paths)
    targets, sources = _load_targets(data_paths)
    total_targets = len(targets)
    print(f"[INFO] loaded targets: {total_targets}")

    num_workers = int(cfg.eval.num_workers)
    group_size = int(cfg.eval.group_size)
    max_turns = int(cfg.env.max_turns)

    start = time.time()
    overall_metrics_acc = _init_metrics_accumulator(total_targets, group_size, num_workers, max_turns)
    source_state: Dict[str, Dict] = {}
    completed = 0

    def ensure_source_state(source_name: str) -> Dict:
        if source_name not in source_state:
            source_dir = os.path.join(output_dir, source_name)
            os.makedirs(source_dir, exist_ok=True)
            source_total_targets = sum(1 for item_source in sources if item_source == source_name)
            source_state[source_name] = {
                "dir": source_dir,
                "metrics_acc": _init_metrics_accumulator(source_total_targets, group_size, num_workers, max_turns),
                "traj_handle": open(
                    os.path.join(source_dir, "trajectories.jsonl"),
                    "a",
                    encoding="utf-8",
                    buffering=1,
                ),
            }
        return source_state[source_name]

    pending = {}
    next_idx = 0
    max_pending = max(num_workers * 4, 1)

    def submit_until_full(executor) -> None:
        nonlocal next_idx
        while next_idx < total_targets and len(pending) < max_pending:
            future = executor.submit(_eval_one_target, next_idx, targets[next_idx], sources[next_idx])
            pending[future] = next_idx
            next_idx += 1

    try:
        with ProcessPoolExecutor(
            max_workers=num_workers, initializer=_init_worker, initargs=(OmegaConf.to_container(cfg, resolve=True),)
        ) as executor:
            submit_until_full(executor)
            while pending:
                done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future, None)
                    res = future.result()
                    completed += 1

                    _update_metrics_accumulator(overall_metrics_acc, res)
                    state = ensure_source_state(res["source"])
                    _update_metrics_accumulator(state["metrics_acc"], res)
                    for attempt in res["attempts"]:
                        state["traj_handle"].write(json.dumps(_build_trajectory_entry(res, attempt), ensure_ascii=False) + "\n")
                    state["traj_handle"].flush()

                    if completed % 10 == 0:
                        print(f"[INFO] completed {completed}/{total_targets}")

                submit_until_full(executor)
    finally:
        for state in source_state.values():
            state["traj_handle"].close()

    duration = time.time() - start

    metrics = _finalize_metrics(overall_metrics_acc, experiment_name, duration)

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    for source, state in source_state.items():
        source_metrics = _finalize_metrics(state["metrics_acc"], experiment_name, duration, source=source)
        source_metrics_path = os.path.join(state["dir"], "metrics.json")
        with open(source_metrics_path, "w") as f:
            json.dump(source_metrics, f, indent=2, ensure_ascii=False)

    print(f"[INFO] done. metrics: {metrics_path}")
    print(f"[INFO] outputs by source: {', '.join(sorted(source_state.keys()))}")


if __name__ == "__main__":
    main()
