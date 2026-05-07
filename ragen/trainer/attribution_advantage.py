from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set

import math
import torch

from ragen.analysis.failure_attribution import (
    FailureAttrConfig,
    build_failure_attr_config,
    compute_failure_blame_for_batch,
)
from ragen.analysis.semantic_attribution import compute_semantic_attribution_for_batch
from ragen.trainer import core_algos


@dataclass
class SemanticAdvConfig:
    harmful_threshold: float = 0.9
    lambda_attr: float = 0.3
    tau: float = 1.0
    z_max: float = 3.0
    eps: float = 1e-6


@dataclass
class SemanticBroadcastResult:
    outcome_adv_token: torch.Tensor
    outcome_ret_token: torch.Tensor
    records_by_sample: List[List[Dict]]
    metrics: Dict[str, float]
    m_list_by_sample: List[List[float]]
    c_map: Dict[int, List[Optional[float]]]
    eligible_indices: Set[int]
    origin_scores: List[Optional[float]]


def get_cfg_value(cfg_obj, key: str, default=None):
    if hasattr(cfg_obj, "get"):
        return cfg_obj.get(key, default)
    return getattr(cfg_obj, key, default)


def build_semantic_adv_config(cfg_obj) -> SemanticAdvConfig:
    return SemanticAdvConfig(
        harmful_threshold=float(get_cfg_value(cfg_obj, "harmful_threshold", 0.9)),
        lambda_attr=float(get_cfg_value(cfg_obj, "lambda", 0.3)),
        tau=float(get_cfg_value(cfg_obj, "tau", 1.0)),
        z_max=float(get_cfg_value(cfg_obj, "z_max", 3.0)),
        eps=float(get_cfg_value(cfg_obj, "eps", 1e-6)),
    )


def sanitize_finite_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if torch.isfinite(tensor).all():
        return tensor
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


def extract_origin_scores(judger_scores, batch_size: int) -> List[Optional[float]]:
    origin_scores: List[Optional[float]] = []
    for i in range(batch_size):
        final_score = None
        if judger_scores is not None and len(judger_scores) > i:
            scores_i = judger_scores[i]
            if scores_i is not None and len(scores_i) > 0:
                final_score = float(scores_i[-1])
        origin_scores.append(final_score)
    return origin_scores


def select_semantic_success_indices(
    response_mask: torch.Tensor,
    outcome_adv_traj: torch.Tensor,
    origin_scores: List[Optional[float]],
    harmful_threshold: float,
    min_num_turns: int = 3,
) -> Set[int]:
    eligible_indices: Set[int] = set()
    batch_size = outcome_adv_traj.shape[0]
    for i in range(batch_size):
        if origin_scores[i] is None:
            continue
        num_turns = len(core_algos.split_trun_from_mask(response_mask[i]))
        if num_turns < min_num_turns:
            continue
        if origin_scores[i] >= harmful_threshold and outcome_adv_traj[i].item() > 0:
            eligible_indices.add(i)
    return eligible_indices


def select_failure_indices(
    response_mask: torch.Tensor,
    outcome_adv_traj: torch.Tensor,
    origin_scores: List[Optional[float]],
    harmful_threshold: float,
    min_num_turns: int = 1,
) -> Set[int]:
    eligible_indices: Set[int] = set()
    batch_size = outcome_adv_traj.shape[0]
    for i in range(batch_size):
        if origin_scores[i] is None:
            continue
        num_turns = len(core_algos.split_trun_from_mask(response_mask[i]))
        if num_turns < min_num_turns:
            continue
        if origin_scores[i] < harmful_threshold and outcome_adv_traj[i].item() < 0:
            eligible_indices.add(i)
    return eligible_indices


def group_records_by_sample(records: List[Dict], batch_size: int) -> List[List[Dict]]:
    records_by_sample: List[List[Dict]] = [[] for _ in range(batch_size)]
    for rec in records:
        idx = rec.get("sample_idx", None)
        if idx is None:
            continue
        if 0 <= idx < batch_size:
            records_by_sample[idx].append(rec)
    return records_by_sample


def merge_records_by_sample(*record_groups: List[List[Dict]]) -> List[List[Dict]]:
    max_len = max((len(group) for group in record_groups), default=0)
    merged: List[List[Dict]] = [[] for _ in range(max_len)]
    for group in record_groups:
        for idx, records in enumerate(group):
            if records:
                merged[idx].extend(records)
    return merged


def _default_multipliers(num_turns: int) -> List[float]:
    return [1.0] * num_turns


def _has_valid_deltas(deltas: Optional[List[Optional[float]]], num_turns: int) -> bool:
    if deltas is None or len(deltas) != num_turns - 1:
        return False
    return all(delta is not None and math.isfinite(delta) for delta in deltas)


def _has_valid_blames(blames: Optional[List[Optional[float]]], num_turns: int) -> bool:
    if blames is None or len(blames) != num_turns:
        return False
    return all(blame is not None and math.isfinite(blame) for blame in blames)


def build_semantic_turn_multipliers(
    deltas: List[float],
    num_turns: int,
    config: SemanticAdvConfig,
    dtype,
    device,
) -> List[float]:
    if num_turns <= 0:
        return []
    if num_turns == 1:
        return [1.0]

    c = torch.tensor(deltas, dtype=dtype, device=device)
    mu = c.mean()
    std = c.std(unbiased=False)
    z = (c - mu) / (std + config.eps)
    z = torch.clamp(z, min=-config.z_max, max=config.z_max)
    weights = torch.softmax(z / config.tau, dim=0)
    m_vals = (1 - config.lambda_attr) + config.lambda_attr * (num_turns - 1) * weights

    m_list = _default_multipliers(num_turns)
    for t in range(num_turns - 1):
        m_list[t] = float(m_vals[t].item())
    m_list[-1] = 1.0
    return m_list


def build_failure_turn_multipliers(
    blames: List[float],
    lambda_attr: float,
    eps: float,
    dtype,
    device,
) -> List[float]:
    if not blames:
        return []
    b = torch.tensor(blames, dtype=dtype, device=device)
    tilde = b + eps
    mean_val = tilde.mean()
    ratio = tilde / mean_val
    m_vals = (1 - lambda_attr) + lambda_attr * ratio
    return [float(val.item()) for val in m_vals]


def build_outcome_tensors_from_multipliers(
    response_mask: torch.Tensor,
    outcome_adv_traj: torch.Tensor,
    m_list_by_sample: List[List[float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    outcome_adv_token = torch.zeros_like(response_mask, dtype=outcome_adv_traj.dtype)
    outcome_ret_token = torch.zeros_like(response_mask, dtype=outcome_adv_traj.dtype)
    batch_size = outcome_adv_traj.shape[0]
    for i in range(batch_size):
        segments = core_algos.split_trun_from_mask(response_mask[i])
        if not segments:
            continue
        m_list = m_list_by_sample[i] if i < len(m_list_by_sample) else None
        if not m_list or len(m_list) != len(segments):
            m_list = _default_multipliers(len(segments))
        for turn_index, seg in enumerate(segments):
            if len(seg) == 0:
                continue
            val = outcome_adv_traj[i] * m_list[turn_index]
            outcome_adv_token[i, seg] = val
            outcome_ret_token[i, seg] = val
    return outcome_adv_token, outcome_ret_token


def attach_outcome_advantages_to_records(
    records_by_sample: List[List[Dict]],
    outcome_adv_traj: torch.Tensor,
    m_list_by_sample: List[List[float]],
) -> None:
    for idx, records in enumerate(records_by_sample):
        m_list = m_list_by_sample[idx] if idx < len(m_list_by_sample) else None
        for rec in records:
            turn_index = rec.get("turn_index", None)
            if m_list is None or turn_index is None or turn_index >= len(m_list):
                rec["outcome_advantage"] = None
                continue
            rec["outcome_advantage"] = float(outcome_adv_traj[idx].item()) * float(m_list[turn_index])


def attach_failure_outcome_advantages_to_records(
    records_by_sample: List[List[Dict]],
    outcome_adv_traj: torch.Tensor,
    m_list_by_sample: List[List[float]],
) -> None:
    for idx, records in enumerate(records_by_sample):
        m_list = m_list_by_sample[idx] if idx < len(m_list_by_sample) else None
        for rec in records:
            rec["trajectory_outcome_advantage"] = float(outcome_adv_traj[idx].item())
            turn_index = rec.get("turn_index", None)
            if m_list is None or turn_index is None or turn_index >= len(m_list):
                rec["turn_failure_outcome_advantage"] = None
                continue
            rec["turn_failure_outcome_advantage"] = float(outcome_adv_traj[idx].item()) * float(m_list[turn_index])


def attach_refusal_advantages_to_records(
    records_by_sample: List[List[Dict]],
    refusal_adv_by_sample: Sequence[Sequence[Optional[float]]],
) -> None:
    for idx, records in enumerate(records_by_sample):
        refusal_vals = refusal_adv_by_sample[idx] if idx < len(refusal_adv_by_sample) else None
        for rec in records:
            turn_index = rec.get("turn_index", None)
            if refusal_vals is None or turn_index is None or turn_index >= len(refusal_vals):
                rec["refusal_advantage"] = None
                continue
            val = refusal_vals[turn_index]
            rec["refusal_advantage"] = None if val is None else float(val)


def summarize_semantic_metrics(
    c_map: Dict[int, List[Optional[float]]],
    eligible_indices: Set[int],
    m_list_by_sample: List[List[float]],
    batch_size: int,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if batch_size <= 0:
        return metrics

    metrics["semantic_active_ratio"] = float(len(eligible_indices)) / float(batch_size)
    task_count = 0
    c_vals: List[float] = []
    top1_shares: List[float] = []

    for idx in eligible_indices:
        deltas = c_map.get(idx, None)
        if deltas:
            task_count += len(deltas)
            for delta in deltas:
                if delta is None or not math.isfinite(delta):
                    continue
                c_vals.append(float(delta))
        m_list = m_list_by_sample[idx] if idx < len(m_list_by_sample) else None
        if m_list and len(m_list) > 1:
            base = m_list[:-1]
            total = sum(base)
            if total > 0:
                top1_shares.append(max(base) / total)

    if c_vals:
        c_mean = sum(c_vals) / len(c_vals)
        c_var = sum((val - c_mean) ** 2 for val in c_vals) / len(c_vals)
        c_pos_frac = sum(1 for val in c_vals if val > 0) / len(c_vals)
    else:
        c_mean = 0.0
        c_var = 0.0
        c_pos_frac = 0.0

    metrics["semantic_task_count"] = int(task_count)
    metrics["semantic_c_mean"] = float(c_mean)
    metrics["semantic_c_var"] = float(c_var)
    metrics["semantic_c_pos_frac"] = float(c_pos_frac)
    metrics["multiplier_top1_share"] = float(sum(top1_shares) / len(top1_shares)) if top1_shares else 0.0
    return metrics


def summarize_failure_metrics(
    blame_map: Dict[int, List[Optional[float]]],
    eligible_indices: Set[int],
    m_list_by_sample: List[List[float]],
    batch_size: int,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if batch_size <= 0:
        return metrics

    metrics["failure_active_ratio"] = float(len(eligible_indices)) / float(batch_size)
    blame_vals: List[float] = []
    top1_shares: List[float] = []
    task_count = 0
    for idx in eligible_indices:
        blames = blame_map.get(idx, None)
        if blames:
            task_count += len(blames)
            for blame in blames:
                if blame is None or not math.isfinite(blame):
                    continue
                blame_vals.append(float(blame))
        m_list = m_list_by_sample[idx] if idx < len(m_list_by_sample) else None
        if m_list:
            total = sum(m_list)
            if total > 0:
                top1_shares.append(max(m_list) / total)
    metrics["failure_task_count"] = int(task_count)
    if blame_vals:
        mean_val = sum(blame_vals) / len(blame_vals)
        var_val = sum((val - mean_val) ** 2 for val in blame_vals) / len(blame_vals)
    else:
        mean_val = 0.0
        var_val = 0.0
    metrics["failure_blame_mean"] = float(mean_val)
    metrics["failure_blame_var"] = float(var_val)
    metrics["failure_multiplier_top1_share"] = float(sum(top1_shares) / len(top1_shares)) if top1_shares else 0.0
    return metrics


def compute_semantic_broadcast(
    response_mask: torch.Tensor,
    outcome_adv_traj: torch.Tensor,
    c_map: Dict[int, List[Optional[float]]],
    eligible_indices: Set[int],
    records: List[Dict],
    config: SemanticAdvConfig,
) -> SemanticBroadcastResult:
    batch_size = outcome_adv_traj.shape[0]
    records_by_sample = group_records_by_sample(records, batch_size)
    m_list_by_sample: List[List[float]] = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        segments = core_algos.split_trun_from_mask(response_mask[i])
        num_turns = len(segments)
        if num_turns == 0:
            continue
        if i in eligible_indices and _has_valid_deltas(c_map.get(i, None), num_turns):
            m_list = build_semantic_turn_multipliers(
                deltas=[float(delta) for delta in c_map[i]],
                num_turns=num_turns,
                config=config,
                dtype=outcome_adv_traj.dtype,
                device=outcome_adv_traj.device,
            )
        else:
            m_list = _default_multipliers(num_turns)
        m_list_by_sample[i] = m_list

    outcome_adv_token, outcome_ret_token = build_outcome_tensors_from_multipliers(
        response_mask=response_mask,
        outcome_adv_traj=outcome_adv_traj,
        m_list_by_sample=m_list_by_sample,
    )
    attach_outcome_advantages_to_records(records_by_sample, outcome_adv_traj, m_list_by_sample)
    metrics = summarize_semantic_metrics(c_map, eligible_indices, m_list_by_sample, batch_size)
    return SemanticBroadcastResult(
        outcome_adv_token=outcome_adv_token,
        outcome_ret_token=outcome_ret_token,
        records_by_sample=records_by_sample,
        metrics=metrics,
        m_list_by_sample=m_list_by_sample,
        c_map=c_map,
        eligible_indices=eligible_indices,
        origin_scores=[],
    )


def compute_grpo_semantic_outcome_broadcast(
    es_manager,
    messages_list,
    judger_scores,
    response_mask: torch.Tensor,
    outcome_adv_traj: torch.Tensor,
    mask_cfg,
    semantic_cfg,
    target_model_profiles=None,
) -> SemanticBroadcastResult:
    if es_manager is None or mask_cfg is None or semantic_cfg is None:
        raise ValueError("GRPO_SEMANTIC requires semantic_es_manager/semantic_mask_cfg/semantic_cfg")
    config = build_semantic_adv_config(semantic_cfg)
    batch_size = outcome_adv_traj.shape[0]

    if hasattr(messages_list, "tolist"):
        messages_list = messages_list.tolist()
    if messages_list is None:
        messages_list = []

    origin_scores = extract_origin_scores(judger_scores, batch_size)
    eligible_indices = select_semantic_success_indices(
        response_mask=response_mask,
        outcome_adv_traj=outcome_adv_traj,
        origin_scores=origin_scores,
        harmful_threshold=config.harmful_threshold,
        min_num_turns=3,
    )
    c_map, records = compute_semantic_attribution_for_batch(
        es_manager=es_manager,
        messages_list_batch=messages_list,
        origin_scores=origin_scores,
        mask_cfg=mask_cfg,
        outcome_advantages=outcome_adv_traj.detach().cpu().tolist(),
        eligible_indices=eligible_indices,
        harmful_threshold=config.harmful_threshold,
        target_model_profiles=target_model_profiles,
    )
    result = compute_semantic_broadcast(
        response_mask=response_mask,
        outcome_adv_traj=outcome_adv_traj,
        c_map=c_map,
        eligible_indices=eligible_indices,
        records=records,
        config=config,
    )
    result.origin_scores = origin_scores
    return result


def compute_grpo_failure_outcome_broadcast(
    semantic_es_manager,
    messages_list,
    judger_scores,
    response_mask: torch.Tensor,
    outcome_adv_traj: torch.Tensor,
    semantic_mask_cfg,
    semantic_cfg,
    failure_minilm_scorer,
    failure_qwen_guard_client,
    failure_cfg,
    target_model_profiles=None,
) -> SemanticBroadcastResult:
    if failure_minilm_scorer is None or failure_qwen_guard_client is None or failure_cfg is None:
        raise ValueError("GRPO_FAILURE requires failure_minilm_scorer/failure_qwen_guard_client/failure_cfg")

    semantic_result = compute_grpo_semantic_outcome_broadcast(
        es_manager=semantic_es_manager,
        messages_list=messages_list,
        judger_scores=judger_scores,
        response_mask=response_mask,
        outcome_adv_traj=outcome_adv_traj,
        mask_cfg=semantic_mask_cfg,
        semantic_cfg=semantic_cfg,
        target_model_profiles=target_model_profiles,
    )
    semantic_config = build_semantic_adv_config(semantic_cfg)
    failure_config: FailureAttrConfig = build_failure_attr_config(
        failure_cfg,
        default_harmful_threshold=semantic_config.harmful_threshold,
        default_lambda=semantic_config.lambda_attr,
        attacker_format=get_cfg_value(failure_cfg, "attacker_format", "legacy_think_answer"),
    )

    batch_size = outcome_adv_traj.shape[0]
    if hasattr(messages_list, "tolist"):
        messages_list = messages_list.tolist()
    if messages_list is None:
        messages_list = []
    if hasattr(judger_scores, "tolist"):
        judger_scores = judger_scores.tolist()

    origin_scores = semantic_result.origin_scores or extract_origin_scores(judger_scores, batch_size)
    failure_indices = select_failure_indices(
        response_mask=response_mask,
        outcome_adv_traj=outcome_adv_traj,
        origin_scores=origin_scores,
        harmful_threshold=failure_config.harmful_threshold,
        min_num_turns=1,
    )
    blame_map, failure_records = compute_failure_blame_for_batch(
        messages_list_batch=messages_list,
        judger_scores=judger_scores,
        eligible_indices=failure_indices,
        minilm_scorer=failure_minilm_scorer,
        qwen_guard_client=failure_qwen_guard_client,
        config=failure_config,
        target_model_profiles=target_model_profiles,
    )

    combined_m_list: List[List[float]] = [list(item) for item in semantic_result.m_list_by_sample]
    if len(combined_m_list) < batch_size:
        combined_m_list.extend([[] for _ in range(batch_size - len(combined_m_list))])
    for i in range(batch_size):
        segments = core_algos.split_trun_from_mask(response_mask[i])
        num_turns = len(segments)
        if num_turns == 0:
            combined_m_list[i] = []
            continue
        if i in failure_indices and _has_valid_blames(blame_map.get(i, None), num_turns):
            combined_m_list[i] = build_failure_turn_multipliers(
                blames=[float(blame) for blame in blame_map[i]],
                lambda_attr=failure_config.lambda_attr,
                eps=failure_config.eps,
                dtype=outcome_adv_traj.dtype,
                device=outcome_adv_traj.device,
            )
        elif not combined_m_list[i]:
            combined_m_list[i] = _default_multipliers(num_turns)

    outcome_adv_token, outcome_ret_token = build_outcome_tensors_from_multipliers(
        response_mask=response_mask,
        outcome_adv_traj=outcome_adv_traj,
        m_list_by_sample=combined_m_list,
    )
    failure_records_by_sample = group_records_by_sample(failure_records, batch_size)
    attach_failure_outcome_advantages_to_records(failure_records_by_sample, outcome_adv_traj, combined_m_list)
    merged_records = merge_records_by_sample(semantic_result.records_by_sample, failure_records_by_sample)

    metrics = dict(semantic_result.metrics)
    metrics.update(summarize_failure_metrics(blame_map, failure_indices, combined_m_list, batch_size))
    return SemanticBroadcastResult(
        outcome_adv_token=outcome_adv_token,
        outcome_ret_token=outcome_ret_token,
        records_by_sample=merged_records,
        metrics=metrics,
        m_list_by_sample=combined_m_list,
        c_map=semantic_result.c_map,
        eligible_indices=set(semantic_result.eligible_indices) | set(failure_indices),
        origin_scores=origin_scores,
    )
