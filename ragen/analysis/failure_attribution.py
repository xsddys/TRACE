# -*- coding: utf-8 -*-
"""Failure-side attribution utilities for GRPO_FAILURE."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ragen.analysis.credit_assignment import messages_list_to_dialogue_history
from ragen.analysis.qwen_guard_helper import QwenGuardClient
from ragen.analysis.success_analysis_lib import (
    MiniLMScorer,
    extract_turns,
    extract_x0_from_init_prompt,
    resolve_x_t,
)


DEFAULT_PHASE_PROFILE = "qwen"
MIXED_PHASE_PROFILE = "mixed"
LEGACY_FIXED_PHASE_PROFILE = "fixed"

DEFAULT_PHASE_PRIORS_BY_PROFILE = {
    "qwen": {
        1: {"safe": 0.78, "controversial": 0.08, "unsafe": 0.14},
        2: {"safe": 0.51, "controversial": 0.14, "unsafe": 0.34},
        3: {"safe": 0.25, "controversial": 0.19, "unsafe": 0.56},
        4: {"safe": 0.15, "controversial": 0.24, "unsafe": 0.61},
        5: {"safe": 0.06, "controversial": 0.14, "unsafe": 0.80},
    },
    "oss": {
        1: {"safe": 0.87, "controversial": 0.11, "unsafe": 0.02},
        2: {"safe": 0.65, "controversial": 0.28, "unsafe": 0.07},
        3: {"safe": 0.56, "controversial": 0.25, "unsafe": 0.18},
        4: {"safe": 0.47, "controversial": 0.26, "unsafe": 0.26},
        5: {"safe": 0.20, "controversial": 0.34, "unsafe": 0.46},
    },
    "llama": {
        1: {"safe": 0.56, "controversial": 0.13, "unsafe": 0.31},
        2: {"safe": 0.40, "controversial": 0.21, "unsafe": 0.38},
        3: {"safe": 0.41, "controversial": 0.18, "unsafe": 0.41},
        4: {"safe": 0.23, "controversial": 0.20, "unsafe": 0.55},
        5: {"safe": 0.13, "controversial": 0.30, "unsafe": 0.47},
    },
}

DEFAULT_PHASE_QUANTILES_BY_PROFILE = {
    "qwen": {
        1: 0.4,
        2: 0.45,
        3: 0.5,
        4: 0.5,
        5: 0.5,
    },
    "oss": {
        1: 0.4,
        2: 0.32,
        3: 0.32,
        4: 0.29,
        5: 0.32,
    },
    "llama": {
        1: 0.52,
        2: 0.52,
        3: 0.46,
        4: 0.50,
        5: 0.48,
    },
}


def _normalize_phase_profile(profile: Optional[str]) -> str:
    normalized = str(profile or DEFAULT_PHASE_PROFILE).strip().lower()
    if normalized in {MIXED_PHASE_PROFILE, LEGACY_FIXED_PHASE_PROFILE}:
        return MIXED_PHASE_PROFILE
    if normalized not in DEFAULT_PHASE_PRIORS_BY_PROFILE:
        return DEFAULT_PHASE_PROFILE
    return normalized


def _copy_default_phase_priors(profile: str) -> Dict[int, Dict[str, float]]:
    normalized = _normalize_phase_profile(profile)
    if normalized == MIXED_PHASE_PROFILE:
        normalized = DEFAULT_PHASE_PROFILE
    return {
        int(phase): {str(label): float(prob) for label, prob in priors.items()}
        for phase, priors in DEFAULT_PHASE_PRIORS_BY_PROFILE[normalized].items()
    }


def _copy_default_phase_quantiles(profile: str) -> Dict[int, float]:
    normalized = _normalize_phase_profile(profile)
    if normalized == MIXED_PHASE_PROFILE:
        normalized = DEFAULT_PHASE_PROFILE
    return {
        int(phase): float(value)
        for phase, value in DEFAULT_PHASE_QUANTILES_BY_PROFILE[normalized].items()
    }


@dataclass
class FailureAttrConfig:
    harmful_threshold: float = 0.9
    lambda_attr: float = 0.3
    beta: float = 1.0
    eps: float = 1e-6
    attacker_format: str = "legacy_think_answer"
    max_phase: int = 5
    phase_profile: str = DEFAULT_PHASE_PROFILE
    phase_priors: Dict[int, Dict[str, float]] = field(default_factory=lambda: _copy_default_phase_priors(DEFAULT_PHASE_PROFILE))
    phase_quantiles: Dict[int, float] = field(default_factory=lambda: _copy_default_phase_quantiles(DEFAULT_PHASE_PROFILE))


def _get(cfg_obj, key, default=None):
    if hasattr(cfg_obj, "get"):
        return cfg_obj.get(key, default)
    return getattr(cfg_obj, key, default)


def _normalize_phase_priors(priors, profile: str) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    if not priors:
        return _copy_default_phase_priors(profile)
    for key, value in priors.items():
        try:
            phase = int(key)
        except Exception:
            continue
        if not isinstance(value, dict):
            continue
        out[phase] = {
            str(label).strip().lower(): float(prob)
            for label, prob in value.items()
        }
    return out or _copy_default_phase_priors(profile)


def _normalize_phase_quantiles(quantiles, profile: str) -> Dict[int, float]:
    out: Dict[int, float] = {}
    if not quantiles:
        return _copy_default_phase_quantiles(profile)
    for key, value in quantiles.items():
        try:
            phase = int(key)
            out[phase] = float(value)
        except Exception:
            continue
    return out or _copy_default_phase_quantiles(profile)


def build_failure_attr_config(cfg_obj, default_harmful_threshold: float = 0.9, default_lambda: float = 0.3, attacker_format: str = "legacy_think_answer") -> FailureAttrConfig:
    phase_profile = _normalize_phase_profile(_get(cfg_obj, "phase_profile", DEFAULT_PHASE_PROFILE))
    return FailureAttrConfig(
        harmful_threshold=float(_get(cfg_obj, "harmful_threshold", default_harmful_threshold)),
        lambda_attr=float(_get(cfg_obj, "lambda", default_lambda)),
        beta=float(_get(cfg_obj, "beta", 1.0)),
        eps=float(_get(cfg_obj, "eps", 1e-6)),
        attacker_format=str(_get(cfg_obj, "attacker_format", attacker_format)),
        max_phase=int(_get(cfg_obj, "max_phase", 5)),
        phase_profile=phase_profile,
        phase_priors=_normalize_phase_priors(_get(cfg_obj, "phase_priors", None), phase_profile),
        phase_quantiles=_normalize_phase_quantiles(_get(cfg_obj, "phase_quantiles", None), phase_profile),
    )


def _resolve_row_phase_profile(config: FailureAttrConfig, target_model_profile: Optional[str]) -> str:
    if config.phase_profile != MIXED_PHASE_PROFILE:
        return _normalize_phase_profile(config.phase_profile)
    resolved = _normalize_phase_profile(target_model_profile)
    if resolved == MIXED_PHASE_PROFILE:
        return DEFAULT_PHASE_PROFILE
    return resolved


def _resolve_row_phase_tables(config: FailureAttrConfig, target_model_profile: Optional[str]) -> Tuple[Dict[int, Dict[str, float]], Dict[int, float], str]:
    row_profile = _resolve_row_phase_profile(config, target_model_profile)
    if config.phase_profile == MIXED_PHASE_PROFILE:
        return _copy_default_phase_priors(row_profile), _copy_default_phase_quantiles(row_profile), row_profile
    return config.phase_priors, config.phase_quantiles, row_profile


def _resolve_phase(turn_index: int, max_phase: int) -> int:
    return min(turn_index + 1, max_phase)


def compute_embedding_blame(e_score: float, lower_quantile: float, eps: float) -> float:
    if lower_quantile <= 0:
        return 0.0
    return max(0.0, float((lower_quantile - e_score) / (lower_quantile + eps)))


def compute_guard_blame(label: Optional[str], priors: Dict[str, float]) -> Optional[float]:
    if label is None:
        return None
    normalized = str(label).strip().lower()
    if normalized not in priors:
        return None
    uncertainty = 1.0 - sum(float(prob) * float(prob) for prob in priors.values())
    return max(0.0, 1.0 - float(priors[normalized]) - float(uncertainty))


def build_failure_records_for_messages_list(
    messages_list: List[Dict],
    sample_idx: int,
    final_score: Optional[float],
    turn_scores: Sequence[Optional[float]],
    attacker_format: str,
    max_phase: int,
    target_model_profile: Optional[str] = None,
) -> List[Dict]:
    dialogue_history = messages_list_to_dialogue_history(messages_list)
    if not dialogue_history:
        return []
    try:
        init_prompt, turns = extract_turns(dialogue_history)
    except Exception:
        return []
    if not turns:
        return []

    init_prompt_text = init_prompt.get("content", "") if isinstance(init_prompt, dict) else ""
    x0 = extract_x0_from_init_prompt(init_prompt_text)
    num_turns = len(turns)
    rows: List[Dict] = []
    for turn_index, (x_msg, _y_msg) in enumerate(turns):
        x_raw = x_msg.get("content", "") if isinstance(x_msg, dict) else ""
        turn_score = None
        if turn_index < len(turn_scores):
            val = turn_scores[turn_index]
            if val is not None:
                turn_score = float(val)
        rows.append(
            {
                "mode": "failure",
                "sample_idx": sample_idx,
                "turn_index": turn_index,
                "num_turns": num_turns,
                "phase": _resolve_phase(turn_index, max_phase=max_phase),
                "phase_profile": None,
                "target_model_profile": target_model_profile,
                "E_t": None,
                "H_t": None,
                "B_t_E": None,
                "B_t_H": None,
                "B_t": None,
                "trajectory_outcome_advantage": None,
                "turn_failure_outcome_advantage": None,
                "x_0": x0,
                "x_t": resolve_x_t(x_raw, attacker_format),
                "final_score": final_score,
                "turn_score": turn_score,
            }
        )
    return rows


def batch_compute_turn_embeddings(rows: List[Dict], embedder: MiniLMScorer) -> None:
    if not rows:
        return
    embedder.annotate_rows(rows)


def batch_classify_turn_harmfulness(rows: List[Dict], qwen_guard_client: QwenGuardClient) -> None:
    if not rows:
        return
    prompts = [(row.get("x_t") or "") for row in rows]
    results = qwen_guard_client.classify_prompt_only_batch(prompts)
    for row, result in zip(rows, results):
        row["H_t"] = result.get("label", None)


def compute_failure_blame_for_batch(
    messages_list_batch: List[List[Dict]],
    judger_scores,
    eligible_indices: Optional[Set[int]],
    minilm_scorer: MiniLMScorer,
    qwen_guard_client: QwenGuardClient,
    config: FailureAttrConfig,
    target_model_profiles: Optional[List[Optional[str]]] = None,
) -> Tuple[Dict[int, List[Optional[float]]], List[Dict]]:
    rows: List[Dict] = []
    for sample_idx, messages_list in enumerate(messages_list_batch):
        if eligible_indices is not None and sample_idx not in eligible_indices:
            continue
        scores_i = []
        if judger_scores is not None and len(judger_scores) > sample_idx and judger_scores[sample_idx] is not None:
            try:
                scores_i = list(judger_scores[sample_idx])
            except Exception:
                scores_i = []
        final_score = float(scores_i[-1]) if scores_i else None
        sample_target_model_profile = None
        if target_model_profiles is not None and sample_idx < len(target_model_profiles):
            sample_target_model_profile = target_model_profiles[sample_idx]
        rows.extend(
            build_failure_records_for_messages_list(
                messages_list=messages_list,
                sample_idx=sample_idx,
                final_score=final_score,
                turn_scores=scores_i,
                attacker_format=config.attacker_format,
                max_phase=config.max_phase,
                target_model_profile=sample_target_model_profile,
            )
        )

    if not rows:
        return {}, []

    batch_compute_turn_embeddings(rows, minilm_scorer)
    batch_classify_turn_harmfulness(rows, qwen_guard_client)

    for row in rows:
        phase = int(row.get("phase", config.max_phase))
        priors_table, quantiles_table, row_profile = _resolve_row_phase_tables(
            config,
            row.get("target_model_profile", None),
        )
        last_quantile = quantiles_table[max(quantiles_table)]
        last_priors = priors_table[max(priors_table)]
        lower_quantile = float(quantiles_table.get(phase, last_quantile))
        priors = priors_table.get(phase, last_priors)
        e_score = float(row.get("E_t", 0.0) or 0.0)
        e_blame = compute_embedding_blame(e_score, lower_quantile, config.eps)
        h_blame = compute_guard_blame(row.get("H_t", None), priors)
        total_blame = None if h_blame is None else float(h_blame + config.beta * e_blame)
        row["phase_profile"] = row_profile
        row["B_t_E"] = float(e_blame)
        row["B_t_H"] = h_blame
        row["B_t"] = total_blame

    rows.sort(key=lambda row: (int(row.get("sample_idx", -1)), int(row.get("turn_index", -1))))
    blame_map: Dict[int, List[Optional[float]]] = {}
    for row in rows:
        sample_idx = int(row["sample_idx"])
        num_turns = int(row["num_turns"])
        blame_map.setdefault(sample_idx, [None] * num_turns)
        turn_index = int(row["turn_index"])
        if 0 <= turn_index < len(blame_map[sample_idx]):
            blame_map[sample_idx][turn_index] = row.get("B_t", None)
    return blame_map, rows
