# -*- coding: utf-8 -*-
"""Semantic attribution utilities for GRPO_SEMANTIC."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ragen.analysis.credit_assignment import (
    MaskConfig,
    build_env_llm_messages,
    extract_turns,
    extract_x0_from_init_prompt,
    messages_list_to_dialogue_history,
)


def build_semantic_tasks_for_messages_list(
    messages_list: List[Dict],
    mask_cfg: MaskConfig,
) -> Tuple[List[Dict], Optional[str], Optional[str], int]:
    """
    Build masked tasks for one trajectory.

    Returns:
        tasks: list of dict with fields {turn_index, messages, x0, y_T_old, x_t, y_t, num_turns}
        x0: parsed harmful objective
        y_T_old: original final response
        num_turns: total turns T
    """
    dialogue_history = messages_list_to_dialogue_history(messages_list)
    if not dialogue_history:
        return [], None, None, 0

    init_prompt = dialogue_history[0].get("content", "")
    x0 = extract_x0_from_init_prompt(init_prompt)

    full_messages, y_T_old, meta = build_env_llm_messages(dialogue_history, None, mask_cfg)
    num_turns = meta.get("num_turns", 0)
    if num_turns <= 1:
        return [], x0, y_T_old, num_turns

    _, turns = extract_turns(dialogue_history)
    tasks = []
    for t in range(num_turns - 1):
        masked_messages, y_T_masked, _ = build_env_llm_messages(dialogue_history, t, mask_cfg)
        if y_T_masked != y_T_old:
            continue
        x_t = turns[t][0].get("content", "") if t < len(turns) else ""
        y_t = turns[t][1].get("content", "") if t < len(turns) else ""
        tasks.append(
            {
                "turn_index": t,
                "messages": masked_messages,
                "x0": x0,
                "y_T_old": y_T_old,
                "x_t": x_t,
                "y_t": y_t,
                "num_turns": num_turns,
            }
        )
    return tasks, x0, y_T_old, num_turns


def compute_semantic_attribution_for_batch(
    es_manager,
    messages_list_batch: List[List[Dict]],
    origin_scores: List[Optional[float]],
    mask_cfg: MaskConfig,
    outcome_advantages: Optional[List[Optional[float]]] = None,
    eligible_indices: Optional[set] = None,
    harmful_threshold: float = 0.9,
    target_model_profiles: Optional[List[Optional[str]]] = None,
) -> Tuple[Dict[int, List[Optional[float]]], List[Dict]]:
    """
    Compute semantic attribution deltas for a batch.
    Returns:
        c_map: {i: [c_{i,t}]} for t < T
        records: list of json-serializable records
    """
    tasks = []
    for i, messages_list in enumerate(messages_list_batch):
        if eligible_indices is not None and i not in eligible_indices:
            continue
        task_list, x0, y_T_old, num_turns = build_semantic_tasks_for_messages_list(messages_list, mask_cfg)
        if num_turns <= 1:
            continue
        origin_score = None
        if origin_scores is not None and len(origin_scores) > i:
            origin_score = origin_scores[i]
        target_model_profile = None
        if target_model_profiles is not None and len(target_model_profiles) > i:
            target_model_profile = target_model_profiles[i]
        for task in task_list:
            task = dict(task)
            task["sample_idx"] = i
            task["origin_score"] = origin_score
            task["target_model_profile"] = target_model_profile
            tasks.append(task)

    if not tasks:
        return {}, []

    messages_batch = [t["messages"] for t in tasks]
    env_profiles = [t.get("target_model_profile", None) for t in tasks]
    y_new_list = es_manager.batch_generate_env_llm(messages_batch, env_profiles=env_profiles)
    x0_list = [t["x0"] for t in tasks]
    new_scores = es_manager.batch_score_with_judger(x0_list, y_new_list)

    c_map: Dict[int, List[Optional[float]]] = {}
    records: List[Dict] = []
    for task, y_new, new_score in zip(tasks, y_new_list, new_scores):
        i = task["sample_idx"]
        t = task["turn_index"]
        num_turns = task["num_turns"]
        origin_score = task["origin_score"]
        outcome_adv = None
        if outcome_advantages is not None and len(outcome_advantages) > i:
            outcome_adv = outcome_advantages[i]
        if origin_score is None or new_score is None:
            c_val = None
        else:
            c_val = float(origin_score) - float(new_score)
        if i not in c_map:
            c_map[i] = [None] * (num_turns - 1)
        if t < len(c_map[i]):
            c_map[i][t] = c_val
        old_is_harm = None if origin_score is None else origin_score >= harmful_threshold
        new_is_harm = None if new_score is None else new_score >= harmful_threshold
        record = {
            "mode": "semantic",
            "sample_idx": i,
            "turn_index": t,
            "num_turns": num_turns,
            "old_is_harm": old_is_harm,
            "new_is_harm": new_is_harm,
            "semantic_score": c_val,
            "origin_score": origin_score,
            "new_score": new_score,
            "outcome_advantage": outcome_adv,
            "mask_strategy": mask_cfg.strategy,
            "x_t": task.get("x_t", ""),
            "y_t": task.get("y_t", ""),
            "y_T_old": task.get("y_T_old", ""),
            "y_T_new": y_new,
            "target_model_profile": task.get("target_model_profile", None),
        }
        records.append(record)
    return c_map, records
