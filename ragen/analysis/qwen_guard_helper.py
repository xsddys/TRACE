# -*- coding: utf-8 -*-
"""Prompt-only QwenGuard helpers."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import re
import time
from typing import Dict, List, Optional, Sequence

from openai import OpenAI


_SAFETY_RE = re.compile(r"safety\s*:\s*(safe|unsafe|controversial)", flags=re.IGNORECASE)
_REFUSAL_RE = re.compile(r"refusal\s*:\s*(yes|no)", flags=re.IGNORECASE)


def parse_qwenguard_output(text: str) -> Optional[str]:
    if text is None:
        return None
    match = _SAFETY_RE.search(text)
    if match:
        return match.group(1).strip().lower()

    stripped = text.strip().lower()
    if stripped in {"safe", "unsafe", "controversial"}:
        return stripped

    for label in ("unsafe", "controversial", "safe"):
        if label in stripped:
            return label
    return None


def parse_qwenguard_refusal(text: str) -> Optional[bool]:
    if text is None:
        return None
    match = _REFUSAL_RE.search(text)
    if match:
        return match.group(1).strip().lower() == "yes"
    stripped = text.strip().lower()
    if "refusal: yes" in stripped:
        return True
    if "refusal: no" in stripped:
        return False
    return None


@dataclass
class QwenGuardConfig:
    base_url: str
    api_model: str = ""
    api_key: str = ""
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 64
    max_workers: int = 16
    max_retries: int = 5
    timeout_sec: int = 60


class QwenGuardClient:
    def __init__(self, config: QwenGuardConfig):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def classify_prompt_only(self, prompt: str) -> Dict:
        return self._moderate_messages([{"role": "user", "content": prompt}])

    def classify_response(self, prompt: str, response: str) -> Dict:
        return self._moderate_messages(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
        )

    def _moderate_messages(self, messages: List[Dict]) -> Dict:
        last_error = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.config.api_model,
                    messages=messages,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout_sec,
                )
                text = completion.choices[0].message.content or ""
                label = parse_qwenguard_output(text)
                refusal = parse_qwenguard_refusal(text)
                return {
                    "label": label,
                    "refusal": refusal,
                    "raw_output": text,
                    "error": None,
                }
            except Exception as exc:
                last_error = exc
                if attempt < self.config.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 8))

        return {
            "label": None,
            "refusal": None,
            "raw_output": None,
            "error": str(last_error) if last_error is not None else "unknown_error",
        }

    def classify_prompt_only_batch(self, prompts: Sequence[str]) -> List[Dict]:
        prompts = list(prompts)
        if not prompts:
            return []

        results: List[Optional[Dict]] = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=max(1, self.config.max_workers)) as executor:
            future_map = {
                executor.submit(self.classify_prompt_only, prompt): idx
                for idx, prompt in enumerate(prompts)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    results[idx] = {
                        "label": None,
                        "refusal": None,
                        "raw_output": None,
                        "error": str(exc),
                    }

        return [
            result if result is not None else {"label": None, "refusal": None, "raw_output": None, "error": "missing_result"}
            for result in results
        ]

    def classify_response_batch(self, pairs: Sequence[tuple[str, str]]) -> List[Dict]:
        pairs = list(pairs)
        if not pairs:
            return []

        results: List[Optional[Dict]] = [None] * len(pairs)
        with ThreadPoolExecutor(max_workers=max(1, self.config.max_workers)) as executor:
            future_map = {
                executor.submit(self.classify_response, prompt, response): idx
                for idx, (prompt, response) in enumerate(pairs)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    results[idx] = {
                        "label": None,
                        "refusal": None,
                        "raw_output": None,
                        "error": str(exc),
                    }

        return [
            result if result is not None else {"label": None, "refusal": None, "raw_output": None, "error": "missing_result"}
            for result in results
        ]
