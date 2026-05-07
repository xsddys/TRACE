# -*- coding: utf-8 -*-
"""Helpers for calling LlamaGuard 4 through an OpenAI-compatible API."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Sequence, Tuple

from openai import OpenAI


def build_llamaguard4_messages(query: str, response: str) -> List[Dict]:
    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": str(query or "")}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": str(response or "")}],
        },
    ]


def parse_llamaguard4_output(text: str) -> Dict:
    raw_text = (text or "").strip()
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    label = lines[0].lower() if lines else None
    categories: List[str] = []
    if len(lines) >= 2:
        categories = [item.strip() for item in lines[1].split(",") if item.strip()]
    return {
        "label": label,
        "categories": categories,
        "raw_output": raw_text,
        "is_unsafe": label == "unsafe" if label is not None else None,
    }


@dataclass
class LlamaGuard4Config:
    base_url: str
    api_model: str = ""
    api_key: str = ""
    temperature: float = 0.0
    max_tokens: int = 32
    max_workers: int = 16
    max_retries: int = 5
    timeout_sec: int = 60


class LlamaGuard4Client:
    def __init__(self, config: LlamaGuard4Config):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def classify_response(self, query: str, response: str) -> Dict:
        if len(str(response or "").strip()) == 0:
            return {
                "label": None,
                "categories": [],
                "raw_output": "",
                "is_unsafe": None,
                "error": "empty_response",
            }

        messages = build_llamaguard4_messages(query=query, response=response)
        last_error = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.config.api_model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout_sec,
                )
                output = completion.choices[0].message.content or ""
                if isinstance(output, list):
                    output = "".join(
                        item.get("text", "") for item in output if isinstance(item, dict)
                    )
                result = parse_llamaguard4_output(str(output))
                result["error"] = None
                return result
            except Exception as exc:
                last_error = exc
                if attempt < self.config.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 8))

        return {
            "label": None,
            "categories": [],
            "raw_output": "",
            "is_unsafe": None,
            "error": str(last_error) if last_error is not None else "unknown_error",
        }

    def classify_response_batch(self, pairs: Sequence[Tuple[str, str]]) -> List[Dict]:
        pairs = list(pairs)
        if not pairs:
            return []

        results: List[Optional[Dict]] = [None] * len(pairs)
        with ThreadPoolExecutor(max_workers=max(1, self.config.max_workers)) as executor:
            future_map = {
                executor.submit(self.classify_response, query, response): idx
                for idx, (query, response) in enumerate(pairs)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    results[idx] = {
                        "label": None,
                        "categories": [],
                        "raw_output": "",
                        "is_unsafe": None,
                        "error": str(exc),
                    }

        return [
            result
            if result is not None
            else {
                "label": None,
                "categories": [],
                "raw_output": "",
                "is_unsafe": None,
                "error": "missing_result",
            }
            for result in results
        ]
