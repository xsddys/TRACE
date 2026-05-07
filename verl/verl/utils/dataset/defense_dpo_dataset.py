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

from typing import Any, List, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask

try:
    from omegaconf import ListConfig
except ImportError:  # pragma: no cover - fallback for lightweight smoke tests
    class ListConfig(list):
        pass


def _series_to_item(obj: Any) -> Any:
    import numpy
    import pandas

    while isinstance(obj, (pandas.core.series.Series, numpy.ndarray)) and len(obj) == 1:
        obj = obj[0]
    return obj


class DefenseDPODataset(Dataset):
    """DPO dataset for defense data.

    Expected preferred schema:
    - prompt: full prefix as message list, ending at current user turn
    - chosen: current assistant preferred answer as message list
    - rejected: current assistant dispreferred answer as message list

    For convenience, prompt_text/chosen_text/rejected_text are also supported.
    """

    def __init__(self, parquet_files: Union[str, List[str]], tokenizer, config):
        if isinstance(parquet_files, ListConfig):
            parquet_files = list(parquet_files)
        elif not isinstance(parquet_files, list):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.prompt_key = config.get("prompt_key", "prompt")
        self.chosen_key = config.get("chosen_key", "chosen")
        self.rejected_key = config.get("rejected_key", "rejected")
        self.prompt_text_key = config.get("prompt_text_key", "prompt_text")
        self.chosen_text_key = config.get("chosen_text_key", "chosen_text")
        self.rejected_text_key = config.get("rejected_text_key", "rejected_text")

        self.max_prompt_length = config.get("max_prompt_length", config.get("max_length", 2048))
        self.max_response_length = config.get("max_response_length", config.get("max_length", 1024))
        self.max_length = config.get("max_length", self.max_prompt_length + self.max_response_length)
        self.prompt_truncation = config.get("prompt_truncation", "left")
        self.response_truncation = config.get("response_truncation", "right")
        self.truncation = config.get("truncation", "error")
        self.append_eos = config.get("append_eos", True)

        assert self.prompt_truncation in ["left", "right"]
        assert self.response_truncation in ["left", "right"]
        assert self.truncation in ["error", "truncate"]

        self._download()
        self._read_files()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True)

    def _read_files(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            if parquet_file.endswith(".parquet"):
                dataframes.append(pd.read_parquet(parquet_file))
            elif parquet_file.endswith(".jsonl"):
                dataframes.append(pd.read_json(parquet_file, lines=True))
            elif parquet_file.endswith(".json"):
                dataframes.append(pd.read_json(parquet_file))
            else:
                raise ValueError(f"Unsupported dataset file: {parquet_file}")
        self.dataframe = pd.concat(dataframes)

    def __len__(self):
        return len(self.dataframe)

    def _truncate_tensor_pair(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, max_len: int, mode: str):
        if input_ids.size(0) <= max_len:
            return input_ids, attention_mask
        if mode == "left":
            return input_ids[-max_len:], attention_mask[-max_len:]
        return input_ids[:max_len], attention_mask[:max_len]

    def _tokenize_prompt(self, prompt_text: str):
        output = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        input_ids = output["input_ids"][0]
        attention_mask = output["attention_mask"][0]
        return self._truncate_tensor_pair(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_len=self.max_prompt_length,
            mode=self.prompt_truncation,
        )

    def _normalize_messages(self, prompt: Any) -> list[dict]:
        prompt = _series_to_item(prompt)
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        if not isinstance(prompt, list):
            raise TypeError(f"Unsupported prompt type: {type(prompt)}")

        messages: list[dict] = []
        for turn in prompt:
            if isinstance(turn, dict) and "role" in turn and "content" in turn:
                messages.append({"role": str(turn["role"]), "content": str(turn["content"])})
                continue
            raise TypeError(f"Unsupported conversational item type: {type(turn)}")
        return messages

    def _extract_response_text(self, value: Any) -> str:
        value = _series_to_item(value)
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            contents = []
            for item in value:
                if isinstance(item, dict) and item.get("role") == "assistant" and item.get("content") is not None:
                    contents.append(str(item["content"]))
                else:
                    raise TypeError(f"Unsupported chosen/rejected item: {item}")
            if not contents:
                raise ValueError("chosen/rejected message list does not contain assistant content.")
            return "\n".join(contents)
        raise TypeError(f"Unsupported chosen/rejected type: {type(value)}")

    def _resolve_prompt_text(self, row: dict) -> str:
        if self.prompt_text_key in row and row[self.prompt_text_key] is not None:
            return str(_series_to_item(row[self.prompt_text_key]))
        prompt_value = row[self.prompt_key]
        messages = self._normalize_messages(prompt_value)
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    def _resolve_response_text(self, row: dict, key: str, text_key: str) -> str:
        if text_key in row and row[text_key] is not None:
            return str(_series_to_item(row[text_key]))
        return self._extract_response_text(row[key])

    def _tokenize_response(self, text: str):
        output = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = output["input_ids"][0]
        attention_mask = output["attention_mask"][0]

        if self.append_eos and self.tokenizer.eos_token_id is not None:
            eos = torch.tensor([self.tokenizer.eos_token_id], dtype=input_ids.dtype)
            input_ids = torch.cat((input_ids, eos), dim=0)
            attention_mask = torch.cat((attention_mask, torch.ones_like(eos)), dim=0)

        input_ids, attention_mask = self._truncate_tensor_pair(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_len=self.max_response_length,
            mode=self.response_truncation,
        )
        if input_ids.numel() == 0:
            raise ValueError("Response becomes empty after truncation. Increase max_response_length or max_length.")
        return input_ids, attention_mask

    def _build_preference_tensors(self, prompt_text: str, response_text: str) -> dict[str, torch.Tensor]:
        prompt_ids, prompt_attention_mask = self._tokenize_prompt(str(_series_to_item(prompt_text)))
        response_ids, response_attention_mask = self._tokenize_response(str(_series_to_item(response_text)))

        total_len = prompt_ids.size(0) + response_ids.size(0)
        if total_len > self.max_length:
            overflow = total_len - self.max_length
            if self.truncation == "error":
                raise ValueError(
                    f"Preference sample is too long: prompt={prompt_ids.size(0)}, response={response_ids.size(0)}, "
                    f"max_length={self.max_length}."
                )

            if overflow > 0 and prompt_ids.size(0) > 1:
                trim_prompt = min(overflow, prompt_ids.size(0) - 1)
                if self.prompt_truncation == "left":
                    prompt_ids = prompt_ids[trim_prompt:]
                    prompt_attention_mask = prompt_attention_mask[trim_prompt:]
                else:
                    prompt_ids = prompt_ids[:-trim_prompt]
                    prompt_attention_mask = prompt_attention_mask[:-trim_prompt]
                overflow -= trim_prompt

            if overflow > 0:
                if self.response_truncation == "left":
                    response_ids = response_ids[overflow:]
                    response_attention_mask = response_attention_mask[overflow:]
                else:
                    response_ids = response_ids[:-overflow]
                    response_attention_mask = response_attention_mask[:-overflow]

        if response_ids.numel() == 0:
            raise ValueError("Response becomes empty after overflow truncation. Increase max_length or reduce prompt length.")

        total_len = prompt_ids.size(0) + response_ids.size(0)
        if total_len > self.max_length:
            raise ValueError(
                f"Preference sample still exceeds max_length after truncation: prompt={prompt_ids.size(0)}, "
                f"response={response_ids.size(0)}, max_length={self.max_length}."
            )

        prompt_length = prompt_ids.size(0)
        response_length = response_ids.size(0)
        input_ids = torch.cat((prompt_ids, response_ids), dim=0)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=0)

        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        if input_ids.size(0) < self.max_length:
            pad_len = self.max_length - input_ids.size(0)
            input_ids = torch.cat((input_ids, torch.full((pad_len,), pad_token_id, dtype=input_ids.dtype)), dim=0)
            attention_mask = torch.cat((attention_mask, torch.zeros((pad_len,), dtype=attention_mask.dtype)), dim=0)

        position_ids = compute_position_id_with_mask(attention_mask)

        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            loss_mask[: prompt_length - 1] = 0
        if prompt_length + response_length > 0:
            loss_mask[prompt_length + response_length - 1] = 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }

    def __getitem__(self, item):
        row = self.dataframe.iloc[item].to_dict()
        prompt_text = self._resolve_prompt_text(row)
        chosen_text = self._resolve_response_text(row, self.chosen_key, self.chosen_text_key)
        rejected_text = self._resolve_response_text(row, self.rejected_key, self.rejected_text_key)

        chosen_tensors = self._build_preference_tensors(prompt_text=prompt_text, response_text=chosen_text)
        rejected_tensors = self._build_preference_tensors(prompt_text=prompt_text, response_text=rejected_text)

        return {
            "chosen_input_ids": chosen_tensors["input_ids"],
            "chosen_attention_mask": chosen_tensors["attention_mask"],
            "chosen_position_ids": chosen_tensors["position_ids"],
            "chosen_loss_mask": chosen_tensors["loss_mask"],
            "rejected_input_ids": rejected_tensors["input_ids"],
            "rejected_attention_mask": rejected_tensors["attention_mask"],
            "rejected_position_ids": rejected_tensors["position_ids"],
            "rejected_loss_mask": rejected_tensors["loss_mask"],
        }
