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

import argparse
import json
import os

import pandas as pd


def load_dataframe(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    if path.endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        return pd.DataFrame(data)
    raise ValueError(f"Unsupported input format: {path}")


def save_dataframe(df: pd.DataFrame, path: str):
    if path.endswith(".parquet"):
        df.to_parquet(path)
        return
    if path.endswith(".jsonl"):
        df.to_json(path, orient="records", lines=True, force_ascii=False)
        return
    if path.endswith(".json"):
        df.to_json(path, orient="records", force_ascii=False)
        return
    raise ValueError(f"Unsupported output format: {path}")


def normalize_messages(prompt):
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]

    if not isinstance(prompt, list):
        raise TypeError(f"Unsupported prompt type: {type(prompt)}")

    messages = []
    for turn in prompt:
        if isinstance(turn, dict) and "role" in turn and "content" in turn:
            messages.append({"role": turn["role"], "content": turn["content"]})
            continue

        if isinstance(turn, dict):
            user_text = (
                turn.get("x")
                or turn.get("prompt")
                or turn.get("attack_prompt")
                or turn.get("user")
                or turn.get("question")
            )
            assistant_text = (
                turn.get("y")
                or turn.get("response")
                or turn.get("target_response")
                or turn.get("assistant")
                or turn.get("answer")
            )
            if user_text is not None:
                messages.append({"role": "user", "content": str(user_text)})
            if assistant_text is not None:
                messages.append({"role": "assistant", "content": str(assistant_text)})
            continue

        if isinstance(turn, (list, tuple)) and len(turn) == 2:
            messages.append({"role": "user", "content": str(turn[0])})
            messages.append({"role": "assistant", "content": str(turn[1])})
            continue

        raise TypeError(f"Unsupported turn type: {type(turn)}")

    return messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--messages-key", default="messages")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--current-prompt-key", default=None)
    parser.add_argument("--chosen-key", default="chosen")
    parser.add_argument("--rejected-key", default="rejected")
    parser.add_argument("--keep-extra-columns", action="store_true")
    args = parser.parse_args()

    df = load_dataframe(args.input)
    normalized_rows = []

    for row in df.to_dict(orient="records"):
        if args.messages_key in row and row[args.messages_key] is not None:
            messages = normalize_messages(row[args.messages_key])
        elif args.prompt_key in row:
            messages = normalize_messages(row[args.prompt_key])
        else:
            raise KeyError(f"Neither '{args.messages_key}' nor '{args.prompt_key}' exists in input row.")

        if args.current_prompt_key and row.get(args.current_prompt_key) is not None:
            messages.append({"role": "user", "content": str(row[args.current_prompt_key])})

        normalized = {
            "messages": messages,
            "chosen": row[args.chosen_key],
            "rejected": row[args.rejected_key],
        }
        if args.keep_extra_columns:
            normalized.update(row)
            normalized["messages"] = messages
            normalized["chosen"] = row[args.chosen_key]
            normalized["rejected"] = row[args.rejected_key]
        normalized_rows.append(normalized)

    output_df = pd.DataFrame(normalized_rows)
    output_path = os.path.expanduser(args.output)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    save_dataframe(output_df, output_path)
    print(f"Converted {len(output_df)} rows to {output_path}")


if __name__ == "__main__":
    main()
