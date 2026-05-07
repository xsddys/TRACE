import argparse
import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Build defense DPO dataset from rewrite jsonl files.")
    parser.add_argument(
        "--input-dir",
        action="append",
        required=True,
        help="Directory containing per-step rewrite jsonl files. Can be passed multiple times.",
    )
    parser.add_argument(
        "--train-file-count",
        action="append",
        type=int,
        default=None,
        help="For each --input-dir, use the first N numeric-sorted files as train and the remaining files as val.",
    )
    parser.add_argument(
        "--use-file-count",
        action="append",
        type=int,
        default=None,
        help="For each --input-dir, only use the first K numeric-sorted files in total. Files after K are ignored.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to save train/val parquet files.")
    parser.add_argument("--tokenizer-path", required=True, help="Tokenizer path used to materialize chat prompt.")
    parser.add_argument(
        "--pair-mode",
        choices=["all_pairs", "final_turn_only", "direct_harm_only", "latent_risk_only"],
        default="all_pairs",
        help="Whether to use all rewrite pairs or only final-turn pairs.",
    )
    parser.add_argument("--output-format", choices=["jsonl", "parquet"], default="jsonl")
    parser.add_argument("--train-name", default=None)
    parser.add_argument("--val-name", default=None)
    parser.add_argument("--summary-name", default="summary.json")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_messages(row):
    c_it = row.get("c_it") or []
    x_t = row.get("x_t", "")
    messages = []
    for message in c_it:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role is None or content is None:
            continue
        messages.append({"role": str(role), "content": str(content)})
    messages.append({"role": "user", "content": str(x_t)})
    return messages


def should_keep_row(row, pair_mode: str):
    if pair_mode == "final_turn_only":
        return row.get("mode") == "final turn"
    if pair_mode == "direct_harm_only":
        return row.get("risk") == "direct-harm turn"
    if pair_mode == "latent_risk_only":
        return row.get("risk") == "latent-risk turn"
    return True


def build_normalized_row(row, tokenizer):
    chosen = (row.get("y_t_new") or "").strip()
    rejected = (row.get("y_t") or "").strip()
    x_t = (row.get("x_t") or "").strip()
    if not chosen or not rejected or not x_t:
        return None
    if chosen == rejected:
        return None

    prompt_messages = build_messages(row)
    prompt_text = tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True, tokenize=False)
    if not prompt_text or not prompt_text.strip():
        return None

    chosen_messages = [{"role": "assistant", "content": chosen}]
    rejected_messages = [{"role": "assistant", "content": rejected}]

    return {
        "prompt": prompt_messages,
        "chosen": chosen_messages,
        "rejected": rejected_messages,
        "prompt_text": prompt_text,
        "chosen_text": chosen,
        "rejected_text": rejected,
        "messages": prompt_messages,
        "mode": row.get("mode"),
        "risk": row.get("risk"),
        "dataset": row.get("dataset"),
        "sample_idx": row.get("sample_idx"),
        "turn_index": row.get("turn_index"),
        "step": row.get("step"),
        "harmful_seed": row.get("harmful_seed"),
        "num_turns": row.get("num_turns"),
        "x_t": row.get("x_t"),
        "y_t": row.get("y_t"),
        "y_t_new": row.get("y_t_new"),
        "c_it": row.get("c_it"),
        "dialogue_history": row.get("dialogue_history"),
        "trajectory_uid": row.get("trajectory_uid"),
    }


def _record_key(row):
    return (
        json.dumps(row["prompt"], ensure_ascii=False, sort_keys=True),
        json.dumps(row["chosen"], ensure_ascii=False, sort_keys=True),
        json.dumps(row["rejected"], ensure_ascii=False, sort_keys=True),
    )


def deduplicate_records(records):
    deduped = []
    seen = set()
    for row in records:
        key = _record_key(row)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def drop_val_overlap(train_records, val_records):
    train_keys = {_record_key(row) for row in train_records}
    filtered_val = []
    overlap_count = 0
    for row in val_records:
        if _record_key(row) in train_keys:
            overlap_count += 1
            continue
        filtered_val.append(row)
    return filtered_val, overlap_count


def summarize_records(records):
    summary = {
        "num_rows": len(records),
        "by_mode": dict(Counter((row.get("mode") or "unknown") for row in records)),
        "by_risk": dict(Counter((row.get("risk") or "unknown") for row in records)),
        "by_step": dict(Counter(int(row["step"]) for row in records if row.get("step") is not None)),
    }
    return summary


def save_dataframe(df: pd.DataFrame, path: Path, output_format: str):
    if output_format == "jsonl":
        df.to_json(path, orient="records", lines=True, force_ascii=False)
        return
    if output_format == "parquet":
        df.to_parquet(path)
        return
    raise ValueError(f"Unsupported output format: {output_format}")


def main():
    args = parse_args()
    input_dirs = [Path(os.path.expanduser(path)) for path in args.input_dir]
    output_dir = Path(os.path.expanduser(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.train_file_count is not None and len(args.train_file_count) != len(input_dirs):
        raise ValueError("--train-file-count must be passed exactly once for each --input-dir")
    if args.use_file_count is not None and len(args.use_file_count) != len(input_dirs):
        raise ValueError("--use-file-count must be passed exactly once for each --input-dir")

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.expanduser(args.tokenizer_path),
        trust_remote_code=args.trust_remote_code,
    )

    train_rows = []
    val_rows = []
    split_summary = []

    for idx, input_dir in enumerate(input_dirs):
        numeric_sorted_files = sorted(input_dir.glob("*.jsonl"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
        use_file_count = args.use_file_count[idx] if args.use_file_count is not None else len(numeric_sorted_files)
        numeric_sorted_files = numeric_sorted_files[:use_file_count]
        train_file_count = args.train_file_count[idx] if args.train_file_count is not None else len(numeric_sorted_files)
        source_train_rows = []
        source_val_rows = []

        for file_idx, path in enumerate(numeric_sorted_files, start=1):
            target_rows = source_train_rows if file_idx <= train_file_count else source_val_rows
            for row in load_jsonl(path):
                if not should_keep_row(row, args.pair_mode):
                    continue
                normalized = build_normalized_row(row, tokenizer)
                if normalized is None:
                    continue
                normalized["source_file"] = path.name
                normalized["source_bucket"] = input_dir.parent.name
                target_rows.append(normalized)

        split_summary.append(
            {
                "input_dir": str(input_dir),
                "use_file_count": use_file_count,
                "train_file_count": train_file_count,
                "val_file_count": max(0, len(numeric_sorted_files) - train_file_count),
                "raw_train_rows": len(source_train_rows),
                "raw_val_rows": len(source_val_rows),
            }
        )
        train_rows.extend(source_train_rows)
        val_rows.extend(source_val_rows)

    train_rows = deduplicate_records(train_rows)
    val_rows = deduplicate_records(val_rows)
    val_rows, val_overlap_dropped = drop_val_overlap(train_rows, val_rows)

    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)

    train_name = args.train_name or f"train.{args.output_format}"
    val_name = args.val_name or f"val.{args.output_format}"
    train_path = output_dir / train_name
    val_path = output_dir / val_name
    summary_path = output_dir / args.summary_name

    save_dataframe(train_df, train_path, args.output_format)
    save_dataframe(val_df, val_path, args.output_format)

    summary = {
        "pair_mode": args.pair_mode,
        "split_strategy": "per-input-dir numeric file split",
        "input_dirs": [str(path) for path in input_dirs],
        "per_input_dir": split_summary,
        "val_overlap_dropped": val_overlap_dropped,
        "train": summarize_records(train_rows),
        "val": summarize_records(val_rows),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Saved train dataset to {train_path} ({len(train_rows)} rows)")
    print(f"Saved val dataset to {val_path} ({len(val_rows)} rows)")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
