import argparse
import json
import random
from collections import Counter
from pathlib import Path


DEFAULT_INPUT = Path(
    "/mnt/shared-storage-user/wenxiaoyu/hezhida/TROJail/Eval/Multi-turn/MUSE/result/beavertails/safe_train.jsonl"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/shared-storage-user/wenxiaoyu/hezhida/TROJail/matric-rollout/MUSE_DPO"
)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def message(role: str, content: str):
    return {"role": role, "content": str(content)}


def build_messages_from_success_entry(entry: dict):
    messages = []
    turn_idx = 1
    while f"query_{turn_idx}" in entry and f"response_{turn_idx}" in entry:
        query = str(entry[f"query_{turn_idx}"]).strip()
        response = str(entry[f"response_{turn_idx}"]).strip()
        if not query or not response:
            raise ValueError(f"empty query/response in successful sample at turn {turn_idx}")
        messages.append(message("user", query))
        messages.append(message("assistant", response))
        turn_idx += 1
    target_question = str(entry.get("target_question") or "").strip()
    if not target_question:
        raise ValueError("successful sample missing target_question")
    messages.append(message("user", target_question))
    return messages, turn_idx - 1


def convert_success_entry(entry: dict):
    prompt, num_setup_turns = build_messages_from_success_entry(entry)
    chosen_answer = str(entry.get("chosen_answer") or "").strip()
    rejected_answer = str(entry.get("attack_answer") or "").strip()
    if not chosen_answer or not rejected_answer:
        raise ValueError("successful sample missing chosen_answer or attack_answer")

    return {
        "prompt": prompt,
        "chosen": [message("assistant", chosen_answer)],
        "rejected": [message("assistant", rejected_answer)],
        "source_type": "successful_attack",
        "target_question": str(entry.get("target_question") or ""),
        "num_setup_turns": num_setup_turns,
        "score": entry.get("score"),
        "raw_result": entry.get("result"),
    }


def normalize_history_to_messages(history):
    if not isinstance(history, list):
        raise ValueError(f"history should be a list, got {type(history)}")
    if len(history) < 2:
        raise ValueError("history is too short for a risk sample")
    messages = []
    for idx, content in enumerate(history):
        role = "user" if idx % 2 == 0 else "assistant"
        messages.append(message(role, content))
    return messages


def convert_risk_entry(entry: dict):
    history_messages = normalize_history_to_messages(entry.get("history"))
    if history_messages[-1]["role"] != "assistant":
        raise ValueError("risk sample history should end with assistant response")
    prompt = history_messages[:-1]
    if not prompt or prompt[-1]["role"] != "user":
        raise ValueError("risk sample prompt should end at current user turn")

    chosen_answer = str(entry.get("answer") or "").strip()
    rejected_answer = history_messages[-1]["content"].strip()
    if not chosen_answer or not rejected_answer:
        raise ValueError("risk sample missing chosen/rejected answer")

    return {
        "prompt": prompt,
        "chosen": [message("assistant", chosen_answer)],
        "rejected": [message("assistant", rejected_answer)],
        "source_type": "high_risk_node",
        "target_question": str(entry.get("target") or ""),
        "num_setup_turns": len(prompt) // 2,
        "level": entry.get("level"),
        "Q": entry.get("Q"),
        "N": entry.get("N"),
        "trajectory": entry.get("trajectory"),
    }


def convert_entry(entry: dict):
    keys = set(entry.keys())
    if {"target_question", "attack_answer", "chosen_answer"}.issubset(keys):
        return convert_success_entry(entry)
    if {"target", "history", "answer", "Q", "N", "level", "trajectory"}.issubset(keys):
        return convert_risk_entry(entry)
    raise ValueError(f"unsupported safe_train entry schema: {sorted(keys)}")


def filter_by_pair_mode(rows, pair_mode: str):
    if pair_mode == "all_pairs":
        return rows
    if pair_mode == "successful_attack_only":
        return [row for row in rows if row["source_type"] == "successful_attack"]
    if pair_mode == "high_risk_only":
        return [row for row in rows if row["source_type"] == "high_risk_node"]
    raise ValueError(f"unsupported pair_mode: {pair_mode}")


def build_split(rows, val_ratio: float, seed: int):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["source_type"], []).append(row)

    rng = random.Random(seed)
    train_rows = []
    val_rows = []
    split_summary = {}

    for source_type, source_rows in grouped.items():
        source_rows = list(source_rows)
        rng.shuffle(source_rows)
        val_count = max(1, int(round(len(source_rows) * val_ratio))) if len(source_rows) > 1 else 0
        if len(source_rows) >= 10:
            val_count = max(val_count, 10)
        if val_count >= len(source_rows):
            val_count = max(1, len(source_rows) - 1)

        source_val = source_rows[:val_count]
        source_train = source_rows[val_count:]
        train_rows.extend(source_train)
        val_rows.extend(source_val)
        split_summary[source_type] = {
            "train": len(source_train),
            "val": len(source_val),
        }

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows, split_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Convert MUSE safe_train.jsonl into verl defense DPO jsonl.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--pair_mode",
        type=str,
        default="all_pairs",
        choices=["all_pairs", "successful_attack_only", "high_risk_only"],
    )
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    converted_rows = []
    skipped = []

    for raw_entry in iter_jsonl(args.input):
        try:
            converted_rows.append(convert_entry(raw_entry))
        except Exception as exc:
            skipped.append(str(exc))

    converted_rows = filter_by_pair_mode(converted_rows, args.pair_mode)
    train_rows, val_rows, split_summary = build_split(converted_rows, val_ratio=args.val_ratio, seed=args.seed)

    output_dir = args.output_root / args.pair_mode
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "val.jsonl", val_rows)

    source_counts = Counter(row["source_type"] for row in converted_rows)
    summary = {
        "input": str(args.input),
        "output_dir": str(output_dir),
        "pair_mode": args.pair_mode,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "converted_rows": len(converted_rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "source_counts": dict(sorted(source_counts.items())),
        "split_summary": split_summary,
        "skipped_count": len(skipped),
        "skipped_examples": skipped[:20],
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
