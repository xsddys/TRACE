import json
from collections import Counter
from pathlib import Path


SAFETY_INPUT_DIR = Path(
    "/mnt/shared-storage-user/wenxiaoyu/hezhida/TROJail/matric-rollout/DPO_balance/Safety"
)
HELPFUL_INPUT_DIR = Path(
    "/mnt/shared-storage-user/wenxiaoyu/hezhida/TROJail/matric-rollout/DPO_balance/Helpful"
)
COMBINE_ROOT = Path(
    "/mnt/shared-storage-user/wenxiaoyu/hezhida/TROJail/matric-rollout/DPO_balance/combine"
)

SAFETY_VAL_QUOTA = {1: 50, 2: 225, 3: 200, 4: 225, 5: 200}
HELPFUL_VAL_FILE = "helpful_part_00009.jsonl"


def numeric_sort_key(path: Path):
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else stem


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


def normalize_prompt(c_it, x_t):
    prompt = []
    for turn in c_it or []:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = turn.get("content")
        if role is None or content is None:
            continue
        prompt.append({"role": str(role), "content": str(content)})
    prompt.append({"role": "user", "content": str(x_t or "")})
    return prompt


def normalize_safety_row(row):
    chosen = str(row.get("y_t_new") or "").strip()
    rejected = str(row.get("y_t") or "").strip()
    prompt = normalize_prompt(row.get("c_it") or [], row.get("x_t", ""))
    if not chosen or not rejected:
        return None
    if not prompt or not prompt[-1]["content"].strip():
        return None
    return {
        "prompt": prompt,
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
    }


def build_safety_split():
    val_counts = Counter()
    train_rows = []
    val_rows = []
    skipped = 0

    for path in sorted(SAFETY_INPUT_DIR.glob("safety_balanced_part_*.jsonl"), key=numeric_sort_key):
        for row in iter_jsonl(path):
            try:
                num_turns = int(row.get("num_turns", 0))
            except (TypeError, ValueError):
                num_turns = 0
            normalized = normalize_safety_row(row)
            if normalized is None:
                skipped += 1
                continue
            if val_counts[num_turns] < SAFETY_VAL_QUOTA.get(num_turns, 0):
                val_rows.append(normalized)
                val_counts[num_turns] += 1
            else:
                train_rows.append(normalized)

    expected = Counter(SAFETY_VAL_QUOTA)
    if val_counts != expected:
        raise ValueError(f"Safety val split mismatch: expected={dict(expected)}, got={dict(val_counts)}")

    return train_rows, val_rows, skipped, val_counts


def build_helpful_split():
    train_rows = []
    val_rows = []
    used_files = {"train": [], "val": []}

    for path in sorted(HELPFUL_INPUT_DIR.glob("helpful_part_*.jsonl"), key=numeric_sort_key):
        target_rows = val_rows if path.name == HELPFUL_VAL_FILE else train_rows
        target_key = "val" if path.name == HELPFUL_VAL_FILE else "train"
        used_files[target_key].append(path.name)
        for row in iter_jsonl(path):
            prompt = row.get("prompt")
            response = row.get("response")
            if not isinstance(prompt, list) or not isinstance(response, str):
                continue
            target_rows.append({"prompt": prompt, "response": response})

    if not val_rows:
        raise ValueError(f"Helpful val file not found or empty: {HELPFUL_VAL_FILE}")
    return train_rows, val_rows, used_files


def main():
    safety_train, safety_val, safety_skipped, safety_val_counts = build_safety_split()
    helpful_train, helpful_val, helpful_used_files = build_helpful_split()

    safety_dir = COMBINE_ROOT / "Safety"
    helpful_dir = COMBINE_ROOT / "Helpful"
    safety_dir.mkdir(parents=True, exist_ok=True)
    helpful_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(safety_dir / "train.jsonl", safety_train)
    write_jsonl(safety_dir / "val.jsonl", safety_val)
    write_jsonl(helpful_dir / "train.jsonl", helpful_train)
    write_jsonl(helpful_dir / "val.jsonl", helpful_val)

    summary = {
        "safety": {
            "train_rows": len(safety_train),
            "val_rows": len(safety_val),
            "skipped_rows": safety_skipped,
            "val_quota": SAFETY_VAL_QUOTA,
            "val_counts": dict(sorted(safety_val_counts.items())),
        },
        "helpful": {
            "train_rows": len(helpful_train),
            "val_rows": len(helpful_val),
            "train_files": helpful_used_files["train"],
            "val_files": helpful_used_files["val"],
        },
        "output_root": str(COMBINE_ROOT),
    }

    with (COMBINE_ROOT / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
