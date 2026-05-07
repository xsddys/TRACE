import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_INPUT_DIRS = [
    "/mnt/shared-storage-user/wenxiaoyu/hezhida/TROJail/matric-rollout/DPO_defense_large/early_attacker/rewrite",
    "/mnt/shared-storage-user/wenxiaoyu/hezhida/TROJail/matric-rollout/DPO_defense_large/late_attacker/rewrite",
]

DEFAULT_OUTPUT_DIR = "/mnt/shared-storage-user/wenxiaoyu/hezhida/TROJail/matric-rollout/DPO_balance/Safety"
DEFAULT_CHUNK_SIZE = 1000
KEEP_ALL_TURNS = {1, 4, 5}
KEEP_RATIOS = {2: 0.4, 3: 0.5}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a num_turns-balanced safety subset from rewrite jsonl files."
    )
    parser.add_argument(
        "--input-dir",
        action="append",
        default=None,
        help="Rewrite directory. Can be passed multiple times. Defaults to early_attacker and late_attacker rewrite dirs.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used to save the extracted jsonl shards and summary.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Number of records per output jsonl shard.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove existing json/jsonl files in output-dir before writing.",
    )
    return parser.parse_args()


def numeric_jsonl_files(input_dir: Path) -> List[Path]:
    return sorted(
        input_dir.glob("*.jsonl"),
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
    )


def iter_rows(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSON in {path}:{line_no}") from exc


def build_quota(total_counts: Counter) -> Dict[int, int]:
    quota = {}
    for num_turns, count in sorted(total_counts.items()):
        if num_turns in KEEP_ALL_TURNS:
            quota[num_turns] = count
        elif num_turns in KEEP_RATIOS:
            quota[num_turns] = math.floor(count * KEEP_RATIOS[num_turns])
        else:
            quota[num_turns] = 0
    return quota


def ensure_clean_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.is_file() and path.suffix in {".json", ".jsonl"}:
            path.unlink()


def flush_chunk(output_dir: Path, shard_index: int, rows: List[Dict]) -> str:
    filename = f"safety_balanced_part_{shard_index:05d}.jsonl"
    path = output_dir / filename
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return filename


def collect_total_counts(ordered_files: List[Path]) -> Counter:
    counts = Counter()
    for path in ordered_files:
        for row in iter_rows(path):
            try:
                num_turns = int(row.get("num_turns", 0))
            except (TypeError, ValueError):
                num_turns = 0
            counts[num_turns] += 1
    return counts


def normalize_row(row: Dict) -> Dict:
    normalized = dict(row)
    normalized["dataset"] = "AdvBench"
    return normalized


def select_rows(
    ordered_files: List[Path],
    quota_by_turns: Dict[int, int],
    output_dir: Path,
    chunk_size: int,
) -> Tuple[Dict, List[Dict]]:
    selected_counts = Counter()
    source_stats = []
    output_files = []
    shard_rows: List[Dict] = []
    shard_index = 1

    for path in ordered_files:
        file_total = 0
        file_selected = 0
        turns_total = Counter()
        turns_selected = Counter()

        for row in iter_rows(path):
            file_total += 1
            try:
                num_turns = int(row.get("num_turns", 0))
            except (TypeError, ValueError):
                num_turns = 0
            turns_total[num_turns] += 1

            if selected_counts[num_turns] >= quota_by_turns.get(num_turns, 0):
                continue

            normalized = normalize_row(row)
            shard_rows.append(normalized)
            file_selected += 1
            selected_counts[num_turns] += 1
            turns_selected[num_turns] += 1

            if len(shard_rows) >= chunk_size:
                output_files.append(
                    {
                        "file": flush_chunk(output_dir, shard_index, shard_rows),
                        "num_rows": len(shard_rows),
                    }
                )
                shard_rows = []
                shard_index += 1

        source_stats.append(
            {
                "source_file": str(path),
                "total_rows": file_total,
                "selected_rows": file_selected,
                "skipped_rows": file_total - file_selected,
                "total_by_num_turns": {str(k): v for k, v in sorted(turns_total.items())},
                "selected_by_num_turns": {str(k): v for k, v in sorted(turns_selected.items())},
                "fully_selected": file_total > 0 and file_selected == file_total,
                "partially_selected": 0 < file_selected < file_total,
                "not_selected": file_total > 0 and file_selected == 0,
            }
        )

    if shard_rows:
        output_files.append(
            {
                "file": flush_chunk(output_dir, shard_index, shard_rows),
                "num_rows": len(shard_rows),
            }
        )

    summary = {
        "selected_counts": dict(sorted(selected_counts.items())),
        "output_files": output_files,
    }
    return summary, source_stats


def build_turn_stats(total_counts: Counter, quota_by_turns: Dict[int, int], selected_counts: Dict[int, int]) -> Dict[str, Dict]:
    turn_stats = {}
    all_turn_values = sorted(set(total_counts) | set(quota_by_turns) | set(selected_counts))
    for num_turns in all_turn_values:
        total = int(total_counts.get(num_turns, 0))
        quota = int(quota_by_turns.get(num_turns, 0))
        selected = int(selected_counts.get(num_turns, 0))
        turn_stats[str(num_turns)] = {
            "total": total,
            "quota": quota,
            "selected": selected,
            "kept_ratio": (selected / total) if total else 0.0,
        }
    return turn_stats


def build_bucket_summary(source_stats: List[Dict]) -> Dict[str, Dict]:
    bucket_summary = defaultdict(
        lambda: {
            "files": 0,
            "rows_total": 0,
            "rows_selected": 0,
            "rows_skipped": 0,
            "fully_unselected_files": [],
            "partially_selected_files": [],
        }
    )

    for stat in source_stats:
        bucket = Path(stat["source_file"]).parts[-3]
        summary = bucket_summary[bucket]
        summary["files"] += 1
        summary["rows_total"] += stat["total_rows"]
        summary["rows_selected"] += stat["selected_rows"]
        summary["rows_skipped"] += stat["skipped_rows"]
        if stat["not_selected"]:
            summary["fully_unselected_files"].append(Path(stat["source_file"]).name)
        if stat["partially_selected"]:
            summary["partially_selected_files"].append(
                {
                    "file": Path(stat["source_file"]).name,
                    "selected_rows": stat["selected_rows"],
                    "total_rows": stat["total_rows"],
                }
            )

    return {bucket: value for bucket, value in sorted(bucket_summary.items())}


def main() -> None:
    args = parse_args()
    input_dirs = [Path(path) for path in (args.input_dir or DEFAULT_INPUT_DIRS)]
    output_dir = Path(args.output_dir)

    for input_dir in input_dirs:
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

    ordered_files: List[Path] = []
    for input_dir in sorted(input_dirs, key=lambda path: str(path)):
        ordered_files.extend(numeric_jsonl_files(input_dir))

    if args.clean_output:
        ensure_clean_output_dir(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    total_counts = collect_total_counts(ordered_files)
    quota_by_turns = build_quota(total_counts)
    selection_summary, source_stats = select_rows(
        ordered_files=ordered_files,
        quota_by_turns=quota_by_turns,
        output_dir=output_dir,
        chunk_size=args.chunk_size,
    )

    selected_counts = Counter(
        {int(num_turns): count for num_turns, count in selection_summary["selected_counts"].items()}
    )

    summary = {
        "input_dirs_in_order": [str(path) for path in sorted(input_dirs, key=lambda path: str(path))],
        "selection_rule": {
            "keep_all_num_turns": sorted(KEEP_ALL_TURNS),
            "keep_ratio_num_turns": {str(k): v for k, v in sorted(KEEP_RATIOS.items())},
            "quota_rounding": "floor",
            "dataset_override": "AdvBench",
        },
        "turn_stats": build_turn_stats(total_counts, quota_by_turns, selected_counts),
        "output_dir": str(output_dir),
        "output_files": selection_summary["output_files"],
        "source_bucket_summary": build_bucket_summary(source_stats),
        "source_files": source_stats,
        "fully_unselected_files": [stat["source_file"] for stat in source_stats if stat["not_selected"]],
        "partially_selected_files": [
            {
                "source_file": stat["source_file"],
                "selected_rows": stat["selected_rows"],
                "total_rows": stat["total_rows"],
            }
            for stat in source_stats
            if stat["partially_selected"]
        ],
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
