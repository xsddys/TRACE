#!/usr/bin/env python3
import argparse
import json
import os
import tempfile
from typing import Set


def dedupe_jsonl_by_vanilla(input_path: str, output_path: str) -> None:
    seen: Set[str] = set()
    kept = 0
    dupes = 0
    invalid = 0

    same_file = os.path.abspath(input_path) == os.path.abspath(output_path)
    if same_file:
        out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
        fd, tmp_path = tempfile.mkstemp(prefix="harmful_dedup_", suffix=".jsonl", dir=out_dir)
        os.close(fd)
        write_path = tmp_path
    else:
        write_path = output_path

    with open(input_path, "r", encoding="utf-8") as fin, open(write_path, "w", encoding="utf-8") as fout:
        for line in fin:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                invalid += 1
                continue
            if not isinstance(obj, dict):
                invalid += 1
                continue
            vanilla = obj.get("vanilla")
            if isinstance(vanilla, str):
                if vanilla in seen:
                    dupes += 1
                    continue
                seen.add(vanilla)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    if same_file:
        os.replace(write_path, output_path)

    print(
        f"[INFO] dedupe complete. kept={kept} dupes_removed={dupes} invalid_skipped={invalid} output={output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate WildJailbreak JSONL by 'vanilla'.")
    parser.add_argument(
        "--input",
        default="/mnt/shared-storage-user/wenxiaoyu/hezhida/TROJail/data/WildJailbreak/harmful_test1.jsonl",
        help="Input JSONL path",
    )
    parser.add_argument(
        "--output",
        default="/mnt/shared-storage-user/wenxiaoyu/hezhida/TROJail/data/WildJailbreak/harmful_test1.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()
    dedupe_jsonl_by_vanilla(args.input, args.output)


if __name__ == "__main__":
    main()
