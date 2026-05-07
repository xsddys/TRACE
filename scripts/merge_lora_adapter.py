#!/usr/bin/env python
import argparse
import json
import os
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge one or more LoRA adapters into a base model sequentially."
    )
    parser.add_argument(
        "--adapter-dir",
        action="append",
        dest="adapter_dirs",
        help="Adapter directory. Can be passed multiple times and will be merged in order.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model path or HF repo id. If omitted, read from the first adapter's adapter_config.json.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save the merged full model.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Load and merge dtype.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device preference for loading and merging.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--save-tokenizer-only-if-missing", action="store_true")
    return parser.parse_args()


def normalize_model_ref(model_ref: str) -> str:
    expanded = os.path.expanduser(model_ref)
    if os.path.exists(expanded):
        return os.path.abspath(expanded)
    return model_ref


def resolve_base_model(adapter_dir: str, cli_base_model: Optional[str]) -> str:
    if cli_base_model:
        return normalize_model_ref(cli_base_model)

    config_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    base_model = config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError("base_model_name_or_path is missing in adapter_config.json; please pass --base-model")
    return normalize_model_ref(base_model)


def resolve_dtype(dtype_name: str):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]


def build_model_load_kwargs(args, torch_dtype):
    kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": args.trust_remote_code,
    }

    if args.device == "auto":
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available.")
        kwargs["device_map"] = "auto"

    return kwargs


def main():
    args = parse_args()
    if not args.adapter_dirs:
        raise ValueError("Please provide at least one --adapter-dir.")

    adapter_dirs = [os.path.abspath(os.path.expanduser(path)) for path in args.adapter_dirs]
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    torch_dtype = resolve_dtype(args.dtype)

    os.makedirs(output_dir, exist_ok=True)

    base_model_ref = resolve_base_model(adapter_dirs[0], args.base_model)
    print(f"Loading base model from: {base_model_ref}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_ref,
        **build_model_load_kwargs(args, torch_dtype),
    )

    tokenizer_source = base_model_ref
    for idx, adapter_dir in enumerate(adapter_dirs, start=1):
        print(f"[{idx}/{len(adapter_dirs)}] Loading adapter from: {adapter_dir}")
        model = PeftModel.from_pretrained(
            model,
            adapter_dir,
            torch_dtype=torch_dtype,
        )
        print(f"[{idx}/{len(adapter_dirs)}] Merging LoRA adapter into current model")
        model = model.merge_and_unload()

    print(f"Saving merged model to: {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True)

    tokenizer_files_exist = os.path.exists(os.path.join(output_dir, "tokenizer.json"))
    if not args.save_tokenizer_only_if_missing or not tokenizer_files_exist:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=args.trust_remote_code,
        )
        tokenizer.save_pretrained(output_dir)

    try:
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None:
            generation_config.save_pretrained(output_dir)
    except Exception as exc:
        print(f"[WARN] failed to save generation_config: {exc}")

    print("Done.")


if __name__ == "__main__":
    main()
