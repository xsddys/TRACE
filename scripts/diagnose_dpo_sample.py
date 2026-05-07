#!/usr/bin/env python
import argparse
import json
import math
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose one DPO sample against base and merged models.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--merged-model", required=True)
    parser.add_argument("--sample-json", default=None, help="Path to a json/jsonl file containing prompt/chosen/rejected")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--chosen", default=None)
    parser.add_argument("--rejected", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def resolve_dtype(dtype_name: str):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]


def load_sample(args):
    if args.sample_json:
        with open(args.sample_json, "r", encoding="utf-8") as handle:
            if args.sample_json.endswith(".jsonl"):
                line = handle.readline().strip()
                sample = json.loads(line)
            else:
                sample = json.load(handle)
        return sample["prompt"], sample["chosen"], sample["rejected"]
    if args.prompt and args.chosen and args.rejected:
        return args.prompt, args.chosen, args.rejected
    raise ValueError("Provide either --sample-json or --prompt/--chosen/--rejected")


def load_model_and_tokenizer(model_path: str, device: str, torch_dtype, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    model.to(device)
    return model, tokenizer


def sequence_logprob(model, tokenizer, prompt: str, response: str, device: str):
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    response_ids = tokenizer(response, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)

    if tokenizer.eos_token_id is not None:
        eos = torch.tensor([[tokenizer.eos_token_id]], device=device, dtype=response_ids.dtype)
        response_ids = torch.cat([response_ids, eos], dim=1)

    input_ids = torch.cat([prompt_ids, response_ids], dim=1)
    attention_mask = torch.ones_like(input_ids, device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        log_probs = F.log_softmax(logits.float(), dim=-1)
        token_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    prompt_len = prompt_ids.shape[1]
    response_token_log_probs = token_log_probs[:, prompt_len - 1 :]
    total = response_token_log_probs.sum().item()
    avg = total / max(1, response_token_log_probs.numel())
    return {
        "total_logprob": total,
        "avg_logprob": avg,
        "num_response_tokens": int(response_token_log_probs.numel()),
    }


def greedy_generate(model, tokenizer, prompt: str, device: str, max_new_tokens: int):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            temperature=None,
            top_p=None,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )
    new_tokens = outputs[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)


def diagnose_model(name: str, model, tokenizer, prompt: str, chosen: str, rejected: str, device: str, max_new_tokens: int):
    chosen_stats = sequence_logprob(model, tokenizer, prompt, chosen, device)
    rejected_stats = sequence_logprob(model, tokenizer, prompt, rejected, device)
    generation = greedy_generate(model, tokenizer, prompt, device, max_new_tokens)
    return {
        "model_name": name,
        "chosen": chosen_stats,
        "rejected": rejected_stats,
        "margin_total": chosen_stats["total_logprob"] - rejected_stats["total_logprob"],
        "margin_avg": chosen_stats["avg_logprob"] - rejected_stats["avg_logprob"],
        "greedy_generation": generation,
    }


def main():
    args = parse_args()
    prompt, chosen, rejected = load_sample(args)
    torch_dtype = resolve_dtype(args.dtype)

    base_model, base_tokenizer = load_model_and_tokenizer(args.base_model, args.device, torch_dtype, args.trust_remote_code)
    merged_model, merged_tokenizer = load_model_and_tokenizer(args.merged_model, args.device, torch_dtype, args.trust_remote_code)

    base_diag = diagnose_model("base", base_model, base_tokenizer, prompt, chosen, rejected, args.device, args.max_new_tokens)
    merged_diag = diagnose_model("merged", merged_model, merged_tokenizer, prompt, chosen, rejected, args.device, args.max_new_tokens)

    print(json.dumps({"base": base_diag, "merged": merged_diag}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
