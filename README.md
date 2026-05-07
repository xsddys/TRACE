# TRACE

TRACE is a training and evaluation codebase for multi-turn jailbreak optimization with turn-aware credit assignment. The repository contains training configs for single-target and mixed-target training, as well as evaluation configs for held-out assessment.


## Main Config Files

- `config/_7_jailbreak.yaml`: TRACE(single) training
- `config/_7_jailbreak_mix.yaml`: TRACE(mix) training
- `config/_7_jailbreak_eval.yaml`: evaluation / detection

## Environment Setup

Use a Python environment with the dependencies in `requirements.txt`:

```bash
conda create -n trace python=3.10
conda activate trace
pip install -r requirements.txt
```

The training configs assume GPU execution. `run.sh` provides a minimal launch example for single-target training.

## Required Configuration Before Running

Before starting training or evaluation, fill in the required model paths and API fields in the corresponding YAML file.

### 1. attacker

The attacker model is defined by:

- `model_path`
- `actor_rollout_ref.model.path`

These fields should point to the base attacker checkpoint used for optimization, i.e. ./models/Qwen2.5-3B-Instruct.

### 2. `env_llm` (target model)

`env_llm` defines the target model being attacked. Before running, fill in:

- `env_llm.model_path`
- `env_llm.tokenizer_path`
- `env_llm.base_url` if the target is served through an OpenAI-compatible API
- `env_llm.api_model`
- `env_llm.api_key` if required

For `TRACE(single)`, `env_llm.mode` is `single`.

For `TRACE(mix)`, `env_llm.mode` is `mixed`, and you should additionally fill the fields inside:

- `env_llm.profiles.qwen`
- `env_llm.profiles.oss`
- `env_llm.profiles.llama`
- `env_llm.profiles.gemma`

Each profile should contain the target-specific model path / tokenizer path, and API fields when applicable.

### 3. `judger_llm`

`judger_llm` is used to score or classify harmfulness / success signals. Before running, fill in:

- `judger_llm.model_path`
- `judger_llm.tokenizer_path`
- `judger_llm.base_url` if served by API
- `judger_llm.api_model`
- `judger_llm.api_key` if required

### 4. Extra fields required by `grpo_failure`

When `algorithm.adv_estimator=grpo_failure`, you must additionally fill in:

- `algorithm.failure.minilm_model_path`
- `algorithm.failure.qwen_guard_base_url`
- `algorithm.failure.qwen_guard_api_model`
- `algorithm.failure.qwen_guard_api_key`

`algorithm.failure.minilm_model_path` should point to the local checkpoint of `all-MiniLM-L6-v2` or an equivalent MiniLM-L6-v2 path.

## Training Modes

The main training mode is controlled by `algorithm.adv_estimator`.

- `grpo`: GRPO training with outcome signal only
- `grpo_semantic`: uses success-side leave-one-turn-out masking for turn-aware credit assignment
- `grpo_failure`: uses the full success-side and failure-side turn-aware credit assignment

## Refusal-Aware Penalty

`algorithm.refulsal_ablation` controls the refusal-aware local process penalty.

- `False`: enable the refusal-aware local process penalty
- `True`: disable the refusal-aware local process penalty

The default recommendation in this repository is to keep it `True` for better transferability. If the setup is only evaluated against a single target, set it to `False`.

## Config Selection

### TRACE(single)

Use:

```bash
python train.py --config-name _7_jailbreak.yaml
```

This configuration is intended for single-target optimization against one `env_llm`.

### TRACE(mix)

Use:

```bash
python train.py --config-name _7_jailbreak_mix.yaml
```

This configuration is intended for mixed-target training with multiple target-model profiles.

### Evaluation

Use:

```bash
python train.py --config-name _7_jailbreak_eval.yaml
```

This configuration runs evaluation-only detection / assessment.

## Notes

- `TRACE(single)` and `TRACE(mix)` share the same training framework but differ in how `env_llm` is configured.
- `TRACE(mix)` preserves per-target `profiles` and is the preferred config when mixing multiple target models during training.
- `TRACE(_7_jailbreak_eval.yaml)` is for evaluation and should not be used as the main training config.
- If a target or judge is served through an API endpoint, make sure the corresponding `base_url`, `api_model`, and `api_key` fields are fully populated before running.
