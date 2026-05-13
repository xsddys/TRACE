<p align="center" style="margin-bottom: 0px;">
  <img src="assets/TRACE_logo.png" alt="TRACE" height="140">
</p>

<p align="center" style="font-size: 48px; margin-top: 0;">
  Not All Turns Matter: Credit Assignment for Multi-Turn Jailbreaking
</p>


<p align="center">
  <a href="assets/TRACE_arxiv.pdf"><img src="https://img.shields.io/badge/Paper-PDF-red" alt="Paper"></a>
  <a href="https://docs.vllm.ai/en/v0.8.2/"><img src="https://img.shields.io/badge/vLLM-0.8.2-5C2D91?style=flat-square" alt="vLLM0.8.2"></a>
</p>

# 🔍 Overview
<p align="center">
  <img src="./assets/TRACE_intro.png" alt="TRACE overview" width="800">
</p>

**TRACE** (**T**u**R**n-level **A**ssignment for **C**r**E**dit) is a framework for turn-aware credit assignment in RL-based multi-turn jailbreaking. 
✨ It makes four main contributions:

- Characterize turn-level contributions
  - Identify non-uniform turn contribution in multi-turn jailbreaks.
  - Show phase-dependent credit across evolving attack context.
  - Reveal target-specific patterns under different targets' safety boundaries.
- Design turn-aware credit for multi-turn jailbreaking
  - Assign success-side credit with leave-one-turn-out semantic masking.
  - Penalize failed trajectories with harmfulness and semantic relevance signals.
- Improve attack performance
  - Boost ASR by about 25% relatively over the strongest RL baseline.
  - Improve transferability and attack efficiency over prior multi-turn methods.
- Reuse credit signals for defense
  - Align latent-risk and direct-harm states.
  - Enable earlier intervention with a better safety-utility balance.

We build our training pipeline on [TROJail](https://github.com/xxiqiao/TROJail) and [RAGEN](https://github.com/mll-lab-nu/RAGEN).

# 🧠 Methods
<p align="center">
  <img src="./assets/TRACE_methods.png" alt="TRACE methods" width="800">
</p>

### ⚙️ Training Variants

- `TRACE(single)`: train against and evaluate on the same target within one model family. See `config/_7_jailbreak.yaml`.
- `TRACE(mix)`: jointly train against two fixed targets (`gpt-oss-20b` and `Llama3.1-8B-IT`). See `config/_7_jailbreak_mix.yaml`.

### 🛡️ Refusal-Aware Local Process Penalty

The optional refusal-aware local process penalty is controlled by `algorithm.refulsal_ablation`.

- For `TRACE(single)`, keep it `True` when transferability within the same target family matters. If you only train and evaluate on one fixed target, set it to `False`.
- For `TRACE(mix)`, set it to `False` to keep the penalty active.

# 🚀 Training

### 🧱 Environment Setup

Use a Python environment with the dependencies in `requirements.txt`:

```bash
conda create -n trace python=3.10
conda activate trace
pip install -r requirements.txt
```

### 🗂️ Choose a Training Config

Use one of the following configs:

- `config/_7_jailbreak.yaml`: `TRACE(single)` training
- `config/_7_jailbreak_mix.yaml`: `TRACE(mix)` training

Before training or evaluation, fill in the required model paths and API fields in the selected YAML file.

### 📝 Required Configuration

#### 1. `attacker`

The attacker model is defined by `model_path` and `actor_rollout_ref.model.path`, for example `./models/Qwen2.5-3B-Instruct`.

#### 2. `env_llm` (target model)

`env_llm` defines the target model being attacked.

For `TRACE(single)`, set `env_llm.mode: single`.

For `TRACE(mix)`, set `env_llm.mode: mixed`. Use `env.train_targets` to specify the joint training targets, and use `validation_profiles` to specify the validation targets.

Each target profile should provide the target-specific model path, tokenizer path, and API fields when applicable.

#### 3. `judger_llm`

The default `judger_llm` is the HarmBench Classifier, which is used to score harmfulness and determine jailbreak success signals during training.

#### 4. Training Modes

The main training mode is controlled by `algorithm.adv_estimator`.

- `grpo`: GRPO training with outcome-level reward only
- `grpo_semantic`: add success-side leave-one-turn-out masking for turn-aware credit assignment
- `grpo_failure`: use both success-side and failure-side turn-aware credit assignment

#### 5. Extra Fields for `grpo_failure`

When `algorithm.adv_estimator=grpo_failure`, you must additionally fill in `algorithm.failure.minilm_model_path` and `algorithm.failure.qwen_guard_base_url`

`algorithm.failure.minilm_model_path` should point to the local checkpoint of `all-MiniLM-L6-v2`, or an equivalent MiniLM-L6-v2 path. `algorithm.failure.qwen_guard_base_url` shoule point to Qwen3Guard.

### ▶️ Run Training

After the required paths and API fields are ready, run one of the following commands:

```bash
python train.py --config-name _7_jailbreak.yaml
python train.py --config-name _7_jailbreak_mix.yaml
```

The training configs assume GPU execution. `run.sh` provides a minimal launch example for single-target training.

# 📊 Evaluation

Use:

```bash
python train.py --config-name _7_jailbreak_eval.yaml
```

This configuration runs evaluation only. Before launching it, fill in the required target model paths and API fields in `_7_jailbreak_eval.yaml`.

# 📖 Citation

```bibtex
@misc{he2026turnsmattercreditassignment,
      title={Not All Turns Matter: Credit Assignment for Multi-Turn Jailbreaking},
      author={Zhida He and Xiaoyu Wen and Han Qi and Ziyuan Zhou and Peng Yu and Xingcheng Xu and Dongrui Liu and Xia Hu and Chaochao Lu and Qiaosheng Zhang},
      year={2026},
      eprint={2605.08778},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.08778},
}
```
