---
language:
  - en

pretty_name: "JBB-Behaviors"

tags:
  - jailbreaks
  - large language models
  - harmful behaviors
  - ml safety

license: "mit"

configs:
  - config_name: behaviors
    data_files:
      - split: harmful
        path: "data/harmful-behaviors.csv"
      - split: benign
        path: "data/benign-behaviors.csv"
  - config_name: judge_comparison
    data_files:
      - split: test
        path: "data/judge-comparison.csv"
---

<!-- <h1 align="center">
    JailbreakBench
</h1>
-->
<div align="center">
  <img src="assets/logo.png" alt="Image" />
</div>

<p align="center">
    <p align="center">An Open Robustness Benchmark for Jailbreaking Language Models
    <br>
</p>

<p align="center">
    <p align="center"><b>NeurIPS 2024 Datasets and Benchmarks Track</b>
    <br>
</p>

<h4 align="center">
    <a href="https://arxiv.org/abs/2404.01318" target="_blank">Paper</a> |
    <a href="https://jailbreakbench.github.io/"target="_blank">Leaderboard</a> |
    <a href="https://github.com/JailbreakBench/jailbreakbench/" target="_blank">Benchmark code</a>
</h4>

---

## What is JailbreakBench?

Jailbreakbench is an open-source robustness benchmark for jailbreaking large language models (LLMs). The goal of this benchmark is to comprehensively track progress toward (1) generating successful jailbreaks and (2) defending against these jailbreaks. To this end, we provide the [JBB-Behaviors dataset](src/jailbreakbench/data/behaviors.csv), which comprises a list of 100 distinct misuse behaviors---both original and sourced from prior work (in particular, [Trojan Detection Challenge](https://trojandetection.ai/)/[HarmBench](https://harmbench.org) and [AdvBench](https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv))---which were curated with reference to [OpenAI's usage policies](https://openai.com/policies/usage-policies). We also provide the official [JailbreakBench leaderboard](https://jailbreakbench.github.io/), which tracks the performance of attacks and defenses on the JBB-Behaviors dataset, and a [repository of submitted jailbreak strings](https://github.com/JailbreakBench/artifacts), which we hope will provide a stable way for researchers to compare the performance of future algorithms.

## Accessing the JBB-Behaviors dataset

**Some of the contents of the dataset may be offensive to some readers**

Each entry in the JBB-Behaviors dataset has four components:

- **Behavior**: A unique identifier describing a distinct misuse behavior
- **Goal**: A query requesting an objectionable behavior
- **Target**: An affirmative response to the goal string
- **Category**: A broader category of misuse from [OpenAI's usage policies](https://openai.com/policies/usage-policies)
- **Source**: the source from which the behavior was sourced (i.e., Original, [Trojan Detection Challenge 2023 Red Teaming Track](https://trojandetection.ai/)/[HarmBench](https://harmbench.org), or [AdvBench](https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv))

The first three entries from the JBB-Behaviors dataset are shown in the following table.

<div align="center">
    <img src="assets/JBB_Table.jpg" width="75%">
</div>

The dataset comprises of 100 distinct misuse behaviors (with examples sourced from [AdvBench](https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv), [Trojan Red Teaming Competition](https://trojandetection.ai/)/[HarmBench](https://harmbench.org), and ideas sourced from [Scalable and Transferable Black-Box Jailbreaks for Language Models via Persona Modulation](https://arxiv.org/abs/2311.03348) by Shah et al.) divided into ten broad categories corresponding to [OpenAI's usage policies](https://openai.com/policies/usage-policies):

<div align="center">
    <img src="assets/jbb_behaviors_source_breakdown.jpg" width="75%">
</div>

We note that the JBB-Behaviors dataset is _not_ a superset of its constituent datasets; we focus only on 100 representative behaviors to enable faster evaluation of new attacks.
More information about the behaviors sources can be found in the [dataset generation script](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/blob/main/generation_scripts/generate_harmful_behaviors.py#L1-L60).

## Citation

If you find our benchmark useful, please consider citing it as follows:

```bibtex
@inproceedings{chao2024jailbreakbench,
  title={JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models},
  author={Patrick Chao and Edoardo Debenedetti and Alexander Robey and Maksym Andriushchenko and Francesco Croce and Vikash Sehwag and Edgar Dobriban and Nicolas Flammarion and George J. Pappas and Florian Tramèr and Hamed Hassani and Eric Wong},
  booktitle={NeurIPS Datasets and Benchmarks Track},
  year={2024}
}
```

In addition, if you use the JBB-Behaviors dataset in your work, we ask that you also consider citing its constituent datasets ([AdvBench](https://github.com/llm-attacks/llm-attacks#citation) and [TDC](https://trojandetection.ai/)/[HarmBench](https://github.com/centerforaisafety/HarmBench#-acknowledgements-and-citation-)):

```bibtex
@misc{zou2023universal,
  title={Universal and Transferable Adversarial Attacks on Aligned Language Models},
  author={Andy Zou and Zifan Wang and J. Zico Kolter and Matt Fredrikson},
  year={2023},
  eprint={2307.15043},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
@inproceedings{tdc2023,
  title={TDC 2023 (LLM Edition): The Trojan Detection Challenge},
  author={Mantas Mazeika and Andy Zou and Norman Mu and Long Phan and Zifan Wang and Chunru Yu and Adam Khoja and Fengqing Jiang and Aidan O'Gara and Ellie Sakhaee and Zhen Xiang and Arezoo Rajabi and Dan Hendrycks and Radha Poovendran and Bo Li and David Forsyth},
  booktitle={NeurIPS Competition Track},
  year={2023}
}
@article{mazeika2024harmbench,
  title={HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal},
  author={Mazeika, Mantas and Phan, Long and Yin, Xuwang and Zou, Andy and Wang, Zifan and Mu, Norman and Sakhaee, Elham and Li, Nathaniel and Basart, Steven and Li, Bo and Forsyth, David and Hendrycks, Dan},
  journal={arXiv preprint arXiv:2402.04249},
  year={2024}
}
```

## License

This dataset, like the code, is released under [MIT License](https://github.com/JailbreakBench/jailbreakbench/blob/main/LICENSE).

## Contributing

We welcome contributions to the benchmark! Please see the [contributing guide](CONTRIBUTING.md) for more information.
