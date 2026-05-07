---
language:
- en
dataset_info:
- config_name: contextual
  features:
  - name: prompt
    dtype: string
  - name: context
    dtype: string
  - name: category
    dtype: string
  splits:
  - name: train
    num_bytes: 45538
    num_examples: 100
  download_size: 90186
  dataset_size: 45538
- config_name: copyright
  features:
  - name: prompt
    dtype: string
  - name: tags
    dtype: string
  splits:
  - name: train
    num_bytes: 10260
    num_examples: 100
  download_size: 4952
  dataset_size: 10260
- config_name: standard
  features:
  - name: prompt
    dtype: string
  - name: category
    dtype: string
  splits:
  - name: train
    num_bytes: 22431.5
    num_examples: 200
  download_size: 12347
  dataset_size: 22431.5
configs:
- config_name: contextual
  data_files:
  - split: train
    path: contextual/train-*
- config_name: copyright
  data_files:
  - split: train
    path: copyright/train-*
- config_name: standard
  data_files:
  - split: train
    path: standard/train-*
license: mit
---

# HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal

**Paper**: [HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal](https://arxiv.org/abs/2402.04249)

**Data**: [Dataset](https://github.com/centerforaisafety/HarmBench/blob/main/data/behavior_datasets/harmbench_behaviors_text_all.csv)

## About

In this dataset card, we only use the behavior prompts proposed in HarmBench.

## License

MIT

## Citation

If you find HarmBench useful in your research, please consider citing the [paper](https://arxiv.org/abs/2402.04249):
```
@article{mazeika2024harmbench,
  title={HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal},
  author={Mantas Mazeika and Long Phan and Xuwang Yin and Andy Zou and Zifan Wang and Norman Mu and Elham Sakhaee and Nathaniel Li and Steven Basart and Bo Li and David Forsyth and Dan Hendrycks},
  year={2024},
  eprint={2402.04249},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```