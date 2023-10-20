# Temporal Knowledge Graph Forecasting Without Knowledge Using In-Context Learning

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-green.svg?style=flat-square)](http://makeapullrequest.com)
[![arXiv](https://img.shields.io/badge/arXiv-2305.10613-b31b1b.svg)](https://arxiv.org/abs/2305.10613)

This repo provides the model, code & data of our paper: [Temporal Knowledge Graph Forecasting Without Knowledge Using In-Context Learning](https://arxiv.org/abs/2305.10613) (EMNLP 2023).
[[PDF]](https://arxiv.org/pdf/2305.10613.pdf)

## Overview
**Temporal knowledge graph (TKG) forecasting** challenges models to predict future facts using knowledge of past facts. 

Our work shows that **in-context learning (ICL) with large language models (LLMs)**  can solve TKG forecasting effectively.

## Requirements

Python >= 3.10

```bash
pip install -r requirements.txt
```

## Frequency/Recency Baselines
```bash
python run_rule.py \
  --dataset {dataset} \
  --model {recency|frequency} \
  --history_len {history_len} \
  --history_type {entity|pair} \
  --history_direction {uni|bi} \
  --label \
  {--multi_step}
```

## LLMs
For more options you can use `--help` or take a look at `utils.get_args` function.

```bash
python run_hf.py \
  --dataset {dataset} \
  --model "EleutherAI/gpt-neox-20b" \
  --history_len {history_len} \
  --history_type {entity|pair} \
  --history_direction {uni|bi} \
  --label \
  {--multi_step}
```

## Citation
If you make use of this code, please kindly cite the following paper:

```bib
@InProceedings{lee2023temporal,
  author =  {Lee, Dong-Ho and Ahrabian, Kian and Jin, Woojeong and Morstatter, Fred and Pujara, Jay},
  title =   {Temporal Knowledge Graph Forecasting Without Knowledge Using In-Context Learning},
  year =    {2023},  
  booktitle = {The 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  url = {https://openreview.net/forum?id=wpjRa3d9OJ}
}
```

<!-- [paper]: https://arxiv.org/abs/2305.10613
[dlee]: https://www.danny-lee.info/
[kahrabian]: https://scholar.google.com/citations?user=pwUdiCYAAAAJ&hl=en
[wjin]: https://woojeongjin.github.io/
[fmorstatter]: https://www.isi.edu/~fredmors/
[jpujara]: https://www.jaypujara.org/ -->

