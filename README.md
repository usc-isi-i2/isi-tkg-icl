# Temporal Knowledge Graph Forecasting Without Knowledge Using In-Context Learning

Code & Data for EMNLP 2023 paper: 

[Temporal Knowledge Graph Forecasting Without Knowledge Using In-Context Learning][paper]

Authors: [Dong-Ho Lee&ast;][dlee], [Kian Ahrabian&ast;][kahrabian], [Woojeong Jin][wjin], [Fred Morstatter][fmorstatter], [Jay Pujara][jpujara].

Temporal knowledge graph (TKG) forecasting benchmarks challenge models to predict future facts using knowledge of past facts. In this paper, we develop an approach to use in-context learning (ICL) with large language models (LLMs) for TKG forecasting. Our extensive evaluation compares diverse baselines, including both simple heuristics and state-of-the-art (SOTA) supervised models, against pre-trained LLMs across several popular benchmarks and experimental settings. We observe that naive LLMs perform on par with SOTA models, which employ carefully designed architectures and supervised training for the forecasting task, falling within the (-3.6%, +1.5%) Hits@1 margin relative to the median performance. To better understand the strengths of LLMs for forecasting, we explore different approaches for selecting historical facts, constructing prompts, controlling information propagation, and parsing outputs into a probability distribution. A surprising finding from our experiments is that LLM performance endures (±0.4% Hit@1) even when semantic information is removed by mapping entities/relations to arbitrary numbers, suggesting that prior semantic knowledge is unnecessary; rather, LLMs can leverage the symbolic patterns in the context to achieve such a strong performance. Our analysis also reveals that ICL enables LLMs to learn irregular patterns from the historical context, going beyond frequency and recency biases

## Requirements

Python >= 3.10

```bash
pip install -r requirements.txt
```

## Frequency/Recency Baselines
```bash
python run_rule.py --dataset {dataset} --model {recency|frequency} --history_len {history_len} --history_type {entity|pair} --history_direction {uni|bi} --label {--multi_step}
```

## LLMs
For more options you can use `--help` or take a look at `utils.get_args` function.

```bash
python run_hf.py --dataset {dataset} --model "EleutherAI/gpt-neox-20b" --history_len {history_len} --history_type {entity|pair} --history_direction {uni|bi} --label {--multi_step}
```

## Citation
If you make use of this code, please kindly cite the following paper:

```bib
@article{lee2023temporal,
  title={Temporal Knowledge Graph Forecasting Without Knowledge Using In-Context Learning},
  author={Lee, Dong-Ho and Ahrabian, Kian and Jin, Woojeong and Morstatter, Fred and Pujara, Jay},
  journal={arXiv preprint arXiv:2305.10613},
  year={2023}
}
```

[paper]: https://arxiv.org/abs/2305.10613
[dlee]: https://www.danny-lee.info/
[kahrabian]: https://scholar.google.com/citations?user=pwUdiCYAAAAJ&hl=en
[wjin]: https://woojeongjin.github.io/
[fmorstatter]: https://www.isi.edu/~fredmors/
[jpujara]: https://www.jaypujara.org/

