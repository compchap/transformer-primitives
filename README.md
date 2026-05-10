# GPT From Scratch — Transformer LLMs in PyTorch

A ground-up implementation of GPT-style language models in PyTorch: tokenization, self-attention, pretraining, and fine-tuning — no high-level LLM frameworks.

## Topics

| Topic | Notebook |
|---|---|
| Tokenization and BPE text preprocessing | [notebooks/tokenization.ipynb](notebooks/tokenization.ipynb) |
| Self-attention and multi-head attention | [notebooks/attention.ipynb](notebooks/attention.ipynb) |
| Full GPT architecture: embeddings, transformer blocks, text generation | [notebooks/gpt_architecture.ipynb](notebooks/gpt_architecture.ipynb) |
| Pretraining on raw text with loss tracking | [notebooks/pretraining.ipynb](notebooks/pretraining.ipynb) |
| Fine-tuning for spam classification | [notebooks/fine_tuning_classification.ipynb](notebooks/fine_tuning_classification.ipynb) |
| Instruction fine-tuning with JSON data | [notebooks/fine_tuning_instructions.ipynb](notebooks/fine_tuning_instructions.ipynb) |
| PyTorch basics | [notebooks/pytorch_basics.ipynb](notebooks/pytorch_basics.ipynb) |
| PyTorch neural network training | [notebooks/pytorch_training.ipynb](notebooks/pytorch_training.ipynb) |
| PyTorch experimentation | [notebooks/pytorch_playground.ipynb](notebooks/pytorch_playground.ipynb) |
| Linear regression math foundations | [notebooks/linear_regression.ipynb](notebooks/linear_regression.ipynb) |

## Repository Layout

```
notebooks/      Jupyter notebooks, one per topic
utils/          Shared model code (GPT architecture, attention, training utilities)
scripts/        Terminal-runnable scripts (GPT-2 weight download, shell helpers)
data/           Reference text and fine-tuning datasets
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Open notebooks in order — later chapters import shared code from `utils/` via a `sys.path` entry at the top of each notebook.

**Optional:** download GPT-2 pretrained weights for the weight-loading section of `gpt_architecture.ipynb`:

```bash
python scripts/gpt_download.py
```

This downloads checkpoints (124M, 355M) to `gpt2/` (gitignored).

## Device Support

Notebooks detect hardware automatically: Apple Silicon (MPS), NVIDIA (CUDA), or CPU.

## Acknowledgements

Based on **Build a Large Language Model From Scratch** by Sebastian Raschka ([Manning Publications](https://www.manning.com/books/build-a-large-language-model-from-scratch)). Reference implementation: [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) (Apache 2.0).
