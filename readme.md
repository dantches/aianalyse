# VulnOpt — Basic Vulnerability Classifier

This repository demonstrates a minimal pipeline for detecting vulnerable code snippets.
The model is a simple neural network trained to recognize dangerous patterns in a small dataset of C functions. The pipeline includes training, quality re-evaluation, and inference via Typer-CLI.

## What the Model Does
- **Purpose.** Determines whether a C code snippet contains typical security flaws (buffer overflow, format string issues, SQL/command injection, etc.).
- **Input.** Raw C function texts. Before feeding into the network, the code is tokenized and transformed into a TF-IDF vector with unigrams and bigrams (max 4,096 features).
- **Architecture.** Fully connected network `4096 → 256 → 128 → 2` with ReLU and a dropout rate of 0.2.
- **Training.** Adam optimizer + CrossEntropyLoss. The pipeline uses pre-split `train/valid` subsets and logs epoch metrics history.
- **Evaluation.** `vulnopt.cli evaluate-cmd` rebuilds features using the saved TF-IDF vectorizer, loads weights, and recalculates loss/accuracy on both training and validation splits.
- **Inference.** `vulnopt.cli infer-cmd` takes a JSON file with a `code` field, restores the TF-IDF vectorization, and returns probabilities for `safe`/`vulnerable` classes.

## Dataset
- **Source.** `data/synthetic_vuln/*.jsonl` — a compact dataset compiled from typical Juliet Test Suite patterns (CWE-121, CWE-134, CWE-78, etc.) alongside their safe counterparts. Each file contains JSONL format with lines like `{ "code": ..., "label": 0|1 }`.
- **Size.** 84 training, 18 validation, and 18 test functions.
- **Format.** `label=1` indicates a vulnerable snippet, `label=0` indicates a safe snippet.

## Performance & Quality
Using default hyperparameters (`epochs=15`, `batch_size=32`, `lr=1e-3`), the model consistently achieves **85–90%** validation accuracy. The exact metric depends on random weight initialization.

## Reproducing Experiments
```bash
# Training
PYTHONPATH=src python -m vulnopt.cli train-cmd --epochs 15 --out runs/vuln_example

# Re-evaluating accuracy (uses the same dataset as training)
PYTHONPATH=src python -m vulnopt.cli evaluate-cmd runs/vuln_example/vuln_classifier.pt

# Inference: analyzing multiple functions from samples.json
PYTHONPATH=src python -m vulnopt.cli infer-cmd runs/vuln_example/vuln_classifier.pt samples.json predictions.json
