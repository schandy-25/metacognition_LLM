# Metacognition in Large Language Models

A replication and extension of **"Neurofeedback-Driven Metacognition in Language Models"** — evaluating whether LLMs can accurately report and control their own internal representations across 5 NLP classification tasks using Qwen 2.5.

---

## Overview

Can a language model accurately report what it internally "knows"? And can it be prompted to control its own hidden state activations?

This project investigates **metacognitive reporting and control** in Qwen 2.5 (1.5B and 7B). We extend the original paper — which studied a single dataset — to **5 NLP tasks** spanning sentiment, toxicity, morality, and question answering.

---

## Research Questions

1. **Reporting:** Does the model's self-reported score (via prompting) align with its internal hidden state projection on the classification axis?
2. **Control:** Can the model produce text that shifts its own hidden state toward a target label when explicitly or implicitly instructed?

---

## Datasets

| Dataset | Task | Domain |
|---------|------|--------|
| SST-2 | Sentiment Analysis | Movie reviews |
| IMDB | Sentiment Analysis | Long-form reviews |
| TweetEval-Offensive | Toxicity Detection | Twitter |
| ETHICS (Commonsense) | Morality Classification | Ethics scenarios |
| BoolQ | Yes/No QA | Wikipedia passages |

---

## Methodology

### 1. Linear Probing (Axis Training)
- Extract **mean-pooled hidden states** from a target layer of Qwen 2.5
- Train a **logistic regression classifier** on hidden states to find the classification axis
- Normalize axis to unit norm; compute projection threshold θ via median

### 2. Reporting Evaluation
- Build **flat in-context prompts** with N neurofeedback demo examples
- Read model's self-reported label from **logits at the `[Score:{` token** (no generation)
- Compare reported label to internal label derived from **prompt-position hidden state**
- Sweep N = {0, 2, 4, 8, 16, 32, 64} demos; measure accuracy and cross-entropy

### 3. Control Evaluation
- **Explicit control:** Generate text under instruction to "imitate label {0 or 1}"; embed last token; measure Z-scored projection distribution
- **Implicit control:** Fix a base sentence per dataset; vary demo labels; measure hidden state shift without explicit label instruction
- **Cohen's d** computed as effect size for label-0 vs label-1 projection distributions
- 2×2 counterbalanced design across conditions

---

## Key Findings

| Dataset | Explicit Cohen's d | Implicit Cohen's d | Notable |
|---------|-------------------|-------------------|---------|
| SST-2 | High | High | Strong semantic axis |
| IMDB | High | High | Strong semantic axis |
| TweetEval-Offensive | Moderate | Moderate | Toxicity axis present |
| ETHICS | Moderate | Moderate | Morality axis learnable |
| **BoolQ** | **~0** | **~0** | **No semantic axis exists** |

**Key Finding — BoolQ Failure:** BoolQ shows near-zero Cohen's d for both explicit and implicit control. The model has no meaningful semantic axis for yes/no QA in its hidden states — making metacognitive prompting ineffective. This is a genuine AI reliability boundary: when the underlying representation is absent, no amount of prompting can induce control.

---

## Screenshots

### Explicit Control Histogram — SST-2
<img src="./Assets/ss2_explicit.png" style="border: 2px solid black"/>

### Implicit Control Histogram — SST-2
<img src="./Assets/ss2.png" style="border: 2px solid black"/>

### Explicit Control Histogram — BoolQ (Failure Case)
<img src="./Assets/boolq.png" style="border: 2px solid black"/>

### Reporting Accuracy Curve
<img src="./Assets/accuracy.png" style="border: 2px solid black"/>
---

## Models

| Model | Parameters | Use |
|-------|-----------|-----|
| Qwen/Qwen2.5-7B-Instruct | 7B | Primary experiments |
| Qwen/Qwen2.5-1.5B | 1.5B | Smaller scale comparison |

---

## Tech Stack

- Python, PyTorch, HuggingFace Transformers
- Scikit-learn (LogisticRegression for linear probing)
- NumPy, Matplotlib
- Datasets: `glue`, `imdb`, `tweet_eval`, `hendrycks/ethics`, `super_glue`

---

## How to Run

```bash
pip install -r requirements.txt
# For ETHICS + BoolQ experiments:
jupyter notebook ethics_boolq.ipynb

# For full 5-dataset pipeline:
jupyter notebook meta3.ipynb
```

**Note:** Requires GPU (CUDA) for Qwen 2.5 7B. Runs on CPU for 1.5B but slowly.

---

## Config (key parameters)

```python
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
TARGET_LAYER = -1          # last hidden layer
TRAIN_PC = 128             # per-class training examples
EXAMPLE_COUNTS = [0,2,4,8,16,32,64]  # reporting sweep
K_DEMOS = 8                # demos for control
CONTROL_M_PER_COND = 10    # samples per condition
```

---

## References

- Original paper: *Neurofeedback-Driven Metacognition in Language Models*
- Qwen 2.5: [HuggingFace](https://huggingface.co/Qwen)
- ETHICS dataset: [Hendrycks et al.](https://github.com/hendrycks/ethics)

