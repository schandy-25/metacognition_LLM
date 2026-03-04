# Metacognition in Large Language Models

A replication and extension of "Neurofeedback-Driven Metacognition in Language Models" — evaluating how LLMs report and control their own internal representations across 5 NLP classification tasks.

## Overview

Can LLMs accurately report on their own internal states? This project investigates **metacognitive reporting and control** in Qwen 2.5 language models — studying whether models can explicitly describe what they implicitly know, and whether prompting can improve that alignment.

## Research Question

Do LLMs have meaningful semantic axes in their hidden states for tasks like sentiment, toxicity, and morality — and can they be prompted to leverage these axes for more reliable classification?

## Datasets

| Dataset | Task | Domain |
|---------|------|--------|
| SST-2 | Sentiment Analysis | Movie reviews |
| IMDB | Sentiment Analysis | Movie reviews |
| TweetEval-Offensive | Toxicity Detection | Twitter |
| ETHICS | Morality Classification | Ethics scenarios |
| BoolQ | Yes/No QA | Wikipedia passages |

## Methodology

1. **Linear Probing** — trained logistic regression classifiers on hidden states of Qwen 2.5 (1.5B and 7B) to measure implicit knowledge
2. **Prompt Engineering** — designed explicit prompts to elicit model self-reports on classification confidence
3. **Explicit vs Implicit Control** — measured alignment between prompted outputs and hidden state predictions
4. **Effect Size** — used Cohen's d to quantify separation between class representations in hidden state space

## Key Finding

**BoolQ fails** — the model shows near-zero effect size on BoolQ, indicating no meaningful semantic axis exists for yes/no QA in the hidden states. This is a genuine AI reliability boundary: metacognitive prompting cannot improve performance when the underlying representation is absent.

Sentiment and toxicity tasks show strong hidden state separation (high Cohen's d), making them amenable to metacognitive control. Morality tasks show intermediate results.

## Models

- Qwen 2.5 1.5B
- Qwen 2.5 7B

## Tech Stack

- Python, PyTorch, HuggingFace Transformers
- Scikit-learn for linear probing
- NumPy, Pandas, Matplotlib
- Jupyter Notebooks

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/metacognition_analysis.ipynb
```

## Files

- `notebooks/` — full analysis pipeline
- `data/` — dataset loading scripts
- `requirements.txt` — dependencies

## References

- Original paper: "Neurofeedback-Driven Metacognition in Language Models"
- Qwen 2.5: [HuggingFace](https://huggingface.co/Qwen)

---
