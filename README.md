
# **Metacognitive Reporting and Control in Large Language Models**  
### *A Replication and Multi-Dataset Extension of “Neurofeedback-Driven Metacognition in Language Models”*

This repository contains a structured replication and extension of the metacognitive **reporting** and **control** framework introduced in:

> **Neurofeedback-Driven Metacognition in Language Models**  
> https://pmc.ncbi.nlm.nih.gov/articles/PMC12136483/pdf/nihpp-2505.13763v1.pdf  

The original work evaluates metacognitive abilities primarily on the **ETHICS – Commonsense Morality** dataset.  
Here, I **replicate** the methodology on ETHICS and **extend** it to four additional binary NLP tasks:

- **SST2** — Sentiment  
- **IMDB** — Sentiment  
- **TweetEval-Offensive** — Toxicity  
- **BoolQ (reformatted)** — Yes/No question answering  

This allows us to test whether reporting and control effects generalize beyond morality.

---

# **Repository Structure**

```
metacognition_LLM/
│
├── ethics.ipynb               # ETHICS replication: axis learning, reporting & control
├── meta3.ipynb                # Multi-dataset experimental pipeline
├── clean_notebooks.py         # Script to clean notebook metadata for GitHub
├── figures/                   # (recommended) place images here
└── README.md
```

---

# **1. Methodology Summary**

Follows the structure from the original paper:  
axis → reporting → explicit control → implicit control.

---

## **1.1 Axis Learning (Linear Probe)**

- Extract hidden states from a **single target layer** (primarily `layer = -1`).
- Train a **balanced logistic regression** classifier.
- Normalize the weight vector to obtain a **semantic axis**.

Score = dot product:  
\[
s = \langle h, w \rangle
\]

---

## **1.2 Reporting Task**

The model is shown N labeled in-context examples:

```
Assistant: text [Score:{1}]
```

Then asked to report the score of a new text.

**Ground-truth:**  
Hidden state at the *prompt-position* token `{` inside `[Score:{`  
(projected onto the axis and thresholded).

Metrics:

- Accuracy vs N  
- Cross-entropy vs N  

---

## **1.3 Control Task**

### **Explicit Control**
Model is directly asked:

> “Now imitate label {0/1}.”

We then:

- Generate multiple samples  
- Embed the last token  
- Project onto axis  
- Compare distributions with **Cohen’s d**

### **Implicit Control**
Model is only given:

> “Respond in the style of examples with score {1}, but do not mention labels.”

No generation needed — only embed the last token.

Expected:

- Explicit → **large effect**
- Implicit → **weak / negligible effect**

---

# **2. Experimental Setup**

## **2.1 Models**

- Qwen/Qwen2.5-1.5B-Instruct  
- Qwen/Qwen2.5-7B-Instruct *(final experiments use 7B)*  

Layer used: **–1**

---

## **2.2 Datasets**

| Dataset | Task | Notes |
|---------|------|--------|
| ETHICS | morality | baseline dataset from the paper |
| SST2 | sentiment | very polarity-aligned |
| IMDB | sentiment | long reviews, rich embedding signal |
| Offensive | toxicity | clear lexical cues |
| BoolQ | yes/no QA | *semantically weak latent axis* |

---

# **3. Results**

(Replace the example filenames with yours.)

---

## **3.1 Reporting – ETHICS**

Accuracy vs N  
![ETHICS Reporting Accuracy](figures/ethics_reporting_acc.png)

Cross-entropy vs N  
![ETHICS Reporting CE](figures/ethics_reporting_ce.png)

---

## **3.2 Explicit Control – ETHICS**

![ETHICS Explicit Histogram](figures/ethics_explicit_hist.png)

Strong separation, high Cohen’s d.

---

## **3.3 Implicit Control – ETHICS**

![ETHICS Implicit Histogram](figures/ethics_implicit_hist.png)

Minimal movement → expected.

---

## **3.4 Cross-Dataset Summary**

For **SST2**, **IMDB**, and **Offensive**, results match ETHICS:

- Reporting accuracy improves with N  
- Explicit control → **large d**  
- Implicit control → **weak**  

Example:  
![SST2 Explicit Control](figures/sst2_explicit_hist.png)

---

## **3.5 BoolQ — Notable Exception**

BoolQ is the only dataset where:

- Logistic regression barely exceeds chance  
- Explicit control has *very small* Cohen’s d  
- Implicit control is effectively zero  

### **Interpretation**

BoolQ is fundamentally just:

> generic question → yes/no answer

So:

- No coherent semantic dimension  
- Long passages + short answers = weak axis  
- Qwen hidden states don’t form a stable “yes/no” direction  

As one reviewer summarized:

> “BoolQ answers are almost arbitrary yes/no decisions over generic passages — there is no meaningful axis. Surprising the LR is even above 50%.”

This supports the conclusion that **metacognition emerges only when the underlying latent dimension is meaningful**.

---

# **4. Differences from the Original Paper**

- Original: **ETHICS only**  
  Here: **ETHICS + 4 new datasets**

- Different model family (Qwen 2.5 instead of proprietary models)

- Improvements:
  - Prompt-position internal labels  
  - Z-scored projections  
  - Stronger counterbalancing  
  - Fixed-sentence implicit prompts  
  - Cleaner split logic

Goal: replicate **qualitative** trends, not exact numbers.

---

# **5. How to Run**

Install:

```bash
pip install torch transformers datasets scikit-learn matplotlib nbformat
```

Run experiments via Jupyter:

- `ethics.ipynb`
- `meta3.ipynb`

Outputs: reporting curves + explicit & implicit control histograms.

---

# **6. Citation**

If referencing:

> **Neurofeedback-Driven Metacognition in Language Models**  
> https://pmc.ncbi.nlm.nih.gov/articles/PMC12136483/pdf/nihpp-2505.13763v1.pdf

This repository is an independent extension and is not affiliated with the authors.

---

# **7. Contact**

If you have questions, ideas, or want to discuss extensions, feel free to open an issue.

```
