# DolphinMind
**Efficient long-context processing for small language models using semantic retrieval**

DolphinMind lets **1–7B parameter models** answer questions from **100k+ token documents** (450k+ tokens tested) with higher accuracy than native 64k (or even 128k) context windows while using **94.5% fewer tokens**.

## Key Results (Varney the Vampire – 329k words / ~453k tokens)

| Method                  | ROUGE-L ↑ | Tokens/Query ↓ | Approach                     |
|-------------------------|-----------|----------------|------------------------------|
| **DolphinMind**         | **0.185** | **3,500**      | Semantic RAG (top-5 chunks)  |
| SmolLM3 (64k native)    | 0.154     | 64,085         | Full context                 |
| RLM-Tools               | 0.154     | 2,100          | Tool-calling                 |
| Truncated (4k)          | 0.146     | 4,000          | Baseline                     |
| Naive Chunking + TF-IDF | 0.135     | 2,800          | Classic retrieval            |

**What we learned:** More context isn’t always better  
SmolLM3 @ 128k context → **0.132 ROUGE-L** (14% worse than 64k)

## Quick Start – Google Colab (no setup)

Just click and run:

1. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/DolphinMind/blob/main/01_load_model_and_profile.ipynb) **01_load_model_and_profile.ipynb** – Load Qwen2.5-3B (4-bit) and test
2. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/DolphinMind/blob/main/02_baseline_comparison.ipynb) **02_baseline_comparison.ipynb** – Compare 3 retrieval methods
3. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/DolphinMind/blob/main/03_rlm_and_more_methods.ipynb) **03_rlm_and_more_methods.ipynb** – Tool-calling, map-reduce, hierarchical
4. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/DolphinMind/blob/main/04_final_evaluation.ipynb) **04_final_evaluation.ipynb** – Full analysis & results

## How DolphinMind Works (4-stage pipeline)

```mermaid
graph LR
    A[Long Document<br>329k words] --> B[Sentence-aware chunking<br>512-word chunks + 128 overlap]
    B --> C[Neural embeddings<br>all-MiniLM-L6-v2]
    C --> D[FAISS top-5 retrieval<br>per query]
    D --> E[Chronological reconstruction<br>≈ 3.5k token context]
    E --> F[Qwen2.5-3B 4-bit]
```

Peak memory usage: ~5.8 GB (model + embeddings + FAISS index)

## Dataset

Varney the Vampire (1847 penny dreadful)
329,160 words ≈ 453k tokens
3 challenging plot & character questions
Human-written reference answers for ROUGE-L evaluation
