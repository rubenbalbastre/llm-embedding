# llm-embedding

## Overview
This project trains an encoder-only embedding model (based on Qwen2.5) for semantic retrieval in the astrophysics domain. The goal is query-to-document retrieval: given a short query (typically a paper title), retrieve the most relevant astrophysics documents from a corpus of arXiv articles. Training uses contrastive InfoNCE loss and evaluation compares against OpenAI's general-purpose embedding model.

## Dataset
Data is pulled from the arXiv API using multiple categories:
- Main topic: `gr-qc` (general relativity and quantum cosmology)
- Soft negatives: `hep-th`, `astro-ph.CO`
- Hard negatives: `math-ph`, `cond-mat.stat-mech`
- Cross-domain negatives: `cs.LG`, `cs.AI`, `stat.ML`
- Different topics: `q-bio`

Each item includes `title`, `abstract`, and `topic`. Datasets are concatenated with Hugging Face `datasets` and tokenized with the Qwen2.5 tokenizer. Abstract token lengths are inspected to inform dynamic batching.

## Model
- Backbone: `Qwen/Qwen2.5-0.5B-Instruct` (AutoModel, no LM head)
- Pooling: masked average pooling over the last hidden state
- Fine-tuning: LoRA adapters injected into linear layers
- Mixed precision: bfloat16 for efficiency

## Training
- Contrastive InfoNCE (title ↔ abstract)
- Gradient accumulation to reach an effective batch size
- Gradient clipping (norm 1.0)
- Optimizer: AdamW
- TensorBoard logging for loss curves

## Evaluation
Retrieval quality is measured with ID-based metrics using `ragas`:
- Recall
- Precision
- MRR

Evaluation runs two baselines:
1. OpenAI embeddings (e.g. `text-embedding-3-small`)
2. The custom LoRA-tuned Qwen2.5 embedding model

FAISS is used to build an index over abstract embeddings; title embeddings are used as queries.

## Notebook Flow (experiment.ipynb)
1. Create and concatenate arXiv datasets
2. Tokenize and analyze abstract lengths
3. Build custom category-aware batches
4. Define encoder + masked pooling + LoRA
5. Train with contrastive loss
6. Evaluate with FAISS + ragas metrics

## Notes / Limitations
- The notebook is written for interactive execution and assumes GPU availability for speed.
- Some cells rely on Colab (e.g. Google Drive checkpointing).
- Evaluation code for OpenAI embeddings assumes `OPENAI_API_KEY` is set.
