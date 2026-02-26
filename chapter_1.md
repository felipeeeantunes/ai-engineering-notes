# Chapter 1 — From Classical ML to Foundation Models

## TL;DR

- Self-supervised learning unlocks training on vast unlabeled data.
- Foundation models are general-purpose; you adapt them per application.
- AI Engineering optimizes systems around models (prompts/RAG/evals), while ML Engineering builds the models.

## 1) The key shift: self-supervised learning

**Self-supervised learning** is one of the core ingredients that enabled the jump from many classical ML systems (task-specific, label-heavy) to today’s large language models.

In self-supervision, the supervision signal comes from the data itself.

- **Next-token prediction (autoregressive LMs)**: given a text prefix, predict the next token.
  - Example: in “I am feeling hungry”, the training objective repeatedly asks the model to predict the next token from context.
- **Masked-token prediction (masked LMs, e.g., BERT)**: hide tokens and predict them.

Why this matters: you can leverage massive amounts of unlabeled text (and other modalities) because you don’t need humans to label every training example.

## 2) Foundation models (and multimodal variants)

**Foundation models** are large, general-purpose models trained on broad data at scale and then adapted to many downstream tasks.

- An LLM is a foundation model focused on text.
- A **multimodal foundation model** (sometimes called an LMM/VLM depending on architecture) is trained over multiple modalities (e.g., text + images).

Multimodal self-supervision example idea:
- Train on (image, text) pairs to learn aligned representations.
- This can support tasks like image classification, captioning, retrieval, and grounding—often with less task-specific labeled data than classical supervised pipelines.

## 3) From task-specific to general-purpose

Classical ML in production often meant separate, narrow models per task:

- sentiment analysis model
- translation model
- intent classifier

Foundation models shift the default toward one general model adapted per use case.

Notes:
- Benchmarks you’ll see in this space include instruction-following datasets/collections (e.g., **Natural Instructions**) and general capability eval suites.

## 4) AI Engineering vs ML Engineering

Working definitions (pragmatic, not absolute):

- **ML Engineering**: build/train models, data pipelines, training infrastructure, evaluation at the model level.
- **AI Engineering**: build applications/systems on top of models; choose models, design prompts, retrieval, tooling, evaluation, safety/quality gates, and product integration.

Why “AI Engineering” grew fast (the “perfect storm”):

- general-purpose capabilities became accessible via APIs and open models
- investment increased
- barrier to building useful apps dropped (you can ship value without training a model from scratch)

## 5) What changes in experimentation

Classical ML experimentation often focuses on:

- feature engineering
- model architecture choice
- hyperparameters
- labeled dataset improvements

Foundation-model-centric experimentation often focuses on:

- model choice (API vs open weights; size; latency/cost)
- prompting patterns (instructions, examples, tool use)
- **RAG** / retrieval strategies (indexing, chunking, ranking)
- sampling parameters (temperature, top-$p$, max tokens)
- eval design (gold sets, regression tests, human review loops)

Common “system” constraints you end up engineering around:

- cost/latency budgets
- privacy/compliance boundaries (what can be sent to a model)
- reliability (regressions when prompts/models change)

## 6) Model adaptation (two big buckets)

### A) Adaptation without changing weights

- prompt engineering (instructions, few-shot examples)
- prompt optimization (systematized prompt search / templates)
- adding context (RAG, memory, tools, structured inputs)

### B) Adaptation by changing weights

- fine-tuning (SFT) for style, format, domain patterns
- preference tuning / alignment-style training (when you control training)
- distillation (teacher → smaller student model)

Rule of thumb: prefer the cheapest lever that meets quality (often starts with better prompts + retrieval + evaluation before weight updates).

## 7) Training lifecycle vocabulary (useful to be precise)

- **Pre-training**: start from (effectively) random initialization and learn general patterns from large-scale data.
- **Fine-tuning**: continue training from an existing checkpoint for a narrower objective/dataset.
- **Post-training**: umbrella term for “after pre-training” stages (often includes instruction tuning and preference/alignment training). People use it loosely—clarify what stage they mean.

## 8) Use cases to keep in mind

- AI-generated books (quality varies; workflows matter)
- multi-agent / society simulations (e.g., Park et al., 2023)
- information aggregation and synthesis (high value, but needs strong eval + source grounding)

## 9) Fun numbers / sanity checks

- Tokenization is model-dependent; a rough heuristic in English: **1 token ≈ 0.75 words** (varies by language and text).
- Labeling cost intuition: if one label costs $0.05, then 1M labels cost ~$50k; double if you require independent cross-check.
- GPT (2018): ~117M parameters
- GPT-2 (2019): ~1.5B parameters

## 10) Fun facts + questions to revisit

- **BERT** = Bidirectional Encoder Representations from Transformers (masked LM; strong for non-generative tasks).
- **Autoregressive** LMs are the dominant generation paradigm.
- **AlexNet** (supervised) kicked off the deep learning boom in vision; revisit: what changed (data, compute, architecture, regularization)?
- “Large” usually refers to parameter count, but quality is not guaranteed—scaling laws have caveats (data quality, training recipe, evaluation domain).
- **CLIP** (2021) uses contrastive training over image-text pairs; it’s primarily an embedding model, not a generative model.
- Foundation models tend to be **open-ended** (outputs not enumerated upfront), unlike many classical “closed set” classifiers.

## 11) Reliability and failure modes (practical)

- hallucinations and overconfident errors → mitigate with grounding (RAG), tool use, and evals
- prompt brittleness → mitigate with templates, tests, and change control
- data leakage / privacy issues → mitigate with policy, redaction, and approved runtimes

## 12) Mini glossary

- **Self-supervised learning**: supervision derived from the input itself.
- **Autoregressive LM**: generates by predicting the next token iteratively.
- **Masked LM**: predicts masked tokens given surrounding context.
- **Foundation model**: broadly trained model adaptable to many tasks.
- **RAG**: Retrieval-Augmented Generation; generate with retrieved external context.
- **Distillation**: train a smaller model to mimic a larger one.

## 13) References to look up later

- Park et al. (2023): “Generative Agents” (multi-agent simulation)
- BERT (Devlin et al., 2018)
- GPT (Radford et al., 2018)
- CLIP (Radford et al., 2021)
