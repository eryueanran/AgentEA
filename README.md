# AgentEA: Multi-Agent Debate for Reliable Entity Alignment

## Code Description

This section describes the execution order and functionality of the main scripts in the project:

1. **Preprocess.py**:
Data preprocessing. This script translates entity names into English and summarizes entity-related information.
2. **Embedding.py**:
Generates embeddings for the processed entity information. We use [LLM2Vec](https://github.com/McGill-NLP/llm2vec) to convert a pretrained language model into an efficient text encoder.
3. **Build_cand.py**:
Constructs an initial candidate set based on the embedding results, and prepares the corresponding entity information for use in the subsequent reasoning stage.
4. **Reasoning.py**:
Performs reasoning over the initial candidate set and computes the relevant experimental metrics.

## Dataset Description

We provide a small, preprocessed example dataset named `data_examples`, which contains 10 pairs of entities. This dataset includes `name.txt`, `att.txt`, and `rel.txt`, and can be directly used to run `preprocess.py`.

In addition, we also provide a preprocessed initial candidate set containing 500 pairs of entities in `candidates_examples`, which can be directly used by `reasoning.py` for inference. These datasets are intended to help users better understand the data formats required by AgentEA and the overall execution pipeline.

All datasets used in our experiments are standard benchmarks commonly adopted in the entity alignment community. Their sources are listed below:

1. **DBP15K**: [Link](https://github.com/kosugi11037/bert-int)
2. **SRPRS**: [Link](https://github.com/DexterZeng/CEA)
3. **ICEWS**: [Link](https://github.com/DataArcTech/Simple-HHEA)
4. **DWY**: [Link](https://github.com/THUDM/SelfKG)
