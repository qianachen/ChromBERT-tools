# ChromBERT-tools: Utilities for ChromBERT-based regulatory analysis

**ChromBERT-tools** is a lightweight companion library built on top of [ChromBERT](https://github.com/TongjiZhanglab/ChromBERT). 
ChromBERT is a pre-trained deep learning model designed to capture the genome-wide co-association patterns of approximately one thousand transcription regulators, thereby enabling accurate representations of context-specific transcriptional regulatory networks (TRNs)

ChromBERT-tools provides convenient command-line and Python utilities for computing ChromBERT-based representations of genomic regions, genes, and regulators, inferring transcriptional regulatory networks (TRNs), imputing missing cistromes, and comparing regulatory programs between cell types, conditions, or functional region sets to identify candidate driver regulators.


This toolkit is intended for users who already have ChromBERT installed and would like to:

- generate ChromBERT-based embeddings for genomic regions;
- generate ChromBERT-based embeddings for genes specified by gene symbols or IDs;
- generate ChromBERT-based embeddings for transcription factors / regulators;
- generate ChromBERT-based embeddings for cistromes;
- generate cell-specific ChromBERT-based embeddings;
- infer cell-specific transcriptional regulatory networks;
- impute missing cistromes for regulators or contexts that lack direct ChIP-like assays;
- rank putative driver regulators underlying specific cell-state transitions;
- identify regulators whose context-specific embeddings best discriminate between two sets of functional regions.