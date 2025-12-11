# ChromBERT-tools: Utilities for ChromBERT-based regulatory analysis

> **ChromBERT** is a pre-trained deep learning model designed to capture genome-wide co-association patterns of ~1,000 transcription regulators and to learn context-specific transcriptional regulatory networks (TRNs) [ChromBERT](https://github.com/TongjiZhanglab/ChromBERT).  
> **ChromBERT-tools** is a lightweight companion library that wraps these capabilities into easy-to-use command-line and Python utilities.

**ChromBERT-tools v1.0 will be released on December 26, 2025**

---

## What you can do with ChromBERT-tools

### 1. General representations

- **Region embeddings** – generate ChromBERT-based embeddings for genomic regions.  
  - Tool: `chrombert_embedding_regions`
- **Gene embeddings** – generate embeddings for genes specified by gene symbols or IDs (using TSS windows).  
  - Tool: `chrombert_embedding_gene_tss`
- **Regulator embeddings** – generate embeddings for transcription factors / regulators.  
  - Tool: `chrombert_embedding_regulators`
- **Cistrome embeddings** – generate embeddings for cistromes.  
  - Tool: `chrombert_embedding_cistromes`
- **TRN inference (bulk / context-level)** – infer transcriptional regulatory networks over user-specified regions.  
  - Tool: `chrombert_inferring_trn`

### 2. Cell- and condition-specific representations

- **Cell-specific region embeddings** – adapt region embeddings to a given cell type or condition.  
  - Tool: `chrombert_embedding_cell_regions`
- **Cell-specific gene embeddings** – adapt gene (TSS-based) embeddings for specific cell types or conditions.  
  - Tool: `chrombert_embedding_cell_gene_tss`
- **Cell-specific regulator embeddings** – adapt regulator embeddings to a given cell type or condition.  
  - Tool: `chrombert_embedding_cell_regulators`
- **Cell-specific TRNs** – infer transcriptional regulatory networks conditioned on a specific cell type or perturbation.  
  - Tool: `chrombert_inferring_cell_trn`

### 3. Cistrome imputation

- **Impute missing cistromes** – predict cistromes for regulators or contexts that lack direct ChIP-like assays.  
  - Tool: `chrombert_imputing_cistromes`

### 4. Dynamic analysis and driver regulator discovery

- **Cell-state transitions** – rank putative driver regulators underlying specific transitions between cell types or conditions.  
  - Tool: `chrombert_finding_driver_factors_in_cell_state_transition`
- **Functional region contrasts** – identify regulators whose context-specific embeddings best discriminate between two sets of functional regions.  
  - Tool: `chrombert_finding_driver_factors_in_functional_regions`
