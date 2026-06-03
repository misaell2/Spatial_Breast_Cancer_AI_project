# Spatial_Breast_Cancer_AI_project

## SpatialNicheAI: Interpretable ML for breast cancer spatial transcriptomics

SpatialNicheAI is an end-to-end bioinformatics and machine learning project using a public 10x Genomics Visium breast cancer dataset to identify, interpret, and classify spatial tumor microenvironment niches.

The project combines:

- 10x Visium spatial transcriptomics analysis
- marker-gene-based biological interpretation
- manual niche annotation
- supervised machine learning
- spatial holdout validation
- Docker reproducibility
- AWS EC2 + S3 cloud execution

The goal is to show how spatial transcriptomics can reveal biologically meaningful regions in breast cancer tissue and how those annotations can be converted into an interpretable ML workflow.

---

## Project Overview

Breast tumors are spatially heterogeneous. Tumor epithelial regions, immune niches, stromal regions, adipose tissue, and metabolically stressed tumor regions can occupy different areas of the same tissue section.

This project asks:

> Can spatial transcriptomics identify biologically meaningful breast cancer tissue niches, and can machine learning learn to classify those niches from expression-derived features?

The workflow starts with public 10x Visium data and produces annotated spatial niche maps, marker-gene evidence, baseline ML models, spatial holdout validation, and cloud-executed results.

---

## Dataset

Primary dataset:

- **10x Genomics Human Breast Cancer: Visium Fresh Frozen, Whole Transcriptome**
- Invasive ductal carcinoma
- ER positive, PR negative, HER2 2+
- Spatial gene expression data generated with 10x Genomics Visium

Dataset page:

```text
https://www.10xgenomics.com/datasets/human-breast-cancer-visium-fresh-frozen-whole-transcriptome-1-standard
```

Large raw data files are not committed to GitHub. They are downloaded locally or stored in S3 for cloud execution.

---

## Workflow Summary

```text
Public 10x Visium breast cancer data
        |
        v
Load data and calculate QC metrics
        |
        v
Filter spots, normalize counts, select highly variable genes
        |
        v
PCA, UMAP, Leiden clustering
        |
        v
Differential expression and marker-gene analysis
        |
        v
Manual biological niche annotation
        |
        v
Baseline ML model training
        |
        v
Random split validation + spatial holdout validation
        |
        v
Dockerized local workflow + AWS EC2/S3 execution
```

---

## Repository Structure

```text
Spatial_Breast_Cancer_AI_project/
├── README.md
├── Dockerfile
├── environment.yml
├── data_manifest/
│   └── annotations/
│       └── leiden_r06_manual_cluster_annotations.csv
├── docs/
│   ├── figures/       # selected README figures
│   └── tables/        # selected summary tables
├── src/
│   ├── preprocessing/
│   │   ├── 01_load_visium_qc.py
│   │   └── 02_preprocess_cluster.py
│   ├── analysis/
│   │   ├── 03_marker_gene_analysis.py
│   │   └── 04_apply_manual_annotations.py
│   └── modeling/
│       ├── 05_train_baseline_niche_classifier.py
│       └── 06_spatial_holdout_validation.py
├── workflows/
│   └── run_local_pipeline.sh
├── aws/
│   └── EC2_S3_RUNBOOK.md
├── data/       # ignored; local raw/processed data
├── results/    # ignored; generated outputs
└── models/     # ignored; trained model artifacts
```

---

## Biological Interpretation

Leiden clusters were interpreted using:

- cluster-level differential expression
- known breast cancer and tumor microenvironment marker genes
- spatial marker expression patterns
- manual biological review

High-confidence spatial niche labels included:

- **Tumor epithelial**
- **Tumor luminal-like**
- **Hypoxic/metabolic tumor-like**
- **Keratin-high tumor**
- **Luminal/secretory epithelial**
- **Antigen-presenting myeloid**
- **B-cell/plasma-cell immune**
- **Adipocyte/fat-associated**

Low-confidence, mixed, mitochondrial/high-oxidative, or review-needed clusters were excluded from supervised ML training.

![Manual spatial niche labels](docs/figures/01_spatial_manual_niche_labels.png)

### Key biological observations

The spatial niche map suggests several biologically meaningful tissue regions:

- Tumor epithelial and luminal-like tumor regions form coherent spatial domains rather than random scattered spots.
- A hypoxic/metabolic tumor-like region appears near tumor-associated epithelial regions, suggesting localized metabolic stress within the tumor area.
- Immune-associated regions include antigen-presenting myeloid/APC-like areas and B-cell/plasma-cell-enriched areas.
- Adipocyte/fat-associated spots occupy a distinct tissue region, consistent with surrounding breast adipose tissue.
- Mixed epithelial/stromal and stress-like clusters were treated cautiously and excluded from high-confidence ML training.

These interpretations are based on marker genes such as:

| Niche | Example markers |
|---|---|
| Tumor epithelial | EPCAM, KRT8, KRT18, KRT19, MUC1 |
| Luminal/secretory | SCGB2A2, SCGB1D2, GATA3, XBP1 |
| Myeloid/APC | CD74, HLA-DRA, HLA-DPA1, C1QA, C1QB, LYZ |
| B-cell/plasma-cell immune | IGKC, IGHG1, IGHG3, IGLC2, JCHAIN |
| Stromal/CAF | COL1A1, COL1A2, DCN, LUM, ACTA2 |
| Adipocyte/fat-associated | FABP4, PLIN1, ADIPOQ, LPL, CFD |
| Hypoxia/glycolysis | GAPDH, PGK1, TPI1, ENO1, LDHA |
| Proliferation | MKI67, TOP2A, PCNA, MCM5 |

![Manual label confidence](docs/figures/02_spatial_manual_label_confidence.png)

---

## Machine Learning Results

The supervised ML task used manually curated high-confidence spatial niche labels as weak supervision.

### Features

The baseline classifier used:

- PCA coordinates from normalized expression
- QC metrics
- spatial coordinates
- marker signature scores for tumor, immune, myeloid, stromal, adipocyte, proliferation, hypoxia/glycolysis, and luminal/secretory programs

### Models compared

- Logistic regression
- Random forest

### Random split performance

| Evaluation | Model | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|---:|
| Random split | Random forest | 0.9409 | 0.9277 | 0.9392 | 0.9407 |
| Random split | Logistic regression | 0.9247 | 0.9278 | 0.9275 | 0.9248 |

The random forest performed best in the random split baseline.

![Random forest confusion matrix](docs/figures/03_random_forest_confusion_matrix.png)

Feature importance suggested that expression-derived PCs and biologically interpretable marker signature scores both contributed to niche classification.

![Random forest feature importance](docs/figures/04_random_forest_feature_importance.png)

The trained model was also used to predict labels across all Visium spots, including low-confidence/review regions.

![Spatial ML predicted labels](docs/figures/05_spatial_ml_predicted_labels.png)

![Spatial ML prediction confidence](docs/figures/06_spatial_ml_prediction_confidence.png)

---

## Spatial Holdout Validation

Random spot-level train/test splits can overestimate performance in spatial transcriptomics because neighboring spots are correlated.

To test robustness more realistically, the tissue was divided into a **3 × 3 spatial grid**, and the model was evaluated using leave-one-spatial-block-out validation.

![Spatial blocks and training labels](docs/figures/07_spatial_blocks_and_training_labels.png)

Spatial holdout validation showed more variable performance than the random split baseline:

- With spatial coordinates: macro F1 ranged from approximately **0.66 to 0.92**
- Without spatial coordinates: macro F1 ranged from approximately **0.71 to 0.92**

This suggests:

- random spot-level splits can overestimate performance
- model performance depends on tissue-region composition and label balance
- the model learns meaningful expression and marker-signature patterns
- spatially aware validation is important for spatial transcriptomics ML workflows

![Spatial holdout macro F1](docs/figures/08_spatial_holdout_macro_f1.png)

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone git@github.com:misaell2/Spatial_Breast_Cancer_AI_project.git
cd Spatial_Breast_Cancer_AI_project
```

### 2. Download the 10x Visium dataset

Create the expected data folder:

```bash
mkdir -p data/raw/10x/Visium_Human_Breast_Cancer
cd data/raw/10x/Visium_Human_Breast_Cancer
```

Download the public 10x files:

```bash
BASE="https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/Visium_Human_Breast_Cancer"

curl -L --fail --retry 3 -O "$BASE/Visium_Human_Breast_Cancer_filtered_feature_bc_matrix.h5"
curl -L --fail --retry 3 -O "$BASE/Visium_Human_Breast_Cancer_spatial.tar.gz"

tar -xzf Visium_Human_Breast_Cancer_spatial.tar.gz
```

Return to the repo root:

```bash
cd ../../../../
```

Expected layout:

```text
data/raw/10x/Visium_Human_Breast_Cancer/
├── Visium_Human_Breast_Cancer_filtered_feature_bc_matrix.h5
├── Visium_Human_Breast_Cancer_spatial.tar.gz
└── spatial/
```

### 3. Run locally with Docker

Build the Docker image:

```bash
docker build -t spatial-bc-ai:local .
```

Test the environment:

```bash
docker run --rm spatial-bc-ai:local
```

Run the full workflow:

```bash
docker run --rm \
  -v "$PWD":/workspace \
  -w /workspace \
  spatial-bc-ai:local \
  bash workflows/run_local_pipeline.sh
```

Generated outputs will be written to:

```text
data/processed/
results/
models/
```

These folders are ignored by Git.

---

## AWS Cloud Execution

The Dockerized workflow was successfully executed on AWS EC2 using S3 for cloud storage.

Completed AWS components:

- uploaded raw 10x Visium input data and annotation manifests to S3
- launched an Ubuntu EC2 instance for cloud compute
- attached an IAM role to EC2 for secure S3 access
- cloned the GitHub repository onto EC2 using SSH
- built the project Docker image on EC2
- ran the full end-to-end workflow inside Docker
- synced generated results, trained models, logs, and processed AnnData files back to S3

Completed AWS run:

```text
RUN_ID=ec2_run_20260602_170837
```

The AWS workflow is documented in:

```text
aws/EC2_S3_RUNBOOK.md
```

The live AWS resources were not kept running after the milestone to avoid ongoing cloud charges.

---

## Reproducibility Notes

Large generated files are intentionally excluded from Git tracking:

```text
data/
results/
models/
*.h5ad
*.h5
*.tar.gz
```

Selected lightweight figures and summary tables are copied into `docs/` for GitHub display.

This keeps the repository readable while preserving a reproducible workflow.

---

## Current Status

Completed:

- 10x Visium breast cancer data loading and QC
- preprocessing and Leiden clustering
- marker-gene analysis
- manual biological niche annotation
- baseline ML classification
- spatial holdout validation
- Dockerized reproducible workflow
- AWS EC2 + S3 workflow execution
- GitHub documentation with selected figures and results

Planned extensions:

- pathway enrichment analysis
- external validation on additional breast cancer spatial transcriptomics samples
- TCGA-BRCA signature comparison
- single-cell reference deconvolution
- AWS Batch, ECR, or Terraform workflow upgrades

---

## License and Use

This repository is released under the MIT License.

This project is intended for research, education, and portfolio demonstration purposes. It is not intended for clinical diagnosis, treatment selection, or medical decision-making.

Users are responsible for complying with the terms of use of any external datasets analyzed with this code.

---

## References and Related Work

### Dataset

10x Genomics. Human Breast Cancer: Visium Fresh Frozen, Whole Transcriptome.

```text
https://www.10xgenomics.com/datasets/human-breast-cancer-visium-fresh-frozen-whole-transcriptome-1-standard
```

### Spatial transcriptomics analysis

Scanpy spatial transcriptomics tutorial.

```text
https://scanpy.readthedocs.io/en/stable/tutorials/spatial/basic-analysis.html
```

Scanpy `read_visium` documentation.

```text
https://scanpy.readthedocs.io/en/stable/api/scanpy.read_visium.html
```

### Related spatial transcriptomics methods

SpaceMarkers: molecular interaction analysis from spatial transcriptomics.

```text
https://www.sciencedirect.com/science/article/pii/S2405471223000807
```

TESLA: machine learning framework for tissue annotation from spatial transcriptomics.

```text
https://pmc.ncbi.nlm.nih.gov/articles/PMC10246692/
```

### Future validation resource

TCGA-BRCA project page.

```text
https://portal.gdc.cancer.gov/projects/TCGA-BRCA
```

---

## Author

**Misael Lazaro**

Bioinformatics and computational biology researcher with experience in NGS analysis, RNA-seq, ChIP-seq, genome assembly, spatial transcriptomics, clinical variant interpretation, Python/R software development, machine learning, and cloud/HPC bioinformatics workflows.
