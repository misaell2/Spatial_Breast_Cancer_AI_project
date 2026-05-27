# Spatial_Breast_Cancer_project

## SpatialNicheAI: Cloud-based ML analysis of breast cancer spatial transcriptomics

This project explores how spatial transcriptomics can be used to identify and interpret biologically meaningful tumor microenvironment niches in breast cancer tissue.

The goal is to build a reproducible AWS-based bioinformatics and machine learning workflow that can analyze public 10x Genomics Visium breast cancer data, classify spatial regions into interpretable biological niches, and validate niche-associated gene signatures using public TCGA-BRCA bulk RNA-seq data.

---

## Project Motivation

Breast tumors are not uniform. Tumor epithelial cells, immune cells, stromal fibroblasts, endothelial cells, and proliferative regions can occupy distinct spatial regions within the same tissue section. These spatial relationships are biologically important because they may reflect tumor invasion, immune infiltration, immune exclusion, stromal remodeling, and tumor microenvironment organization.

This project asks:

> Can spatial transcriptomics reveal distinct tumor, immune, and stromal niches in breast cancer tissue, and can machine learning automate the biological interpretation of these spatial regions?

---

## Main Objectives

1. Analyze public breast cancer spatial transcriptomics data.
2. Identify spatially organized tumor microenvironment niches.
3. Assign biological interpretations using marker genes and pathway enrichment.
4. Build a machine learning model that predicts spatial niche labels from gene expression and spatial-neighborhood features.
5. Validate niche-associated gene signatures using TCGA-BRCA bulk RNA-seq and clinical metadata.
6. Deploy the analysis as a reproducible cloud workflow using AWS, Docker, and workflow orchestration.

---

## Biological Questions

This project focuses on several biological questions:

1. What spatially distinct regions are present in breast cancer tissue?
2. Which genes and pathways define tumor, stromal, immune, and proliferative regions?
3. Are immune-rich or stromal-rich regions spatially adjacent to tumor-rich regions?
4. Can a machine learning model classify biological niches from spatial transcriptomics data?
5. Do spatial niche signatures appear in TCGA-BRCA bulk RNA-seq samples?
6. Are immune, stromal, or proliferative signatures associated with breast cancer subtype or clinical features?

---

## Planned Datasets

### Primary spatial transcriptomics datasets

Public 10x Genomics Visium breast cancer datasets:

- Human Breast Cancer: Visium Fresh Frozen, Whole Transcriptome
  - Invasive ductal carcinoma
  - ER positive, PR negative, HER2 2+
  - 10x Genomics public dataset
  - URL: https://www.10xgenomics.com/datasets/human-breast-cancer-visium-fresh-frozen-whole-transcriptome-1-standard

- Human Breast Cancer: Ductal Carcinoma In Situ and Invasive Carcinoma, FFPE
  - FFPE breast cancer tissue annotated with DCIS and invasive carcinoma
  - 10x Genomics public dataset
  - URL: https://www.10xgenomics.com/datasets/human-breast-cancer-ductal-carcinoma-in-situ-invasive-carcinoma-ffpe-1-standard-1-3-0

### Validation dataset

- TCGA-BRCA
  - Bulk RNA-seq and clinical metadata for breast cancer samples
  - Accessed through the NCI Genomic Data Commons
  - URL: https://portal.gdc.cancer.gov/projects/TCGA-BRCA

---

## Methods Overview

The workflow will include:

1. Data ingestion from public sources
2. Storage of raw and processed files in AWS S3
3. Quality control of spatial transcriptomics data
4. Normalization and feature selection
5. Dimensionality reduction and clustering
6. Marker gene detection
7. Biological niche annotation
8. Pathway enrichment analysis
9. Spatial neighborhood analysis
10. Machine learning model training
11. Model evaluation and interpretation
12. TCGA-BRCA signature validation
13. Automated report generation

---

## Machine Learning Component

The ML component, called **SpatialNicheAI**, is designed to automate biological interpretation of spatial transcriptomics data.

### Model task

Given a Visium spatial transcriptomics sample, predict a biological niche label for each spatial spot.

Example niche labels:

- Tumor epithelial
- CAF/stromal
- Immune-enriched
- T-cell-enriched
- Myeloid/macrophage-enriched
- Proliferative tumor
- Endothelial/vascular
- Hypoxic or stress-associated
- Mixed or uncertain

### Input features

The model will use features derived from:

- Gene expression
- PCA or latent embeddings
- Marker gene signature scores
- Spatial-neighborhood averages
- Spatial graph features
- Optional H&E image patch features in a future extension

### Planned models

Baseline models:

- Logistic regression
- Random forest
- XGBoost or LightGBM

Spatial models:

- Graph-based feature model
- Graph neural network using spatial spot adjacency

### Labeling strategy

Because public spatial datasets may not contain curated spot-level labels, this project will use a weak supervision strategy:

1. Cluster spots using unsupervised analysis.
2. Identify marker genes for each cluster.
3. Assign provisional biological labels based on known cell-type and pathway markers.
4. Train models to reproduce and generalize these marker-informed labels.
5. Evaluate predictions using held-out samples, marker enrichment, and spatial coherence.

---

## Example Marker Genes

| Niche | Example markers |
|---|---|
| Tumor epithelial | EPCAM, KRT8, KRT18, KRT19 |
| T cells | CD3D, CD3E, CD8A |
| B cells | MS4A1, CD79A |
| Myeloid/macrophage | LST1, C1QA, C1QB, CD68 |
| Fibroblast/CAF | COL1A1, COL1A2, DCN, LUM, ACTA2 |
| Endothelial | PECAM1, VWF, KDR |
| Proliferation | MKI67, TOP2A, PCNA |
| Hypoxia/stress | VEGFA, CA9, HIF1A-associated genes |

---

## Planned AWS Architecture

The project will be built as a reproducible cloud workflow.

```text
Public datasets
      |
      v
AWS S3 raw data bucket
      |
      v
Dockerized analysis environment
      |
      v
EC2 / AWS Batch compute
      |
      v
Nextflow or Snakemake workflow
      |
      v
Preprocessing + feature engineering
      |
      v
ML model training and inference
      |
      v
Pathway enrichment + spatial analysis
      |
      v
Automated HTML report
      |
      v
AWS S3 results bucket
