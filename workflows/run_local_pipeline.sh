#!/usr/bin/env bash

set -euo pipefail

echo "Running Spatial_Breast_Cancer_AI_project local workflow"

echo "Step 1: Load Visium data and calculate QC metrics"
python src/preprocessing/01_load_visium_qc.py

echo "Step 2: Preprocess, normalize, cluster"
python src/preprocessing/02_preprocess_cluster.py

echo "Step 3: Marker gene analysis"
python src/analysis/03_marker_gene_analysis.py

echo "Step 4: Apply manual annotations"
python src/analysis/04_apply_manual_annotations.py

echo "Step 5: Train baseline niche classifier"
python src/modeling/05_train_baseline_niche_classifier.py

echo "Step 6: Run spatial holdout validation"
python src/modeling/06_spatial_holdout_validation.py

echo "Workflow complete"
