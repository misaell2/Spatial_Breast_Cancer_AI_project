"""
Milestone 1: Load 10x Visium breast cancer data and perform initial QC.

Purpose
-------
This script is the first checkpoint in the project. It loads the public
10x Genomics Visium breast cancer dataset, checks that the spatial data loaded
correctly, calculates basic quality-control metrics, and generates initial
figures.

Input
-----
Expected local data folder:

    data/raw/10x/Visium_Human_Breast_Cancer/

Expected files inside that folder:

    Visium_Human_Breast_Cancer_filtered_feature_bc_matrix.h5
    spatial/

The `spatial/` directory should contain files such as:

    tissue_hires_image.png
    tissue_lowres_image.png
    scalefactors_json.json
    tissue_positions_list.csv or tissue_positions.csv

Output
------
This script saves:

    data/processed/visium_human_breast_cancer_raw_qc.h5ad

and figures in:

    results/figures/01_initial_qc/

Analysis notes
--------------
This script intentionally performs only initial loading and QC. It does not
filter spots, normalize counts, run PCA, or cluster the data. Those steps are
handled in:

    src/preprocessing/02_preprocess_cluster.py

Possible alternative choices
----------------------------
- `scanpy.read_visium()` is used here because it works with the current
  environment. In newer workflows, `squidpy.read.visium()` is often preferred.
- Additional QC metrics could include ribosomal gene percentage, hemoglobin
  gene percentage, or spot-level tissue image features.
- More detailed QC could also include spatial inspection of tissue edges,
  spot count distributions, and comparison of high/low-quality tissue regions.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import scanpy as sc


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
# `__file__` points to this script:
#   src/preprocessing/01_load_visium_qc.py
#
# `.parents[2]` moves up two levels:
#   src/preprocessing/ -> src/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Folder containing the downloaded and extracted 10x Visium dataset.
DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "10x"
    / "Visium_Human_Breast_Cancer"
)

# 10x filtered feature-barcode matrix file.
# This is the processed Space Ranger count matrix used for downstream analysis.
COUNT_FILE = "Visium_Human_Breast_Cancer_filtered_feature_bc_matrix.h5"

# Output folder for QC figures.
OUT_DIR = PROJECT_ROOT / "results" / "figures" / "01_initial_qc"

# Output folder for processed AnnData files.
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Create output directories if they do not already exist.
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def save_current_fig(filename: str) -> None:
    """
    Save the active matplotlib figure and close it.

    Parameters
    ----------
    filename:
        Name of the figure file to save inside OUT_DIR.

    Why close the figure?
    ---------------------
    Closing figures prevents later plots from overlapping and avoids memory
    buildup when multiple figures are generated in one script.
    """
    out_path = OUT_DIR / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {out_path}")


def main() -> None:
    """
    Load Visium data, calculate initial QC metrics, and save QC outputs.

    Workflow
    --------
    1. Load the 10x Visium dataset.
    2. Confirm expected metadata and spatial fields are present.
    3. Identify mitochondrial genes.
    4. Calculate basic QC metrics.
    5. Save a raw-QC AnnData checkpoint.
    6. Generate QC and marker-gene spatial plots.
    """
    print(f"Loading Visium data from: {DATA_DIR}")

    # ------------------------------------------------------------------
    # 1. Load 10x Visium data
    # ------------------------------------------------------------------
    # `sc.read_visium` reads:
    #   - the filtered expression matrix
    #   - spatial coordinates
    #   - tissue image files
    #   - scale factors
    #
    # `library_id` is used by Scanpy to store and retrieve the spatial image
    # information under `adata.uns["spatial"][library_id]`.
    #
    # Alternative:
    #   Newer spatial workflows often use `squidpy.read.visium`, but Squidpy
    #   was skipped early in this project because the local install was slow.
    adata = sc.read_visium(
        path=DATA_DIR,
        count_file=COUNT_FILE,
        library_id="Visium_Human_Breast_Cancer",
        load_images=True,
    )

    # Ensure gene names are unique.
    # Some downstream Scanpy operations require unique variable names.
    # If duplicate gene symbols exist, Scanpy appends suffixes to make them
    # unique.
    adata.var_names_make_unique()

    # ------------------------------------------------------------------
    # 2. Print basic object information
    # ------------------------------------------------------------------
    # AnnData stores:
    #   - adata.X: expression matrix
    #   - adata.obs: spot/cell metadata
    #   - adata.var: gene metadata
    #   - adata.obsm["spatial"]: spatial coordinates
    #   - adata.uns["spatial"]: image and scale-factor metadata
    print("\nLoaded AnnData object:")
    print(adata)

    print("\nObservation metadata columns:")
    print(adata.obs.columns.tolist())

    print("\nVariable metadata columns:")
    print(adata.var.columns.tolist())

    print("\nSpatial keys:")
    print(adata.uns["spatial"].keys())

    print("\nSpatial coordinate matrix shape:")
    print(adata.obsm["spatial"].shape)

    # ------------------------------------------------------------------
    # 3. Identify mitochondrial genes
    # ------------------------------------------------------------------
    # Human mitochondrial genes usually begin with "MT-", such as:
    #   MT-CO1, MT-ND1, MT-CYB
    #
    # Mitochondrial percentage is a common QC metric. High mitochondrial
    # fraction can indicate stressed, damaged, or low-quality spots/cells.
    #
    # Note:
    #   In spatial transcriptomics, mitochondrial signal can also reflect
    #   tissue biology or local metabolic state, so it should not be used too
    #   aggressively without spatial inspection.
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")

    # ------------------------------------------------------------------
    # 4. Calculate QC metrics
    # ------------------------------------------------------------------
    # This adds useful columns to `adata.obs`, including:
    #   total_counts       = total UMI counts per spot
    #   n_genes_by_counts  = number of detected genes per spot
    #   total_counts_mt    = mitochondrial counts per spot
    #   pct_counts_mt      = percent mitochondrial counts per spot
    #
    # `percent_top=None` skips calculating percent counts in top N genes.
    # Alternative:
    #   percent_top=[50, 100, 200] can be useful to detect spots dominated by
    #   a small number of highly expressed genes.
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    # Print descriptive statistics to guide filtering thresholds in the next
    # preprocessing script.
    print("\nQC summary:")
    print(adata.obs[["total_counts", "n_genes_by_counts", "pct_counts_mt"]].describe())

    # ------------------------------------------------------------------
    # 5. Save raw-QC AnnData checkpoint
    # ------------------------------------------------------------------
    # This file is the input to Milestone 2:
    #   src/preprocessing/02_preprocess_cluster.py
    #
    # It contains raw counts plus QC metrics, but it is not yet filtered,
    # normalized, or clustered.
    out_h5ad = PROCESSED_DIR / "visium_human_breast_cancer_raw_qc.h5ad"
    adata.write_h5ad(out_h5ad)
    print(f"\nSaved QC AnnData object to: {out_h5ad}")

    # ------------------------------------------------------------------
    # 6. Plot QC distributions
    # ------------------------------------------------------------------
    # Violin plots summarize key QC metrics across all spots.
    #
    # Interpretation:
    #   - Low total_counts may indicate weak capture or tissue-free spots.
    #   - Low n_genes_by_counts may indicate low-complexity spots.
    #   - High pct_counts_mt may indicate stress/damage or strong metabolic
    #     signal.
    #
    # Alternative plots:
    #   - histograms
    #   - density plots
    #   - scatter total_counts vs n_genes_by_counts
    #   - spatial QC maps by tissue region
    sc.pl.violin(
        adata,
        keys=["total_counts", "n_genes_by_counts", "pct_counts_mt"],
        jitter=0.4,
        multi_panel=True,
        show=False,
    )
    save_current_fig("qc_violin.png")

    # ------------------------------------------------------------------
    # 7. Plot QC metrics on tissue coordinates
    # ------------------------------------------------------------------
    # Spatial QC plots help determine whether low-quality spots are randomly
    # distributed or localized to tissue edges, folds, damaged regions, or
    # areas outside tissue.
    #
    # Alternative:
    #   Squidpy's `sq.pl.spatial_scatter` can be used in workflows where
    #   Squidpy is installed.
    sc.pl.spatial(
        adata,
        color=["total_counts", "n_genes_by_counts", "pct_counts_mt"],
        library_id="Visium_Human_Breast_Cancer",
        show=False,
    )
    save_current_fig("spatial_qc_metrics.png")

    # ------------------------------------------------------------------
    # 8. Plot a small set of biological marker genes
    # ------------------------------------------------------------------
    # These markers provide an early sanity check that biologically relevant
    # signals are spatially structured.
    #
    # Marker examples:
    #   EPCAM  = epithelial/tumor-associated
    #   KRT18  = epithelial marker
    #   COL1A1 = stromal/fibroblast/extracellular matrix
    #   PTPRC  = pan-immune marker, also known as CD45
    #   CD3D   = T-cell marker
    #   MKI67  = proliferation marker
    #
    # These are not final annotations. They are only an initial biological
    # inspection before clustering and formal marker-gene analysis.
    marker_genes = [
        "EPCAM",
        "KRT18",
        "COL1A1",
        "PTPRC",
        "CD3D",
        "MKI67",
    ]

    # Some requested markers may not be present in the dataset after gene-name
    # processing. Only plot genes that are available.
    available_markers = [gene for gene in marker_genes if gene in adata.var_names]

    print("\nAvailable marker genes:")
    print(available_markers)

    # Generate spatial marker plots only if at least one marker is available.
    if available_markers:
        sc.pl.spatial(
            adata,
            color=available_markers,
            library_id="Visium_Human_Breast_Cancer",
            show=False,
        )
        save_current_fig("spatial_marker_genes.png")

    print("\nDone. Initial QC figures saved to:")
    print(OUT_DIR)

print("\nMilestone 1 complete.")


if __name__ == "__main__":
    main()
