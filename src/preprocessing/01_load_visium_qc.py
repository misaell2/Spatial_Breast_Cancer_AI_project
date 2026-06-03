"""
Milestone 1: Load 10x Visium breast cancer data and perform initial QC.

Purpose
-------
This script is the first checkpoint in the SpatialNicheAI workflow.

It loads the public 10x Genomics Visium breast cancer dataset, confirms that
the spatial metadata and image information loaded correctly, calculates basic
quality-control metrics, and generates first-pass QC and marker-gene figures.

Why Squidpy is used here
------------------------
This project originally used `scanpy.read_visium()`, which worked well for
loading the 10x Visium data. Now that Squidpy has been added to the project,
this script uses `squidpy.read.visium()` instead.

Squidpy is useful because it is built specifically for spatial omics workflows.
For this script, the practical benefit is modest: it loads the same 10x Visium
components used by Scanpy, including:

    - count matrix
    - spatial coordinates
    - tissue images
    - scale factors from the `spatial/` directory

The larger advantage comes in later milestones, where Squidpy can be used for:

    - spatial neighbor graph construction
    - neighborhood enrichment analysis
    - niche co-occurrence analysis
    - spatial autocorrelation of genes/signatures
    - H&E image feature extraction

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

Possible alternative analysis choices
-------------------------------------
QC metrics:
    - ribosomal gene percentage
    - hemoglobin gene percentage
    - percent counts in top expressed genes
    - spot-level image features from H&E tissue morphology

QC plots:
    - histograms
    - density plots
    - total_counts vs n_genes_by_counts scatter plots
    - spatial plots of QC metrics over tissue
    - spatial inspection of tissue edges and damaged regions

Spatial analysis:
    - Squidpy spatial neighbor graph can be added after QC
    - spatial autocorrelation could be used to identify spatially patterned
      QC metrics or marker genes
"""

from pathlib import Path

import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq


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
#
# Squidpy's parameter name is `counts_file`.
# Scanpy's older reader uses `count_file`.
COUNT_FILE = "Visium_Human_Breast_Cancer_filtered_feature_bc_matrix.h5"

# Stable library ID used throughout the project.
# This key is stored inside adata.uns["spatial"] and is needed for spatial plots.
LIBRARY_ID = "Visium_Human_Breast_Cancer"

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

    Why this helper exists
    ----------------------
    This script creates several figures. Centralizing the save logic keeps
    output settings consistent and avoids repeating save/close code.

    Parameters
    ----------
    filename:
        Name of the figure file to save inside OUT_DIR.

    Alternative parameters
    ----------------------
    dpi=150:
        Smaller files for quick debugging.

    dpi=300:
        Good balance for GitHub, reports, and presentations. Used here.

    dpi=600:
        Print-quality figures, but much larger files.
    """
    out_path = OUT_DIR / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {out_path}")


def validate_visium_input_files() -> None:
    """
    Confirm that the expected Visium input files exist before loading.

    Why this function exists
    ------------------------
    A missing count matrix or missing `spatial/` folder will cause the reader
    to fail. Checking upfront gives a clearer error message and helps users
    fix the data layout quickly.

    Expected layout
    ---------------
    data/raw/10x/Visium_Human_Breast_Cancer/
    ├── Visium_Human_Breast_Cancer_filtered_feature_bc_matrix.h5
    └── spatial/

    Alternative
    -----------
    If the workflow later supports multiple 10x samples, this function could
    be generalized to validate any dataset path and count filename.
    """
    count_path = DATA_DIR / COUNT_FILE
    spatial_dir = DATA_DIR / "spatial"

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory does not exist: {DATA_DIR}")

    if not count_path.exists():
        raise FileNotFoundError(f"Count matrix not found: {count_path}")

    if not spatial_dir.exists():
        raise FileNotFoundError(f"Spatial directory not found: {spatial_dir}")


def load_visium_with_squidpy():
    """
    Load the 10x Visium dataset using Squidpy.

    Why Squidpy is used here
    ------------------------
    Squidpy's `read.visium()` is designed for 10x Visium-formatted spatial
    transcriptomics data. It reads the expression matrix and also looks for the
    `spatial/` directory so that images, spatial coordinates, and scale factors
    are available in the AnnData object.

    The resulting AnnData object is still compatible with Scanpy. This is
    useful because the project uses:

        - Squidpy for spatial-aware loading/plotting/analysis
        - Scanpy for standard preprocessing and QC metrics

    Important parameter choices
    ---------------------------
    path=DATA_DIR:
        Points to the directory containing the 10x count file and spatial
        directory.

    counts_file=COUNT_FILE:
        Tells Squidpy which matrix file to load. The filtered matrix is used
        because this project starts from 10x's filtered Space Ranger output.

    library_id=LIBRARY_ID:
        Gives the spatial library a stable name. This is helpful for plotting
        and for future multi-sample workflows.

    load_images=True:
        Loads the tissue images and scale factors. This is important because
        the project relies on spatial plots over the tissue image.

    Alternative parameter choices
    -----------------------------
    counts_file="raw_feature_bc_matrix.h5":
        Could be used if starting from the raw Space Ranger matrix instead of
        the filtered matrix.

    load_images=False:
        Faster and lighter, but not appropriate here because spatial tissue
        plots are central to the project.

    source_image_path=<path>:
        Can be used when a higher-quality image is stored outside the standard
        10x `spatial/` directory.
    """
    adata = sq.read.visium(
        path=DATA_DIR,
        counts_file=COUNT_FILE,
        library_id=LIBRARY_ID,
        load_images=True,
    )

    # Ensure gene names are unique.
    #
    # Why this matters:
    # Some downstream Scanpy and AnnData operations expect unique var_names.
    # If duplicate gene symbols are present, Scanpy appends suffixes to make
    # them unique.
    adata.var_names_make_unique()

    return adata


def add_basic_qc_metrics(adata):
    """
    Add basic QC metrics to the AnnData object.

    Why these QC metrics are included
    ---------------------------------
    total_counts:
        Total UMI counts per spot. Very low counts may indicate weak capture,
        tissue-free spots, or poor-quality regions.

    n_genes_by_counts:
        Number of detected genes per spot. Low values may indicate low
        complexity or poor-quality spots.

    pct_counts_mt:
        Percent of counts from mitochondrial genes. High mitochondrial signal
        can indicate tissue stress or damage, but in spatial transcriptomics it
        can also reflect biology or local metabolic state.

    Why this step still uses Scanpy
    -------------------------------
    Squidpy and Scanpy work together in the scverse ecosystem. Squidpy focuses
    on spatial functionality, while Scanpy provides mature preprocessing and QC
    functions such as `sc.pp.calculate_qc_metrics()`.

    Alternative QC parameters
    -------------------------
    percent_top=[50, 100, 200]:
        Adds the percent of counts contributed by the top expressed genes.
        Useful for identifying spots dominated by a small number of genes.

    qc_vars=["mt", "ribo", "hb"]:
        Could add ribosomal and hemoglobin gene QC once those gene sets are
        defined in adata.var.
    """
    # Human mitochondrial genes usually begin with "MT-".
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")

    # Optional additional QC gene sets.
    #
    # Ribosomal genes:
    #   RPS* and RPL* genes can be useful QC features.
    #
    # Hemoglobin genes:
    #   HBA* and HBB* genes can indicate blood contamination or erythrocyte
    #   signal in some datasets.
    #
    # These are added here even if not used in filtering yet, because they make
    # future QC expansion straightforward.
    adata.var["ribo"] = adata.var_names.str.upper().str.startswith(("RPS", "RPL"))
    adata.var["hb"] = adata.var_names.str.upper().str.startswith(("HBA", "HBB"))

    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt", "ribo", "hb"],
        percent_top=[50, 100, 200],
        log1p=False,
        inplace=True,
    )

    return adata


def print_dataset_summary(adata) -> None:
    """
    Print a compact summary of the loaded Visium object.

    Why this function exists
    ------------------------
    Early in the workflow, it is useful to confirm that the object contains
    expression data, observation metadata, gene metadata, spatial coordinates,
    and image metadata.

    This catches data loading problems before downstream analysis begins.
    """
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

    print("\nQC summary:")
    qc_cols = [
        "total_counts",
        "n_genes_by_counts",
        "pct_counts_mt",
        "pct_counts_ribo",
        "pct_counts_hb",
        "pct_counts_in_top_50_genes",
        "pct_counts_in_top_100_genes",
        "pct_counts_in_top_200_genes",
    ]

    available_qc_cols = [col for col in qc_cols if col in adata.obs.columns]
    print(adata.obs[available_qc_cols].describe())


def plot_qc_distributions(adata) -> None:
    """
    Plot QC metric distributions across all spots.

    Why this plot is useful
    -----------------------
    Distribution plots help decide filtering thresholds for the next
    preprocessing script.

    Interpretation examples
    -----------------------
    Low total_counts:
        May indicate weak capture or poor-quality/tissue-free spots.

    Low n_genes_by_counts:
        May indicate low-complexity spots.

    High pct_counts_mt:
        May indicate tissue stress, damage, or metabolic biology.

    High pct_counts_in_top_50_genes:
        May indicate that a small number of highly expressed genes dominate
        the spot.

    Alternative plots
    -----------------
    - histograms
    - density plots
    - total_counts vs n_genes_by_counts scatter
    - QC metrics stratified by tissue region
    """
    qc_plot_keys = [
        "total_counts",
        "n_genes_by_counts",
        "pct_counts_mt",
        "pct_counts_ribo",
        "pct_counts_hb",
        "pct_counts_in_top_50_genes",
    ]

    available_qc_plot_keys = [key for key in qc_plot_keys if key in adata.obs.columns]

    sc.pl.violin(
        adata,
        keys=available_qc_plot_keys,
        jitter=0.4,
        multi_panel=True,
        show=False,
    )
    save_current_fig("qc_violin.png")


def plot_spatial_qc_metrics(adata) -> None:
    """
    Plot QC metrics on the tissue image using Squidpy.

    Why use spatial plots for QC?
    -----------------------------
    QC metrics are not only numerical distributions. In spatial data, it is
    critical to inspect whether low-quality or high-mitochondrial spots are
    localized to:

        - tissue edges
        - folds
        - damaged areas
        - low tissue coverage areas
        - regions outside tissue

    Why Squidpy plotting is used
    ----------------------------
    Squidpy's `spatial_scatter()` is designed for spatial omics visualization
    and will be used throughout the project as Squidpy becomes more central.

    Alternative parameters
    ----------------------
    img=True:
        Shows the tissue image behind the spots.

    img=False:
        Plots only spots and coordinates. Useful if the image makes the plot
        too visually busy.

    size:
        Can be adjusted if spots appear too small or too large.
    """
    spatial_qc_keys = [
        "total_counts",
        "n_genes_by_counts",
        "pct_counts_mt",
        "pct_counts_ribo",
        "pct_counts_hb",
    ]

    available_spatial_qc_keys = [
        key for key in spatial_qc_keys if key in adata.obs.columns
    ]

    sq.pl.spatial_scatter(
        adata,
        color=available_spatial_qc_keys,
        library_id=LIBRARY_ID,
        img=True,
    )
    save_current_fig("spatial_qc_metrics.png")


def plot_initial_marker_genes(adata) -> None:
    """
    Plot a small panel of biological marker genes.

    Why this plot is included
    -------------------------
    Before formal clustering or differential expression, it is useful to check
    whether expected breast cancer and tumor microenvironment signals are
    spatially structured.

    Marker examples
    ---------------
    EPCAM:
        epithelial/tumor-associated

    KRT18:
        epithelial marker

    COL1A1:
        stromal/fibroblast/extracellular matrix

    PTPRC:
        pan-immune marker, also known as CD45

    CD3D:
        T-cell marker

    MKI67:
        proliferation marker

    Alternative marker panels
    -------------------------
    A broader first-pass panel could include:

        - KRT8, KRT19, MUC1 for epithelial/tumor signal
        - CD74, HLA-DRA, C1QA for antigen-presenting/myeloid signal
        - IGKC, JCHAIN for plasma-cell/B-cell signal
        - FABP4, PLIN1 for adipocyte/fat-associated regions
        - PGK1, LDHA, CA9 for hypoxia/glycolysis
    """
    marker_genes = [
        "EPCAM",
        "KRT18",
        "COL1A1",
        "PTPRC",
        "CD3D",
        "MKI67",
    ]

    available_markers = [gene for gene in marker_genes if gene in adata.var_names]

    print("\nAvailable marker genes:")
    print(available_markers)

    if not available_markers:
        print("No requested marker genes were found; skipping marker plot.")
        return

    sq.pl.spatial_scatter(
        adata,
        color=available_markers,
        library_id=LIBRARY_ID,
        img=True,
    )
    save_current_fig("spatial_marker_genes.png")


def main() -> None:
    """
    Run Milestone 1.

    Workflow
    --------
    1. Validate expected 10x Visium input files.
    2. Load the dataset with Squidpy.
    3. Add QC metrics with Scanpy.
    4. Print dataset and QC summaries.
    5. Save the raw-QC AnnData checkpoint.
    6. Generate QC and marker-gene plots.
    """
    print(f"Loading Visium data from: {DATA_DIR}")

    validate_visium_input_files()

    adata = load_visium_with_squidpy()

    adata = add_basic_qc_metrics(adata)

    print_dataset_summary(adata)

    out_h5ad = PROCESSED_DIR / "visium_human_breast_cancer_raw_qc.h5ad"
    adata.write_h5ad(out_h5ad)
    print(f"\nSaved QC AnnData object to: {out_h5ad}")

    plot_qc_distributions(adata)
    plot_spatial_qc_metrics(adata)
    plot_initial_marker_genes(adata)

    print("\nDone. Initial QC figures saved to:")
    print(OUT_DIR)

    print("\nMilestone 1 complete.")


if __name__ == "__main__":
    main()
