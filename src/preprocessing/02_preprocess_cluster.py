"""
Milestone 2: Preprocess, cluster, and add preliminary marker-score labels.

Purpose
-------
This script takes the raw-QC AnnData object from Milestone 1 and performs the
main preprocessing and unsupervised clustering workflow.

It also adds a first-pass marker-score interpretation layer so that Leiden
clusters are easier to interpret before formal marker-gene analysis.

Why Squidpy is used here
------------------------
This project now uses Squidpy for spatial-specific functionality.

In this script, Squidpy is used for:

    - spatial plotting with `sq.pl.spatial_scatter`
    - construction of a Visium spot-neighborhood graph with
      `sq.gr.spatial_neighbors`

Scanpy is still used for standard expression preprocessing:

    - normalization
    - log transformation
    - highly variable gene selection
    - scaling
    - PCA
    - graph-based expression neighbors
    - UMAP
    - Leiden clustering
    - marker-gene signature scoring

Why both Scanpy and Squidpy?
----------------------------
Scanpy provides mature single-cell/spatial expression preprocessing tools.
Squidpy extends AnnData/Scanpy workflows with spatial graphs, spatial
statistics, and spatial plotting.

This division keeps the workflow readable:

    Scanpy  = expression preprocessing and clustering
    Squidpy = spatial visualization and spatial neighborhood graph

Input
-----
    data/processed/visium_human_breast_cancer_raw_qc.h5ad

Outputs
-------
Processed AnnData object:

    data/processed/visium_human_breast_cancer_processed_clustered.h5ad

Figures:

    results/figures/02_preprocessing_clustering/

Tables:

    results/tables/

Main analysis steps
-------------------
1. Load raw-QC AnnData object.
2. Apply conservative QC filtering.
3. Preserve raw counts.
4. Normalize counts and log-transform expression.
5. Select highly variable genes.
6. Run scaling, PCA, expression-neighbor graph, UMAP, and Leiden clustering.
7. Build a Squidpy spatial-neighbor graph.
8. Plot QC, clustering, marker genes, and spatial neighbor structure.
9. Score biological marker-gene signatures.
10. Assign preliminary biological labels to Leiden clusters.
11. Save processed AnnData object.

Analysis notes
--------------
The preliminary marker-score labels are intentionally conservative. They are
useful for quick interpretation, but the more rigorous biological annotation is
performed later in:

    src/analysis/03_marker_gene_analysis.py

The Squidpy spatial-neighbor graph saved here is not used for Leiden clustering
in this script. Leiden clustering is still based on expression PCs. The spatial
graph is saved for downstream spatial biology analyses such as:

    - neighborhood enrichment
    - co-occurrence analysis
    - spatial autocorrelation
    - spatially aware feature engineering

Possible alternatives
---------------------
QC thresholds:
    - min_genes_by_counts could be raised to 750 or 1000 for stricter filtering.
    - min_total_counts could be raised if low-count tissue-edge spots dominate.
    - max_pct_counts_mt could be lowered to 10 or raised to 20 depending on
      tissue quality and whether mitochondrial signal is biological.

HVG selection:
    - `flavor="seurat"` is stable and avoids extra dependencies.
    - `flavor="seurat_v3"` could be used if `scikit-misc` is installed.

Dimensionality reduction:
    - n_top_genes=2000, 3000, or 5000 could be compared.
    - n_pcs=20, 30, 50 could be compared.
    - n_neighbors=8, 12, 20 could be compared.

Clustering:
    - Leiden resolution could be tuned, e.g. 0.3, 0.6, 0.8, 1.0.
    - Louvain could be used as an alternative graph-clustering method.
    - Spatial domain methods could be tested later.

Squidpy spatial graph:
    - n_rings=1 focuses on immediate Visium neighbors.
    - n_rings=2 captures a broader local tissue neighborhood.
    - coord_type="grid" is appropriate for Visium-like grid/hex layouts.

Marker scoring:
    - Scanpy `score_genes` is a simple interpretable baseline.
    - Alternatives include AUCell, ssGSEA, GSVA, decoupler, pathway scoring,
      or single-cell-reference deconvolution.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import squidpy as sq


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
# This script lives in:
#   src/preprocessing/02_preprocess_cluster.py
#
# parents[2] moves from:
#   src/preprocessing/ -> src/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_H5AD = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "visium_human_breast_cancer_raw_qc.h5ad"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures" / "02_preprocessing_clustering"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------
LIBRARY_ID = "Visium_Human_Breast_Cancer"

# Leiden cluster key created in this script.
CLUSTER_KEY = "leiden_r06"

# Squidpy stores the spatial-neighbor graph in adata.obsp using these keys.
SPATIAL_CONNECTIVITY_KEY = "spatial_connectivities"
SPATIAL_DISTANCE_KEY = "spatial_distances"


# ---------------------------------------------------------------------
# Marker sets for preliminary cluster interpretation
# ---------------------------------------------------------------------
MARKER_SETS = {
    "tumor_epithelial": [
        "EPCAM",
        "KRT8",
        "KRT18",
        "KRT19",
        "MUC1",
        "KRT7",
    ],
    "stromal_caf": [
        "COL1A1",
        "COL1A2",
        "COL3A1",
        "DCN",
        "LUM",
        "ACTA2",
        "TAGLN",
    ],
    "pan_immune": [
        "PTPRC",
        "LCP1",
        "CD52",
        "CORO1A",
    ],
    "t_cell": [
        "CD3D",
        "CD3E",
        "CD2",
        "TRAC",
        "CD8A",
        "CD8B",
    ],
    "b_cell": [
        "MS4A1",
        "CD79A",
        "CD79B",
        "BANK1",
    ],
    "myeloid_macrophage": [
        "LST1",
        "C1QA",
        "C1QB",
        "C1QC",
        "CD68",
        "TYROBP",
        "FCER1G",
    ],
    "endothelial": [
        "PECAM1",
        "VWF",
        "KDR",
        "ENG",
        "PLVAP",
    ],
    "proliferation": [
        "MKI67",
        "TOP2A",
        "PCNA",
        "MCM5",
        "STMN1",
    ],
}


LABEL_DISPLAY_NAMES = {
    "tumor_epithelial": "Tumor epithelial",
    "stromal_caf": "Stromal/CAF",
    "pan_immune": "Immune-enriched",
    "t_cell": "T-cell enriched",
    "b_cell": "B-cell enriched",
    "myeloid_macrophage": "Myeloid/macrophage",
    "endothelial": "Endothelial/vascular",
    "proliferation": "Proliferative",
}


def save_current_fig(filename: str) -> None:
    """
    Save the active matplotlib figure and close it.

    Why this helper exists
    ----------------------
    The script creates many plots using both Scanpy and Squidpy. Centralizing
    save behavior keeps output resolution and formatting consistent.

    Alternative parameters
    ----------------------
    dpi=150:
        smaller debugging figures

    dpi=300:
        good for GitHub and reports; used here

    dpi=600:
        higher-quality print figures, but larger files
    """
    out_path = FIGURE_DIR / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {out_path}")


def plot_spatial_scatter(
    adata,
    color,
    filename: str,
    use_raw: bool | None = None,
    img: bool = True,
    legend_loc: str | None = "right margin",
    title=None,
    ncols: int = 4,
    connectivity_key: str | None = None,
    edges_width: float = 0.2,
) -> None:
    """
    Wrapper around Squidpy spatial plotting.

    Why this helper exists
    ----------------------
    Squidpy's `sq.pl.spatial_scatter()` is now the preferred spatial plotting
    function in this project. This wrapper keeps plotting calls consistent and
    avoids repeating the same arguments across the script.

    Important detail
    ----------------
    Unlike Scanpy plotting functions, `sq.pl.spatial_scatter()` does not use
    `show=False` in the same way. Passing unsupported plotting arguments can be
    forwarded to Matplotlib and cause errors. Therefore this wrapper does not
    pass `show=False`.

    Parameters
    ----------
    color:
        AnnData `.obs` column, gene name, or list of columns/genes to plot.

    filename:
        Output filename.

    use_raw:
        If True, gene expression is pulled from `adata.raw`.

    img:
        Whether to show the tissue image underneath the spots.

    legend_loc:
        Legend location for categorical labels.

    ncols:
        Number of subplot columns when plotting multiple features.

    connectivity_key:
        If provided, Squidpy can draw graph edges from adata.obsp.

    edges_width:
        Width of plotted graph edges.

    Alternative parameters
    ----------------------
    img=False:
        useful when tissue image makes spots hard to see

    img_alpha=0.6:
        useful if the image is too visually dominant

    size:
        can be adjusted if spots look too large or small

    crop_coord:
        can crop the tissue image to a region of interest
    """
    sq.pl.spatial_scatter(
        adata,
        color=color,
        library_id=LIBRARY_ID,
        img=img,
        use_raw=use_raw,
        legend_loc=legend_loc,
        ncols=ncols,
        title=title,
        connectivity_key=connectivity_key,
        edges_width=edges_width,
    )
    save_current_fig(filename)


def get_available_markers(adata, genes: list[str]) -> list[str]:
    """
    Return marker genes available in the AnnData object.

    Why this function exists
    ------------------------
    Some marker genes may not be present after filtering, gene-symbol handling,
    or highly variable gene subsetting. Filtering to available genes prevents
    marker scoring and plotting errors.

    Why use adata.raw?
    ------------------
    The object is later subset to highly variable genes for PCA and clustering.
    Marker genes are not guaranteed to be highly variable, so `adata.raw` is
    preferred when available because it stores the full normalized/logged gene
    matrix before HVG subsetting.

    Alternative
    -----------
    If using Ensembl IDs instead of gene symbols, this function could be
    extended to map gene symbols to Ensembl IDs before checking availability.
    """
    if adata.raw is not None:
        gene_index = set(adata.raw.var_names)
    else:
        gene_index = set(adata.var_names)

    return [gene for gene in genes if gene in gene_index]


def assign_cluster_labels(cluster_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Assign preliminary biological labels to Leiden clusters.

    Why this function exists
    ------------------------
    Leiden clusters are numeric labels. This function gives them a first-pass
    biological interpretation using marker signature scores.

    Labeling strategy
    -----------------
    For each cluster:
        1. Sort average marker signature scores.
        2. Use the highest-scoring signature as the first label candidate.
        3. Apply simple biological refinement rules.
        4. Mark clusters as mixed if the top two signatures are too close.

    Important limitation
    --------------------
    These are provisional labels. Final biological labels should be assigned
    after formal differential expression and manual review.

    Alternative strategies
    ----------------------
    - Use formal marker gene tables instead of marker scores.
    - Use pathway enrichment per cluster.
    - Use single-cell reference deconvolution.
    - Use probabilistic/soft labels instead of hard labels.
    - Use expert/pathologist review of tissue morphology.
    """
    rows = []

    for cluster_id, row in cluster_scores.iterrows():
        sorted_scores = row.sort_values(ascending=False)

        top_signature = sorted_scores.index[0]
        second_signature = sorted_scores.index[1]

        top_score = sorted_scores.iloc[0]
        second_score = sorted_scores.iloc[1]
        margin = top_score - second_score

        label = LABEL_DISPLAY_NAMES[top_signature]

        # Proliferation can appear inside tumor regions. If the proliferation
        # signature is strongest but epithelial signal is also above the median,
        # we flag the label as proliferative tumor rather than generic
        # proliferation.
        if (
            top_signature == "proliferation"
            and row.get("tumor_epithelial", -999) > row.median()
        ):
            label = "Proliferative tumor"

        # Tumor epithelial clusters with high proliferation are flagged because
        # they may represent cycling tumor-enriched regions.
        elif (
            top_signature == "tumor_epithelial"
            and row.get("proliferation", -999) > sorted_scores.quantile(0.75)
        ):
            label = "Tumor epithelial/proliferative"

        # Immune signatures often overlap. If a myeloid/macrophage score is as
        # strong as broad immune signal, the cluster is interpreted as
        # myeloid/immune enriched.
        elif (
            top_signature in ["pan_immune", "t_cell", "b_cell", "myeloid_macrophage"]
            and row.get("myeloid_macrophage", -999) >= row.get("pan_immune", -999)
        ):
            label = "Myeloid/immune enriched"

        # If the top two signatures are very close, treat the cluster as mixed.
        #
        # The 0.05 margin is heuristic. Other options:
        #   - use a relative margin
        #   - z-score cluster scores first
        #   - require manual marker inspection before assigning mixed labels
        if margin < 0.05:
            label = (
                f"Mixed: {LABEL_DISPLAY_NAMES[top_signature]} / "
                f"{LABEL_DISPLAY_NAMES[second_signature]}"
            )

        rows.append(
            {
                "cluster": cluster_id,
                "preliminary_label": label,
                "top_signature": top_signature,
                "top_score": top_score,
                "second_signature": second_signature,
                "second_score": second_score,
                "score_margin": margin,
            }
        )

    return pd.DataFrame(rows)


def add_preliminary_marker_score_labels(adata):
    """
    Score marker-gene signatures and assign preliminary cluster labels.

    Why marker scoring is included here
    -----------------------------------
    Marker scoring provides a quick biological interpretation layer after
    unsupervised clustering. It helps answer:

        "What might each cluster represent biologically?"

    This is not a replacement for formal marker-gene analysis. It is a
    preliminary guide before Milestone 3.

    Returns
    -------
    adata:
        AnnData object with marker score columns and preliminary labels.

    score_columns:
        List of new score columns added to `adata.obs`.

    cluster_scores:
        Cluster-level mean marker signature scores.

    annotations:
        Preliminary cluster annotation table.

    Alternative scoring methods
    ---------------------------
    - AUCell
    - ssGSEA
    - GSVA
    - decoupler pathway scoring
    - single-cell reference deconvolution
    """
    score_columns = []

    for signature_name, genes in MARKER_SETS.items():
        available_genes = get_available_markers(adata, genes)

        print(f"\nMarker signature: {signature_name}")
        print(f"  Requested genes: {genes}")
        print(f"  Available genes: {available_genes}")

        # Require at least two genes so a score is not driven by one marker.
        # A one-gene score may be useful for highly specific markers, but is
        # more fragile and should be interpreted cautiously.
        if len(available_genes) < 2:
            print(f"  Skipping {signature_name}: fewer than 2 genes available.")
            continue

        score_name = f"{signature_name}_score"

        sc.tl.score_genes(
            adata,
            gene_list=available_genes,
            score_name=score_name,
            use_raw=True if adata.raw is not None else False,
        )

        score_columns.append(score_name)

    print("\nCreated marker score columns:")
    print(score_columns)

    # Average marker scores within each Leiden cluster.
    #
    # Alternative summary statistics:
    #   - median score per cluster
    #   - fraction of spots above a score threshold
    #   - z-scored mean score
    #   - trimmed mean to reduce outlier influence
    cluster_scores = (
        adata.obs[[CLUSTER_KEY] + score_columns]
        .groupby(CLUSTER_KEY, observed=True)
        .mean()
    )

    cluster_scores.columns = [
        col.replace("_score", "") for col in cluster_scores.columns
    ]

    cluster_scores.to_csv(TABLE_DIR / "cluster_marker_signature_scores.csv")

    annotations = assign_cluster_labels(cluster_scores)
    annotations.to_csv(
        TABLE_DIR / "preliminary_cluster_annotations.csv",
        index=False,
    )

    cluster_to_label = dict(
        zip(
            annotations["cluster"].astype(str),
            annotations["preliminary_label"],
        )
    )

    adata.obs["preliminary_niche_label"] = (
        adata.obs[CLUSTER_KEY]
        .astype(str)
        .map(cluster_to_label)
        .astype("category")
    )

    cluster_to_combined_label = {
        cluster: f"{cluster}: {label}"
        for cluster, label in cluster_to_label.items()
    }

    adata.obs["cluster_preliminary_label"] = (
        adata.obs[CLUSTER_KEY]
        .astype(str)
        .map(cluster_to_combined_label)
        .astype("category")
    )

    adata.obs[
        [
            CLUSTER_KEY,
            "preliminary_niche_label",
            "cluster_preliminary_label",
        ]
        + score_columns
    ].to_csv(TABLE_DIR / "spot_marker_scores_and_labels.csv")

    return adata, score_columns, cluster_scores, annotations


def add_squidpy_spatial_neighbors(adata):
    """
    Build a spatial-neighbor graph using Squidpy.

    Why this step is added in Milestone 2
    -------------------------------------
    This script already creates an expression-neighbor graph with Scanpy for
    UMAP and Leiden clustering. That graph connects spots based on expression
    similarity.

    Squidpy's spatial-neighbor graph is different. It connects spots based on
    physical tissue location.

    Expression graph:
        "Which spots have similar transcriptomes?"

    Spatial graph:
        "Which spots are physically adjacent in the tissue?"

    Why save the spatial graph now?
    -------------------------------
    The spatial graph will be useful for downstream analyses, including:

        - neighborhood enrichment between niche labels
        - co-occurrence across distance
        - Moran's I spatial autocorrelation
        - spatially aware ML features

    Current parameters
    ------------------
    coord_type="grid":
        Appropriate for Visium-style regularly arranged spots.

    n_neighs=6:
        Approximate immediate hexagonal Visium neighborhood.

    n_rings=1:
        Uses immediate neighboring spots. This is conservative and focused on
        local tissue adjacency.

    Alternative parameters
    ----------------------
    n_rings=2:
        captures a broader local neighborhood

    n_rings=3:
        captures larger tissue context but may blur fine boundaries

    coord_type="generic":
        useful for non-grid spatial platforms, often with radius or Delaunay
        graph construction
    """
    print("\nBuilding Squidpy spatial-neighbor graph...")

    sq.gr.spatial_neighbors(
        adata,
        spatial_key="spatial",
        coord_type="grid",
        n_neighs=6,
        n_rings=1,
    )

    if SPATIAL_CONNECTIVITY_KEY in adata.obsp:
        n_edges = int(adata.obsp[SPATIAL_CONNECTIVITY_KEY].nnz)
        print(f"  Spatial connectivity graph stored in adata.obsp['{SPATIAL_CONNECTIVITY_KEY}']")
        print(f"  Non-zero graph entries: {n_edges}")

    if SPATIAL_DISTANCE_KEY in adata.obsp:
        print(f"  Spatial distance graph stored in adata.obsp['{SPATIAL_DISTANCE_KEY}']")

    return adata


def plot_preliminary_marker_score_outputs(adata, score_columns, cluster_scores) -> None:
    """
    Generate plots for marker scores and preliminary cluster labels.

    Why these plots are included
    ----------------------------
    Marker score maps help check whether biological signatures occupy coherent
    tissue regions. Preliminary label maps help check whether automated labels
    make spatial sense before formal marker-gene review.

    Alternative plots
    -----------------
    - dotplots grouped by Leiden cluster
    - violin plots of marker scores by cluster
    - spatial maps without tissue image background
    - spatial maps with graph edges overlaid
    """
    if score_columns:
        plot_spatial_scatter(
            adata,
            color=score_columns,
            filename="spatial_marker_signature_scores.png",
            use_raw=False,
            img=True,
            ncols=4,
        )

    plot_spatial_scatter(
        adata,
        color=["cluster_preliminary_label"],
        filename="spatial_clusters_with_preliminary_labels.png",
        use_raw=False,
        img=True,
        legend_loc="right margin",
    )

    sc.pl.umap(
        adata,
        color=["cluster_preliminary_label"],
        legend_loc="right margin",
        frameon=False,
        show=False,
    )
    save_current_fig("umap_clusters_with_preliminary_labels.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(cluster_scores.values, aspect="auto")

    ax.set_xticks(range(cluster_scores.shape[1]))
    ax.set_xticklabels(cluster_scores.columns, rotation=45, ha="right")

    ax.set_yticks(range(cluster_scores.shape[0]))
    ax.set_yticklabels(cluster_scores.index)

    ax.set_xlabel("Marker signature")
    ax.set_ylabel("Leiden cluster")
    ax.set_title("Mean marker signature score by Leiden cluster")

    fig.colorbar(im, ax=ax, label="Mean signature score")
    save_current_fig("cluster_marker_signature_score_heatmap.png")


def main() -> None:
    """
    Run preprocessing, clustering, spatial graph construction, and preliminary labeling.

    Workflow
    --------
    1. Load raw-QC AnnData object.
    2. Apply conservative spot filtering.
    3. Save QC filtering summary.
    4. Plot post-filtering QC metrics spatially.
    5. Preserve raw counts.
    6. Normalize counts and log-transform expression.
    7. Store full normalized expression in adata.raw.
    8. Select highly variable genes.
    9. Run PCA, Scanpy expression-neighbor graph, UMAP, and Leiden clustering.
    10. Build Squidpy spatial-neighbor graph.
    11. Plot clusters, marker genes, spatial graph, and marker signatures.
    12. Add preliminary marker-score labels.
    13. Save processed AnnData object.
    """
    print(f"Loading QC AnnData object from: {INPUT_H5AD}")
    adata = sc.read_h5ad(INPUT_H5AD)

    print("\nOriginal AnnData object:")
    print(adata)

    original_n_spots = adata.n_obs
    original_n_genes = adata.n_vars

    qc_summary_cols = [
        "total_counts",
        "n_genes_by_counts",
        "pct_counts_mt",
        "pct_counts_ribo",
        "pct_counts_hb",
    ]
    available_qc_summary_cols = [
        col for col in qc_summary_cols if col in adata.obs.columns
    ]

    print("\nOriginal QC summary:")
    print(adata.obs[available_qc_summary_cols].describe())

    # ------------------------------------------------------------------
    # 1. Conservative QC filtering
    # ------------------------------------------------------------------
    # These thresholds are intentionally conservative. The goal is to remove
    # clearly poor-quality spots without aggressively discarding biologically
    # interesting tissue regions.
    #
    # Possible alternatives:
    #   min_genes_by_counts = 750 or 1000 for stricter filtering
    #   min_total_counts = 2000 if low-count spots are clearly low-quality
    #   max_pct_counts_mt = 10 for stricter mitochondrial filtering
    #   max_pct_counts_mt = 20 if mitochondrial signal appears biological
    min_genes_by_counts = 500
    min_total_counts = 1000
    max_pct_counts_mt = 15

    qc_mask = (
        (adata.obs["n_genes_by_counts"] >= min_genes_by_counts)
        & (adata.obs["total_counts"] >= min_total_counts)
        & (adata.obs["pct_counts_mt"] < max_pct_counts_mt)
    )

    adata = adata[qc_mask].copy()

    filtered_n_spots = adata.n_obs

    print("\nQC filtering thresholds:")
    print(f"  n_genes_by_counts >= {min_genes_by_counts}")
    print(f"  total_counts >= {min_total_counts}")
    print(f"  pct_counts_mt < {max_pct_counts_mt}")

    print("\nFiltering summary:")
    print(f"  Original spots: {original_n_spots}")
    print(f"  Retained spots: {filtered_n_spots}")
    print(f"  Removed spots: {original_n_spots - filtered_n_spots}")
    print(f"  Original genes: {original_n_genes}")

    filtering_summary = pd.DataFrame(
        {
            "metric": [
                "original_spots",
                "retained_spots",
                "removed_spots",
                "original_genes",
                "min_genes_by_counts",
                "min_total_counts",
                "max_pct_counts_mt",
            ],
            "value": [
                original_n_spots,
                filtered_n_spots,
                original_n_spots - filtered_n_spots,
                original_n_genes,
                min_genes_by_counts,
                min_total_counts,
                max_pct_counts_mt,
            ],
        }
    )

    filtering_summary.to_csv(
        TABLE_DIR / "qc_filtering_summary.csv",
        index=False,
    )

    # Spatial QC after filtering confirms whether remaining spots still cover
    # the tissue and whether high/low QC regions are spatially localized.
    post_filter_qc_cols = [
        "total_counts",
        "n_genes_by_counts",
        "pct_counts_mt",
        "pct_counts_ribo",
        "pct_counts_hb",
    ]
    available_post_filter_qc_cols = [
        col for col in post_filter_qc_cols if col in adata.obs.columns
    ]

    plot_spatial_scatter(
        adata,
        color=available_post_filter_qc_cols,
        filename="spatial_qc_after_filtering.png",
        img=True,
        ncols=3,
    )

    # ------------------------------------------------------------------
    # 2. Preserve raw counts
    # ------------------------------------------------------------------
    # Raw counts are saved in a layer before normalization. This gives future
    # analyses the option to return to count-scale data.
    adata.layers["counts"] = adata.X.copy()

    # ------------------------------------------------------------------
    # 3. Normalize and log-transform
    # ------------------------------------------------------------------
    # Normalize each spot to a common library size and log-transform.
    #
    # target_sum=1e4 is a common single-cell/spatial RNA-seq convention.
    #
    # Alternatives:
    #   target_sum=1e6 for CPM-like scaling
    #   scran-style normalization for more complex workflows
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Store full normalized/log-transformed expression before HVG subsetting.
    # This allows marker plotting and marker scoring to use genes that may not
    # be selected as highly variable genes.
    adata.raw = adata

    # ------------------------------------------------------------------
    # 4. Highly variable gene selection
    # ------------------------------------------------------------------
    # HVG selection reduces dimensionality and focuses PCA/clustering on genes
    # with informative variation.
    #
    # Current choice:
    #   n_top_genes=3000, flavor="seurat"
    #
    # Alternatives:
    #   n_top_genes=2000 for a smaller feature set
    #   n_top_genes=5000 for more expression variation
    #   flavor="seurat_v3" if scikit-misc is installed
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=3000,
        flavor="seurat",
    )

    n_hvgs = int(adata.var["highly_variable"].sum())
    print(f"\nHighly variable genes selected: {n_hvgs}")

    hvgs = adata.var[adata.var["highly_variable"]].copy()
    hvgs.to_csv(TABLE_DIR / "highly_variable_genes.csv")

    sc.pl.highly_variable_genes(adata, show=False)
    save_current_fig("highly_variable_genes.png")

    # Subset to HVGs for dimensionality reduction and clustering.
    # adata.raw still preserves the full normalized expression matrix.
    adata = adata[:, adata.var["highly_variable"]].copy()

    # ------------------------------------------------------------------
    # 5. Scaling, PCA, expression neighbors, UMAP, and Leiden clustering
    # ------------------------------------------------------------------
    # Scaling gives each gene comparable variance for PCA.
    #
    # max_value=10 caps extreme scaled values to reduce the influence of
    # outliers.
    sc.pp.scale(adata, max_value=10)

    # PCA compresses expression variation into a smaller feature space.
    #
    # Alternatives:
    #   n_comps=30 for a smaller representation
    #   n_comps=100 for more complex datasets
    sc.tl.pca(
        adata,
        n_comps=50,
        svd_solver="arpack",
    )

    sc.pl.pca_variance_ratio(
        adata,
        n_pcs=50,
        log=True,
        show=False,
    )
    save_current_fig("pca_variance_ratio.png")

    # Scanpy expression-neighbor graph.
    #
    # This is different from Squidpy's spatial-neighbor graph:
    #
    #   sc.pp.neighbors = connects transcriptionally similar spots
    #   sq.gr.spatial_neighbors = connects physically adjacent spots
    #
    # Expression-neighbor parameters:
    #   n_neighbors=12 controls local expression neighborhood size.
    #   n_pcs=30 controls how many PCs define expression similarity.
    #
    # Alternatives:
    #   n_neighbors=8 for finer local clusters
    #   n_neighbors=20 for smoother/larger clusters
    #   n_pcs=20 or 50 depending on PCA variance structure
    sc.pp.neighbors(
        adata,
        n_neighbors=12,
        n_pcs=30,
    )

    sc.tl.umap(adata, random_state=42)

    # Leiden clustering is run on the expression-neighbor graph, not the
    # Squidpy spatial graph. This keeps the main clusters expression-driven.
    #
    # Alternatives:
    #   resolution=0.3 for fewer/broader clusters
    #   resolution=0.8 or 1.0 for more/finer clusters
    sc.tl.leiden(
        adata,
        resolution=0.6,
        key_added=CLUSTER_KEY,
        random_state=42,
    )

    print("\nLeiden cluster counts:")
    print(adata.obs[CLUSTER_KEY].value_counts().sort_index())

    cluster_counts = (
        adata.obs[CLUSTER_KEY]
        .value_counts()
        .sort_index()
        .rename_axis("cluster")
        .reset_index(name="n_spots")
    )
    cluster_counts.to_csv(TABLE_DIR / "leiden_r06_cluster_counts.csv", index=False)

    # ------------------------------------------------------------------
    # 6. Build Squidpy spatial-neighbor graph
    # ------------------------------------------------------------------
    # This graph is saved in the AnnData object and can be reused later for
    # neighborhood enrichment, co-occurrence, and spatial autocorrelation.
    adata = add_squidpy_spatial_neighbors(adata)

    # ------------------------------------------------------------------
    # 7. Core clustering plots
    # ------------------------------------------------------------------
    sc.pl.umap(
        adata,
        color=[CLUSTER_KEY],
        legend_loc="on data",
        frameon=False,
        show=False,
    )
    save_current_fig("umap_leiden_r06.png")

    plot_spatial_scatter(
        adata,
        color=[CLUSTER_KEY],
        filename="spatial_leiden_r06.png",
        img=True,
        legend_loc="right margin",
    )

    # Plot the spatial-neighbor graph over tissue using Leiden labels.
    # This confirms the graph was created and shows the local spatial structure.
    plot_spatial_scatter(
        adata,
        color=[CLUSTER_KEY],
        filename="spatial_leiden_r06_with_squidpy_neighbor_edges.png",
        img=True,
        legend_loc="right margin",
        connectivity_key=SPATIAL_CONNECTIVITY_KEY,
        edges_width=0.15,
    )

    marker_genes = [
        "EPCAM",
        "KRT8",
        "KRT18",
        "KRT19",
        "COL1A1",
        "COL1A2",
        "DCN",
        "LUM",
        "PTPRC",
        "CD3D",
        "CD8A",
        "MS4A1",
        "LST1",
        "C1QA",
        "CD68",
        "MKI67",
        "TOP2A",
        "PECAM1",
        "VWF",
    ]

    available_markers = [
        gene for gene in marker_genes if gene in adata.raw.var_names
    ]

    print("\nAvailable marker genes for plotting:")
    print(available_markers)

    if available_markers:
        plot_spatial_scatter(
            adata,
            color=available_markers,
            filename="spatial_marker_genes_normalized.png",
            use_raw=True,
            img=True,
            ncols=4,
        )

    # ------------------------------------------------------------------
    # 8. Preliminary marker-score annotation
    # ------------------------------------------------------------------
    adata, score_columns, cluster_scores, annotations = (
        add_preliminary_marker_score_labels(adata)
    )

    print("\nPreliminary cluster annotations:")
    print(annotations)

    plot_preliminary_marker_score_outputs(
        adata=adata,
        score_columns=score_columns,
        cluster_scores=cluster_scores,
    )

    # ------------------------------------------------------------------
    # 9. Save processed object
    # ------------------------------------------------------------------
    # This object now contains:
    #   - filtered spots
    #   - normalized/logged expression
    #   - adata.raw with full normalized expression
    #   - HVG-subset matrix for PCA/clustering
    #   - PCA/UMAP
    #   - Leiden clusters
    #   - Squidpy spatial-neighbor graph in adata.obsp
    #   - preliminary marker-score labels
    output_h5ad = PROCESSED_DIR / "visium_human_breast_cancer_processed_clustered.h5ad"
    adata.write_h5ad(output_h5ad)

    print(f"\nSaved processed AnnData object to: {output_h5ad}")
    print("\nMilestone 2 complete.")


if __name__ == "__main__":
    main()
