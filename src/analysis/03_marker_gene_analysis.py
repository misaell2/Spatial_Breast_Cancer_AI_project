"""
Milestone 3: Marker gene discovery, spatial marker analysis, and annotation template creation.

Purpose
-------
This script performs formal marker gene analysis for each Leiden cluster
identified during preprocessing and clustering.

It also adds Squidpy-based spatial analyses that help move the interpretation
beyond visual inspection:

    - neighborhood enrichment between Leiden clusters
    - Moran's I spatial autocorrelation for selected biological marker genes
    - Squidpy spatial marker and label plots

The previous script, `02_preprocess_cluster.py`, performs QC, normalization,
dimensionality reduction, Leiden clustering, preliminary marker-score labeling,
and Squidpy spatial-neighbor graph construction.

This script asks:

    1. Which genes are statistically enriched in each Leiden cluster?
    2. Which clusters are spatially enriched near each other?
    3. Which marker genes show non-random spatial organization?
    4. Which clusters should be manually reviewed as biological niches?

Input
-----
    data/processed/visium_human_breast_cancer_processed_clustered.h5ad

Expected contents of the input AnnData object:
    - Leiden cluster labels in `adata.obs["leiden_r06"]`
    - normalized/log-transformed expression in `adata.raw`
    - UMAP coordinates in `adata.obsm["X_umap"]`
    - spatial coordinates in `adata.obsm["spatial"]`
    - Visium image metadata in `adata.uns["spatial"]`
    - Squidpy spatial graph in:
        adata.obsp["spatial_connectivities"]
        adata.obsp["spatial_distances"]

Outputs
-------
AnnData:

    data/processed/visium_human_breast_cancer_marker_annotated.h5ad

Tables:

    results/tables/leiden_r06_full_marker_gene_table.csv
    results/tables/leiden_r06_top20_marker_genes_per_cluster.csv
    results/tables/leiden_r06_cluster_annotation_template.csv
    results/tables/leiden_r06_spatial_neighborhood_enrichment_zscore.csv
    results/tables/leiden_r06_spatial_neighborhood_enrichment_count.csv
    results/tables/selected_marker_gene_morans_i.csv

Figures:

    results/figures/03_marker_gene_analysis/

Main analysis steps
-------------------
1. Load the clustered AnnData object.
2. Confirm Leiden labels and spatial graph are present.
3. Run differential expression for each Leiden cluster.
4. Export full and filtered marker gene tables.
5. Suggest first-pass labels based on known marker overlap.
6. Run Squidpy neighborhood enrichment between Leiden clusters.
7. Run Moran's I spatial autocorrelation for selected marker genes.
8. Generate marker gene, spatial label, and spatial statistics plots.
9. Save an AnnData object with suggested labels and spatial analysis results.

Important notes
---------------
The automated labels created here are only first-pass suggestions. Final
biological labels should be assigned manually in:

    data_manifest/annotations/leiden_r06_manual_cluster_annotations.csv

and then applied in:

    src/analysis/04_apply_manual_annotations.py

Because the updated Squidpy-enabled preprocessing may produce a different
number of Leiden clusters than the previous run, the manual annotation CSV
should be regenerated/reviewed after this script.

Alternative analysis options
----------------------------
Differential expression:
    - method="wilcoxon" is used here
    - method="t-test" is faster but less robust
    - method="t-test_overestim_var" is more conservative
    - method="logreg" is useful for classifier-like marker selection

Marker filtering:
    - min_logfc could be raised to 0.5 or 1.0
    - max_adj_pval could be lowered to 0.01
    - pct_nz_group and pct_nz_reference could be used for additional filtering

Squidpy spatial analyses:
    - nhood_enrichment n_perms=1000 is used for a solid default
    - n_perms=100 is faster for testing
    - n_perms=5000 is slower but more robust
    - Moran's I is used for spatial autocorrelation
    - Geary's C could also be tested with mode="geary"

Biological interpretation:
    - marker overlap is useful but incomplete
    - spatial enrichment helps interpret cluster adjacency
    - Moran's I helps identify spatially patterned marker genes
    - pathway enrichment and single-cell reference deconvolution could be added
      in later milestones
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
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_H5AD = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "visium_human_breast_cancer_processed_clustered.h5ad"
)

OUTPUT_H5AD = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "visium_human_breast_cancer_marker_annotated.h5ad"
)

FIGURE_DIR = PROJECT_ROOT / "results" / "figures" / "03_marker_gene_analysis"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Dataset and clustering metadata
# ---------------------------------------------------------------------
LIBRARY_ID = "Visium_Human_Breast_Cancer"
CLUSTER_KEY = "leiden_r06"

SPATIAL_CONNECTIVITY_KEY = "spatial_connectivities"
SPATIAL_DISTANCE_KEY = "spatial_distances"


# ---------------------------------------------------------------------
# Known biological marker sets
# ---------------------------------------------------------------------
KNOWN_MARKER_SETS = {
    "Tumor epithelial": [
        "EPCAM",
        "KRT8",
        "KRT18",
        "KRT19",
        "KRT7",
        "MUC1",
        "TACSTD2",
    ],
    "Luminal/secretory epithelial": [
        "SCGB2A2",
        "SCGB1D2",
        "GATA3",
        "XBP1",
        "CSTA",
        "S100G",
    ],
    "Stromal/CAF": [
        "COL1A1",
        "COL1A2",
        "COL3A1",
        "DCN",
        "LUM",
        "ACTA2",
        "TAGLN",
        "FAP",
        "PDGFRA",
    ],
    "Pan-immune": [
        "PTPRC",
        "LCP1",
        "CD52",
        "CORO1A",
        "CXCL13",
    ],
    "T-cell enriched": [
        "CD3D",
        "CD3E",
        "CD2",
        "TRAC",
        "CD8A",
        "CD8B",
        "GZMB",
        "NKG7",
    ],
    "B-cell/plasma-cell enriched": [
        "MS4A1",
        "CD79A",
        "CD79B",
        "BANK1",
        "MZB1",
        "JCHAIN",
        "IGKC",
        "IGHG1",
        "IGHG3",
        "IGLC2",
    ],
    "Myeloid/APC": [
        "CD74",
        "HLA-DRA",
        "HLA-DPA1",
        "HLA-DPB1",
        "LST1",
        "C1QA",
        "C1QB",
        "C1QC",
        "CD68",
        "TYROBP",
        "FCER1G",
        "AIF1",
        "LYZ",
    ],
    "Endothelial/vascular": [
        "PECAM1",
        "VWF",
        "KDR",
        "ENG",
        "PLVAP",
        "CLDN5",
    ],
    "Proliferative": [
        "MKI67",
        "TOP2A",
        "PCNA",
        "MCM5",
        "STMN1",
        "UBE2C",
    ],
    "Adipocyte/fat-associated": [
        "ADIPOQ",
        "PLIN1",
        "PLIN4",
        "FABP4",
        "LPL",
        "APOE",
        "CFD",
        "G0S2",
    ],
    "Hypoxia/glycolysis/stress": [
        "VEGFA",
        "CA9",
        "LDHA",
        "ENO1",
        "SLC2A1",
        "GAPDH",
        "PGK1",
        "TPI1",
        "MIF",
    ],
}


def save_current_fig(filename: str) -> None:
    """
    Save the active matplotlib figure and close it.

    Why this helper exists
    ----------------------
    This script creates figures using Scanpy, Squidpy, and Matplotlib. Keeping
    the save behavior centralized avoids inconsistent figure resolution and
    prevents open figures from accumulating.

    Alternative parameters:
        dpi=150 for quick debugging
        dpi=300 for GitHub/report-quality figures
        dpi=600 for print-quality output
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
    ncols: int = 4,
    title=None,
) -> None:
    """
    Plot spatial data using Squidpy and save the figure.

    Why this helper exists
    ----------------------
    Scanpy's spatial plotting is deprecated in newer Scanpy versions in favor
    of Squidpy spatial plotting. This wrapper makes the transition cleaner and
    keeps all spatial plots consistent.

    Important detail
    ----------------
    `sq.pl.spatial_scatter()` should not be called with `show=False` in this
    project environment because that argument can be passed to Matplotlib and
    cause plotting errors.

    Parameters
    ----------
    color:
        Gene name, obs column, or list of genes/columns.

    filename:
        Output filename.

    use_raw:
        Whether gene expression should be read from `adata.raw`.

    img:
        Whether to draw the tissue image behind the spots.

    legend_loc:
        Legend location for categorical labels.

    ncols:
        Number of columns for multi-panel plots.

    Alternative parameters
    ----------------------
    img=False:
        Useful if the tissue image makes spots hard to see.

    crop_coord:
        Could focus on a region of interest.

    size:
        Could adjust spot size if plotting looks too dense.
    """
    kwargs = {
        "adata": adata,
        "color": color,
        "library_id": LIBRARY_ID,
        "img": img,
        "ncols": ncols,
    }

    if use_raw is not None:
        kwargs["use_raw"] = use_raw

    if legend_loc is not None:
        kwargs["legend_loc"] = legend_loc

    if title is not None:
        kwargs["title"] = title

    sq.pl.spatial_scatter(**kwargs)
    save_current_fig(filename)


def get_available_genes(adata, genes: list[str]) -> list[str]:
    """
    Return genes that are available in the AnnData object.

    Why use adata.raw?
    ------------------
    The processed object is subset to highly variable genes for PCA and
    clustering, but `adata.raw` stores the full normalized/log-transformed
    expression matrix before HVG subsetting. Marker genes may not be HVGs, so
    checking `adata.raw` preserves access to more biologically relevant genes.
    """
    if adata.raw is not None:
        available = set(adata.raw.var_names)
    else:
        available = set(adata.var_names)

    return [gene for gene in genes if gene in available]


def validate_required_fields(adata) -> None:
    """
    Validate that required columns and spatial graph fields exist.

    Why this function exists
    ------------------------
    This script depends on outputs from Milestone 2. Failing early with a clear
    message is better than producing partial or misleading marker/spatial
    results.

    Required fields
    ---------------
    - Leiden cluster labels
    - UMAP coordinates
    - spatial coordinates
    - spatial graph from Squidpy
    """
    if CLUSTER_KEY not in adata.obs.columns:
        raise ValueError(f"Could not find {CLUSTER_KEY} in adata.obs")

    if "X_umap" not in adata.obsm:
        raise ValueError("Could not find UMAP coordinates in adata.obsm['X_umap']")

    if "spatial" not in adata.obsm:
        raise ValueError("Could not find spatial coordinates in adata.obsm['spatial']")

    if SPATIAL_CONNECTIVITY_KEY not in adata.obsp:
        raise ValueError(
            f"Could not find {SPATIAL_CONNECTIVITY_KEY} in adata.obsp. "
            "Rerun src/preprocessing/02_preprocess_cluster.py with Squidpy spatial neighbors."
        )

    if SPATIAL_DISTANCE_KEY not in adata.obsp:
        print(
            f"Warning: {SPATIAL_DISTANCE_KEY} not found in adata.obsp. "
            "Neighborhood enrichment can still use spatial_connectivities."
        )


def extract_top_markers(
    marker_df: pd.DataFrame,
    top_n: int = 20,
    min_logfc: float = 0.25,
    max_adj_pval: float = 0.05,
) -> pd.DataFrame:
    """
    Filter the full marker table and return top markers per cluster.

    Why these thresholds are used
    -----------------------------
    This is an exploratory spatial transcriptomics workflow. The thresholds are
    permissive enough to preserve candidate markers for manual interpretation,
    while still removing weak or non-significant hits.

    Alternative thresholds:
        min_logfc=0.5 for more clearly enriched markers
        min_logfc=1.0 for very strong markers
        max_adj_pval=0.01 for stricter significance

    Additional possible filters:
        pct_nz_group > 0.10 or 0.25
        pct_nz_group - pct_nz_reference > 0.10
        remove mitochondrial or ribosomal genes from annotation candidates
    """
    df = marker_df.copy()
    df = df.rename(columns={"names": "gene"})

    if "logfoldchanges" in df.columns:
        df = df[df["logfoldchanges"] >= min_logfc]

    if "pvals_adj" in df.columns:
        df = df[df["pvals_adj"] <= max_adj_pval]

    top_rows = []
    for _, sub_df in df.groupby("group"):
        top_rows.append(sub_df.head(top_n))

    if not top_rows:
        return pd.DataFrame()

    return pd.concat(top_rows, axis=0).reset_index(drop=True)


def suggest_cluster_labels(
    top_markers_df: pd.DataFrame,
    cluster_counts: pd.Series,
) -> pd.DataFrame:
    """
    Suggest biological labels using overlap with known marker sets.

    Why this function exists
    ------------------------
    Differential expression returns ranked genes, but the annotation process
    needs a structured table that connects those genes to biological labels.

    Labeling strategy
    -----------------
    For each cluster:
        1. Take the top DE markers.
        2. Compare them to known marker sets.
        3. Count overlaps with each marker set.
        4. Suggest the label with the highest overlap.

    Limitations
    -----------
    Marker overlap is a heuristic. It can miss biologically meaningful clusters
    if the biology is pathway/state-driven, the cluster is mixed, or markers are
    not included in the curated list.

    Alternative approaches
    ----------------------
    - pathway enrichment on marker genes
    - average marker signature scores
    - single-cell reference deconvolution
    - manual review of spatial marker plots
    """
    if top_markers_df.empty:
        raise ValueError("Top marker table is empty; cannot suggest cluster labels.")

    annotation_rows = []

    for cluster in sorted(
        top_markers_df["group"].astype(str).unique(),
        key=lambda x: int(x),
    ):
        cluster_markers = (
            top_markers_df[top_markers_df["group"].astype(str) == cluster]["gene"]
            .head(50)
            .tolist()
        )
        cluster_marker_set = set(cluster_markers)

        scores = {}
        matched_markers = {}

        for label, markers in KNOWN_MARKER_SETS.items():
            overlap = sorted(cluster_marker_set.intersection(set(markers)))
            scores[label] = len(overlap)
            matched_markers[label] = overlap

        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]
        best_matches = matched_markers[best_label]

        if best_score == 0:
            suggested_label = "Uncertain/mixed"
        else:
            suggested_label = best_label

        top_10_markers = cluster_markers[:10]

        annotation_rows.append(
            {
                "cluster": cluster,
                "n_spots": int(cluster_counts.loc[cluster]),
                "suggested_label": suggested_label,
                "marker_set_overlap_count": best_score,
                "matched_known_markers": ";".join(best_matches),
                "top_10_de_markers": ";".join(top_10_markers),
                "manual_label": "",
                "notes": "",
            }
        )

    return pd.DataFrame(annotation_rows)


def add_manual_or_suggested_labels(
    adata,
    annotation_df: pd.DataFrame,
):
    """
    Add suggested or manually edited labels to `adata.obs`.

    Behavior
    --------
    If `manual_label` is filled in, it is used. Otherwise, this function falls
    back to `suggested_label`.

    Why this is useful
    ------------------
    This allows the same function to work before and after manual curation.
    Final manual labels are applied more formally in script 4.
    """
    label_map = {}

    for _, row in annotation_df.iterrows():
        cluster = str(row["cluster"])

        if isinstance(row.get("manual_label", ""), str) and row["manual_label"].strip():
            label = row["manual_label"].strip()
        else:
            label = row["suggested_label"]

        label_map[cluster] = label

    combined_label_map = {
        cluster: f"{cluster}: {label}" for cluster, label in label_map.items()
    }

    adata.obs["suggested_niche_label"] = (
        adata.obs[CLUSTER_KEY].astype(str).map(label_map).astype("category")
    )

    adata.obs["cluster_suggested_niche_label"] = (
        adata.obs[CLUSTER_KEY].astype(str).map(combined_label_map).astype("category")
    )

    return adata


def run_neighborhood_enrichment(adata) -> None:
    """
    Run Squidpy neighborhood enrichment between Leiden clusters.

    Why this analysis is useful
    ---------------------------
    Marker genes tell us what a cluster may represent. Neighborhood enrichment
    asks whether clusters are physically near each other more often than
    expected by chance.

    This can support biological interpretations such as:

        - tumor epithelial regions adjacent to hypoxic/metabolic regions
        - immune-enriched regions adjacent to tumor regions
        - adipocyte/fat-associated regions spatially separated from tumor cores

    Current parameters
    ------------------
    n_perms=1000:
        A reasonable default for stable enrichment z-scores.

    seed=42:
        Reproducible permutation results.

    n_jobs=1:
        Conservative setting that avoids oversubscription issues on laptops.

    Alternative parameters
    ----------------------
    n_perms=100:
        Faster for testing.

    n_perms=5000:
        Slower but more stable for final reporting.

    cluster_key="suggested_niche_label":
        Could be used after manual/suggested labels are available, but here we
        run on Leiden clusters because manual labels are not final yet.
    """
    print("\nRunning Squidpy neighborhood enrichment between Leiden clusters...")

    adata.obs[CLUSTER_KEY] = adata.obs[CLUSTER_KEY].astype("category")

    sq.gr.nhood_enrichment(
        adata,
        cluster_key=CLUSTER_KEY,
        n_perms=1000,
        seed=42,
        n_jobs=1,
        show_progress_bar=True,
    )

    result_key = f"{CLUSTER_KEY}_nhood_enrichment"

    if result_key not in adata.uns:
        print(f"Warning: expected {result_key} in adata.uns but did not find it.")
        return

    enrichment = adata.uns[result_key]

    categories = adata.obs[CLUSTER_KEY].cat.categories.astype(str).tolist()

    if "zscore" in enrichment:
        zscore_df = pd.DataFrame(
            enrichment["zscore"],
            index=categories,
            columns=categories,
        )
        zscore_path = (
            TABLE_DIR / "leiden_r06_spatial_neighborhood_enrichment_zscore.csv"
        )
        zscore_df.to_csv(zscore_path)
        print(f"Saved neighborhood enrichment z-scores to: {zscore_path}")

    if "count" in enrichment:
        count_df = pd.DataFrame(
            enrichment["count"],
            index=categories,
            columns=categories,
        )
        count_path = (
            TABLE_DIR / "leiden_r06_spatial_neighborhood_enrichment_count.csv"
        )
        count_df.to_csv(count_path)
        print(f"Saved neighborhood enrichment counts to: {count_path}")

    sq.pl.nhood_enrichment(
        adata,
        cluster_key=CLUSTER_KEY,
        annotate=True,
        method="average",
        figsize=(8, 8),
    )
    save_current_fig("leiden_r06_spatial_neighborhood_enrichment.png")


def run_marker_spatial_autocorrelation(
    adata,
    candidate_genes: list[str],
) -> None:
    """
    Run Moran's I spatial autocorrelation on selected marker genes.

    Why this analysis is useful
    ---------------------------
    Spatial marker plots are visually informative, but Moran's I provides a
    quantitative measure of whether a gene is spatially structured.

    This helps distinguish:

        - genes with clear spatial organization
        - genes that are expressed but spatially diffuse
        - genes that may not be informative for spatial niche interpretation

    Current parameters
    ------------------
    mode="moran":
        Moran's I is a common global spatial autocorrelation statistic.

    n_perms=100:
        Adds permutation-based significance while keeping runtime manageable.

    use_raw=True:
        Uses the full normalized expression matrix stored in adata.raw.

    Alternative parameters
    ----------------------
    mode="geary":
        Geary's C can be used as an alternative spatial autocorrelation metric.

    n_perms=None:
        Faster analytic-only p-values.

    n_perms=1000:
        More robust but slower.

    attr="obs":
        Could be used later to test spatial autocorrelation of marker signature
        scores or QC metrics stored in adata.obs.
    """
    autocorr_genes = get_available_genes(adata, candidate_genes)
    autocorr_genes = list(dict.fromkeys(autocorr_genes))

    if not autocorr_genes:
        print("\nNo candidate genes available for Moran's I; skipping.")
        return

    print("\nRunning Moran's I spatial autocorrelation for selected marker genes:")
    print(autocorr_genes)

    sq.gr.spatial_autocorr(
        adata,
        genes=autocorr_genes,
        mode="moran",
        n_perms=100,
        seed=42,
        use_raw=True if adata.raw is not None else False,
        n_jobs=1,
        show_progress_bar=True,
    )

    if "moranI" not in adata.uns:
        print("Warning: Moran's I result not found in adata.uns['moranI'].")
        return

    moran_df = adata.uns["moranI"].copy()
    moran_path = TABLE_DIR / "selected_marker_gene_morans_i.csv"
    moran_df.to_csv(moran_path)
    print(f"Saved Moran's I results to: {moran_path}")

    # Plot the top spatially autocorrelated marker genes.
    score_col = "I" if "I" in moran_df.columns else moran_df.columns[0]
    top_moran_genes = moran_df.sort_values(score_col, ascending=False).head(12).index.tolist()

    print("\nTop spatially autocorrelated marker genes:")
    print(top_moran_genes)

    plot_spatial_scatter(
        adata,
        color=top_moran_genes,
        filename="spatial_top_morans_i_marker_genes.png",
        use_raw=True if adata.raw is not None else False,
        img=True,
        ncols=4,
        legend_loc=None,
    )


def main() -> None:
    """
    Run formal marker gene analysis and Squidpy spatial marker analyses.

    Workflow
    --------
    1. Load clustered AnnData object.
    2. Validate required cluster and spatial graph fields.
    3. Run differential expression using Scanpy.
    4. Export full and filtered marker gene tables.
    5. Create an annotation template for manual review.
    6. Add suggested labels to AnnData.
    7. Run Squidpy neighborhood enrichment.
    8. Run Moran's I for selected marker genes.
    9. Generate marker plots and spatial label plots.
    10. Save marker-annotated AnnData object.
    """
    print(f"Loading clustered AnnData object from: {INPUT_H5AD}")

    adata = sc.read_h5ad(INPUT_H5AD)

    print("\nLoaded AnnData object:")
    print(adata)

    validate_required_fields(adata)

    print("\nCluster counts:")
    cluster_counts = adata.obs[CLUSTER_KEY].astype(str).value_counts().sort_index()
    print(cluster_counts)

    # ------------------------------------------------------------------
    # 1. Run marker gene discovery
    # ------------------------------------------------------------------
    print("\nRunning differential expression with Wilcoxon rank-sum test...")

    sc.tl.rank_genes_groups(
        adata,
        groupby=CLUSTER_KEY,
        method="wilcoxon",
        use_raw=True if adata.raw is not None else False,
        pts=True,
        key_added="rank_genes_leiden_r06",
    )

    print("Differential expression complete.")

    # ------------------------------------------------------------------
    # 2. Export full marker table
    # ------------------------------------------------------------------
    marker_df = sc.get.rank_genes_groups_df(
        adata,
        group=None,
        key="rank_genes_leiden_r06",
    )

    marker_df = marker_df.rename(columns={"names": "gene"})

    marker_table_path = TABLE_DIR / "leiden_r06_full_marker_gene_table.csv"
    marker_df.to_csv(marker_table_path, index=False)

    print(f"\nSaved full marker gene table to: {marker_table_path}")

    # ------------------------------------------------------------------
    # 3. Export filtered top markers
    # ------------------------------------------------------------------
    top_markers_df = extract_top_markers(
        marker_df.rename(columns={"gene": "names"}),
        top_n=20,
        min_logfc=0.25,
        max_adj_pval=0.05,
    )

    top_marker_path = TABLE_DIR / "leiden_r06_top20_marker_genes_per_cluster.csv"
    top_markers_df.to_csv(top_marker_path, index=False)

    print(f"Saved top marker genes to: {top_marker_path}")

    print("\nTop 5 markers per cluster:")
    for cluster, sub_df in top_markers_df.groupby("group"):
        genes = sub_df["gene"].head(5).tolist()
        print(f"  Cluster {cluster}: {', '.join(genes)}")

    # ------------------------------------------------------------------
    # 4. Create preliminary annotation table
    # ------------------------------------------------------------------
    annotation_df = suggest_cluster_labels(top_markers_df, cluster_counts)

    annotation_path = TABLE_DIR / "leiden_r06_cluster_annotation_template.csv"
    annotation_df.to_csv(annotation_path, index=False)

    print(f"\nSaved cluster annotation template to: {annotation_path}")
    print("\nPreliminary annotation table:")
    print(annotation_df)

    adata = add_manual_or_suggested_labels(adata, annotation_df)

    # ------------------------------------------------------------------
    # 5. Plot ranked marker genes
    # ------------------------------------------------------------------
    sc.pl.rank_genes_groups(
        adata,
        key="rank_genes_leiden_r06",
        n_genes=5,
        sharey=False,
        show=False,
    )
    save_current_fig("rank_genes_groups_top5_per_cluster.png")

    # ------------------------------------------------------------------
    # 6. Dotplot and matrixplot of top markers
    # ------------------------------------------------------------------
    plot_genes = []

    for _, sub_df in top_markers_df.groupby("group"):
        plot_genes.extend(sub_df["gene"].head(3).tolist())

    plot_genes = list(dict.fromkeys(plot_genes))
    plot_genes = plot_genes[:40]

    print("\nTop marker genes used for dotplot/matrixplot:")
    print(plot_genes)

    if plot_genes:
        sc.pl.dotplot(
            adata,
            var_names=plot_genes,
            groupby=CLUSTER_KEY,
            use_raw=True if adata.raw is not None else False,
            standard_scale="var",
            show=False,
        )
        save_current_fig("dotplot_top_cluster_markers.png")

        sc.pl.matrixplot(
            adata,
            var_names=plot_genes,
            groupby=CLUSTER_KEY,
            use_raw=True if adata.raw is not None else False,
            standard_scale="var",
            show=False,
        )
        save_current_fig("matrixplot_top_cluster_markers.png")

    # ------------------------------------------------------------------
    # 7. Plot known biological marker genes
    # ------------------------------------------------------------------
    known_marker_genes = []

    for genes in KNOWN_MARKER_SETS.values():
        known_marker_genes.extend(genes)

    known_marker_genes = list(dict.fromkeys(known_marker_genes))
    known_marker_genes = get_available_genes(adata, known_marker_genes)

    known_marker_genes_for_plot = known_marker_genes[:50]

    if known_marker_genes_for_plot:
        sc.pl.dotplot(
            adata,
            var_names=known_marker_genes_for_plot,
            groupby=CLUSTER_KEY,
            use_raw=True if adata.raw is not None else False,
            standard_scale="var",
            show=False,
        )
        save_current_fig("dotplot_known_biological_markers.png")

    # ------------------------------------------------------------------
    # 8. Squidpy spatial neighborhood enrichment
    # ------------------------------------------------------------------
    run_neighborhood_enrichment(adata)

    # ------------------------------------------------------------------
    # 9. Plot suggested labels spatially and on UMAP
    # ------------------------------------------------------------------
    plot_spatial_scatter(
        adata,
        color=["cluster_suggested_niche_label"],
        filename="spatial_suggested_niche_labels.png",
        img=True,
        legend_loc="right margin",
    )

    sc.pl.umap(
        adata,
        color=["cluster_suggested_niche_label"],
        legend_loc="right margin",
        frameon=False,
        show=False,
    )
    save_current_fig("umap_suggested_niche_labels.png")

    # ------------------------------------------------------------------
    # 10. Plot selected key spatial markers with Squidpy
    # ------------------------------------------------------------------
    selected_spatial_markers = [
        "EPCAM",
        "KRT18",
        "KRT19",
        "MUC1",
        "SCGB2A2",
        "SCGB1D2",
        "COL1A1",
        "DCN",
        "LUM",
        "PTPRC",
        "CD3D",
        "CD8A",
        "MS4A1",
        "IGKC",
        "JCHAIN",
        "CD74",
        "HLA-DRA",
        "LST1",
        "C1QA",
        "CD68",
        "MKI67",
        "TOP2A",
        "FABP4",
        "PLIN1",
        "GAPDH",
        "PGK1",
        "LDHA",
        "ENO1",
        "PECAM1",
        "VWF",
    ]

    selected_spatial_markers = get_available_genes(adata, selected_spatial_markers)

    if selected_spatial_markers:
        plot_spatial_scatter(
            adata,
            color=selected_spatial_markers,
            filename="spatial_selected_biological_markers.png",
            use_raw=True if adata.raw is not None else False,
            img=True,
            ncols=4,
            legend_loc=None,
        )

    # ------------------------------------------------------------------
    # 11. Squidpy Moran's I spatial autocorrelation
    # ------------------------------------------------------------------
    autocorr_candidates = list(dict.fromkeys(selected_spatial_markers + plot_genes[:20]))
    run_marker_spatial_autocorrelation(adata, autocorr_candidates)

    # ------------------------------------------------------------------
    # 12. Save updated AnnData object
    # ------------------------------------------------------------------
    adata.write_h5ad(OUTPUT_H5AD)

    print(f"\nSaved marker-annotated AnnData object to: {OUTPUT_H5AD}")
    print("\nMilestone 3 complete.")


if __name__ == "__main__":
    main()
