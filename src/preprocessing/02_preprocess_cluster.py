from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc


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

LIBRARY_ID = "Visium_Human_Breast_Cancer"


def save_current_fig(filename: str):
    """Save the current matplotlib figure and close it."""
    out_path = FIGURE_DIR / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {out_path}")


def main():
    print(f"Loading QC AnnData object from: {INPUT_H5AD}")
    adata = sc.read_h5ad(INPUT_H5AD)

    print("\nOriginal AnnData object:")
    print(adata)

    original_n_spots = adata.n_obs
    original_n_genes = adata.n_vars

    print("\nOriginal QC summary:")
    print(
        adata.obs[
            ["total_counts", "n_genes_by_counts", "pct_counts_mt"]
        ].describe()
    )

    # ------------------------------------------------------------------
    # 1. Conservative QC filtering
    # ------------------------------------------------------------------
    # Based on the first QC run:
    # - Some spots have very low counts and low detected genes.
    # - Mitochondrial percentages are generally reasonable.
    # This keeps filtering conservative so we do not remove useful tissue
    # regions too early.
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

    # Spatial QC after filtering
    sc.pl.spatial(
        adata,
        color=["total_counts", "n_genes_by_counts", "pct_counts_mt"],
        library_id=LIBRARY_ID,
        show=False,
    )
    save_current_fig("spatial_qc_after_filtering.png")

    # ------------------------------------------------------------------
    # 2. Preserve raw counts
    # ------------------------------------------------------------------
    # Keep a copy of the count matrix before normalization.
    adata.layers["counts"] = adata.X.copy()

    # ------------------------------------------------------------------
    # 3. Normalize and log-transform
    # ------------------------------------------------------------------
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Store normalized/log-transformed values for later marker plotting.
    adata.raw = adata

    # ------------------------------------------------------------------
    # 4. Highly variable genes
    # ------------------------------------------------------------------
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

    # Subset to highly variable genes for PCA/clustering.
    adata = adata[:, adata.var["highly_variable"]].copy()

    # ------------------------------------------------------------------
    # 5. Scaling, PCA, neighbors, UMAP, Leiden
    # ------------------------------------------------------------------
    sc.pp.scale(adata, max_value=10)

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

    sc.pp.neighbors(
        adata,
        n_neighbors=12,
        n_pcs=30,
    )

    sc.tl.umap(adata, random_state=42)

    # Try a moderate resolution first. We can tune this later.
    sc.tl.leiden(
        adata,
        resolution=0.6,
        key_added="leiden_r06",
        random_state=42,
    )

    print("\nLeiden cluster counts:")
    print(adata.obs["leiden_r06"].value_counts().sort_index())

    cluster_counts = (
        adata.obs["leiden_r06"]
        .value_counts()
        .sort_index()
        .rename_axis("cluster")
        .reset_index(name="n_spots")
    )
    cluster_counts.to_csv(TABLE_DIR / "leiden_r06_cluster_counts.csv", index=False)

    # ------------------------------------------------------------------
    # 6. Plots
    # ------------------------------------------------------------------
    sc.pl.umap(
        adata,
        color=["leiden_r06"],
        legend_loc="on data",
        frameon=False,
        show=False,
    )
    save_current_fig("umap_leiden_r06.png")

    sc.pl.spatial(
        adata,
        color=["leiden_r06"],
        library_id=LIBRARY_ID,
        show=False,
    )
    save_current_fig("spatial_leiden_r06.png")

    # Plot key marker genes after normalization/log transform.
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

    # After HVG subsetting, some marker genes may no longer be in adata.var_names.
    # Use adata.raw for plotting marker genes from the full normalized dataset.
    available_markers = [
        gene for gene in marker_genes if gene in adata.raw.var_names
    ]

    print("\nAvailable marker genes for plotting:")
    print(available_markers)

    if available_markers:
        sc.pl.spatial(
            adata,
            color=available_markers,
            library_id=LIBRARY_ID,
            use_raw=True,
            show=False,
        )
        save_current_fig("spatial_marker_genes_normalized.png")

    # ------------------------------------------------------------------
    # 7. Save processed object
    # ------------------------------------------------------------------
    output_h5ad = PROCESSED_DIR / "visium_human_breast_cancer_processed_clustered.h5ad"
    adata.write_h5ad(output_h5ad)

    print(f"\nSaved processed AnnData object to: {output_h5ad}")
    print("\nMilestone 2 complete.")


if __name__ == "__main__":
    main()
