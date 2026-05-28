from pathlib import Path

import matplotlib.pyplot as plt
import scanpy as sc


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "10x"
    / "Visium_Human_Breast_Cancer"
)

COUNT_FILE = "Visium_Human_Breast_Cancer_filtered_feature_bc_matrix.h5"

OUT_DIR = PROJECT_ROOT / "results" / "figures" / "01_initial_qc"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OUT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print(f"Loading Visium data from: {DATA_DIR}")

    adata = sc.read_visium(
        path=DATA_DIR,
        count_file=COUNT_FILE,
        library_id="Visium_Human_Breast_Cancer",
        load_images=True,
    )

    adata.var_names_make_unique()

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

    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")

    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    print("\nQC summary:")
    print(adata.obs[["total_counts", "n_genes_by_counts", "pct_counts_mt"]].describe())

    out_h5ad = PROCESSED_DIR / "visium_human_breast_cancer_raw_qc.h5ad"
    adata.write_h5ad(out_h5ad)
    print(f"\nSaved QC AnnData object to: {out_h5ad}")

    sc.pl.violin(
        adata,
        keys=["total_counts", "n_genes_by_counts", "pct_counts_mt"],
        jitter=0.4,
        multi_panel=True,
        show=False,
    )
    plt.savefig(OUT_DIR / "qc_violin.png", dpi=300, bbox_inches="tight")
    plt.close()

    sc.pl.spatial(
        adata,
        color=["total_counts", "n_genes_by_counts", "pct_counts_mt"],
        library_id="Visium_Human_Breast_Cancer",
        show=False,
    )
    plt.savefig(OUT_DIR / "spatial_qc_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()

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

    if available_markers:
        sc.pl.spatial(
            adata,
            color=available_markers,
            library_id="Visium_Human_Breast_Cancer",
            show=False,
        )
        plt.savefig(OUT_DIR / "spatial_marker_genes.png", dpi=300, bbox_inches="tight")
        plt.close()

    print("\nDone. Initial QC figures saved to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
