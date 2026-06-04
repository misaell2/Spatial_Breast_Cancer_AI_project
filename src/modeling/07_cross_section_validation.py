"""
Milestone 7: Cross-section validation and multi-section model training.

Purpose
-------
This script evaluates whether spatial niche classifiers trained on one breast
cancer Visium section can generalize to another section from the same tissue
block / serial-section pair.

This milestone is different from the previous validation scripts:

    Script 05:
        Random spot-level train/test split within one section.

    Script 06:
        Spatial holdout validation within one section.

    Script 07:
        Cross-section validation between two tissue sections.

Why this matters
----------------
The current project already shows that a model can classify spatial niches
within one Visium section. However, random splits and within-section spatial
holdout validation do not fully answer whether the model generalizes to another
tissue section.

This script asks a stronger question:

    If I train on Section 1, can I predict niches in Section 2?
    If I train on Section 2, can I predict niches in Section 1?

This is especially useful for the newly added 10x Genomics Block A pair:

    V1_Breast_Cancer_Block_A_Section_1
    V1_Breast_Cancer_Block_A_Section_2

These are well suited for a same-block / serial-section validation milestone.

Expected input
--------------
This script expects each section to already have a final labeled AnnData object
created by the earlier project workflow.

Recommended layout:

    data/processed/V1_Breast_Cancer_Block_A_Section_1/final_labeled.h5ad
    data/processed/V1_Breast_Cancer_Block_A_Section_2/final_labeled.h5ad

Each final labeled AnnData object should contain:

    adata.obs["ml_training_label"]
    adata.obs["manual_niche_label_short"]
    adata.obsm["spatial"]
    adata.raw
    adata.obsp["spatial_connectivities"]  # recommended, can be rebuilt

Why final labeled objects are required
--------------------------------------
Cluster IDs are not comparable across sections. For example:

    Section 1 cluster 3 may be Tumor epithelial
    Section 2 cluster 7 may be Tumor epithelial

Therefore, this script does not use Leiden cluster IDs as labels. It uses
harmonized biological labels stored in:

    adata.obs["ml_training_label"]

This means Section 1 and Section 2 must be annotated using the same label
vocabulary before cross-section validation.

Outputs
-------
Tables:

    results/tables/cross_section_validation_metrics.csv
    results/tables/cross_section_summary_by_model.csv
    results/tables/cross_section_summary_by_feature_set.csv
    results/tables/cross_section_summary_by_model_and_feature_set.csv
    results/tables/cross_section_label_counts.csv
    results/tables/cross_section_feature_columns.csv
    results/tables/cross_section_*_classification_report.csv

Figures:

    results/figures/07_cross_section_validation/

Models:

    models/cross_section_spatial_niche_classifier.joblib
    models/cross_section_model_metadata.json

Validation design
-----------------
The script performs two directions:

    1. Train Section 1 -> Test Section 2
    2. Train Section 2 -> Test Section 1

For each direction, it evaluates multiple models and feature sets.

Why not use precomputed adata.obsm["X_pca"]?
--------------------------------------------
Precomputed PCA coordinates from Section 1 and Section 2 are usually not in the
same coordinate system if PCA was fit independently per section.

For cross-section validation, expression-derived components should be learned
from the training section and then applied to the test section. This script uses
TruncatedSVD on common genes for that reason.

Scientific caution
------------------
This is stronger than within-section validation, but it is still not full
multi-patient validation. A true external validation milestone would require
additional tissue blocks or patients.
"""

from pathlib import Path
import json
import re
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq

from scipy import sparse

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

FIGURE_DIR = PROJECT_ROOT / "results" / "figures" / "07_cross_section_validation"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
MODEL_DIR = PROJECT_ROOT / "models"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Section configuration
# ---------------------------------------------------------------------
# These paths are intentionally explicit.
#
# Why:
#   Cross-section validation should not accidentally use the single-section
#   processed object from the earlier project. Each section should have its own
#   processed/labeled AnnData file.
#
# Before running this script, process and annotate each section independently
# and save the final labeled object to these locations.
SECTION_CONFIGS = [
    {
        "section_id": "V1_Breast_Cancer_Block_A_Section_1",
        "display_name": "Block A Section 1",
        "h5ad_path": (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "V1_Breast_Cancer_Block_A_Section_1"
            / "final_labeled.h5ad"
        ),
    },
    {
        "section_id": "V1_Breast_Cancer_Block_A_Section_2",
        "display_name": "Block A Section 2",
        "h5ad_path": (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "V1_Breast_Cancer_Block_A_Section_2"
            / "final_labeled.h5ad"
        ),
    },
]


# ---------------------------------------------------------------------
# Label and spatial metadata
# ---------------------------------------------------------------------
LABEL_KEY = "ml_training_label"
EXCLUDE_LABEL = "Exclude_low_confidence"

MANUAL_LABEL_KEY = "manual_niche_label_short"
SPATIAL_CONNECTIVITY_KEY = "spatial_connectivities"


# ---------------------------------------------------------------------
# Cross-section validation settings
# ---------------------------------------------------------------------
# Restricting evaluation to labels present in both sections makes the
# validation fairer. A model cannot learn a class that is absent from the
# training section.
#
# Alternative:
#   Set REQUIRE_SHARED_LABELS = False to keep all high-confidence labels and
#   report labels that are missing from the training section.
REQUIRE_SHARED_LABELS = True

# Top variable genes selected from the training section for expression SVD.
# Alternatives: 1000 for faster testing, 5000 for richer expression features.
N_TOP_EXPRESSION_GENES = 3000

# Number of expression components learned by TruncatedSVD.
N_EXPRESSION_COMPONENTS = 30


# ---------------------------------------------------------------------
# Marker sets used as interpretable cross-section features
# ---------------------------------------------------------------------
MARKER_SETS = {
    "tumor_epithelial": [
        "EPCAM",
        "KRT8",
        "KRT18",
        "KRT19",
        "KRT7",
        "MUC1",
        "TACSTD2",
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
        "CXCL13",
    ],
    "t_cell": [
        "CD3D",
        "CD3E",
        "CD2",
        "TRAC",
        "CD8A",
        "CD8B",
        "GZMB",
        "NKG7",
    ],
    "b_cell_plasma": [
        "MS4A1",
        "CD79A",
        "CD79B",
        "BANK1",
        "MZB1",
        "JCHAIN",
        "IGKC",
        "IGHG1",
        "IGHG3",
    ],
    "myeloid_apc": [
        "CD74",
        "HLA-DRA",
        "HLA-DPA1",
        "HLA-DPB1",
        "C1QA",
        "C1QB",
        "LYZ",
        "LST1",
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
        "UBE2C",
    ],
    "adipocyte_fat": [
        "FABP4",
        "PLIN1",
        "ADIPOQ",
        "LPL",
        "G0S2",
        "CFD",
    ],
    "hypoxia_glycolysis": [
        "GAPDH",
        "PGK1",
        "TPI1",
        "ENO1",
        "LDHA",
        "VEGFA",
        "CA9",
    ],
    "luminal_secretory": [
        "SCGB2A2",
        "SCGB1D2",
        "CSTA",
        "S100G",
        "GATA3",
        "XBP1",
    ],
}


def safe_name(text: str) -> str:
    """
    Convert arbitrary text into a filename-safe string.

    Why this helper exists
    ----------------------
    Model names, feature set names, and validation directions are used in file
    names. This function avoids spaces and punctuation in output paths.
    """
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
    return text.strip("_")


def save_current_fig(filename: str) -> None:
    """
    Save the active matplotlib figure and close it.
    """
    out_path = FIGURE_DIR / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {out_path}")


def validate_section_input(adata, section_id: str) -> None:
    """
    Validate that one section has the fields needed for cross-section modeling.

    Why this function exists
    ------------------------
    This script assumes the earlier project workflow has already produced
    labeled AnnData objects. If a field is missing, we want a clear error that
    explains what needs to be rerun.
    """
    required_obs = [LABEL_KEY, MANUAL_LABEL_KEY]
    missing_obs = [col for col in required_obs if col not in adata.obs.columns]

    if missing_obs:
        raise ValueError(
            f"{section_id} is missing required adata.obs columns: {missing_obs}. "
            "Rerun manual annotation/application for this section."
        )

    if "spatial" not in adata.obsm:
        raise ValueError(
            f"{section_id} is missing adata.obsm['spatial']. "
            "Rerun the Visium loading/preprocessing workflow."
        )

    if adata.raw is None:
        warnings.warn(
            f"{section_id} does not have adata.raw. Expression SVD and marker "
            "scoring will use adata.X/adata.var_names, which may be HVG-subset.",
            UserWarning,
        )


def load_sections() -> dict[str, sc.AnnData]:
    """
    Load all configured sections.

    Why this is separated into a function
    -------------------------------------
    It gives one clear place to control expected paths and section IDs. Later,
    this can be replaced by reading `data_manifest/datasets.csv`.
    """
    sections = {}

    for cfg in SECTION_CONFIGS:
        section_id = cfg["section_id"]
        h5ad_path = cfg["h5ad_path"]

        print(f"\nLoading {section_id} from: {h5ad_path}")

        if not h5ad_path.exists():
            raise FileNotFoundError(
                f"Could not find final labeled AnnData for {section_id}:\n"
                f"  {h5ad_path}\n\n"
                "Expected workflow before running script 07:\n"
                "  1. Run QC/preprocessing for this section.\n"
                "  2. Run marker analysis for this section.\n"
                "  3. Apply manual harmonized labels for this section.\n"
                "  4. Save the result as final_labeled.h5ad in the section directory."
            )

        adata = sc.read_h5ad(h5ad_path)
        validate_section_input(adata, section_id)

        adata.obs["section_id"] = section_id
        adata.obs["section_display_name"] = cfg["display_name"]

        print(adata)
        print("\nLabel counts:")
        print(adata.obs[LABEL_KEY].astype(str).value_counts())

        sections[section_id] = adata

    return sections


def get_available_genes(adata, genes: list[str]) -> list[str]:
    """
    Return genes available in the AnnData object.

    Why use adata.raw?
    ------------------
    The final labeled object may be subset to highly variable genes, but
    `adata.raw` usually preserves the full normalized/log-transformed
    expression matrix before HVG subsetting.
    """
    if adata.raw is not None:
        available = set(adata.raw.var_names)
    else:
        available = set(adata.var_names)

    return [gene for gene in genes if gene in available]


def add_marker_signature_scores(adata, section_id: str):
    """
    Add marker signature scores to one section.

    Why this is done per section
    ----------------------------
    Marker score values are calculated within each AnnData object. The same gene
    sets are used across sections, which makes the features biologically
    comparable even if the sections differ in local composition.
    """
    score_columns = []

    for signature_name, genes in MARKER_SETS.items():
        available_genes = get_available_genes(adata, genes)

        print(f"\n{section_id} signature: {signature_name}")
        print(f"  Available genes: {available_genes}")

        if len(available_genes) < 2:
            print("  Skipping: fewer than 2 genes available.")
            continue

        score_col = f"score_{signature_name}"

        sc.tl.score_genes(
            adata,
            gene_list=available_genes,
            score_name=score_col,
            use_raw=True if adata.raw is not None else False,
        )

        score_columns.append(score_col)

    return adata, score_columns


def ensure_spatial_neighbors(adata, section_id: str):
    """
    Ensure a Squidpy spatial-neighbor graph exists for one section.

    Why this matters
    ----------------
    The neighbor-averaged marker features depend on
    `adata.obsp["spatial_connectivities"]`.

    If the graph is missing, this function builds a Visium-style grid graph.
    """
    if SPATIAL_CONNECTIVITY_KEY in adata.obsp:
        return adata

    print(
        f"\n{section_id}: {SPATIAL_CONNECTIVITY_KEY} not found. "
        "Building Squidpy spatial-neighbor graph..."
    )

    sq.gr.spatial_neighbors(
        adata,
        spatial_key="spatial",
        coord_type="grid",
        n_neighs=6,
        n_rings=1,
    )

    return adata


def make_neighbor_score_features(adata, score_columns: list[str]) -> pd.DataFrame:
    """
    Calculate neighbor-averaged marker score features.

    Why this feature is useful
    --------------------------
    A spot's biological niche may depend on local tissue context, not only its
    own expression. Neighbor-averaged marker scores summarize nearby tissue
    programs using the Squidpy spatial graph.
    """
    if SPATIAL_CONNECTIVITY_KEY not in adata.obsp or not score_columns:
        return pd.DataFrame(index=adata.obs_names)

    graph = adata.obsp[SPATIAL_CONNECTIVITY_KEY].copy()

    row_sums = np.asarray(graph.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0

    graph_normalized = graph.multiply(1.0 / row_sums[:, None])

    score_matrix = adata.obs[score_columns].astype(float).to_numpy()
    neighbor_scores = graph_normalized.dot(score_matrix)

    neighbor_columns = [f"neighbor_mean_{col}" for col in score_columns]

    neighbor_df = pd.DataFrame(
        neighbor_scores,
        index=adata.obs_names,
        columns=neighbor_columns,
    )

    return neighbor_df


def make_qc_features(adata) -> pd.DataFrame:
    """
    Build QC feature table for one section.

    Why include QC metrics?
    -----------------------
    QC metrics can capture technical variation that affects expression-based
    models. Including them lets the model adjust for library size, detected
    genes, mitochondrial signal, and related quality features.
    """
    possible_qc_cols = [
        "total_counts",
        "n_genes_by_counts",
        "pct_counts_mt",
        "pct_counts_ribo",
        "pct_counts_hb",
        "pct_counts_in_top_50_genes",
        "pct_counts_in_top_100_genes",
        "pct_counts_in_top_200_genes",
    ]

    qc_cols = [col for col in possible_qc_cols if col in adata.obs.columns]

    if not qc_cols:
        warnings.warn(
            "No QC columns found. Returning an empty QC feature table.",
            UserWarning,
        )
        return pd.DataFrame(index=adata.obs_names)

    return adata.obs[qc_cols].astype(float).copy()


def make_normalized_spatial_features(adata) -> pd.DataFrame:
    """
    Make section-normalized spatial coordinate features.

    Why normalize spatial coordinates?
    ----------------------------------
    Raw pixel coordinates are not directly comparable across serial sections.
    Normalizing x and y into a 0-1 range within each section makes them relative
    tissue-position features.

    Important caution
    -----------------
    Even normalized coordinates may not align perfectly across sections if
    tissue orientation or section placement differs. For this reason, spatial
    coordinates are tested as a separate feature set rather than assumed to be
    universally helpful.
    """
    spatial = adata.obsm["spatial"].astype(float)

    x = spatial[:, 0]
    y = spatial[:, 1]

    x_range = x.max() - x.min()
    y_range = y.max() - y.min()

    x_norm = np.zeros_like(x) if x_range == 0 else (x - x.min()) / x_range
    y_norm = np.zeros_like(y) if y_range == 0 else (y - y.min()) / y_range

    return pd.DataFrame(
        {
            "spatial_x_norm": x_norm,
            "spatial_y_norm": y_norm,
        },
        index=adata.obs_names,
    )


def get_raw_gene_names(adata) -> pd.Index:
    """Return the gene names available for expression features."""
    if adata.raw is not None:
        return pd.Index(adata.raw.var_names)
    return pd.Index(adata.var_names)


def get_expression_matrix_for_genes(adata, genes: list[str]):
    """
    Extract expression matrix for a specific ordered gene list.

    Why this function exists
    ------------------------
    Cross-section expression features must use the same genes in the same order
    for training and testing. This helper centralizes that extraction.
    """
    if adata.raw is not None:
        X = adata.raw[:, genes].X
    else:
        X = adata[:, genes].X

    return X


def calculate_sparse_variance(X) -> np.ndarray:
    """
    Calculate per-gene variance for dense or sparse expression matrices.
    """
    if sparse.issparse(X):
        mean = np.asarray(X.mean(axis=0)).ravel()
        mean_sq = np.asarray(X.power(2).mean(axis=0)).ravel()
        variance = mean_sq - mean**2
    else:
        variance = np.asarray(X).var(axis=0)

    return variance


def select_training_variable_genes(
    train_adata,
    test_adata,
    n_top_genes: int = N_TOP_EXPRESSION_GENES,
) -> list[str]:
    """
    Select top variable genes from the training section that also exist in test.

    Why use training-only selection?
    --------------------------------
    Feature selection must not use information from the test section. Selecting
    variable genes using only the training section avoids test-set leakage.
    """
    train_genes = get_raw_gene_names(train_adata)
    test_genes = get_raw_gene_names(test_adata)

    common_genes = train_genes.intersection(test_genes).tolist()

    if len(common_genes) < 100:
        raise ValueError(
            "Too few common genes between train and test sections for "
            "expression SVD."
        )

    X_train_common = get_expression_matrix_for_genes(train_adata, common_genes)
    variances = calculate_sparse_variance(X_train_common)

    n_select = min(n_top_genes, len(common_genes))
    top_idx = np.argsort(variances)[::-1][:n_select]

    selected_genes = [common_genes[i] for i in top_idx]

    return selected_genes


def fit_transform_expression_svd(
    train_adata,
    test_adata,
    n_components: int = N_EXPRESSION_COMPONENTS,
):
    """
    Fit expression SVD on training section and transform test section.

    Why this replaces precomputed PCA
    ---------------------------------
    PCA coordinates computed independently per section are not directly
    comparable. Here, expression components are learned from the training
    section and then applied to the test section, which creates a valid
    cross-section feature space.
    """
    selected_genes = select_training_variable_genes(
        train_adata=train_adata,
        test_adata=test_adata,
        n_top_genes=N_TOP_EXPRESSION_GENES,
    )

    X_train_expr = get_expression_matrix_for_genes(train_adata, selected_genes)
    X_test_expr = get_expression_matrix_for_genes(test_adata, selected_genes)

    max_components = min(
        n_components,
        X_train_expr.shape[0] - 1,
        X_train_expr.shape[1] - 1,
    )

    if max_components < 2:
        raise ValueError("Not enough observations/features for expression SVD.")

    svd = TruncatedSVD(
        n_components=max_components,
        random_state=42,
    )

    train_svd = svd.fit_transform(X_train_expr)
    test_svd = svd.transform(X_test_expr)

    svd_columns = [f"expr_svd_{i + 1}" for i in range(max_components)]

    train_svd_df = pd.DataFrame(
        train_svd,
        index=train_adata.obs_names,
        columns=svd_columns,
    )

    test_svd_df = pd.DataFrame(
        test_svd,
        index=test_adata.obs_names,
        columns=svd_columns,
    )

    metadata = {
        "selected_genes": selected_genes,
        "n_selected_genes": len(selected_genes),
        "n_components": max_components,
        "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
    }

    return train_svd_df, test_svd_df, svd, metadata


def build_static_feature_tables(adata, score_columns: list[str]) -> dict[str, pd.DataFrame]:
    """
    Build non-expression-SVD feature tables for one section.

    Static means these features can be computed independently for each section:

        - QC features
        - marker signature scores
        - normalized spatial coordinates
        - Squidpy neighbor marker scores
    """
    qc_df = make_qc_features(adata)
    marker_df = adata.obs[score_columns].astype(float).copy()
    spatial_df = make_normalized_spatial_features(adata)
    neighbor_df = make_neighbor_score_features(adata, score_columns)

    static_tables = {
        "qc": qc_df,
        "marker": marker_df,
        "spatial": spatial_df,
        "neighbor": neighbor_df,
    }

    return static_tables


def clean_feature_table(features: pd.DataFrame) -> pd.DataFrame:
    """
    Replace infinite values and impute missing numeric values.
    """
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True))
    features = features.fillna(0.0)

    return features


def align_feature_columns(train_features: pd.DataFrame, test_features: pd.DataFrame):
    """
    Align train and test feature columns.

    Why this matters
    ----------------
    Some QC or marker features may be missing in one section. Models require the
    same feature columns in train and test, in the same order.
    """
    all_columns = sorted(set(train_features.columns).union(test_features.columns))

    train_aligned = train_features.reindex(columns=all_columns, fill_value=0.0)
    test_aligned = test_features.reindex(columns=all_columns, fill_value=0.0)

    train_aligned = clean_feature_table(train_aligned)
    test_aligned = clean_feature_table(test_aligned)

    return train_aligned, test_aligned


def build_direction_feature_sets(
    train_adata,
    test_adata,
    train_static: dict[str, pd.DataFrame],
    test_static: dict[str, pd.DataFrame],
):
    """
    Build all feature sets for one train-section -> test-section direction.

    Why direction-specific?
    -----------------------
    Expression SVD is fit on the training section. Therefore, expression-based
    features depend on which section is train and which section is test.
    """
    train_svd_df, test_svd_df, _svd, svd_metadata = fit_transform_expression_svd(
        train_adata=train_adata,
        test_adata=test_adata,
    )

    train_base = pd.concat([train_static["qc"], train_static["marker"]], axis=1)
    test_base = pd.concat([test_static["qc"], test_static["marker"]], axis=1)

    raw_feature_sets = {
        "marker_qc": (train_base, test_base),
        "marker_qc_spatial": (
            pd.concat([train_base, train_static["spatial"]], axis=1),
            pd.concat([test_base, test_static["spatial"]], axis=1),
        ),
        "marker_qc_neighbor": (
            pd.concat([train_base, train_static["neighbor"]], axis=1),
            pd.concat([test_base, test_static["neighbor"]], axis=1),
        ),
        "expression_svd_marker_qc": (
            pd.concat([train_svd_df, train_base], axis=1),
            pd.concat([test_svd_df, test_base], axis=1),
        ),
        "full_cross_section_context": (
            pd.concat(
                [train_svd_df, train_base, train_static["spatial"], train_static["neighbor"]],
                axis=1,
            ),
            pd.concat(
                [test_svd_df, test_base, test_static["spatial"], test_static["neighbor"]],
                axis=1,
            ),
        ),
    }

    feature_sets = {}

    for feature_set_name, (train_features, test_features) in raw_feature_sets.items():
        train_aligned, test_aligned = align_feature_columns(
            train_features=train_features,
            test_features=test_features,
        )

        uses_svd = feature_set_name in [
            "expression_svd_marker_qc",
            "full_cross_section_context",
        ]

        feature_sets[feature_set_name] = {
            "X_train": train_aligned,
            "X_test": test_aligned,
            "feature_columns": train_aligned.columns.tolist(),
            "svd_metadata": svd_metadata if uses_svd else None,
        }

    return feature_sets


def define_models() -> dict:
    """
    Define candidate classifiers for cross-section validation.

    Why these models?
    -----------------
    They match the model families already tested in the project:

        - linear probabilistic baseline
        - linear margin-based classifier
        - bagged tree ensemble
        - randomized tree ensemble
        - boosted tree ensemble
    """
    logistic_regression = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )

    calibrated_linear_svm = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                CalibratedClassifierCV(
                    estimator=LinearSVC(
                        class_weight="balanced",
                        max_iter=10000,
                        random_state=42,
                    ),
                    cv=3,
                ),
            ),
        ]
    )

    random_forest = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
        min_samples_leaf=3,
    )

    extra_trees = ExtraTreesClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
        min_samples_leaf=3,
    )

    hist_gradient_boosting = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        class_weight="balanced",
        random_state=42,
    )

    return {
        "logistic_regression": logistic_regression,
        "calibrated_linear_svm": calibrated_linear_svm,
        "random_forest": random_forest,
        "extra_trees": extra_trees,
        "hist_gradient_boosting": hist_gradient_boosting,
    }


def get_trainable_labels(sections: dict[str, sc.AnnData]) -> list[str]:
    """
    Determine which labels should be used for cross-section validation.

    Why shared labels by default?
    -----------------------------
    If a label appears only in the test section, a model trained on the other
    section cannot learn that class. Restricting to shared labels makes the
    validation focus on reproducible niches across sections.
    """
    label_sets = []

    for section_id, adata in sections.items():
        labels = set(
            adata.obs.loc[
                adata.obs[LABEL_KEY].astype(str) != EXCLUDE_LABEL,
                LABEL_KEY,
            ].astype(str)
        )
        label_sets.append(labels)

        print(f"\nHigh-confidence labels in {section_id}:")
        for label in sorted(labels):
            print(f"  {label}")

    if REQUIRE_SHARED_LABELS:
        shared_labels = sorted(set.intersection(*label_sets))
        print("\nUsing shared high-confidence labels only:")
        for label in shared_labels:
            print(f"  {label}")
        return shared_labels

    all_labels = sorted(set.union(*label_sets))
    print("\nUsing all high-confidence labels:")
    for label in all_labels:
        print(f"  {label}")
    return all_labels


def save_label_count_table(sections: dict[str, sc.AnnData]) -> None:
    """
    Save label counts by section.
    """
    rows = []

    for section_id, adata in sections.items():
        counts = adata.obs[LABEL_KEY].astype(str).value_counts()

        for label, count in counts.items():
            rows.append(
                {
                    "section_id": section_id,
                    "label": label,
                    "n_spots": int(count),
                }
            )

    label_counts = pd.DataFrame(rows)

    out_path = TABLE_DIR / "cross_section_label_counts.csv"
    label_counts.to_csv(out_path, index=False)

    print(f"\nSaved cross-section label counts to: {out_path}")


def filter_to_trainable_spots(adata, trainable_labels: list[str]) -> np.ndarray:
    """
    Create a boolean mask for spots included in cross-section validation.
    """
    labels = adata.obs[LABEL_KEY].astype(str)

    mask = (labels != EXCLUDE_LABEL) & labels.isin(trainable_labels)

    return mask.to_numpy()


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: list[str],
    title: str,
    filename: str,
) -> None:
    """
    Plot a confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest")

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax)
    save_current_fig(filename)


def evaluate_model_direction(
    model_name: str,
    model,
    feature_set_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    train_section: str,
    test_section: str,
) -> dict:
    """
    Train one model on one section and evaluate on the other section.
    """
    print("\n----------------------------------------")
    print(f"Model:       {model_name}")
    print(f"Feature set: {feature_set_name}")
    print(f"Train:       {train_section}")
    print(f"Test:        {test_section}")
    print("----------------------------------------")

    model.fit(X_train, y_train)
    y_pred = pd.Series(model.predict(X_test), index=y_test.index, name="predicted_label")

    labels = sorted(set(y_train.unique()).union(set(y_test.unique())).union(set(y_pred.unique())))

    metrics = {
        "train_section": train_section,
        "test_section": test_section,
        "direction": f"{train_section}_to_{test_section}",
        "model": model_name,
        "feature_set": feature_set_name,
        "n_train_spots": int(X_train.shape[0]),
        "n_test_spots": int(X_test.shape[0]),
        "n_train_labels": int(y_train.nunique()),
        "n_test_labels": int(y_test.nunique()),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "macro_f1": float(
            f1_score(y_test, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_test, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
    }

    print("Metrics:")
    for key in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]:
        print(f"  {key}: {metrics[key]:.4f}")

    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    report_path = (
        TABLE_DIR
        / (
            "cross_section_"
            f"{safe_name(model_name)}_"
            f"{safe_name(feature_set_name)}_"
            f"{safe_name(train_section)}_to_"
            f"{safe_name(test_section)}_classification_report.csv"
        )
    )

    pd.DataFrame(report).transpose().to_csv(report_path)

    plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        labels=labels,
        title=f"{model_name} | {feature_set_name}\n{train_section} -> {test_section}",
        filename=(
            "cross_section_"
            f"{safe_name(model_name)}_"
            f"{safe_name(feature_set_name)}_"
            f"{safe_name(train_section)}_to_"
            f"{safe_name(test_section)}_confusion_matrix.png"
        ),
    )

    return metrics


def run_cross_section_validation(
    sections: dict[str, sc.AnnData],
    static_features: dict[str, dict[str, pd.DataFrame]],
    trainable_labels: list[str],
) -> pd.DataFrame:
    """
    Run Section 1 -> Section 2 and Section 2 -> Section 1 validation.
    """
    section_ids = list(sections.keys())

    if len(section_ids) != 2:
        raise ValueError(
            "This version of script 07 expects exactly two sections. "
            "For more than two sections, extend this to leave-one-section-out."
        )

    directions = [(section_ids[0], section_ids[1]), (section_ids[1], section_ids[0])]

    models = define_models()
    all_metrics = []
    feature_column_rows = []

    for train_section, test_section in directions:
        train_adata = sections[train_section]
        test_adata = sections[test_section]

        train_mask = filter_to_trainable_spots(train_adata, trainable_labels)
        test_mask = filter_to_trainable_spots(test_adata, trainable_labels)

        y_train = train_adata.obs.loc[train_mask, LABEL_KEY].astype(str)
        y_test = test_adata.obs.loc[test_mask, LABEL_KEY].astype(str)

        print("\n========================================")
        print(f"Direction: {train_section} -> {test_section}")
        print("========================================")
        print("\nTraining label counts:")
        print(y_train.value_counts())
        print("\nTesting label counts:")
        print(y_test.value_counts())

        direction_feature_sets = build_direction_feature_sets(
            train_adata=train_adata,
            test_adata=test_adata,
            train_static=static_features[train_section],
            test_static=static_features[test_section],
        )

        for feature_set_name, payload in direction_feature_sets.items():
            X_train_all = payload["X_train"]
            X_test_all = payload["X_test"]

            X_train = X_train_all.loc[train_mask].copy()
            X_test = X_test_all.loc[test_mask].copy()

            for feature in payload["feature_columns"]:
                feature_column_rows.append(
                    {
                        "direction": f"{train_section}_to_{test_section}",
                        "feature_set": feature_set_name,
                        "feature": feature,
                    }
                )

            for model_name, model in models.items():
                metrics = evaluate_model_direction(
                    model_name=model_name,
                    model=clone(model),
                    feature_set_name=feature_set_name,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    train_section=train_section,
                    test_section=test_section,
                )
                all_metrics.append(metrics)

    feature_columns_path = TABLE_DIR / "cross_section_feature_columns.csv"
    pd.DataFrame(feature_column_rows).drop_duplicates().to_csv(feature_columns_path, index=False)
    print(f"\nSaved cross-section feature columns to: {feature_columns_path}")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = TABLE_DIR / "cross_section_validation_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved cross-section validation metrics to: {metrics_path}")

    return metrics_df


def save_summary_tables(metrics_df: pd.DataFrame) -> None:
    """
    Save model and feature-set summary tables.
    """
    if metrics_df.empty:
        print("No cross-section metrics available.")
        return

    metric_cols = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]

    summary_by_model = (
        metrics_df.groupby("model")[metric_cols]
        .agg(["mean", "std", "min", "max"])
        .sort_values(("macro_f1", "mean"), ascending=False)
    )
    summary_by_model_path = TABLE_DIR / "cross_section_summary_by_model.csv"
    summary_by_model.to_csv(summary_by_model_path)
    print(f"Saved summary by model to: {summary_by_model_path}")

    summary_by_feature_set = (
        metrics_df.groupby("feature_set")[metric_cols]
        .agg(["mean", "std", "min", "max"])
        .sort_values(("macro_f1", "mean"), ascending=False)
    )
    summary_by_feature_set_path = TABLE_DIR / "cross_section_summary_by_feature_set.csv"
    summary_by_feature_set.to_csv(summary_by_feature_set_path)
    print(f"Saved summary by feature set to: {summary_by_feature_set_path}")

    summary_by_model_feature = (
        metrics_df.groupby(["model", "feature_set"])[metric_cols]
        .agg(["mean", "std", "min", "max"])
        .sort_values(("macro_f1", "mean"), ascending=False)
    )
    summary_by_model_feature_path = TABLE_DIR / "cross_section_summary_by_model_and_feature_set.csv"
    summary_by_model_feature.to_csv(summary_by_model_feature_path)
    print(f"Saved summary by model and feature set to: {summary_by_model_feature_path}")

    print("\nCross-section summary by model:")
    print(summary_by_model)

    print("\nCross-section summary by feature set:")
    print(summary_by_feature_set)

    print("\nCross-section summary by model and feature set:")
    print(summary_by_model_feature)


def plot_summary_figures(metrics_df: pd.DataFrame) -> None:
    """
    Generate cross-section validation summary plots.
    """
    if metrics_df.empty:
        return

    model_summary = metrics_df.groupby("model")["macro_f1"].mean().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    model_summary.plot(kind="barh", ax=ax)
    ax.set_xlabel("Mean macro F1")
    ax.set_ylabel("Model")
    ax.set_title("Cross-section validation: mean macro F1 by model")
    ax.set_xlim(0, 1.05)
    save_current_fig("cross_section_macro_f1_by_model.png")

    feature_summary = metrics_df.groupby("feature_set")["macro_f1"].mean().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    feature_summary.plot(kind="barh", ax=ax)
    ax.set_xlabel("Mean macro F1")
    ax.set_ylabel("Feature set")
    ax.set_title("Cross-section validation: mean macro F1 by feature set")
    ax.set_xlim(0, 1.05)
    save_current_fig("cross_section_macro_f1_by_feature_set.png")

    model_feature_summary = (
        metrics_df.groupby(["model", "feature_set"])["macro_f1"].mean().reset_index()
    )
    model_feature_summary["model_feature"] = (
        model_feature_summary["model"] + " | " + model_feature_summary["feature_set"]
    )
    model_feature_summary = model_feature_summary.sort_values("macro_f1", ascending=True)

    fig_height = max(8, 0.35 * model_feature_summary.shape[0])
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.barh(model_feature_summary["model_feature"], model_feature_summary["macro_f1"])
    ax.set_xlabel("Mean macro F1")
    ax.set_ylabel("Model | Feature set")
    ax.set_title("Cross-section validation: model and feature-set comparison")
    ax.set_xlim(0, 1.05)
    save_current_fig("cross_section_macro_f1_by_model_and_feature_set.png")

    direction_summary = metrics_df.groupby("direction")["accuracy"].mean().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    direction_summary.plot(kind="barh", ax=ax)
    ax.set_xlabel("Mean accuracy")
    ax.set_ylabel("Validation direction")
    ax.set_title("Cross-section validation: mean accuracy by direction")
    ax.set_xlim(0, 1.05)
    save_current_fig("cross_section_accuracy_by_direction.png")


def choose_best_model_feature_set(metrics_df: pd.DataFrame) -> tuple[str, str]:
    """
    Choose the best model and feature set by mean macro F1.
    """
    summary = (
        metrics_df.groupby(["model", "feature_set"])["macro_f1"]
        .mean()
        .reset_index()
        .sort_values("macro_f1", ascending=False)
    )

    best = summary.iloc[0]

    best_model_name = best["model"]
    best_feature_set = best["feature_set"]

    print("\nBest cross-section model/feature set by mean macro F1:")
    print(summary.head(10))

    return best_model_name, best_feature_set


def build_final_combined_feature_table(
    sections: dict[str, sc.AnnData],
    static_features: dict[str, dict[str, pd.DataFrame]],
    trainable_labels: list[str],
    feature_set_name: str,
):
    """
    Build a final combined feature table for training on both sections.

    Why this is done after validation
    ---------------------------------
    Cross-section validation estimates generalization. After that, it is
    reasonable to train a final model on all available high-confidence labeled
    sections for future prediction.
    """
    section_ids = list(sections.keys())

    combined_feature_tables = []
    combined_labels = []

    if feature_set_name in ["expression_svd_marker_qc", "full_cross_section_context"]:
        common_genes = get_raw_gene_names(sections[section_ids[0]])
        for section_id in section_ids[1:]:
            common_genes = common_genes.intersection(get_raw_gene_names(sections[section_id]))
        common_genes = common_genes.tolist()

        if len(common_genes) < 100:
            raise ValueError("Too few common genes for final combined expression SVD.")

        expression_matrices = []
        for section_id in section_ids:
            adata = sections[section_id]
            mask = filter_to_trainable_spots(adata, trainable_labels)
            X_expr = get_expression_matrix_for_genes(adata, common_genes)
            expression_matrices.append(X_expr[mask])

        if sparse.issparse(expression_matrices[0]):
            X_combined_expr_all_genes = sparse.vstack(expression_matrices)
        else:
            X_combined_expr_all_genes = np.vstack(expression_matrices)

        variances = calculate_sparse_variance(X_combined_expr_all_genes)
        n_select = min(N_TOP_EXPRESSION_GENES, len(common_genes))
        top_idx = np.argsort(variances)[::-1][:n_select]
        selected_genes = [common_genes[i] for i in top_idx]

        selected_expression_matrices = []
        for section_id in section_ids:
            adata = sections[section_id]
            mask = filter_to_trainable_spots(adata, trainable_labels)
            X_expr = get_expression_matrix_for_genes(adata, selected_genes)
            selected_expression_matrices.append(X_expr[mask])

        if sparse.issparse(selected_expression_matrices[0]):
            X_combined_selected = sparse.vstack(selected_expression_matrices)
        else:
            X_combined_selected = np.vstack(selected_expression_matrices)

        max_components = min(
            N_EXPRESSION_COMPONENTS,
            X_combined_selected.shape[0] - 1,
            X_combined_selected.shape[1] - 1,
        )

        final_svd = TruncatedSVD(n_components=max_components, random_state=42)
        combined_svd_values = final_svd.fit_transform(X_combined_selected)
        svd_columns = [f"expr_svd_{i + 1}" for i in range(max_components)]

        svd_metadata = {
            "selected_genes": selected_genes,
            "n_selected_genes": len(selected_genes),
            "n_components": max_components,
            "explained_variance_ratio_sum": float(final_svd.explained_variance_ratio_.sum()),
        }
    else:
        combined_svd_values = None
        svd_columns = []
        final_svd = None
        svd_metadata = None

    svd_row_start = 0

    for section_id in section_ids:
        adata = sections[section_id]
        mask = filter_to_trainable_spots(adata, trainable_labels)
        y_section = adata.obs.loc[mask, LABEL_KEY].astype(str)

        base = pd.concat(
            [static_features[section_id]["qc"], static_features[section_id]["marker"]],
            axis=1,
        ).loc[mask]

        if feature_set_name == "marker_qc":
            features = base
        elif feature_set_name == "marker_qc_spatial":
            features = pd.concat([base, static_features[section_id]["spatial"].loc[mask]], axis=1)
        elif feature_set_name == "marker_qc_neighbor":
            features = pd.concat([base, static_features[section_id]["neighbor"].loc[mask]], axis=1)
        elif feature_set_name in ["expression_svd_marker_qc", "full_cross_section_context"]:
            n_rows = y_section.shape[0]
            svd_values = combined_svd_values[svd_row_start : svd_row_start + n_rows]
            svd_row_start += n_rows

            svd_df = pd.DataFrame(svd_values, index=y_section.index, columns=svd_columns)

            if feature_set_name == "expression_svd_marker_qc":
                features = pd.concat([svd_df, base], axis=1)
            else:
                features = pd.concat(
                    [
                        svd_df,
                        base,
                        static_features[section_id]["spatial"].loc[mask],
                        static_features[section_id]["neighbor"].loc[mask],
                    ],
                    axis=1,
                )
        else:
            raise ValueError(f"Unknown feature set: {feature_set_name}")

        combined_feature_tables.append(features)
        combined_labels.append(y_section)

    X_combined = pd.concat(combined_feature_tables, axis=0)
    y_combined = pd.concat(combined_labels, axis=0)

    X_combined = clean_feature_table(X_combined)

    metadata = {
        "feature_set": feature_set_name,
        "feature_columns": X_combined.columns.tolist(),
        "svd_metadata": svd_metadata,
        "section_ids": section_ids,
        "n_training_spots": int(X_combined.shape[0]),
        "label_names": sorted(y_combined.unique().tolist()),
    }

    return X_combined, y_combined, final_svd, metadata


def train_final_combined_model(
    sections: dict[str, sc.AnnData],
    static_features: dict[str, dict[str, pd.DataFrame]],
    trainable_labels: list[str],
    best_model_name: str,
    best_feature_set: str,
    metrics_df: pd.DataFrame,
) -> None:
    """
    Train a final model using high-confidence spots from both sections.

    Why train a final combined model?
    ---------------------------------
    Cross-section validation tells us whether models generalize between
    sections. After validation, the most useful production/portfolio model is
    trained on all available labeled sections.
    """
    print("\nTraining final combined cross-section model...")
    print(f"Best model: {best_model_name}")
    print(f"Best feature set: {best_feature_set}")

    X_combined, y_combined, final_svd, metadata = build_final_combined_feature_table(
        sections=sections,
        static_features=static_features,
        trainable_labels=trainable_labels,
        feature_set_name=best_feature_set,
    )

    model = clone(define_models()[best_model_name])
    model.fit(X_combined, y_combined)

    model_bundle = {
        "model": model,
        "feature_set": best_feature_set,
        "model_name": best_model_name,
        "feature_columns": X_combined.columns.tolist(),
        "expression_svd": final_svd,
        "metadata": metadata,
    }

    model_path = MODEL_DIR / "cross_section_spatial_niche_classifier.joblib"
    joblib.dump(model_bundle, model_path)

    metadata_out = {
        "best_model": best_model_name,
        "best_feature_set": best_feature_set,
        "label_key": LABEL_KEY,
        "excluded_label": EXCLUDE_LABEL,
        "require_shared_labels": REQUIRE_SHARED_LABELS,
        "trainable_labels": trainable_labels,
        "final_model_metadata": metadata,
        "cross_section_metrics": metrics_df.to_dict(orient="records"),
    }

    metadata_path = MODEL_DIR / "cross_section_model_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata_out, handle, indent=2)

    print(f"Saved final combined model bundle to: {model_path}")
    print(f"Saved final combined model metadata to: {metadata_path}")


def prepare_sections_for_modeling(sections: dict[str, sc.AnnData]):
    """
    Add marker scores, spatial neighbors, and static features for all sections.
    """
    all_score_columns = None
    static_features = {}

    for section_id, adata in sections.items():
        print("\n========================================")
        print(f"Preparing section: {section_id}")
        print("========================================")

        adata, score_columns = add_marker_signature_scores(adata, section_id)
        adata = ensure_spatial_neighbors(adata, section_id)

        if all_score_columns is None:
            all_score_columns = set(score_columns)
        else:
            all_score_columns = all_score_columns.intersection(score_columns)

        sections[section_id] = adata

    shared_score_columns = sorted(all_score_columns)

    print("\nShared marker score columns across sections:")
    print(shared_score_columns)

    for section_id, adata in sections.items():
        static_features[section_id] = build_static_feature_tables(
            adata=adata,
            score_columns=shared_score_columns,
        )

    return sections, static_features, shared_score_columns


def main() -> None:
    """
    Run cross-section validation and train a final multi-section model.

    Workflow
    --------
    1. Load final labeled AnnData objects for both sections.
    2. Validate required fields.
    3. Add marker signature scores.
    4. Ensure Squidpy spatial neighbor graphs exist.
    5. Build static QC/marker/spatial/neighbor features.
    6. Identify shared high-confidence labels.
    7. Run Section 1 -> Section 2 validation.
    8. Run Section 2 -> Section 1 validation.
    9. Save metrics, reports, and plots.
    10. Select best model-feature combination by mean macro F1.
    11. Train a final model using both sections.
    """
    print("Starting Milestone 7: cross-section validation")

    sections = load_sections()
    save_label_count_table(sections)

    sections, static_features, _shared_score_columns = prepare_sections_for_modeling(sections)
    trainable_labels = get_trainable_labels(sections)

    if len(trainable_labels) < 2:
        raise ValueError(
            "Fewer than two shared trainable labels were found. "
            "Review manual labels or set REQUIRE_SHARED_LABELS = False."
        )

    metrics_df = run_cross_section_validation(
        sections=sections,
        static_features=static_features,
        trainable_labels=trainable_labels,
    )

    save_summary_tables(metrics_df)
    plot_summary_figures(metrics_df)

    best_model_name, best_feature_set = choose_best_model_feature_set(metrics_df)

    train_final_combined_model(
        sections=sections,
        static_features=static_features,
        trainable_labels=trainable_labels,
        best_model_name=best_model_name,
        best_feature_set=best_feature_set,
        metrics_df=metrics_df,
    )

    print("\nMilestone 7 complete.")


if __name__ == "__main__":
    main()
