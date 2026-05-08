"""
Utility functions for the Quantum-Sat Classification pipeline.

Provides: result saving/loading (JSON + CSV + NPY), publication-quality plotting,
confusion matrix visualization, per-class F1 bar charts, logging setup, and
imbalance-aware formatting (Rule I7).

All plots: 300 DPI, labeled axes with units, legends, fonts 12+, PNG + PDF,
seaborn "colorblind" palette.
"""

import os
import sys
import json
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/WSL
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score,
    cohen_kappa_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ============ GLOBAL PLOT SETTINGS ============
sns.set_palette("colorblind")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.figsize": (10, 6),
})


def setup_logging(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure a logger with console and optional file output.

    Args:
        name: Logger name (typically __name__ or experiment name).
        log_file: Optional path to a log file. If None, only console output.
        level: Logging level (default: INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s — %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ============ RESULT SAVING / LOADING ============

def save_results(
    results: Dict[str, Any],
    filepath: str,
    fmt: str = "json",
) -> str:
    """
    Save experiment results to disk.

    Supports JSON (dicts/lists), CSV (DataFrames/dicts-of-lists), and NPY (arrays).
    Creates parent directories automatically.

    Args:
        results: Data to save (dict for JSON, DataFrame/dict for CSV, ndarray for NPY).
        filepath: Output file path (extension determines format if fmt='auto').
        fmt: Format — 'json', 'csv', 'npy', or 'auto' (infer from extension).

    Returns:
        str: Absolute path of saved file.

    Raises:
        ValueError: If format is unsupported.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if fmt == "auto":
        ext = os.path.splitext(filepath)[1].lower()
        fmt = {"json": "json", ".csv": "csv", ".npy": "npy"}.get(ext, "json")

    if fmt == "json":
        # Convert numpy types for JSON serialization
        serializable = _make_json_serializable(results)
        with open(filepath, "w") as f:
            json.dump(serializable, f, indent=2)
    elif fmt == "csv":
        if isinstance(results, pd.DataFrame):
            results.to_csv(filepath, index=False)
        elif isinstance(results, dict):
            pd.DataFrame(results).to_csv(filepath, index=False)
        else:
            raise ValueError(f"Cannot save type {type(results)} as CSV")
    elif fmt == "npy":
        np.save(filepath, results)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    return os.path.abspath(filepath)


def load_results(filepath: str, fmt: str = "auto") -> Any:
    """
    Load experiment results from disk.

    Args:
        filepath: Path to the file to load.
        fmt: Format — 'json', 'csv', 'npy', or 'auto' (infer from extension).

    Returns:
        Loaded data (dict, DataFrame, or ndarray).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If format is unsupported.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Results file not found: {filepath}")

    if fmt == "auto":
        ext = os.path.splitext(filepath)[1].lower()
        fmt = {".json": "json", ".csv": "csv", ".npy": "npy", ".npz": "npz"}.get(ext, "json")

    if fmt == "json":
        with open(filepath, "r") as f:
            return json.load(f)
    elif fmt == "csv":
        return pd.read_csv(filepath)
    elif fmt == "npy":
        return np.load(filepath, allow_pickle=True)
    elif fmt == "npz":
        return np.load(filepath, allow_pickle=True)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def _make_json_serializable(obj: Any) -> Any:
    """
    Recursively convert numpy types to Python-native types for JSON.

    Args:
        obj: Object to convert.

    Returns:
        JSON-serializable version of obj.
    """
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (datetime,)):
        return obj.isoformat()
    else:
        return obj


# ============ PUBLICATION-QUALITY PLOTTING ============

def create_publication_plot(
    fig: plt.Figure,
    filepath: str,
    title: Optional[str] = None,
    tight_layout: bool = True,
) -> Tuple[str, str]:
    """
    Save a matplotlib figure in both PNG (300 DPI) and PDF formats.

    Args:
        fig: Matplotlib figure to save.
        filepath: Base path (without extension) or with .png extension.
        title: Optional suptitle for the figure.
        tight_layout: Whether to apply tight_layout.

    Returns:
        Tuple[str, str]: Paths to PNG and PDF files.
    """
    base = os.path.splitext(filepath)[0]
    png_path = base + ".png"
    pdf_path = base + ".pdf"

    os.makedirs(os.path.dirname(png_path) if os.path.dirname(png_path) else ".", exist_ok=True)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    if tight_layout:
        try:
            fig.tight_layout()
        except Exception:
            pass  # Some complex layouts can't use tight_layout

    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return png_path, pdf_path


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    filepath: str,
    title: str = "Confusion Matrix",
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    figsize: Tuple[int, int] = (14, 12),
) -> Tuple[str, str]:
    """
    Plot and save a row-normalized confusion matrix (Rule I7).

    Confusion matrices are ALWAYS row-normalized to show recall per class,
    preventing dominant classes from masking rare class performance.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        filepath: Output file path (base).
        title: Plot title.
        class_names: Class labels (defaults to config.LCZ_CLASS_NAMES).
        normalize: If True, row-normalize (shows recall). ALWAYS True per Rule I7.
        figsize: Figure size in inches.

    Returns:
        Tuple[str, str]: Paths to PNG and PDF files.
    """
    if class_names is None:
        class_names = config.LCZ_CLASS_NAMES

    # Get unique labels present in data
    unique_labels = sorted(set(y_true) | set(y_pred))

    # Compute confusion matrix (row-normalized per Rule I7)
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # avoid division by zero
        cm_normalized = cm.astype(float) / row_sums
    else:
        cm_normalized = cm.astype(float)

    # Map label indices to class names
    tick_labels = [
        class_names[i] if i < len(class_names) else str(i)
        for i in unique_labels
    ]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        ax=ax,
        vmin=0,
        vmax=1 if normalize else None,
        cbar_kws={"label": "Recall" if normalize else "Count"},
    )
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    return create_publication_plot(fig, filepath)


def plot_per_class_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    filepath: str,
    title: str = "Per-Class F1 Score",
    class_names: Optional[List[str]] = None,
    sort_ascending: bool = True,
    figsize: Tuple[int, int] = (12, 8),
) -> Tuple[str, str]:
    """
    Plot per-class F1 scores as horizontal bar chart, sorted lowest-first (Rule I7).

    Sorting by lowest F1 first highlights the hardest classes (most important
    for class-imbalanced datasets like So2Sat with 67:1 ratio).

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        filepath: Output file path (base).
        title: Plot title.
        class_names: Class labels (defaults to config.LCZ_CLASS_NAMES).
        sort_ascending: If True, sort lowest F1 first (hardest classes at top).
        figsize: Figure size.

    Returns:
        Tuple[str, str]: Paths to PNG and PDF files.
    """
    if class_names is None:
        class_names = config.LCZ_CLASS_NAMES

    unique_labels = sorted(set(y_true) | set(y_pred))
    per_class_f1 = f1_score(y_true, y_pred, labels=unique_labels, average=None, zero_division=0)

    label_names = [
        class_names[i] if i < len(class_names) else f"Class {i}"
        for i in unique_labels
    ]

    # Create DataFrame for sorting
    df = pd.DataFrame({"class": label_names, "f1": per_class_f1})
    if sort_ascending:
        df = df.sort_values("f1", ascending=True)

    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("colorblind", len(df))
    bars = ax.barh(df["class"], df["f1"], color=colors)

    # Add value labels on bars
    for bar, val in zip(bars, df["f1"]):
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9,
        )

    ax.set_xlabel("F1 Score", fontsize=12)
    ax.set_xlim(0, 1.1)
    ax.axvline(x=df["f1"].mean(), color="red", linestyle="--", alpha=0.7,
               label=f"Macro-F1 = {df['f1'].mean():.3f}")
    ax.legend(fontsize=10)
    ax.set_title(title, fontsize=14, fontweight="bold")

    return create_publication_plot(fig, filepath)


def plot_per_class_f1_comparison(
    y_true: np.ndarray,
    y_pred_quantum: np.ndarray,
    y_pred_classical: np.ndarray,
    filepath: str,
    title: str = "Per-Class F1: Quantum vs Classical",
    quantum_label: str = "FQK-SVM",
    classical_label: str = "RBF-SVM",
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 8),
) -> Tuple[str, str]:
    """
    Plot per-class F1 delta (quantum - classical) as horizontal bar chart.

    Positive bars = quantum better, negative = classical better.

    Args:
        y_true: Ground truth labels.
        y_pred_quantum: Quantum model predictions.
        y_pred_classical: Classical model predictions.
        filepath: Output file path.
        title: Plot title.
        quantum_label: Label for quantum predictions.
        classical_label: Label for classical predictions.
        class_names: Class labels.
        figsize: Figure size.

    Returns:
        Tuple[str, str]: Paths to PNG and PDF files.
    """
    if class_names is None:
        class_names = config.LCZ_CLASS_NAMES

    unique_labels = sorted(set(y_true) | set(y_pred_quantum) | set(y_pred_classical))

    f1_q = f1_score(y_true, y_pred_quantum, labels=unique_labels, average=None, zero_division=0)
    f1_c = f1_score(y_true, y_pred_classical, labels=unique_labels, average=None, zero_division=0)
    delta_f1 = f1_q - f1_c

    label_names = [
        class_names[i] if i < len(class_names) else f"Class {i}"
        for i in unique_labels
    ]

    df = pd.DataFrame({"class": label_names, "delta_f1": delta_f1})
    df = df.sort_values("delta_f1", ascending=True)

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in df["delta_f1"]]
    ax.barh(df["class"], df["delta_f1"], color=colors)

    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    ax.set_xlabel(f"ΔF1 ({quantum_label} − {classical_label})", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add value labels
    for idx, (_, row) in enumerate(df.iterrows()):
        ax.text(
            row["delta_f1"] + (0.005 if row["delta_f1"] >= 0 else -0.005),
            idx,
            f"{row['delta_f1']:+.3f}",
            va="center",
            ha="left" if row["delta_f1"] >= 0 else "right",
            fontsize=9,
        )

    return create_publication_plot(fig, filepath)


def plot_learning_curves(
    n_samples_list: List[int],
    results_dict: Dict[str, Dict[str, np.ndarray]],
    filepath: str,
    metric: str = "macro_f1",
    title: str = "Few-Shot Learning Curves",
    ylabel: str = "Macro-F1",
    figsize: Tuple[int, int] = (10, 7),
) -> Tuple[str, str]:
    """
    Plot few-shot learning curves with error bands.

    Args:
        n_samples_list: List of training set sizes.
        results_dict: {model_name: {"mean": array, "std": array}}.
        filepath: Output file path.
        metric: Metric name for labeling.
        title: Plot title.
        ylabel: Y-axis label.
        figsize: Figure size.

    Returns:
        Tuple[str, str]: Paths to PNG and PDF files.
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("colorblind", len(results_dict))

    for (name, data), color in zip(results_dict.items(), colors):
        mean = np.array(data["mean"])
        std = np.array(data["std"])
        ax.plot(n_samples_list, mean, "o-", label=name, color=color, linewidth=2, markersize=6)
        ax.fill_between(
            n_samples_list, mean - std, mean + std,
            alpha=0.2, color=color,
        )

    ax.set_xlabel("Number of Training Samples", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xscale("log")
    ax.set_xticks(n_samples_list)
    ax.set_xticklabels([str(n) for n in n_samples_list])
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=14, fontweight="bold")

    return create_publication_plot(fig, filepath)


def plot_eigenvalue_spectrum(
    eigenvalues_dict: Dict[str, np.ndarray],
    filepath: str,
    title: str = "Kernel Eigenvalue Spectrum",
    figsize: Tuple[int, int] = (10, 6),
    top_k: int = 50,
) -> Tuple[str, str]:
    """
    Plot eigenvalue spectra for multiple kernels.

    Args:
        eigenvalues_dict: {kernel_name: eigenvalues_array (sorted descending)}.
        filepath: Output file path.
        title: Plot title.
        figsize: Figure size.
        top_k: Number of top eigenvalues to show.

    Returns:
        Tuple[str, str]: Paths to PNG and PDF files.
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("colorblind", len(eigenvalues_dict))

    for (name, eigs), color in zip(eigenvalues_dict.items(), colors):
        sorted_eigs = np.sort(eigs)[::-1][:top_k]
        # Normalize
        sorted_eigs = sorted_eigs / sorted_eigs.sum() if sorted_eigs.sum() > 0 else sorted_eigs
        ax.plot(range(1, len(sorted_eigs) + 1), sorted_eigs, "o-",
                label=name, color=color, linewidth=2, markersize=4)

    ax.set_xlabel("Eigenvalue Index", fontsize=12)
    ax.set_ylabel("Normalized Eigenvalue", fontsize=12)
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=14, fontweight="bold")

    return create_publication_plot(fig, filepath)


def plot_kernel_heatmap(
    K: np.ndarray,
    filepath: str,
    title: str = "Kernel Matrix",
    figsize: Tuple[int, int] = (10, 8),
) -> Tuple[str, str]:
    """
    Plot kernel matrix as a heatmap.

    Args:
        K: Kernel matrix (n × n).
        filepath: Output file path.
        title: Plot title.
        figsize: Figure size.

    Returns:
        Tuple[str, str]: Paths to PNG and PDF files.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(K, cmap="viridis", ax=ax, square=True,
                cbar_kws={"label": "Kernel value"})
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Sample index", fontsize=12)
    ax.set_ylabel("Sample index", fontsize=12)

    return create_publication_plot(fig, filepath)


# ============ METRICS COMPUTATION ============

def compute_full_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute ALL required metrics per Rule I3.

    Returns accuracy, macro-F1 (PRIMARY), weighted-F1, per-class F1 (all 17),
    Cohen's kappa, and classification report.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Class label names.

    Returns:
        dict: Full metrics dictionary.
    """
    if class_names is None:
        class_names = config.LCZ_CLASS_NAMES

    unique_labels = sorted(set(y_true) | set(y_pred))
    target_names = [
        class_names[i] if i < len(class_names) else f"Class {i}"
        for i in unique_labels
    ]

    per_class_f1 = f1_score(y_true, y_pred, labels=unique_labels,
                            average=None, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=unique_labels,
                        average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=unique_labels,
                          average="weighted", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    report = classification_report(
        y_true, y_pred, labels=unique_labels,
        target_names=target_names, zero_division=0, output_dict=True,
    )

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),  # PRIMARY metric per Rule I3
        "weighted_f1": float(weighted_f1),
        "cohen_kappa": float(kappa),
        "per_class_f1": {
            target_names[i]: float(per_class_f1[i])
            for i in range(len(unique_labels))
        },
        "classification_report": report,
        "n_samples": int(len(y_true)),
        "n_classes_present": int(len(unique_labels)),
    }


# ============ TIMING ============

class Timer:
    """
    Context manager for wall-clock timing of experiment sections.

    Usage:
        with Timer("Kernel computation") as t:
            compute_kernel(...)
        print(f"Took {t.elapsed:.1f}s")
    """

    def __init__(self, description: str = "", logger: Optional[logging.Logger] = None):
        self.description = description
        self.logger = logger
        self.start_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self):
        self.start_time = time.time()
        msg = f"Starting: {self.description}"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        msg = f"Completed: {self.description} ({self.elapsed:.1f}s)"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)


# ============ LATEX TABLE GENERATION ============

def results_to_latex_table(
    results: pd.DataFrame,
    filepath: str,
    caption: str = "Experimental Results",
    label: str = "tab:results",
    bold_best: bool = True,
    metric_cols: Optional[List[str]] = None,
) -> str:
    """
    Convert results DataFrame to a LaTeX table and save.

    Args:
        results: DataFrame with experiment results.
        filepath: Output .tex file path.
        caption: Table caption for LaTeX.
        label: Table label for referencing.
        bold_best: If True, bold the best value in metric columns.
        metric_cols: Columns to consider for bolding best values.

    Returns:
        str: LaTeX table string.
    """
    df = results.copy()

    if bold_best and metric_cols:
        for col in metric_cols:
            if col in df.columns:
                best_idx = df[col].idxmax()
                df.loc[best_idx, col] = f"\\textbf{{{df.loc[best_idx, col]:.4f}}}"

    latex = df.to_latex(index=False, escape=False, caption=caption, label=label)

    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    with open(filepath, "w") as f:
        f.write(latex)

    return latex


def check_and_skip(filepath: str, logger: Optional[logging.Logger] = None) -> bool:
    """
    Check if a result file already exists (for resumability).

    Per Rule 10: every experiment must be resumable, checking os.path.exists()
    before recomputing.

    Args:
        filepath: Path to check.
        logger: Optional logger for messaging.

    Returns:
        bool: True if file exists and should be skipped, False otherwise.
    """
    if os.path.exists(filepath):
        msg = f"[SKIP] Already exists: {filepath}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
        return True
    return False


def ensure_dir(path: str) -> str:
    """
    Create directory (and parents) if it doesn't exist.

    Args:
        path: Directory path to create.

    Returns:
        str: The same path, for convenience.
    """
    os.makedirs(path, exist_ok=True)
    return path


def load_bandwidth_gamma(feature_set: str, results_dir: str = None) -> float:
    """
    Load CV-selected encoding bandwidth γ for a given feature set.

    Args:
        feature_set: "pca" or "physics"
        results_dir: path to results directory (default: config.RESULTS_DIR)

    Returns:
        γ (float): encoding bandwidth scalar. Features should be multiplied
                   by γ before passing to quantum kernels.
                   Features are stored in [0,π], so encoded range is [0, γπ].

    Raises:
        FileNotFoundError if CV has not been run yet.
        ValueError if feature_set is not "pca" or "physics".
    """
    import json, os
    if results_dir is None:
        import config
        results_dir = config.RESULTS_DIR

    gamma_path = os.path.join(results_dir, "bandwidth_cv", "selected_gamma.json")

    if not os.path.exists(gamma_path):
        raise FileNotFoundError(
            f"Bandwidth CV results not found: {gamma_path}\n"
            f"Run: python scripts/select_bandwidth_cv.py"
        )

    with open(gamma_path) as f:
        data = json.load(f)

    key_map = {"pca": "gamma_pca", "physics": "gamma_physics"}
    if feature_set not in key_map:
        raise ValueError(f"feature_set must be 'pca' or 'physics', got '{feature_set}'")

    gamma = float(data[key_map[feature_set]])
    return gamma


def validate_test_kernel_shape(K_test, X_train, X_test, name):
    """
    Assert test kernel has shape (n_test, n_train) as required by sklearn SVM.
    If shape is (n_train, n_test) (transposed), the cached file is from the
    pre-bugfix version. Delete it and raise so the experiment re-computes it.
    """
    expected = (len(X_test), len(X_train))
    if K_test.shape == expected:
        return  # correct
    transposed = (len(X_train), len(X_test))
    if K_test.shape == transposed:
        raise ValueError(
            f"{name} test kernel has TRANSPOSED shape {K_test.shape}. "
            f"Expected {expected}. This is a pre-bugfix cached file. "
            f"Delete the .npy file and rerun."
        )
    raise ValueError(
        f"{name} test kernel unexpected shape {K_test.shape}, expected {expected}."
    )


if __name__ == "__main__":
    print("=" * 60)
    print("  src/utils.py — Self-test")
    print("=" * 60)

    # Test JSON serialization
    test_data = {
        "accuracy": np.float64(0.85),
        "predictions": np.array([0, 1, 2, 3]),
        "nested": {"val": np.int32(42)},
    }
    test_path = os.path.join(config.RESULTS_DIR, "_test_utils.json")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    saved = save_results(test_data, test_path, fmt="json")
    loaded = load_results(saved)
    print(f"  JSON save/load: OK ({saved})")
    os.remove(saved)

    # Test timer
    with Timer("Test operation"):
        import time as _t
        _t.sleep(0.1)

    # Test metrics
    np.random.seed(42)
    y_true = np.random.randint(0, 17, 100)
    y_pred = np.random.randint(0, 17, 100)
    metrics = compute_full_metrics(y_true, y_pred)
    print(f"  Metrics: macro-F1={metrics['macro_f1']:.3f}, kappa={metrics['cohen_kappa']:.3f}")

    # Test logging
    logger = setup_logging("test_logger")
    logger.info("Logger works correctly")

    print()
    print("  All self-tests passed.")
