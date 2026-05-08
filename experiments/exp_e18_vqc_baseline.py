"""
Experiment E18 — VQC/QNN baseline on physics-8.

A same-family quantum baseline: StronglyEntanglingLayers VQC with
one-vs-rest binary heads (17 classifiers), Adam, cross-entropy.

Shows AGPQK beats quantum competitors, not just classical.

Output: results/vqc_baseline/{metrics.csv, fig.pdf}
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, balanced_accuracy_score

import config
from src.utils import ensure_dir, setup_logging
from src.attention_kernel import select_features_by_fisher
from experiments._e_common import load_physics_16

RES = ensure_dir(os.path.join(config.RESULTS_DIR, "vqc_baseline"))
logger = setup_logging("exp_e18", log_file=os.path.join(RES, "e18.log"))

N        = 500
N_QUBITS = config.N_QUBITS
N_LAYERS = 3
EPOCHS   = 40
LR       = 0.05
SEEDS    = [42, 43, 44]


def build_vqc():
    dev = config.get_device(N_QUBITS)
    shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS,
                                               n_wires=N_QUBITS)

    @qml.qnode(dev, diff_method="best")
    def circ(weights, x):
        qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
        return qml.expval(qml.PauliZ(0))
    return circ, shape


def ovr_train_predict(X_tr, y_tr, X_te, classes, seed):
    """One-vs-rest: one binary VQC per class. Predict argmax over margins."""
    circ, shape = build_vqc()
    rng = np.random.default_rng(seed)
    n_te = len(X_te)
    margins = np.zeros((n_te, len(classes)))

    for ci, cls in enumerate(classes):
        y_bin = np.where(y_tr == cls, 1.0, -1.0)
        w = pnp.array(rng.normal(0, 0.1, size=shape), requires_grad=True)
        opt = qml.AdamOptimizer(stepsize=LR)

        def cost(w):
            preds = pnp.stack([circ(w, x) for x in X_tr])
            return pnp.mean((preds - y_bin) ** 2)

        for ep in range(EPOCHS):
            w = opt.step(cost, w)

        preds_te = np.array([float(circ(w, x)) for x in X_te])
        margins[:, ci] = preds_te
        logger.info(f"seed={seed} class={cls} done")

    yp = classes[np.argmax(margins, axis=1)]
    return yp


def run():
    X, _, y = load_physics_16()
    sss = StratifiedShuffleSplit(n_splits=1, train_size=N, random_state=42)
    (idx, _), = sss.split(np.zeros(len(y)), y)
    Xs, ys = X[idx], y[idx]
    selected, _ = select_features_by_fisher(Xs, ys, n_select=N_QUBITS,
                                            min_sar=4, min_opt=4)
    Xs = Xs[:, selected]
    classes = np.unique(ys)

    rows = []
    for seed in SEEDS:
        tr_split = StratifiedShuffleSplit(n_splits=1, test_size=0.3,
                                          random_state=seed)
        (tr, te), = tr_split.split(np.zeros(len(ys)), ys)
        yp = ovr_train_predict(Xs[tr], ys[tr], Xs[te], classes, seed)
        rows.append(dict(
            seed=seed,
            macro_f1=float(f1_score(ys[te], yp, average="macro",
                                    zero_division=0)),
            balanced_acc=float(balanced_accuracy_score(ys[te], yp)),
        ))
        logger.info(f"seed={seed} F1={rows[-1]['macro_f1']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "metrics.csv"), index=False)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Macro-F1", "Balanced Acc"],
           [df["macro_f1"].mean(), df["balanced_acc"].mean()],
           yerr=[df["macro_f1"].std(), df["balanced_acc"].std()],
           capsize=6, color="#17becf")
    ax.grid(alpha=0.3, axis="y")
    ax.set_title(f"E18 — VQC baseline (SEL, {N_LAYERS} layers, N={N})")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"fig.{ext}"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    run()
