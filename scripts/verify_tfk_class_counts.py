"""
Verify that all 17 LCZ classes have enough samples in the training subsample
to support TFK stratified sampling with subset_size=68 (4 per class).

Run before TFK training:
    python scripts/verify_tfk_class_counts.py

Passes if every class has >= 4 samples. Reports which classes need
replace=True fallback.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import config
from src.data_loader import load_subsample

def verify_tfk_class_counts(
    subset_size: int = 68,
    n_classes: int = 17,
) -> None:
    """
    Load training subsample and verify per-class sample counts.

    Prints a table of all 17 classes with their sample counts,
    whether they meet the minimum (subset_size // n_classes = 4),
    and whether replace=True fallback is needed.

    Args:
        subset_size: Planned TFK_SUBSET_SIZE (default 68).
        n_classes: Number of LCZ classes (default 17).
    """
    n_per_class_needed = subset_size // n_classes   # = 4 for 68 // 17

    print("=" * 65)
    print(f"  TFK Stratified Sampler Verification")
    print(f"  subset_size={subset_size}, n_classes={n_classes}")
    print(f"  n_per_class_needed={n_per_class_needed}")
    print("=" * 65)

    subsample = load_subsample()
    y_train = subsample["y_train"]

    classes, counts = np.unique(y_train, return_counts=True)

    print(f"\n{'Class':<5} {'Name':<28} {'Count':>6} {'Min OK':>7} {'Replace?':>9}")
    print("-" * 65)

    n_need_replace = 0
    n_ok = 0
    min_count = counts.min()
    max_count = counts.max()

    for c, count in zip(classes, counts):
        name = config.LCZ_CLASS_NAMES[c] if c < len(config.LCZ_CLASS_NAMES) else f"Class {c}"
        ok = count >= n_per_class_needed
        needs_replace = not ok
        if needs_replace:
            n_need_replace += 1
        else:
            n_ok += 1

        flag = "YES — use replace=True" if needs_replace else "no"
        marker = "  ✗" if needs_replace else "  ✓"
        print(f"  {c:<3} {name:<28} {count:>6} {str(ok):>7}{marker}  {flag}")

    print("-" * 65)
    print(f"\nSummary:")
    print(f"  Total training samples: {len(y_train)}")
    print(f"  Classes with count >= {n_per_class_needed}: {n_ok} / {len(classes)}")
    print(f"  Classes needing replace=True: {n_need_replace} / {len(classes)}")
    print(f"  Min class count: {min_count}")
    print(f"  Max class count: {max_count}")
    print(f"  Imbalance ratio: {max_count/min_count:.1f}:1")

    print(f"\nConclusion:")
    if n_need_replace == 0:
        print(f"  ✓ ALL classes have >= {n_per_class_needed} samples.")
        print(f"  ✓ replace=False is safe for all classes at subset_size={subset_size}.")
        print(f"  ✓ No fallback needed.")
    else:
        print(f"  ✗ {n_need_replace} class(es) have < {n_per_class_needed} samples.")
        print(f"  ✗ replace=True fallback IS needed for those classes.")
        print(f"  The updated stratified sampler in train_quantum_kernel_kta")
        print(f"  handles this automatically — no manual action required.")

    print("=" * 65)


if __name__ == "__main__":
    verify_tfk_class_counts()
