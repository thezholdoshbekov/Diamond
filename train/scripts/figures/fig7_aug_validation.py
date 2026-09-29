"""Fig. 3 in the report — where the augmentation lands in feature space (PCA).

Data: validation_v4.csv from the augmentor repo (features of studio / our
augmentation / real degraded recordings, produced by validate_v4_pca.py).

    python scripts/figures/fig7_aug_validation.py
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SIBLING = Path.home() / "emolia_dermo_recognition_version_almaz"

LABELS = {
    "studio": "studio target",
    "aug_v4": "our augmentation",
    "emolia": "real degraded",
}
COLORS = {"studio": "tab:blue", "aug_v4": "tab:green", "emolia": "tab:orange"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=SIBLING / "validation_v4.csv")
    ap.add_argument("--out", type=Path, default=Path("figures/fig7_aug_validation.png"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    feats = [c for c in df.columns if not c.startswith("__")]
    df = df.dropna(subset=feats).reset_index(drop=True)

    Xs = StandardScaler().fit_transform(df[feats].values)
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(Xs)
    df["pc1"], df["pc2"] = Xp[:, 0], Xp[:, 1]

    centers = {s: Xs[(df["__source"] == s).values].mean(axis=0) for s in LABELS}
    d_se = np.linalg.norm(centers["studio"] - centers["emolia"])
    d_sa = np.linalg.norm(centers["studio"] - centers["aug_v4"])
    d_ae = np.linalg.norm(centers["aug_v4"] - centers["emolia"])
    closed = (1 - d_ae / d_se) * 100

    fig, ax = plt.subplots(figsize=(10.4, 7.2))
    for src in ("studio", "emolia", "aug_v4"):
        m = df["__source"] == src
        ax.scatter(df.loc[m, "pc1"], df.loc[m, "pc2"], s=34, alpha=0.6,
                   c=COLORS[src], edgecolors="none",
                   label=f"{LABELS[src]} ({int(m.sum())})")
    for src in ("studio", "aug_v4", "emolia"):
        m = df["__source"] == src
        ax.scatter(df.loc[m, "pc1"].mean(), df.loc[m, "pc2"].mean(),
                   c=COLORS[src], s=420, marker="X",
                   edgecolors="black", linewidths=2, zorder=5)

    ax.annotate(
        f"d(studio, real)        = {d_se:.2f}\n"
        f"d(studio, augmented)   = {d_sa:.2f}\n"
        f"d(augmented, real)     = {d_ae:.2f}\n"
        f"gap closed: {closed:.0f}%",
        xy=(0.02, 0.98), xycoords="axes fraction", ha="left", va="top",
        fontsize=10, family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )
    # Legend goes to the empty lower-left region (the point clouds sit right of it).
    ax.legend(loc="lower left", fontsize=12, markerscale=2.0,
              labelspacing=0.9, borderpad=1.0, framealpha=0.95)
    ax.set_xlabel(f"PC1 ({100 * pca.explained_variance_ratio_[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({100 * pca.explained_variance_ratio_[1]:.1f}%)")
    ax.set_title("Where the augmentation lands: studio / augmented / real recordings")
    ax.grid(alpha=0.3)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"saved {args.out}  (d_se={d_se:.2f} d_sa={d_sa:.2f} d_ae={d_ae:.2f} closed={closed:.0f}%)")


if __name__ == "__main__":
    main()
