"""Fig. 5 in the report — per-sample retrieval vs random/median combo selection.

Pure numerics over the calibration palette and the real target profiles from the
augmentor repo (no DSP is run). Left panel: histogram + KDE of the per-target
reproduction error. Right panel: the same strategies against the intrinsic
spread of the target corpus.

    python scripts/figures/fig6_retrieval_validation.py
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler

SIBLING = Path.home() / "emolia_dermo_recognition_version_almaz"

MATCH_FEATURES = [
    "snr_estimate_db", "dynamic_range_db", "noise_floor_db",
    "spectral_centroid_mean", "spectral_bandwidth_mean", "spectral_rolloff_mean",
    "hnr_db", "harmonic_ratio",
    "band_300_3400_ratio", "band_3400_8000_ratio", "band_8k_plus_ratio",
    "spectral_flatness_mean", "hf_energy_ratio", "proximity_bass_boost_db",
    "mic_resonance_peak_db", "crest_factor", "hf_temporal_std",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--palette", type=Path, default=SIBLING / "calibration_results.csv")
    ap.add_argument("--targets", type=Path, default=SIBLING / "class_separation_features.csv")
    ap.add_argument("--out", type=Path, default=Path("figures/fig6_retrieval_validation.png"))
    args = ap.parse_args()

    palette = pd.read_csv(args.palette)
    targets = pd.read_csv(args.targets)
    targets = targets[targets["__verdict"].isin(["bad", "borderline"])]

    avail = [f for f in MATCH_FEATURES if f in palette.columns and f in targets.columns]
    P = palette[avail].fillna(0).values
    T = targets[avail].fillna(0).values
    scaler = StandardScaler().fit(np.vstack([P, T]))
    Ps, Ts = scaler.transform(P), scaler.transform(T)

    # (A) per-sample retrieval: nearest palette combo for each target
    a = np.array([np.sqrt(((Ps - t) ** 2).sum(axis=1)).min() for t in Ts])
    # (B) a random combo per target
    rng = np.random.default_rng(42)
    b = np.array([np.sqrt(((Ps[rng.integers(0, len(Ps))] - t) ** 2).sum()) for t in Ts])
    # (C) one median combo for everyone
    c_idx = int(np.argmin(np.sqrt(((Ps - Ps.mean(axis=0)) ** 2).sum(axis=1))))
    c = np.sqrt(((Ts - Ps[c_idx]) ** 2).sum(axis=1))
    # (D) intrinsic spread of the target corpus
    d = np.sqrt(((Ts - Ts.mean(axis=0)) ** 2).sum(axis=1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bins = np.linspace(0, max(b.max(), c.max()) * 1.05, 40)
    width = bins[1] - bins[0]
    grid = np.linspace(bins[0], bins[-1], 400)
    series = [
        (a, "tab:green", 0.55, "(A) Per-sample retrieval"),
        (b, "tab:gray", 0.45, "(B) Random combo"),
        (c, "tab:red", 0.35, "(C) Median combo"),
    ]
    for arr, color, alpha, name in series:
        axes[0].hist(arr, bins=bins, alpha=alpha, color=color, edgecolor="none")
        # KDE scaled to the histogram counts (density x n x bin width).
        kde = gaussian_kde(arr)(grid) * len(arr) * width
        axes[0].plot(grid, kde, color=color, lw=2.2,
                     label=f"{name} (mean={arr.mean():.2f})")
        axes[0].axvline(arr.mean(), color=color, ls="--", lw=1.2, alpha=0.8)
    axes[0].set_xlabel("Distance to target (scaled, 17-D)")
    axes[0].set_ylabel("count")
    axes[0].set_title("Per-target reproduction error")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].boxplot([a, b, c, d], tick_labels=["(A)\nper-sample\nretrieval", "(B)\nrandom\ncombo",
                                               "(C)\nmedian\ncombo", "(D)\ntarget\nspread"])
    axes[1].axhline(d.mean(), color="blue", ls=":", alpha=0.6, label="target intra-spread")
    axes[1].set_ylabel("Distance")
    axes[1].set_title("Strategies vs intrinsic spread")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"saved {args.out}  A={a.mean():.2f} B={b.mean():.2f} C={c.mean():.2f} D={d.mean():.2f}")


if __name__ == "__main__":
    main()
