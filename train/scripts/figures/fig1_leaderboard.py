"""Fig. 6 in the report — DNSMOS-P.835 OVRL leaderboard on Diamond-Bench (n=750).

    python scripts/figures/fig1_leaderboard.py
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

INPUT_OVRL = 3.575  # mean OVRL of the degraded inputs on the same 750 clips


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("benchmarks/bench750_results.csv"))
    ap.add_argument("--out", type=Path, default=Path("figures/fig1_leaderboard.png"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv).sort_values("dnsmos_ovrl")
    df["model"] = df["model"].str.replace(r"^Diamond.*", "Diamond (ours)", regex=True)
    colors = ["tab:blue" if m.startswith("Diamond") else "darkgray" for m in df["model"]]

    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    bars = ax.barh(df["model"], df["dnsmos_ovrl"], color=colors, height=0.62)

    lo = min(df["dnsmos_ovrl"].min(), INPUT_OVRL) - 0.13
    hi = df["dnsmos_ovrl"].max() + 0.09
    ax.set_xlim(lo, hi)

    ax.axvline(INPUT_OVRL, color="firebrick", ls="--", lw=1.5, alpha=0.9)
    # Keep the input annotation inside the axes so it cannot collide with the title.
    ax.annotate(f"input {INPUT_OVRL:.3f}", xy=(INPUT_OVRL, 0.985),
                xycoords=("data", "axes fraction"), xytext=(5, 0),
                textcoords="offset points", ha="left", va="top",
                color="firebrick", fontsize=10)

    for bar, val in zip(bars, df["dnsmos_ovrl"]):
        # A bar ending near the input line would put its label on top of the line —
        # push such labels past it.
        x = val if val > INPUT_OVRL + 0.02 else INPUT_OVRL + 0.02
        ax.text(x + 0.006, bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
                va="center", ha="left", fontsize=10)

    ax.set_xlabel("DNSMOS-P.835  OVRL")
    ax.set_title("Diamond-Bench — perceptual quality (DNSMOS)", pad=16)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
