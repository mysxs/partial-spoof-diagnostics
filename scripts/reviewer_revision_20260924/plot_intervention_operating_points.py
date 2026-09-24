#!/usr/bin/env python3
"""Plot the completed, pre-fixed development-FPR grid; no curve fitting."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.summary_csv, newline="")))
    points = [.001, .005, .01, .02, .05, .1, .2]
    colors = {"original": "#25355A", "augmentation": "#0A8A7A", "consistency": "#C76B2A"}
    names = {"original": "Original", "augmentation": "Augmentation", "consistency": "+ Consistency"}
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.15), sharex=True,
                             gridspec_kw={"wspace": .22, "hspace": .18})
    for i, arch in enumerate(("frame", "phone")):
        for j, (metric, label) in enumerate((("gg_minus_genuine_fpr", "GG excess FPR (%)"),
                                             ("gf_raw_fnr_core", "GF core FNR (%)"))):
            ax = axes[i, j]
            for condition in colors:
                values = []
                for point in points:
                    key = f"dev_fpr_{point}"
                    matches = [r for r in rows if r["architecture"] == arch and
                               r["condition"] == condition and r["operating_point"] == key and
                               r["metric"] == metric]
                    assert len(matches) == 1, (arch, condition, key, metric)
                    values.append(float(matches[0]["mean_percent"]))
                ax.plot([p * 100 for p in points], values, marker="o", markersize=3.5,
                        linewidth=1.6, color=colors[condition], label=names[condition])
            ax.set_xscale("log")
            ax.set_xticks([.1, .5, 1, 2, 5, 10, 20])
            ax.set_xticklabels(["0.1", "0.5", "1", "2", "5", "10", "20"])
            ax.grid(alpha=.2, linewidth=.5)
            ax.set_ylabel(label)
            if i == 1:
                ax.set_xlabel("PS-dev target FPR (%)")
            ax.set_title(f"{arch.capitalize()} model", fontsize=9, loc="left")
    axes[0, 0].legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.34),
                      fontsize=8, frameon=False)
    fig.subplots_adjust(top=.84, left=.11, right=.98, bottom=.11)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", metadata={"Title": "Source-calibrated GG/GF operating-point diagnostic"})
    print(output)


if __name__ == "__main__":
    main()
