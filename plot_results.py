"""
Plot experiment results from results.tsv (Karpathy-style progress chart).

Usage:
    python plot_results.py              # show plot
    python plot_results.py --save       # save to progress.png
    python plot_results.py --watch      # auto-refresh every 60s (for overnight runs)
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_results(path="results.tsv"):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        return None
    return df


def plot(df, save_path=None):
    fig, ax = plt.subplots(figsize=(14, 7))

    kept = df[df["status"] == "keep"].copy()
    discarded = df[df["status"].isin(["discard", "crash"])].copy()
    num_kept = len(kept)

    fig.suptitle(
        f"Autoresearch Progress: {len(df)} Experiments, {num_kept} Kept Improvements",
        fontsize=14, fontweight="bold",
    )

    # Plot discarded experiments as gray dots
    if not discarded.empty:
        valid_disc = discarded[discarded["val_bpb"] > 0]
        if not valid_disc.empty:
            ax.scatter(
                valid_disc.index, valid_disc["val_bpb"],
                c="#cccccc", s=30, zorder=2, alpha=0.6,
                edgecolors="white", linewidth=0.3, label="Discarded",
            )

    # Plot kept experiments as green dots
    if not kept.empty:
        kept["best_so_far"] = kept["val_bpb"].cummin()

        ax.scatter(
            kept.index, kept["val_bpb"],
            c="#2ecc71", s=60, zorder=4, edgecolors="white", linewidth=0.5,
            label="Kept",
        )

        # Running best step line
        ax.step(
            kept.index, kept["best_so_far"],
            where="post", color="#2ecc71", linewidth=2, zorder=3,
            label="Running best",
        )

        # Label each kept experiment with its description
        for idx, row in kept.iterrows():
            desc = str(row.get("description", ""))
            # Truncate long descriptions
            if len(desc) > 45:
                desc = desc[:42] + "..."
            ax.annotate(
                desc,
                xy=(idx, row["val_bpb"]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=6.5, color="#333333", rotation=25,
                ha="left", va="bottom",
            )

    ax.set_xlabel("Experiment #", fontsize=12)
    ax.set_ylabel("Validation BPB (lower is better)", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.2)

    # Add summary stats as text box
    if not kept.empty:
        best_bpb = kept["val_bpb"].min()
        best_row = kept.loc[kept["val_bpb"].idxmin()]
        stats_lines = [f"Best: {best_bpb:.6f}"]
        if "mfu" in df.columns and pd.notna(best_row.get("mfu")):
            stats_lines.append(f"MFU: {best_row['mfu']:.1f}%")
        if "tok_per_sec" in df.columns and pd.notna(best_row.get("tok_per_sec")):
            stats_lines.append(f"Throughput: {best_row['tok_per_sec']:.0f} tok/s")
        if "memory_gb" in df.columns and pd.notna(best_row.get("memory_gb")):
            stats_lines.append(f"VRAM: {best_row['memory_gb']:.1f}/12.0 GB")
        stats_text = "\n".join(stats_lines)
        ax.text(
            0.02, 0.02, stats_text,
            transform=ax.transAxes, fontsize=9,
            verticalalignment="bottom", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="#cccccc"),
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot autoresearch results")
    parser.add_argument("--save", action="store_true", help="Save to progress.png")
    parser.add_argument("--watch", action="store_true", help="Auto-refresh every 60s")
    parser.add_argument("--file", default="results.tsv", help="Path to results TSV")
    args = parser.parse_args()

    if args.watch:
        plt.ion()
        print("Watching results.tsv — Ctrl+C to stop")
        while True:
            df = load_results(args.file)
            if df is not None:
                plt.clf()
                plot(df, save_path="progress.png" if args.save else None)
                plt.pause(0.1)
                kept = df[df["status"] == "keep"]
                if not kept.empty:
                    print(
                        f"\r[{time.strftime('%H:%M:%S')}] {len(df)} experiments, "
                        f"best={kept['val_bpb'].min():.6f}",
                        end="", flush=True,
                    )
            time.sleep(60)
    else:
        df = load_results(args.file)
        if df is None:
            print("No results.tsv found or it's empty. Run some experiments first.")
            return
        print(f"Loaded {len(df)} experiments from {args.file}")
        kept = df[df["status"] == "keep"]
        if not kept.empty:
            best = kept.loc[kept["val_bpb"].idxmin()]
            print(f"Best val_bpb: {best['val_bpb']:.6f} (experiment #{best.name}, {best['description']})")
        plot(df, save_path="progress.png" if args.save else None)
        if not args.save:
            plt.show()


if __name__ == "__main__":
    main()
