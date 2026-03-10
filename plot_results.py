"""
Plot experiment results from results.tsv.

Usage:
    python plot_results.py              # show plot
    python plot_results.py --save       # save to progress.png
    python plot_results.py --watch      # auto-refresh every 60s (for overnight runs)
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import pandas as pd


def load_results(path="results.tsv"):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        return None
    return df


def plot(df, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Autoresearch Experiment Progress", fontsize=14, fontweight="bold")

    # Color by status
    colors = {"keep": "#2ecc71", "discard": "#e74c3c", "crash": "#95a5a6"}
    df["color"] = df["status"].map(colors).fillna("#3498db")

    # Track the "best so far" line (cumulative min of kept experiments)
    kept = df[df["status"] == "keep"].copy()
    if not kept.empty:
        kept["best_so_far"] = kept["val_bpb"].cummin()

    # 1. val_bpb over experiments
    ax = axes[0, 0]
    valid = df[df["val_bpb"] > 0]
    ax.scatter(valid.index, valid["val_bpb"], c=valid["color"], s=40, zorder=3, edgecolors="white", linewidth=0.5)
    if not kept.empty:
        ax.step(kept.index, kept["best_so_far"], where="post", color="#2c3e50", linewidth=2, label="best (kept)")
        ax.legend(fontsize=8)
    ax.set_xlabel("Experiment #")
    ax.set_ylabel("val_bpb")
    ax.set_title("Validation BPB (lower is better)")
    ax.grid(True, alpha=0.3)

    # 2. Memory usage
    ax = axes[0, 1]
    valid_mem = df[df["memory_gb"] > 0]
    ax.bar(valid_mem.index, valid_mem["memory_gb"], color=valid_mem["color"], alpha=0.7, edgecolor="white")
    ax.axhline(y=12.0, color="red", linestyle="--", alpha=0.5, label="12GB VRAM limit")
    ax.set_xlabel("Experiment #")
    ax.set_ylabel("Peak VRAM (GB)")
    ax.set_title("Memory Usage")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Status distribution
    ax = axes[1, 0]
    status_counts = df["status"].value_counts()
    pie_colors = [colors.get(s, "#3498db") for s in status_counts.index]
    ax.pie(status_counts, labels=status_counts.index, colors=pie_colors, autopct="%1.0f%%", startangle=90)
    ax.set_title(f"Outcomes ({len(df)} experiments)")

    # 4. Improvement timeline
    ax = axes[1, 1]
    if not kept.empty and len(kept) > 1:
        improvements = kept["best_so_far"].diff().dropna()
        imp_idx = improvements[improvements < 0]
        ax.bar(imp_idx.index, -imp_idx, color="#2ecc71", alpha=0.8, edgecolor="white")
        ax.set_xlabel("Experiment #")
        ax.set_ylabel("BPB improvement")
        ax.set_title("Improvements (when best updated)")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Need 2+ kept\nexperiments", ha="center", va="center", fontsize=12, color="#95a5a6")
        ax.set_title("Improvements")

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
                print(f"\r[{time.strftime('%H:%M:%S')}] {len(df)} experiments, "
                      f"best={df[df['status']=='keep']['val_bpb'].min():.6f}" if not df[df['status']=='keep'].empty else "",
                      end="", flush=True)
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
