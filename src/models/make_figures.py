"""
Plot the overfitting evidence for both models from the saved CSV files.

  outputs/figures/cnn_loss_curves.png      train vs val loss per epoch
  outputs/figures/cnn_accuracy_curves.png  train vs val accuracy per epoch
  outputs/figures/rf_train_vs_val.png      train vs val macro-F1 per RF setting

Nothing is retrained; the plots only read the CSV files.

Run from the project root (after train_rf.py and train_cnn.py):
    python -m src.models.make_figures
"""

import matplotlib
matplotlib.use("Agg")   # draw to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd

from src.models.data_utils import OUTPUT_DIR

plt.style.use("dark_background")


FIGURE_DIR = OUTPUT_DIR / "figures"
TRAIN_COLOR = "#2a78d6"     # blue
VAL_COLOR = "#eb6834"       # orange
TEXT_COLOR = "#d9d9d9"      # light gray for labels and the best-epoch marker
GRID_COLOR = "#444444"
DPI = 200


# Shared clean style: muted grid, no top/right borders, light gray text.
def style_axes(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=12, color=TEXT_COLOR)
    ax.set_xlabel(xlabel, color=TEXT_COLOR)
    ax.set_ylabel(ylabel, color=TEXT_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR)


# Plot one train-vs-val curve from the CNN history and mark the best epoch
# (the epoch with the lowest validation loss, whose weights were restored).
def plot_cnn_curve(history, train_column, val_column, ylabel, filename):
    best = history.loc[history["val_loss"].idxmin()]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(history["epoch"], history[train_column], color=TRAIN_COLOR,
            linewidth=2, marker="o", markersize=4, label="Train")
    ax.plot(history["epoch"], history[val_column], color=VAL_COLOR,
            linewidth=2, marker="o", markersize=4, label="Validation")

    # Dashed line at the best epoch, labelled just above the plot area
    ax.axvline(best["epoch"], color=TEXT_COLOR, linestyle="--", linewidth=1)
    ax.text(best["epoch"], 1.01, f"best epoch {int(best['epoch'])} (lowest val loss)",
            transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=9, color=TEXT_COLOR)

    style_axes(ax, f"CNN {ylabel.lower()} per epoch", "Epoch", ylabel)
    ax.title.set_position((0, 1.08))
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=DPI)
    plt.close(fig)


# Plot train and val macro-F1 for every RF grid setting as two dots joined by
# a line. A long line means the forest fits training windows much better
# than unseen validation windows (overfitting).
def plot_rf_train_vs_val(results):
    results = results.copy()
    results["max_depth"] = results["max_depth"].fillna("None")
    results["name"] = (
        "trees=" + results["n_estimators"].astype(str)
        + ", depth=" + results["max_depth"].astype(str).str.replace(".0", "", regex=False)
        + ", leaf=" + results["min_samples_leaf"].astype(str)
    )
    best_index = results["val_macro_f1"].idxmax()
    rows = range(len(results))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.hlines(rows, results["val_macro_f1"], results["train_macro_f1"],
              color=GRID_COLOR, linewidth=2)
    ax.scatter(results["train_macro_f1"], rows, color=TRAIN_COLOR, s=40,
               zorder=3, label="Train")
    ax.scatter(results["val_macro_f1"], rows, color=VAL_COLOR, s=40,
               zorder=3, label="Validation")

    ax.set_yticks(list(rows))
    labels = [name + ("  (selected)" if i == best_index else "")
              for i, name in enumerate(results["name"])]
    ax.set_yticklabels(labels, fontsize=9)
    ax.get_yticklabels()[best_index].set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)

    style_axes(ax, "Random Forest: train vs val macro-F1 (window level)",
               "Macro-F1", "")
    ax.title.set_position((-0.5, 1.0))   # start the title above the row labels
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "rf_train_vs_val.png", dpi=DPI)
    plt.close(fig)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    history = pd.read_csv(OUTPUT_DIR / "cnn_training_history.csv")
    plot_cnn_curve(history, "loss", "val_loss", "Loss", "cnn_loss_curves.png")
    plot_cnn_curve(history, "accuracy", "val_accuracy", "Accuracy", "cnn_accuracy_curves.png")

    rf_results = pd.read_csv(OUTPUT_DIR / "rf_tuning_results.csv")
    plot_rf_train_vs_val(rf_results)

    print(f"Saved figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
