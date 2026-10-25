"""Plotting and Visualization Utilities.

Generates:
1. Overfitting curve (train vs. test accuracy across Decision Tree max_depth values).
2. Decision tree visualization (sklearn plot_tree with feature names and classes).
3. Feature importance horizontal bar chart.
4. Confusion matrix heatmaps for best models.
5. Overall model comparison bar chart.
"""

import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.tree import plot_tree

# Styling configuration
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_overfitting_curve(
    depths: list[int],
    train_scores: list[float],
    test_scores: list[float],
    save_path: str = "results/overfitting_curve.png",
):
    """Plots training vs test accuracy as max_depth varies from 1 to 20,

    demonstrating overfitting and the effect of pre-pruning.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6), dpi=300)

    plt.plot(depths, train_scores, "o-", label="Train Accuracy", color="#1f77b4", linewidth=2.5, markersize=6)
    plt.plot(depths, test_scores, "s-", label="Test Accuracy", color="#d62728", linewidth=2.5, markersize=6)

    # Highlight optimal test depth
    best_idx = int(np.argmax(test_scores))
    best_depth = depths[best_idx]
    best_acc = test_scores[best_idx]

    plt.axvline(x=best_depth, color="#2ca02c", linestyle="--", alpha=0.8, label=f"Optimal Test Depth = {best_depth} ({best_acc*100:.1f}%)")
    plt.scatter([best_depth], [best_acc], color="#2ca02c", s=140, zorder=5)

    plt.title("Decision Tree Overfitting Analysis: Train vs Test Accuracy vs Tree Depth\n(Handcrafted Features - Condition B: Truncated 50 Words)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Tree Max Depth (Pre-pruning parameter)", fontsize=11, fontweight="semibold")
    plt.ylabel("Accuracy", fontsize=11, fontweight="semibold")
    plt.xticks(depths)
    plt.ylim(min(min(test_scores), min(train_scores)) - 0.03, 1.02)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10, loc="lower right")

    # Annotate region of underfitting vs overfitting
    plt.annotate(
        "Underfitting\n(High Bias)",
        xy=(2, train_scores[1]),
        xytext=(2, train_scores[1] - 0.10),
        arrowprops=dict(facecolor="#555555", arrowstyle="->", lw=1),
        fontsize=9,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="#bbbbbb"),
    )
    plt.annotate(
        "Overfitting\n(High Variance,\n100% Train memorization)",
        xy=(depths[-2], train_scores[-2]),
        xytext=(depths[-4], 0.90),
        arrowprops=dict(facecolor="#555555", arrowstyle="->", lw=1),
        fontsize=9,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fdf2f2", ec="#e0b4b4"),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plotting] Overfitting curve saved to: {save_path}")


def plot_decision_tree_structure(
    tree_clf,
    feature_names: list[str],
    class_names: list[str] = ["Human", "AI"],
    save_path: str = "results/decision_tree.png",
):
    """Visualizes an interpretable shallow decision tree (max_depth=3 or 4)

    using sklearn plot_tree with feature names and class labels.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(24, 12), dpi=300)

    plot_tree(
        tree_clf,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        impurity=True,
        proportion=False,
        precision=3,
        fontsize=10,
        ax=ax,
    )

    plt.title(
        "Interpretable Decision Tree (max_depth=3, Handcrafted Features, Condition B)\nHuman (0) vs AI (1)",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plotting] Decision tree plot saved to: {save_path}")


def plot_feature_importances(
    importances: np.ndarray,
    feature_names: list[str],
    save_path: str = "results/feature_importance.png",
):
    """Generates a horizontal bar chart displaying feature importances from the decision tree."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Sort descending
    indices = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in indices]
    sorted_vals = importances[indices]

    plt.figure(figsize=(10, 6), dpi=300)
    colors = sns.color_palette("viridis", len(sorted_names))[::-1]
    bars = plt.barh(range(len(sorted_names)), sorted_vals, color=colors, edgecolor="#333333", height=0.6)

    plt.yticks(range(len(sorted_names)), sorted_names, fontsize=10)
    plt.gca().invert_yaxis()  # Top feature at top
    plt.xlabel("Gini Feature Importance", fontsize=11, fontweight="semibold")
    plt.title("Decision Tree Feature Importances (Condition B: Handcrafted Features)", fontsize=13, fontweight="bold", pad=12)
    plt.grid(axis="x", linestyle="--", alpha=0.6)

    # Value labels on bars
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.005, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", ha="left", fontsize=9, fontweight="bold")

    plt.xlim(0, max(sorted_vals) * 1.15 if max(sorted_vals) > 0 else 1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plotting] Feature importance chart saved to: {save_path}")


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    model_name: str,
    condition: str,
    save_path: str,
    class_names: list[str] = ["Human", "AI"],
):
    """Draws a confusion matrix heatmap showing raw counts and class-wise percentages."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(6.5, 5.5), dpi=300)

    # Calculate percentages per true class (row)
    cm_pct = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)"

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"fontsize": 12, "fontweight": "bold"},
        linewidths=1.5,
        linecolor="#dddddd",
    )

    plt.title(f"Confusion Matrix: {model_name}\n({condition})", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=11, fontweight="semibold")
    plt.ylabel("Actual Label", fontsize=11, fontweight="semibold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plotting] Confusion matrix saved to: {save_path}")


def plot_model_comparison_bar(
    comparison_df: pd.DataFrame,
    save_path: str = "results/model_comparison_chart.png",
):
    """Creates a grouped bar chart comparing test accuracies across all models and conditions."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(12, 6.5), dpi=300)

    # Build clean label: Model + Feature Set
    df = comparison_df.copy()
    df["Model_Features"] = df["Model"] + " (" + df["Feature_Set"] + ")"

    palette = {"Condition A (Full Text)": "#3b82f6", "Condition B (Truncated 50 Words)": "#10b981"}

    ax = sns.barplot(
        data=df,
        x="Model_Features",
        y="Test_Accuracy",
        hue="Condition",
        palette=palette,
        edgecolor="#222222",
    )

    plt.title("Model Accuracy Comparison: Full Text (Condition A) vs Truncated (Condition B)", fontsize=13, fontweight="bold", pad=14)
    plt.xlabel("Model & Feature Representation", fontsize=11, fontweight="semibold")
    plt.ylabel("Test Set Accuracy", fontsize=11, fontweight="semibold")
    plt.xticks(rotation=25, ha="right", fontsize=9.5)
    plt.ylim(0.4, 1.02)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(title="Evaluation Condition", fontsize=10, title_fontsize=10)

    # Annotate heights
    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h) and h > 0:
            ax.annotate(f"{h*100:.1f}%", (p.get_x() + p.get_width() / 2.0, h), ha="center", va="bottom", fontsize=8, fontweight="bold", xytext=(0, 2), textcoords="offset points")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plotting] Model comparison bar chart saved to: {save_path}")
