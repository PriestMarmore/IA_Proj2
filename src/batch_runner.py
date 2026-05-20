import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
import time
warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ChurnModel import generate_data, prepare_data


# ─────────────────────────────────────────
# 1. BATCH SETTINGS
# ─────────────────────────────────────────

DATASET_SIZES = [100, 500, 1000, 5000, 10000, 50000, 100000]

HYPERPARAMS = {
    "Decision Tree": [
        {"max_depth": d} for d in [2, 3, 4, 5, 6, 8, 10]
    ],
    "Logistic Regression": [
        {"C": c} for c in [0.01, 0.1, 1, 10]
    ],
    "KNN": [
        {"n_neighbors": k} for k in [3, 5, 7, 10, 15]
    ],
    "Neural Network": [
        {"hidden_layer_sizes": s} for s in [(32,), (64, 32), (128, 64), (64, 32, 16)]
    ],
}


# ─────────────────────────────────────────
# 2. MODEL FACTORY
# ─────────────────────────────────────────

def build_model(name, params):
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=42, **params)
    elif name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=42, **params)
    elif name == "KNN":
        return KNeighborsClassifier(algorithm="ball_tree", **params)
    elif name == "Neural Network":
        return MLPClassifier(max_iter=500, random_state=42, **params)


# ─────────────────────────────────────────
# 3. EVALUATE
# ─────────────────────────────────────────

def evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")

    return {
        "Accuracy":   round(accuracy_score(y_test, y_pred), 4),
        "Precision":  round(precision_score(y_test, y_pred), 4),
        "Recall":     round(recall_score(y_test, y_pred), 4),
        "F1 Score":   round(f1_score(y_test, y_pred), 4),
        "CV F1 Mean": round(cv_scores.mean(), 4),
        "CV F1 Std":  round(cv_scores.std(), 4),
    }


# ─────────────────────────────────────────
# 4. RUN BATCH
# ─────────────────────────────────────────

def run_batch():
    results = []

    total = sum(len(DATASET_SIZES) * len(params) for params in HYPERPARAMS.values())
    current = 0

    for algo_name, param_list in HYPERPARAMS.items():
        for params in param_list:
            for size in DATASET_SIZES:
                current += 1
                param_str = ", ".join(f"{k}={v}" for k, v in params.items())
                print(f"[{current}/{total}] {algo_name} | {param_str} | n={size}")

                start = time.time()
                df = generate_data(n_customers=size)
                X_train, X_test, y_train, y_test, _ = prepare_data(df)

                model = build_model(algo_name, params)
                metrics = evaluate(model, X_train, X_test, y_train, y_test)
                elapsed = round(time.time() - start, 2)

                row = {
                    "Algorithm":    algo_name,
                    "Dataset Size": size,
                    **params,
                    **metrics,
                    "Time (s)":     elapsed,
                }
                results.append(row)

    return pd.DataFrame(results)


# ─────────────────────────────────────────
# 5. PLOTS
# ─────────────────────────────────────────

def plot_f1_vs_size(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    algos = df["Algorithm"].unique()

    for i, algo in enumerate(algos):
        ax = axes[i]
        subset = df[df["Algorithm"] == algo]

        param_col = [c for c in df.columns if c not in [
            "Algorithm", "Dataset Size", "Accuracy", "Precision",
            "Recall", "F1 Score", "CV F1 Mean", "CV F1 Std", "Time (s)"
        ]][0]

        for val in subset[param_col].unique():
            group = subset[subset[param_col] == val].sort_values("Dataset Size")
            ax.plot(group["Dataset Size"], group["F1 Score"],
                    marker="o", label=f"{param_col}={val}")

        ax.set_title(algo)
        ax.set_xlabel("Dataset Size")
        ax.set_ylabel("F1 Score")
        ax.legend(fontsize=7)
        ax.grid(True)

    plt.suptitle("F1 Score vs Dataset Size by Algorithm and Hyperparameter", fontsize=13)
    plt.tight_layout()
    plt.savefig("batch_f1_vs_size.png", dpi=150)
    plt.close()
    print("Saved: batch_f1_vs_size.png")


def plot_best_per_algo(df):
    best_rows = (
        df.sort_values("F1 Score", ascending=False)
          .groupby("Algorithm")
          .first()
          .reset_index()
    )

    plt.figure(figsize=(8, 5))
    bars = plt.bar(best_rows["Algorithm"], best_rows["F1 Score"], color="steelblue")
    plt.ylim(0.5, 1.0)
    plt.ylabel("Best F1 Score")
    plt.title("Best F1 Score per Algorithm (across all settings)")

    for bar, val in zip(bars, best_rows["F1 Score"]):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig("batch_best_per_algo.png", dpi=150)
    plt.close()
    print("Saved: batch_best_per_algo.png")


# ─────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("Starting batch run...\n")
    start_total = time.time()

    df_results = run_batch()

    df_results.to_csv("batch_results.csv", index=False)
    print("\nbatch_results.csv saved.")

    plot_f1_vs_size(df_results)
    plot_best_per_algo(df_results)

    total_time = round(time.time() - start_total, 1)
    print(f"\nDone in {total_time}s.")

    print("\nBest overall settings per algorithm:")
    print("=" * 60)
    best = (
        df_results.sort_values("F1 Score", ascending=False)
                  .groupby("Algorithm")
                  .first()
                  .reset_index()
    )
    print(best[["Algorithm", "Dataset Size", "F1 Score", "CV F1 Mean"]].to_string(index=False))