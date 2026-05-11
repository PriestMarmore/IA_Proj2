"""
E-Commerce Customer Churn Prediction
=====================================
Generates synthetic customer data and trains a Decision Tree classifier
to predict whether a customer will churn in the next 30 days.

To add more algorithms later, see the MODELS dictionary at the bottom.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use("Agg")

# ─────────────────────────────────────────
# 1. GENERATE SYNTHETIC DATA
# ─────────────────────────────────────────

def generate_data(n_customers=1000, random_state=42):
    """
    Generate realistic fake customer data for churn prediction.

    Features (RFM + extras):
      - recency_days       : days since last purchase
      - frequency          : number of purchases in last 6 months
      - monetary           : total amount spent (USD)
      - avg_order_value    : average value per order
      - num_returns        : number of returned items
      - email_open_rate    : fraction of marketing emails opened (0–1)
      - days_since_signup  : how long they've been a customer

    Target:
      - churned            : 1 if customer churned, 0 if not
    """
    rng = np.random.default_rng(random_state)
    n = n_customers

    recency_days     = rng.integers(1, 365, n)
    frequency        = rng.integers(1, 50, n)
    monetary         = rng.uniform(10, 5000, n).round(2)
    avg_order_value  = (monetary / frequency).round(2)
    num_returns      = rng.integers(0, 10, n)
    email_open_rate  = rng.uniform(0, 1, n).round(2)
    days_since_signup= rng.integers(30, 1825, n)

    # Churn logic: high recency, low frequency, low email engagement → more likely to churn
    churn_score = (
        0.4 * (recency_days / 365) +
        0.3 * (1 - frequency / 50) +
        0.2 * (1 - email_open_rate) +
        0.1 * (num_returns / 10)
    )
    noise = rng.uniform(0, 0.2, n)
    churned = ((churn_score + noise) > 0.55).astype(int)

    df = pd.DataFrame({
        "recency_days":      recency_days,
        "frequency":         frequency,
        "monetary":          monetary,
        "avg_order_value":   avg_order_value,
        "num_returns":       num_returns,
        "email_open_rate":   email_open_rate,
        "days_since_signup": days_since_signup,
        "churned":           churned,
    })

    return df


# ─────────────────────────────────────────
# 2. PREPARE DATA
# ─────────────────────────────────────────

def prepare_data(df):
    FEATURES = [
        "recency_days", "frequency", "monetary",
        "avg_order_value", "num_returns",
        "email_open_rate", "days_since_signup"
    ]
    TARGET = "churned"

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, FEATURES


# ─────────────────────────────────────────
# 3. MODELS
# ─────────────────────────────────────────

MODELS = {
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
}



# ─────────────────────────────────────────
# 4. TRAIN & EVALUATE
# ─────────────────────────────────────────

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")

    results = {
        "Model":     name,
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "F1 Score":  f1_score(y_test, y_pred),
        "CV F1 Mean": cv_scores.mean(),
        "CV F1 Std":  cv_scores.std(),
    }

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy : {results['Accuracy']:.3f}")
    print(f"  Precision: {results['Precision']:.3f}")
    print(f"  Recall   : {results['Recall']:.3f}")
    print(f"  F1 Score : {results['F1 Score']:.3f}")
    print(f"  CV F1    : {results['CV F1 Mean']:.3f} ± {results['CV F1 Std']:.3f}")
    print(f"\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["Not Churned", "Churned"]))

    return results, model, y_pred


# ─────────────────────────────────────────
# 5. PLOTS
# ─────────────────────────────────────────

def plot_confusion_matrix(name, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Churned", "Churned"],
                yticklabels=["Not Churned", "Churned"])
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{name.replace(' ', '_').lower()}.png", dpi=150)
    # plt.show()
    print(f"Saved: confusion_matrix_{name.replace(' ', '_').lower()}.png")


def plot_feature_importance(name, model, feature_names):
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(feature_names)), importances[indices], color="steelblue")
    plt.xticks(range(len(feature_names)),
               [feature_names[i] for i in indices], rotation=45, ha="right")
    plt.title(f"Feature Importances — {name}")
    plt.tight_layout()
    plt.savefig(f"feature_importance_{name.replace(' ', '_').lower()}.png", dpi=150)
    # plt.show()
    print(f"Saved: feature_importance_{name.replace(' ', '_').lower()}.png")


# ─────────────────────────────────────────
# 6. SAVE MODEL
# ─────────────────────────────────────────

def save_best_model(all_results, trained_models, feature_names):
    best = max(all_results, key=lambda r: r["F1 Score"])
    best_model = trained_models[best["Model"]]

    with open("best_model.pkl", "wb") as f:
        pickle.dump({"model": best_model, "features": feature_names}, f)

    print(f"\n✅ Best model: {best['Model']} (F1={best['F1 Score']:.3f})")
    print("   Saved to best_model.pkl")
    return best_model


# ─────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("Generating synthetic customer data...")
    df = generate_data(n_customers=10000)
    print(f"Dataset: {len(df)} customers | Churn rate: {df['churned'].mean():.1%}")
    print(df.describe().round(2))

    X_train, X_test, y_train, y_test, feature_names = prepare_data(df)

    all_results = []
    trained_models = {}

    for name, model in MODELS.items():
        results, trained_model, y_pred = evaluate_model(
            name, model, X_train, X_test, y_train, y_test
        )
        all_results.append(results)
        trained_models[name] = trained_model

        plot_confusion_matrix(name, y_test, y_pred)
        plot_feature_importance(name, trained_model, feature_names)

    # Summary table
    print("\n\n📊 RESULTS SUMMARY")
    print("=" * 60)
    summary = pd.DataFrame(all_results).set_index("Model")
    print(summary[["Accuracy", "Precision", "Recall", "F1 Score", "CV F1 Mean"]].round(3))

    save_best_model(all_results, trained_models, feature_names)

    # Save dataset for inspection / web app use
    df.to_csv("customer_data.csv", index=False)
    print("\n📁 customer_data.csv saved.")