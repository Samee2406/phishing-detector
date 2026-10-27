"""
Phishing Website Detection using Machine Learning
-------------------------------------------------
Trains and compares three classifiers on the UCI Phishing Websites dataset
(11,055 websites, 30 features). Reports accuracy, precision, recall, F1,
shows a confusion matrix and feature importances, and saves the best model.

Run:
    pip install -r requirements.txt
    python phishing_detector.py

Optional: use a local CSV instead of downloading
    python phishing_detector.py --csv path/to/phishing.csv --target Result
"""

import argparse

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def load_data(csv_path=None, target="Result"):
    """Load the dataset from a CSV, or download it from OpenML."""
    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        from sklearn.datasets import fetch_openml

        # UCI Phishing Websites dataset hosted on OpenML (id 4534)
        data = fetch_openml(data_id=4534, as_frame=True)
        df = data.frame

    # Some sources store the -1/0/1 values as text or categories
    df = df.astype(str).apply(pd.to_numeric, errors="coerce").dropna()
    if target not in df.columns:
        print(f"Column '{target}' not found; using last column "
              f"'{df.columns[-1]}' as the label.")
        target = df.columns[-1]
    # Normalise labels to 0 = legitimate, 1 = phishing
    # In the UCI dataset: -1 = phishing, 1 = legitimate
    y = (df[target] == -1).astype(int)
    X = df.drop(columns=[target])
    return X, y


def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="Path to a local CSV")
    parser.add_argument("--target", default="Result", help="Label column name")
    args = parser.parse_args()

    X, y = load_data(args.csv, args.target)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Phishing share: {y.mean():.1%}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        results.append(evaluate(name, model, X_test, y_test))

    results_df = pd.DataFrame(results).set_index("Model").round(4)
    print(results_df, "\n")
    results_df.to_csv("results.csv")

    best_name = results_df["F1"].idxmax()
    best = models[best_name]
    print(f"Best model: {best_name}\n")
    print(classification_report(y_test, best.predict(X_test),
                                target_names=["Legitimate", "Phishing"]))

    # Confusion matrix
    ConfusionMatrixDisplay.from_estimator(
        best, X_test, y_test, display_labels=["Legitimate", "Phishing"],
        cmap="Blues",
    )
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()

    # Feature importance (tree-based models)
    if hasattr(best, "feature_importances_"):
        imp = pd.Series(best.feature_importances_, index=X.columns)
        imp = imp.sort_values().tail(10)
        imp.plot(kind="barh", figsize=(7, 5), color="#2b6cb0")
        plt.title("Top 10 Features for Detecting Phishing")
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=150)
        plt.close()
        print("\nTop features:")
        print(imp.sort_values(ascending=False).round(4).to_string())

    joblib.dump(best, "phishing_model.joblib")
    print("\nSaved: results.csv, confusion_matrix.png, "
          "feature_importance.png, phishing_model.joblib")


if __name__ == "__main__":
    main()
