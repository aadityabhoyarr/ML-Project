"""Model Training and Evaluation Pipeline.

Executes 12 systematic experiments:
  3 Classifiers: Naive Bayes, Decision Tree, k-NN
  x 2 Feature Sets: TF-IDF vs. Handcrafted
  x 2 Conditions: Condition A (Full Text) vs. Condition B (Truncated 50 Words)

Computes 5-fold cross-validation, test accuracy, precision, recall, F1,
generates confusion matrices, overfitting curves, decision tree plots,
and saves the best models.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Ensure local imports work regardless of working directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from preprocess import preprocess_and_split
from features import (
    HandcraftedFeatureExtractor,
    get_tfidf_vectorizer,
    HANDCRAFTED_FEATURE_LABELS,
)
from plots import (
    plot_overfitting_curve,
    plot_decision_tree_structure,
    plot_feature_importances,
    plot_confusion_matrix_heatmap,
    plot_model_comparison_bar,
)


def ensure_data_ready() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ensures train.csv and test.csv exist. If not, runs preprocessing."""
    train_path = os.path.join("data", "train.csv")
    test_path = os.path.join("data", "test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("[Pipeline] Preprocessed data not found. Running preprocess_and_split()...")
        train_df, test_df = preprocess_and_split()
    else:
        print(f"[Pipeline] Loading existing train/test data from '{train_path}' and '{test_path}'...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

    return train_df, test_df


def run_experiments():
    """Runs all 12 experiments, logs metrics, generates visual assets,

    and saves model checkpoints.
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    train_df, test_df = ensure_data_ready()
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    # Experiment definitions
    experiments = [
        # --- CONDITION A: FULL TEXT ---
        {
            "Condition": "Condition A (Full Text)",
            "Model": "Naive Bayes",
            "Feature_Set": "TF-IDF",
            "pipeline": Pipeline([
                ("tfidf", get_tfidf_vectorizer(5000)),
                ("clf", MultinomialNB()),
            ]),
            "train_col": "text_clean",
            "test_col": "text_clean",
        },
        {
            "Condition": "Condition A (Full Text)",
            "Model": "Decision Tree",
            "Feature_Set": "TF-IDF",
            "pipeline": Pipeline([
                ("tfidf", get_tfidf_vectorizer(5000)),
                ("clf", DecisionTreeClassifier(random_state=42)),
            ]),
            "train_col": "text_clean",
            "test_col": "text_clean",
        },
        {
            "Condition": "Condition A (Full Text)",
            "Model": "k-NN",
            "Feature_Set": "TF-IDF",
            "pipeline": Pipeline([
                ("tfidf", get_tfidf_vectorizer(5000)),
                ("clf", KNeighborsClassifier(n_neighbors=5, metric="cosine")),
            ]),
            "train_col": "text_clean",
            "test_col": "text_clean",
        },
        {
            "Condition": "Condition A (Full Text)",
            "Model": "Naive Bayes",
            "Feature_Set": "Handcrafted",
            "pipeline": Pipeline([
                ("features", HandcraftedFeatureExtractor()),
                ("scaler", StandardScaler()),
                ("clf", GaussianNB()),
            ]),
            "train_col": "text_raw",
            "test_col": "text_raw",
        },
        {
            "Condition": "Condition A (Full Text)",
            "Model": "Decision Tree",
            "Feature_Set": "Handcrafted",
            "pipeline": Pipeline([
                ("features", HandcraftedFeatureExtractor()),
                ("clf", DecisionTreeClassifier(random_state=42)),
            ]),
            "train_col": "text_raw",
            "test_col": "text_raw",
        },
        {
            "Condition": "Condition A (Full Text)",
            "Model": "k-NN",
            "Feature_Set": "Handcrafted",
            "pipeline": Pipeline([
                ("features", HandcraftedFeatureExtractor()),
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=5, metric="minkowski")),
            ]),
            "train_col": "text_raw",
            "test_col": "text_raw",
        },

        # --- CONDITION B: TRUNCATED (FIRST 50 WORDS) ---
        {
            "Condition": "Condition B (Truncated 50 Words)",
            "Model": "Naive Bayes",
            "Feature_Set": "TF-IDF",
            "pipeline": Pipeline([
                ("tfidf", get_tfidf_vectorizer(5000)),
                ("clf", MultinomialNB()),
            ]),
            "train_col": "text_clean_trunc",
            "test_col": "text_clean_trunc",
        },
        {
            "Condition": "Condition B (Truncated 50 Words)",
            "Model": "Decision Tree",
            "Feature_Set": "TF-IDF",
            "pipeline": Pipeline([
                ("tfidf", get_tfidf_vectorizer(5000)),
                ("clf", DecisionTreeClassifier(random_state=42)),
            ]),
            "train_col": "text_clean_trunc",
            "test_col": "text_clean_trunc",
        },
        {
            "Condition": "Condition B (Truncated 50 Words)",
            "Model": "k-NN",
            "Feature_Set": "TF-IDF",
            "pipeline": Pipeline([
                ("tfidf", get_tfidf_vectorizer(5000)),
                ("clf", KNeighborsClassifier(n_neighbors=5, metric="cosine")),
            ]),
            "train_col": "text_clean_trunc",
            "test_col": "text_clean_trunc",
        },
        {
            "Condition": "Condition B (Truncated 50 Words)",
            "Model": "Naive Bayes",
            "Feature_Set": "Handcrafted",
            "pipeline": Pipeline([
                ("features", HandcraftedFeatureExtractor()),
                ("scaler", StandardScaler()),
                ("clf", GaussianNB()),
            ]),
            "train_col": "text_raw_trunc",
            "test_col": "text_raw_trunc",
        },
        {
            "Condition": "Condition B (Truncated 50 Words)",
            "Model": "Decision Tree",
            "Feature_Set": "Handcrafted",
            "pipeline": Pipeline([
                ("features", HandcraftedFeatureExtractor()),
                ("clf", DecisionTreeClassifier(random_state=42)),
            ]),
            "train_col": "text_raw_trunc",
            "test_col": "text_raw_trunc",
        },
        {
            "Condition": "Condition B (Truncated 50 Words)",
            "Model": "k-NN",
            "Feature_Set": "Handcrafted",
            "pipeline": Pipeline([
                ("features", HandcraftedFeatureExtractor()),
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=5, metric="minkowski")),
            ]),
            "train_col": "text_raw_trunc",
            "test_col": "text_raw_trunc",
        },
    ]

    results_records = []
    fitted_pipelines = []
    cms = []

    print("\n" + "=" * 80)
    print("STARTING 12 SYSTEMATIC EXPERIMENTS (3 Models x 2 Feature Sets x 2 Conditions)")
    print("=" * 80)

    for idx, exp in enumerate(experiments, 1):
        condition = exp["Condition"]
        model_name = exp["Model"]
        feat_name = exp["Feature_Set"]
        pipeline = exp["pipeline"]
        train_col = exp["train_col"]
        test_col = exp["test_col"]

        print(f"\n[{idx}/12] Running: {condition} | {feat_name} | {model_name}...")
        X_tr = train_df[train_col]
        X_te = test_df[test_col]

        # 5-fold cross-validation on train set
        print("  -> Performing 5-fold Cross-Validation on training set...")
        cv_scores = cross_val_score(pipeline, X_tr, y_train, cv=5, scoring="accuracy")
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        # Fit on entire training set
        print("  -> Fitting pipeline on full training set...")
        pipeline.fit(X_tr, y_train)

        # Predict on test set
        y_pred = pipeline.predict(X_te)

        # Compute test metrics
        test_acc = float(accuracy_score(y_test, y_pred))
        test_prec = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
        test_rec = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
        test_f1 = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))
        cm = confusion_matrix(y_test, y_pred)

        cms.append(cm)
        fitted_pipelines.append(pipeline)

        results_records.append({
            "Experiment": idx,
            "Condition": condition,
            "Feature_Set": feat_name,
            "Model": model_name,
            "CV_Accuracy_Mean": round(cv_mean, 4),
            "CV_Accuracy_Std": round(cv_std, 4),
            "CV_Score": f"{cv_mean:.4f} +/- {cv_std:.4f}",
            "Test_Accuracy": round(test_acc, 4),
            "Test_Precision": round(test_prec, 4),
            "Test_Recall": round(test_rec, 4),
            "Test_F1": round(test_f1, 4),
        })

        print(f"  -> Results: Test Acc = {test_acc:.4f} | F1 = {test_f1:.4f} | 5-Fold CV = {cv_mean:.4f} +/- {cv_std:.4f}")

    # Build results DataFrame
    results_df = pd.DataFrame(results_records)
    results_csv_path = os.path.join("results", "model_comparison.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\n[Results] Saved full comparison table to: {results_csv_path}")

    # Display clean formatted table
    print("\n" + "=" * 110)
    print("FINAL MODEL BENCHMARK RESULTS TABLE")
    print("=" * 110)
    display_cols = [
        "Experiment",
        "Condition",
        "Feature_Set",
        "Model",
        "CV_Score",
        "Test_Accuracy",
        "Test_Precision",
        "Test_Recall",
        "Test_F1",
    ]
    print(results_df[display_cols].to_string(index=False))
    print("=" * 110)

    # Plot overall model comparison bar chart
    plot_model_comparison_bar(results_df, "results/model_comparison_chart.png")

    # Select Best Model for Condition A and Condition B based on Test_F1 then Test_Accuracy
    df_cond_a = results_df[results_df["Condition"] == "Condition A (Full Text)"]
    best_idx_a = df_cond_a.sort_values(by=["Test_F1", "Test_Accuracy"], ascending=False).index[0]
    best_pipe_a = fitted_pipelines[best_idx_a]
    best_rec_a = results_df.iloc[best_idx_a]

    df_cond_b = results_df[results_df["Condition"] == "Condition B (Truncated 50 Words)"]
    best_idx_b = df_cond_b.sort_values(by=["Test_F1", "Test_Accuracy"], ascending=False).index[0]
    best_pipe_b = fitted_pipelines[best_idx_b]
    best_rec_b = results_df.iloc[best_idx_b]

    print(f"\n[Selection] Best Model for Condition A: {best_rec_a['Model']} ({best_rec_a['Feature_Set']}) with Test F1 = {best_rec_a['Test_F1']}")
    print(f"[Selection] Best Model for Condition B: {best_rec_b['Model']} ({best_rec_b['Feature_Set']}) with Test F1 = {best_rec_b['Test_F1']}")

    # Save models using joblib
    model_a_path = os.path.join("models", "best_model_condition_a.joblib")
    model_b_path = os.path.join("models", "best_model_condition_b.joblib")

    # Save with metadata dict wrapper
    joblib.dump({
        "pipeline": best_pipe_a,
        "model_name": best_rec_a["Model"],
        "feature_set": best_rec_a["Feature_Set"],
        "condition": best_rec_a["Condition"],
        "test_accuracy": best_rec_a["Test_Accuracy"],
        "test_f1": best_rec_a["Test_F1"],
    }, model_a_path)

    joblib.dump({
        "pipeline": best_pipe_b,
        "model_name": best_rec_b["Model"],
        "feature_set": best_rec_b["Feature_Set"],
        "condition": best_rec_b["Condition"],
        "test_accuracy": best_rec_b["Test_Accuracy"],
        "test_f1": best_rec_b["Test_F1"],
    }, model_b_path)

    print(f"[Checkpoint] Best Model Condition A saved to: {model_a_path}")
    print(f"[Checkpoint] Best Model Condition B saved to: {model_b_path}")

    # Generate Confusion Matrix Heatmaps for best models
    plot_confusion_matrix_heatmap(
        cms[best_idx_a],
        f"{best_rec_a['Model']} ({best_rec_a['Feature_Set']})",
        best_rec_a["Condition"],
        "results/confusion_matrix_condition_a.png",
    )
    plot_confusion_matrix_heatmap(
        cms[best_idx_b],
        f"{best_rec_b['Model']} ({best_rec_b['Feature_Set']})",
        best_rec_b["Condition"],
        "results/confusion_matrix_condition_b.png",
    )

    # --- STEP 4: OVERFITTING DEMO ---
    print("\n" + "=" * 80)
    print("RUNNING STEP 4: DECISION TREE OVERFITTING DEMO (max_depth=1 to 20)")
    print("=" * 80)
    X_tr_raw_b = train_df["text_raw_trunc"]
    X_te_raw_b = test_df["text_raw_trunc"]

    # Pre-extract features for fast depth sweep
    print("[Overfitting Demo] Pre-extracting handcrafted features for depth sweep...")
    extractor = HandcraftedFeatureExtractor()
    X_tr_feats = extractor.transform(X_tr_raw_b)
    X_te_feats = extractor.transform(X_te_raw_b)

    depths = list(range(1, 21))
    train_scores = []
    test_scores = []

    for d in depths:
        dt = DecisionTreeClassifier(max_depth=d, random_state=42)
        dt.fit(X_tr_feats, y_train)
        tr_acc = accuracy_score(y_train, dt.predict(X_tr_feats))
        te_acc = accuracy_score(y_test, dt.predict(X_te_feats))
        train_scores.append(tr_acc)
        test_scores.append(te_acc)

    plot_overfitting_curve(depths, train_scores, test_scores, "results/overfitting_curve.png")

    # --- STEP 5: INTERPRETABILITY DEMO ---
    print("\n" + "=" * 80)
    print("RUNNING STEP 5: INTERPRETABILITY & DECISION TREE VISUALIZATION")
    print("=" * 80)
    # Train small interpretable tree (depth=3)
    small_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    small_tree.fit(X_tr_feats, y_train)

    plot_decision_tree_structure(
        small_tree,
        feature_names=HANDCRAFTED_FEATURE_LABELS,
        class_names=["Human", "AI"],
        save_path="results/decision_tree.png",
    )

    # Feature importances from the unconstrained/fair decision tree
    plot_feature_importances(
        small_tree.feature_importances_,
        feature_names=HANDCRAFTED_FEATURE_LABELS,
        save_path="results/feature_importance.png",
    )

    print("\n[Pipeline Complete] All 12 models trained, evaluated, and assets saved!")
    return results_df


if __name__ == "__main__":
    run_experiments()
