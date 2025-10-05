# Model evaluation script - Comprehensive performance analysis and visualization
# src/evaluate.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import warnings

warnings.filterwarnings("ignore")


def load_test_data():
    """Load test data from preprocessing pipeline"""
    print(" Loading test data...")

    try:
        # Load from preprocessing pipeline
        with open("models/train_test_data.pkl", "rb") as f:
            data = pickle.load(f)

        X_test = data["X_test"]
        y_test = data["y_test"]

        print(f" Loaded test data: {X_test.shape[0]} samples")
        return X_test, y_test

    except FileNotFoundError:
        print(" Test data not found! Run preprocess.py and train.py first.")
        return None, None


def load_models():
    """Load all trained models and vectorizer"""
    print(" Loading trained models...")

    try:
        # Load all models
        with open("models/all_models.pkl", "rb") as f:
            all_models = pickle.load(f)

        # Load best model
        with open("models/best_model.pkl", "rb") as f:
            best_model = pickle.load(f)

        # Load model info
        with open("models/model_info.pkl", "rb") as f:
            model_info = pickle.load(f)

        # Load vectorizer
        with open("models/tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)

        print(f" Loaded {len(all_models)} models")
        print(f"   Best model: {model_info['best_model_name']}")

        return all_models, best_model, model_info, vectorizer

    except FileNotFoundError as e:
        print(f" Model files not found: {e}")
        print(" Run train.py first!")
        return None, None, None, None


def evaluate_single_model(model, X_test, y_test, model_name):
    """Evaluate a single model and return metrics"""
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability for class 1

    # Calculate metrics
    metrics = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "predictions": y_pred,
        "probabilities": y_pred_proba,
    }

    return metrics


def evaluate_all_models(all_models, X_test, y_test):
    """Evaluate all models and return comparison"""
    print("\n Evaluating All Models...")
    print("=" * 60)

    results = []

    for model_name, model in all_models.items():
        print(f"\n Evaluating {model_name}...")
        metrics = evaluate_single_model(model, X_test, y_test, model_name)
        results.append(metrics)

        # Print metrics
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1-Score:  {metrics['f1_score']:.4f}")
        print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")

    return results


def create_comparison_table(results):
    """Create performance comparison table"""
    print("\n Model Performance Comparison")
    print("=" * 80)

    # Create DataFrame
    comparison_data = []
    for result in results:
        comparison_data.append(
            {
                "Model": result["model_name"],
                "Accuracy": f"{result['accuracy']:.4f}",
                "Precision": f"{result['precision']:.4f}",
                "Recall": f"{result['recall']:.4f}",
                "F1-Score": f"{result['f1_score']:.4f}",
                "ROC-AUC": f"{result['roc_auc']:.4f}",
            }
        )

    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))

    # Find best model for each metric
    print(f"\n Best Performing Models:")
    metrics_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

    for metric in metrics_cols:
        # Convert to float for comparison
        df_comparison[f"{metric}_float"] = df_comparison[metric].astype(float)
        best_idx = df_comparison[f"{metric}_float"].idxmax()
        best_model = df_comparison.loc[best_idx, "Model"]
        best_score = df_comparison.loc[best_idx, metric]
        print(f"   {metric}: {best_model} ({best_score})")

    return df_comparison


def plot_model_comparison(results):
    """Create visualization comparing model performance"""
    print("\n Creating performance visualization...")

    # Extract data for plotting
    models = [r["model_name"] for r in results]
    accuracy = [r["accuracy"] for r in results]
    precision = [r["precision"] for r in results]
    recall = [r["recall"] for r in results]
    f1 = [r["f1_score"] for r in results]
    roc_auc = [r["roc_auc"] for r in results]

    # Create subplot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold")

    # Individual metric plots
    metrics = [accuracy, precision, recall, f1, roc_auc]
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    colors = ["skyblue", "lightgreen", "salmon", "gold", "lightcoral"]

    for i, (metric, name, color) in enumerate(zip(metrics, metric_names, colors)):
        row, col = divmod(i, 3)
        ax = axes[row, col]

        bars = ax.bar(models, metric, color=color, alpha=0.7, edgecolor="black")
        ax.set_title(f"{name} Comparison", fontweight="bold")
        ax.set_ylabel(name)
        ax.set_ylim(0, 1.1)

        # Add value labels on bars
        for bar, value in zip(bars, metric):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Rotate x-axis labels if needed
        ax.tick_params(axis="x", rotation=45)

    # Overall comparison plot
    ax = axes[1, 2]
    x_pos = np.arange(len(models))
    width = 0.15

    ax.bar(x_pos - 2 * width, accuracy, width, label="Accuracy", alpha=0.7)
    ax.bar(x_pos - width, precision, width, label="Precision", alpha=0.7)
    ax.bar(x_pos, recall, width, label="Recall", alpha=0.7)
    ax.bar(x_pos + width, f1, width, label="F1-Score", alpha=0.7)
    ax.bar(x_pos + 2 * width, roc_auc, width, label="ROC-AUC", alpha=0.7)

    ax.set_title("All Metrics Comparison", fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig("models/model_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(" Visualization saved to: models/model_comparison.png")


def generate_detailed_report(best_result, y_test):
    """Generate detailed classification report for best model"""
    print(f"\n Detailed Report - Best Model: {best_result['model_name']}")
    print("=" * 70)

    # Classification report
    print("\n Classification Report:")
    print(
        classification_report(
            y_test, best_result["predictions"], target_names=["Fake News", "Real News"]
        )
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, best_result["predictions"])
    print(f"\n Confusion Matrix:")
    print(f"                 Predicted")
    print(f"               Fake   Real")
    print(f"Actual  Fake   {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"        Real   {cm[1,0]:4d}   {cm[1,1]:4d}")

    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)

    print(f"\n Additional Metrics:")
    print(f"   True Positives:  {tp}")
    print(f"   True Negatives:  {tn}")
    print(f"   False Positives: {fp}")
    print(f"   False Negatives: {fn}")
    print(f"   Specificity:     {specificity:.4f}")


def save_evaluation_results(results, comparison_df):
    """Save evaluation results to files"""
    print("\n Saving evaluation results...")

    # Create evaluation directory
    os.makedirs("models/evaluation", exist_ok=True)

    # Save comparison table
    comparison_df.to_csv("models/evaluation/model_comparison.csv", index=False)

    # Save detailed results
    detailed_results = []
    for result in results:
        detailed_results.append(
            {
                "model": result["model_name"],
                "accuracy": result["accuracy"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1_score": result["f1_score"],
                "roc_auc": result["roc_auc"],
            }
        )

    pd.DataFrame(detailed_results).to_csv(
        "models/evaluation/detailed_metrics.csv", index=False
    )

    print(" Results saved to:")
    print("   - models/evaluation/model_comparison.csv")
    print("   - models/evaluation/detailed_metrics.csv")
    print("   - models/model_comparison.png")


def main():
    """Main evaluation pipeline"""
    print(" Starting Model Evaluation Pipeline")
    print("=" * 50)

    # Load test data
    X_test, y_test = load_test_data()
    if X_test is None:
        return

    # Load models
    all_models, best_model, model_info, vectorizer = load_models()
    if all_models is None:
        return

    # Evaluate all models
    results = evaluate_all_models(all_models, X_test, y_test)

    # Create comparison table
    comparison_df = create_comparison_table(results)

    # Plot comparisons
    plot_model_comparison(results)

    # Find best result for detailed report
    best_result = max(results, key=lambda x: x["f1_score"])
    generate_detailed_report(best_result, y_test)

    # Save results
    save_evaluation_results(results, comparison_df)

    print(f"\n Evaluation Complete!")
    print(f" Best performing model: {best_result['model_name']}")
    print(f" F1-Score: {best_result['f1_score']:.4f}")


if __name__ == "__main__":
    main()
