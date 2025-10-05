# Model training script - Loads data → preprocess → vectorize → train → save model and vectorizer
# src/train.py
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings("ignore")


def load_processed_data():
    """Load preprocessed data"""
    print(" Loading processed data...")

    try:
        # Try loading from preprocessing pipeline
        with open("models/train_test_data.pkl", "rb") as f:
            data = pickle.load(f)

        X_train = data["X_train"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        y_test = data["y_test"]

        print(f" Loaded train/test data from preprocessing pipeline")
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Testing samples: {X_test.shape[0]}")
        print(f"   Feature dimensions: {X_train.shape[1]}")

        return X_train, X_test, y_train, y_test

    except FileNotFoundError:
        print("  Preprocessed train/test data not found. Loading from CSV...")

        # Fallback: load from processed CSV
        df = pd.read_csv("data/processed/processed_data.csv")

        # Combine title and text (matching preprocessing)
        df["combined_text"] = df["title"] + " " + df["text"]
        X = df["combined_text"]
        y = df["label"].astype(int)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f" Loaded and split data from CSV")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")

        return X_train, X_test, y_train, y_test


def train_models(X_train, X_test, y_train, y_test):
    """Train all 4 models and return best model"""
    print("\n Training Models...")

    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    results = {}
    trained_models = {}

    print("\n Training Results:")
    print("-" * 60)

    for name, model in models.items():
        print(f"\n Training {name}...")

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Calculate accuracies
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)

        # Store results
        results[name] = {
            "model": model,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "predictions": y_pred_test,
        }

        trained_models[name] = model

        print(f"    {name}")
        print(f"      Train Accuracy: {train_acc:.4f}")
        print(f"      Test Accuracy:  {test_acc:.4f}")

    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]["test_accuracy"])
    best_model = results[best_model_name]["model"]
    best_accuracy = results[best_model_name]["test_accuracy"]

    print(f"\n Best Model: {best_model_name}")
    print(f"   Test Accuracy: {best_accuracy:.4f}")

    return trained_models, results, best_model, best_model_name


def save_models(trained_models, best_model, best_model_name, vectorizer):
    """Save all models and vectorizer"""
    print(f"\n Saving Models...")

    # Create models directory
    os.makedirs("models", exist_ok=True)

    # Save best model
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    # Save all trained models
    with open("models/all_models.pkl", "wb") as f:
        pickle.dump(trained_models, f)

    # Save vectorizer
    with open("models/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    # Save model info
    model_info = {
        "best_model_name": best_model_name,
        "models_available": list(trained_models.keys()),
    }

    with open("models/model_info.pkl", "wb") as f:
        pickle.dump(model_info, f)

    print(f" Saved:")
    print(f"   - Best model: {best_model_name}")
    print(f"   - All models: {len(trained_models)} models")
    print(f"   - TF-IDF vectorizer")


def generate_detailed_report(results, y_test):
    """Generate detailed classification report"""
    print(f"\n Detailed Model Performance Report")
    print("=" * 80)

    for name, result in results.items():
        print(f"\n {name} Detailed Results:")
        print("-" * 50)
        print(
            classification_report(
                y_test, result["predictions"], target_names=["Fake News", "Real News"]
            )
        )


def main():
    """Main training pipeline"""
    print(" Starting Model Training Pipeline")
    print("=" * 50)

    # Load data
    try:
        X_train, X_test, y_train, y_test = load_processed_data()
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print(" Make sure to run preprocess.py first!")
        return

    # Check if we need to vectorize (in case data is raw text)
    # If data is already sparse matrix (from preprocessing), skip vectorization
    try:
        # Check if data is already vectorized (sparse matrix)
        if hasattr(X_train, "toarray"):  # Simple check for sparse matrix
            print(" Data is already vectorized (sparse matrix)")
            # Load vectorizer for later use
            try:
                with open("models/tfidf_vectorizer.pkl", "rb") as f:
                    vectorizer = pickle.load(f)
                print(" Loaded existing TF-IDF vectorizer")
            except FileNotFoundError:
                print(" Vectorizer not found! Run preprocess.py first.")
                return
        elif isinstance(
            X_train.iloc[0] if hasattr(X_train, "iloc") else X_train[0], str
        ):
            print("\n Vectorizing text data...")

            # Load existing vectorizer or create new one
            try:
                with open("models/tfidf_vectorizer.pkl", "rb") as f:
                    vectorizer = pickle.load(f)
                print(" Loaded existing TF-IDF vectorizer")
                X_train = vectorizer.transform(X_train)
                X_test = vectorizer.transform(X_test)
            except FileNotFoundError:
                print("  Creating new TF-IDF vectorizer...")
                vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
                X_train = vectorizer.fit_transform(X_train)
                X_test = vectorizer.transform(X_test)
        else:
            # Data is already vectorized
            try:
                with open("models/tfidf_vectorizer.pkl", "rb") as f:
                    vectorizer = pickle.load(f)
            except FileNotFoundError:
                print(" Vectorizer not found! Run preprocess.py first.")
                return
    except Exception as e:
        print(f" Error processing data: {e}")
        return

    # Train models
    trained_models, results, best_model, best_model_name = train_models(
        X_train, X_test, y_train, y_test
    )

    # Save models
    save_models(trained_models, best_model, best_model_name, vectorizer)

    # Generate detailed report
    generate_detailed_report(results, y_test)

    print(f"\n Training Complete!")
    print(f" Best model ({best_model_name}) saved to models/best_model.pkl")
    print(f" Ready for predictions!")


if __name__ == "__main__":
    main()
