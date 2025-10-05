# Prediction script --> New input → preprocess → load pickle files → predict
# src/predict.py
import pandas as pd
import pickle
import re
import string
import os
from typing import List, Union
import warnings

warnings.filterwarnings("ignore")


def load_model_artifacts():
    """Load trained model and vectorizer"""
    try:
        # Load best model
        with open("models/best_model.pkl", "rb") as f:
            model = pickle.load(f)

        # Load vectorizer
        with open("models/tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)

        # Load model info
        with open("models/model_info.pkl", "rb") as f:
            model_info = pickle.load(f)

        print(f" Loaded: {model_info['best_model_name']}")
        return model, vectorizer, model_info

    except FileNotFoundError as e:
        print(f" Model files not found: {e}")
        print(" Run train.py first to train models!")
        return None, None, None


def clean_text(text):
    """Clean and preprocess text (matching preprocessing pipeline)"""
    if pd.isna(text) or text == "":
        return ""

    # Convert to string and lowercase
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove special characters and digits
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra whitespaces
    text = " ".join(text.split())

    return text


def preprocess_texts(texts: List[str]) -> List[str]:
    """Preprocess a list of texts"""
    cleaned_texts = []

    for text in texts:
        cleaned = clean_text(text)
        cleaned_texts.append(cleaned)

    return cleaned_texts


def predict_single(title: str, text: str, model, vectorizer) -> dict:
    """Predict single news article"""
    # Combine title and text (matching training pipeline)
    combined_text = f"{title} {text}"

    # Clean text
    cleaned_text = clean_text(combined_text)

    # Vectorize
    text_vector = vectorizer.transform([cleaned_text])

    # Predict
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]

    # Format result
    result = {
        "prediction": int(prediction),
        "label": "Real News" if prediction == 1 else "Fake News",
        "confidence": float(max(probability)),
        "probabilities": {"fake": float(probability[0]), "real": float(probability[1])},
    }

    return result


def predict_batch(data: Union[List[str], List[dict]], model, vectorizer) -> List[dict]:
    """Predict multiple texts"""
    results = []

    for item in data:
        if isinstance(item, str):
            # Single text string
            result = predict_single("", item, model, vectorizer)
        elif isinstance(item, dict):
            # Dictionary with title and text
            title = item.get("title", "")
            text = item.get("text", "")
            result = predict_single(title, text, model, vectorizer)
        else:
            result = {"error": "Invalid input format"}

        results.append(result)

    return results


def predict_texts(
    texts: Union[str, dict, List[str], List[dict]],
) -> Union[dict, List[dict]]:
    """Main prediction function"""
    # Load model artifacts
    model, vectorizer, model_info = load_model_artifacts()

    if model is None:
        return {"error": "Model not loaded"}

    # Handle single string input
    if isinstance(texts, str):
        return predict_single("", texts, model, vectorizer)

    # Handle single dictionary input
    if isinstance(texts, dict):
        title = texts.get("title", "")
        text = texts.get("text", "")
        return predict_single(title, text, model, vectorizer)

    # Handle list input
    if isinstance(texts, list):
        return predict_batch(texts, model, vectorizer)

    return {"error": "Invalid input type"}


def interactive_prediction():
    """Interactive prediction mode"""
    print(" Interactive Fake News Detection")
    print("=" * 40)

    # Load model
    model, vectorizer, model_info = load_model_artifacts()

    if model is None:
        return

    print(f"Using: {model_info['best_model_name']}")
    print("\nEnter news details (or 'quit' to exit):")

    while True:
        print("\n" + "-" * 40)
        title = input(" News Title: ").strip()

        if title.lower() == "quit":
            break

        text = input(" News Text: ").strip()

        if text.lower() == "quit":
            break

        # Make prediction
        result = predict_single(title, text, model, vectorizer)

        # Display result
        print(f"\n Prediction: {result['label']}")
        print(f" Confidence: {result['confidence']:.2%}")
        print(f" Probabilities:")
        print(f"   Fake: {result['probabilities']['fake']:.2%}")
        print(f"   Real: {result['probabilities']['real']:.2%}")


def main():
    """Main function for testing"""
    print(" Testing Prediction Pipeline")
    print("=" * 40)

    # Test samples
    test_samples = [
        {
            "title": "Breaking: New COVID vaccine shows 95% efficacy",
            "text": "Scientists at major pharmaceutical company announced today that their new vaccine candidate showed 95% efficacy in preventing COVID-19 in clinical trials.",
        },
        {
            "title": "You won $1 Million! Click here now!",
            "text": "Congratulations! You have been selected to receive $1 million dollars. Click this link immediately to claim your prize before it expires.",
        },
        "Scientists discover new planet in nearby solar system",
        "Aliens have landed in New York City according to secret government documents",
    ]

    print(" Testing with sample data...")

    for i, sample in enumerate(test_samples, 1):
        print(f"\n--- Test {i} ---")

        if isinstance(sample, dict):
            print(f"Title: {sample['title'][:50]}...")
            print(f"Text: {sample['text'][:50]}...")
        else:
            print(f"Text: {sample[:50]}...")

        result = predict_texts(sample)

        if "error" not in result:
            print(f"Prediction: {result['label']} ({result['confidence']:.2%})")
        else:
            print(f"Error: {result['error']}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_prediction()
    else:
        main()
