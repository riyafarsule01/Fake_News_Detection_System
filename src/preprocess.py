# Preprocessing functions - Handles cleaning + vectorizer functionality for fake news detection
# src/preprocess.py

import re
import string
import pickle
import pandas as pd
import numpy as np
import os
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# ---------- TEXT CLEANING ----------
def clean_text(text):
    """
    Lowercase, remove punctuation, numbers, extra spaces, and stopwords.
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    try:
        stop_words = set(stopwords.words("english"))
        text = " ".join([word for word in text.split() if word not in stop_words])
    except LookupError:
        print(
            "NLTK stopwords not found. Please run: python -m nltk.downloader stopwords"
        )
        # Continue without stopword removal
        pass

    return text


def preprocess_dataframe(df, text_col="text"):
    """
    Applies text cleaning to the specified column in a DataFrame.
    """
    df = df.copy()
    df[text_col] = df[text_col].astype(str).apply(clean_text)
    return df


# ---------- VECTORIZER UTILITIES ----------
def create_vectorizer():
    """
    Returns a TF-IDF vectorizer.
    """
    return TfidfVectorizer(max_features=5000, stop_words="english")


def save_pickle(obj, path):
    """Save object to pickle file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    """Load object from pickle file"""
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------- MAIN PREPROCESSING PIPELINE ----------
def load_and_preprocess_data():
    """
    Load raw data, combine, preprocess, and save processed data.
    """
    print("\n========== Starting data preprocessing pipeline... ==========")

    # Load raw datasets
    try:
        fake_df = pd.read_csv("data/raw_data/Fake.csv")
        true_df = pd.read_csv("data/raw_data/True.csv")
        print(
            f"\n========== Loaded {len(fake_df)} fake news and {len(true_df)} true news articles ==========="
        )
    except FileNotFoundError as e:
        print(f"Error loading datasets: {e}")
        return None

    # Add labels
    fake_df["label"] = 0  # Fake news
    true_df["label"] = 1  # True news

    # Combine datasets
    df_combined = pd.concat([fake_df, true_df], ignore_index=True)
    print(f"\n ========== Combined dataset shape: {df_combined.shape} ==========")

    # Check for required columns
    required_cols = ["title", "text"]
    missing_cols = [col for col in required_cols if col not in df_combined.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        print(f"Available columns: {df_combined.columns.tolist()}")
        return None

    # Handle missing values
    df_combined["title"] = df_combined["title"].fillna("")
    df_combined["text"] = df_combined["text"].fillna("")

    # Combine title and text for better feature extraction
    df_combined["combined_text"] = df_combined["title"] + " " + df_combined["text"]

    # Clean the combined text
    print("\n========== Cleaning text data... ==========")
    df_processed = preprocess_dataframe(df_combined, text_col="combined_text")

    # Remove duplicate records
    print("\n========== Removing duplicate records... ==========")
    initial_count = len(df_processed)
    df_processed = df_processed.drop_duplicates(subset=["combined_text"], keep="first")
    duplicates_removed = initial_count - len(df_processed)
    print(f"\n ========== Removed {duplicates_removed} duplicate records ==========")

    # Remove empty texts after cleaning
    df_processed = df_processed[df_processed["combined_text"].str.len() > 0]
    print(
        f"\n ==========After cleaning and deduplication, dataset shape: {df_processed.shape}=========="
    )

    # Shuffle the data
    df_processed = df_processed.sample(frac=1, random_state=42).reset_index(drop=True)

    # Create output directory
    os.makedirs("data/processed", exist_ok=True)

    # Save processed data
    output_path = "data/processed/processed_data.csv"
    df_processed.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")

    # Display basic statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(df_processed)}")
    print(f"Fake news: {len(df_processed[df_processed['label'] == 0])}")
    print(f"True news: {len(df_processed[df_processed['label'] == 1])}")
    print(f"Average text length: {df_processed['combined_text'].str.len().mean():.2f}")

    return df_processed


def create_train_test_split():
    """
    Create train/test split and save TF-IDF features.
    """
    print("\nCreating train/test split and TF-IDF features...")

    # Load processed data
    try:
        df = pd.read_csv("data/processed/processed_data.csv")
    except FileNotFoundError:
        print("Processed data not found. Please run load_and_preprocess_data() first.")
        return None

    # Split features and labels
    X = df["combined_text"]
    y = df["label"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Create and fit TF-IDF vectorizer
    vectorizer = create_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"TF-IDF feature shape: {X_train_tfidf.shape}")

    # Save vectorizer
    os.makedirs("models", exist_ok=True)
    save_pickle(vectorizer, "models/tfidf_vectorizer.pkl")
    print("TF-IDF vectorizer saved to: models/tfidf_vectorizer.pkl")

    # Save train/test splits
    train_data = {
        "X_train": X_train_tfidf,
        "X_test": X_test_tfidf,
        "y_train": y_train,
        "y_test": y_test,
    }
    save_pickle(train_data, "models/train_test_data.pkl")
    print("Train/test data saved to: models/train_test_data.pkl")

    return train_data


if __name__ == "__main__":
    # Run the complete preprocessing pipeline
    print("=== FAKE NEWS DETECTION - PREPROCESSING PIPELINE ===")

    # Step 1: Load and preprocess data
    processed_df = load_and_preprocess_data()

    if processed_df is not None:
        # Step 2: Create train/test split and TF-IDF features
        train_data = create_train_test_split()

        if train_data is not None:
            print(
                "\n========== Preprocessing pipeline completed successfully! =========="
            )
            print("Next step: Run src/train.py to train models")
        else:
            print("\n Error in creating train/test split")
    else:
        print("\n Error in data preprocessing")
