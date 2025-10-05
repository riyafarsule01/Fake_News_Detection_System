
# Fake News Detection – End-to-End ML & MLOps Project

**Data Source:** [Kaggle Fake News Detection Dataset](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)

![Python](https://img.shields.io/badge/python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.95-green)![CI/CD](https://img.shields.io/badge/GitHub%20Actions-Active-yellow) ![Status](https://img.shields.io/badge/Status-Development-orange)

## Table of Contents

1. [Project Overview](#project-overview)
2. [Current Status](#current-status)
3. [Next Steps Checklist](#next-steps-checklist)
4. [Features](#features)
5. [Project Structure](#project-structure)
6. [Installation & Setup](#installation--setup)
7. [Data](#data)
8. [Model Training & Evaluation](#model-training--evaluation)
9. [API](#api)
10. [Deployment](deployment)
11. [Testing](#testing)


---

## Current Status

✅ **Completed:**
- ✅ **Data Preparation & Validation** - Raw datasets processed and cleaned
- ✅ **Feature Engineering Pipeline** - TF-IDF vectorization with 5000 features
- ✅ **Model Training & Selection** - Random Forest achieved 99.67% accuracy
- ✅ **Model Evaluation & Metrics** - Comprehensive performance analysis completed
- ✅ **FastAPI Application** - Modern web interface with beautiful UI
- ✅ **Template Separation** - Clean code structure with HTML templates
- ✅ **API Endpoints** - Functional GET /, POST /predict, and JSON API


---

## Next Steps Checklist

### Phase 1: Data & Model Development ✅ COMPLETED
- [x] **Data Preparation** - ✅ Raw datasets processed (5,799 duplicates removed)
- [x] **Feature Engineering** - ✅ TF-IDF vectorization with 5000 features implemented
- [x] **Model Training** - ✅ 4 models trained (Random Forest selected as best)
- [x] **Model Evaluation** - ✅ 99.67% accuracy achieved with comprehensive metrics

### Phase 2: API & Integration ✅ COMPLETED
- [x] **FastAPI Application** - ✅ Modern web interface with gradient design
- [x] **Template update** - ✅ Clean HTML templates in app/frontend/
- [x] **API Testing** - ✅ All endpoints functional (GET /, POST /predict, JSON API)
- [x] **Local Environment** - ✅ Running successfully on localhost:3000
- [ ] **Unit Testing** - Execute comprehensive test suite with pytest



---

## Project Overview

The **Fake News Detection** project is a full-stack machine learning pipeline designed to detect whether a news article is **real or fake**. It includes:

* Data preprocessing
* Feature engineering (TF-IDF / embeddings)
* Model training & evaluation
* REST API for real-time predictions with **FastAPI**
* CI/CD workflow with **GitHub Actions**
* Deployment-ready setup for **Render**

This project demonstrates **production-ready MLOps practices**, including modular code, automated pipelines, and scalable architecture.

---

## Features

* Clean and modular folder structure
* Preprocessing & feature engineering scripts
* End-to-end ML pipeline: train → evaluate → save → predict
* REST API for predictions (`POST /predict`)
* CI/CD pipeline for testing and deployment
* Support for additional datasets via `data/raw_data/`

---

## Project Structure

```
fake-news-detection/
│
├── data/
│   ├── raw_data/           # Original datasets
│   │   ├── True.csv        # Real news articles
│   │   └── Fake.csv        # Fake news articles
│   └── processed/          # Cleaned datasets
│       ├── processed_data.csv      # Final processed dataset
│       └── pre_processed_data.csv  # Intermediate preprocessing
│
├── notebooks/              # EDA and experiments
│   └── eda_fake_news_detection.ipynb
│
├── src/                    # ML pipeline (✅ COMPLETED)
│   ├── __init__.py
│   ├── preprocess.py      # Data cleaning & TF-IDF feature extraction
│   ├── train.py           # Model training (4 algorithms)
│   ├── predict.py         # Prediction logic with model loading
│   └── evaluate.py        # Model evaluation & metrics
│
├── models/                 # Saved trained models (✅ TRAINED)
│   ├── best_model.pkl           # Random Forest (99.67% accuracy)
│   ├── all_models.pkl           # All 4 trained models
│   ├── tfidf_vectorizer.pkl     # TF-IDF feature extractor
│   ├── train_test_data.pkl      # Training/test data splits
│   ├── model_info.pkl           # Model metadata
│   ├── model_comparison.png     # Performance comparison chart
│   └── evaluation/              # Detailed metrics
│       ├── detailed_metrics.csv
│       └── model_comparison.csv
│
├── app/                    # FastAPI web application 
│   ├── __init__.py
│   ├── main.py            # Clean FastAPI app with template loading
│   └── docs/        
│       └── home.html
        └── style.css
        └── script.js     # Beautiful modern UI with gradient design
│
├── config/
│   ├── __init__.py
│   └── config.yaml        # Configuration settings
│
├── tests/                  # Unit tests
│   ├── __init__.py
│   ├── test_main.py       # FastAPI endpoint tests
│   ├── test_model.py      # Model functionality tests
│
|── requirements.txt       # Python dependencies
|
├── README.md             # Project documentation
└── .gitignore            # Git ignore rules
```

---

## Installation & Setup

1. **Clone the repository**

```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Setup project (preprocessing + train model)**

```bash
bash scripts/setup.sh
```

---

## Data

* Place all raw datasets in `data/raw_data/`.
* Supported format: CSV with at least `title` and `text` columns.
* Preprocessing outputs will be saved in `data/processed/`.

**Example files:**

```
data/raw_data/true.csv
data/raw_data/fake.csv
```

---

## Model Training & Evaluation ✅ COMPLETED

**ML Models Implemented & Results:**
- **Random Forest** - 99.67% accuracy ⭐ (Selected as best model)
- **Logistic Regression** - High performance baseline
- **Decision Tree Classifier** - Good interpretability
- **Gradient Boost Classifier** - Advanced ensemble method

**Feature Engineering:**
- **TF-IDF Vectorization** with 5000 features
- **Text Preprocessing** - Removed 5,799 duplicate articles
- **Data Cleaning** - Standardized text format and encoding

**Model Performance:**
- **Best Model**: Random Forest with 99.67% accuracy
- **Dataset Size**: 44,898 articles after preprocessing
- **Feature Count**: 5000 TF-IDF features
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score

**Pipeline Scripts:**
- Training: `python src/train.py` ✅
- Preprocessing: `python src/preprocess.py` ✅
- Evaluation: `python src/evaluate.py` ✅
- Prediction: `python src/predict.py` ✅

**Model Storage:**
Trained models and artifacts saved to `models/` directory:
- `best_model.pkl` - Random Forest classifier
- `tfidf_vectorizer.pkl` - Feature extraction pipeline
- `model_info.pkl` - Performance metadata

---

## API ✅ COMPLETED

**FastAPI Web Application:**
- **Beautiful Modern UI** with gradient design and animations
- **Template Separation** - HTML,CSS,Javascript in `app/docs/home.html/Style.css/Script.js`
- **Clean Code Structure** - No code duplication
- **Responsive Design** - Works on mobile and desktop

**Local Development:**
```bash
# Start the application
python app/main.py
# or
uvicorn app.main:app --reload --host 0.0.0.0 --port 3000
```

**Access Points:**
- **Web Interface**: http://localhost:3000/
- **API Documentation**: http://localhost:3000/docs

**Available Endpoints:**
- `GET /` → Beautiful web interface with form
- `GET /health` → API health status and model information
- `POST /predict` → Web form submission (returns HTML)
- `POST /api/predict` → JSON API endpoint

**JSON API Example:**
```bash
# Request
curl -X POST "http://localhost:3000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Breaking News Example",
    "text": "This is the article content..."
  }'

# Response
{
  "label": "Real News",
  "confidence": 0.856,
  "probabilities": {
    "fake": 0.144,
    "real": 0.856
  }
}
```

**Deployment:**

* Ready to deploy on **Render**.
* Ensure `uvicorn main:app` is used as entrypoint in Dockerfile.

---

## CI/CD

* GitHub Actions workflow: `.github/workflows/ci-cd.yml`
* Automates:

  * Testing (`pytest`)
  * Deployment to Render

---

## Testing

* Tests are located in `tests/`
* Run all tests:

```bash
pytest -v
```

