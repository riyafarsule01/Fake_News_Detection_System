
# Fake News Detection – End-to-End ML & MLOps Project

**Data Source:** [Kaggle Fake News Detection Dataset](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)

![Python](https://img.shields.io/badge/python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.95-green) ![Docker](https://img.shields.io/badge/Docker-20.10-blue) ![CI/CD](https://img.shields.io/badge/GitHub%20Actions-Active-yellow) ![Status](https://img.shields.io/badge/Status-Development-orange)

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
10. [Docker & Deployment](#docker--deployment)
11. [CI/CD](#cicd)
12. [Testing](#testing)
13. [Contributing](#contributing)
14. [License](#license)

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

🚧 **In Progress:**
- Unit testing suite execution
- Enhanced application testing

⏳ **Pending:**
- Docker containerization
- CI/CD pipeline validation
- Production deployment

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

### Phase 3: Deployment & Production
- [ ] **Docker Container** - Build and test containerized application
- [ ] **CI/CD Pipeline** - Validate GitHub Actions workflow
- [ ] **Production Deployment** - Deploy to cloud provider (Render)
- [ ] **Documentation** - Final cleanup and documentation updates

---

## Project Overview

The **Fake News Detection** project is a full-stack machine learning pipeline designed to detect whether a news article is **real or fake**. It includes:

* Data preprocessing
* Feature engineering (TF-IDF / embeddings)
* Model training & evaluation
* REST API for real-time predictions with **FastAPI**
* Containerization with **Docker**
* CI/CD workflow with **GitHub Actions**
* Deployment-ready setup for **Render**

This project demonstrates **production-ready MLOps practices**, including modular code, automated pipelines, and scalable architecture.

---

## Features

* Clean and modular folder structure
* Preprocessing & feature engineering scripts
* End-to-end ML pipeline: train → evaluate → save → predict
* REST API for predictions (`POST /predict`)
* Dockerized environment for reproducibility
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
├── app/                    # FastAPI web application (✅ COMPLETED)
│   ├── __init__.py
│   ├── main.py            # Clean FastAPI app with template loading
│   └── templates/         # HTML templates
│       └── home.html      # Beautiful modern UI with gradient design
│
├── config/
│   ├── __init__.py
│   └── config.yaml        # Configuration settings
│
├── tests/                  # Unit tests
│   ├── __init__.py
│   ├── test_main.py       # FastAPI endpoint tests
│   ├── test_model.py      # Model functionality tests
│   └── manual_test.py     # Manual testing scripts
│
├── scripts/                # Helper scripts
│   ├── run.sh             # Application startup script
│   └── setup.sh           # Environment setup script
│
├── Dockerfile             # Docker containerization
├── docker-compose.yml     # Multi-container setup
├── requirements.txt       # Python dependencies
├── Makefile              # Build automation
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
- **Template Separation** - HTML in `app/templates/home.html`
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
- **Health Check**: http://localhost:3000/health
- **API Documentation**: http://localhost:3000/docs

**Available Endpoints:**
- `GET /` → Beautiful web interface with form
- `GET /health` → API health status and model information
- `POST /predict` → Web form submission (returns HTML)
- `POST /api/predict` → JSON API endpoint

**Web Form Features:**
- **Title Field** (optional) - News headline input
- **Content Field** (required) - Full article text
- **Real-time Results** - Instant prediction with confidence
- **Visual Feedback** - Color-coded results (Green=Real, Red=Fake)
- **Confidence Bar** - Visual confidence percentage

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

---

## Docker & Deployment

**Build Docker image:**

```bash
docker build -t fake-news-detection .
```

**Run Docker container:**

```bash
docker run -p 8000:8000 fake-news-detection
```

**Deployment:**

* Ready to deploy on **Render** or any Docker-compatible cloud provider.
* Ensure `uvicorn main:app` is used as entrypoint in Dockerfile.

---

## CI/CD

* GitHub Actions workflow: `.github/workflows/ci-cd.yml`
* Automates:

  * Testing (`pytest`)
  * Docker build & push
  * Deployment to Render

---

## Testing

* Tests are located in `tests/`
* Run all tests:

```bash
pytest -v
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m "Description"`
4. Push branch: `git push origin feature-name`
5. Open a Pull Request

---

## License

This project is licensed under **MIT License**.

---

I can also create a **Markdown table with all scripts and their purpose** for even better clarity in this README if you want.

Do you want me to do that next?
