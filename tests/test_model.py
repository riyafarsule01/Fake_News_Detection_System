# Tests for model training and inference
# This file can focus on unit-testing your ML model and vectorizer separately from FastAPI
# tests/test_model.py
import pytest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import predict_texts

# Check if we're in CI environment
CI_ENVIRONMENT = os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true"


def test_single_text_prediction():
    """Test single text prediction"""
    result = predict_texts("Breaking news: President announces new policy")

    if CI_ENVIRONMENT and "error" in result and "Model not loaded" in result["error"]:
        # In CI, just check that the function runs without crashing
        assert isinstance(result, dict)
        assert "error" in result
        pytest.skip("Models not available in CI environment")
    else:
        assert isinstance(result, dict)
        assert "label" in result
        assert "confidence" in result
        assert "prediction" in result
        assert "probabilities" in result
        assert result["label"] in ["Real News", "Fake News"]
        assert result["prediction"] in [0, 1]
        assert 0 <= result["confidence"] <= 1


def test_dict_input_prediction():
    """Test prediction with dict input (title + text)"""
    test_input = {
        "title": "Science News",
        "text": "Scientists discover new breakthrough in renewable energy",
    }
    result = predict_texts(test_input)

    if CI_ENVIRONMENT and "error" in result and "Model not loaded" in result["error"]:
        assert isinstance(result, dict)
        assert "error" in result
        pytest.skip("Models not available in CI environment")
    else:
        assert isinstance(result, dict)
        assert "label" in result
        assert "confidence" in result
        assert result["label"] in ["Real News", "Fake News"]


def test_batch_prediction():
    """Test batch prediction"""
    test_texts = [
        "Breaking news: President announces new policy",
        "You won $5000! Click the link now!",
        "Scientists develop new medical treatment",
    ]
    results = predict_texts(test_texts)

    if CI_ENVIRONMENT and isinstance(results, dict) and "error" in results:
        assert isinstance(results, dict)
        assert "error" in results
        pytest.skip("Models not available in CI environment")
    else:
        assert isinstance(results, list)
        assert len(results) == len(test_texts)
        for result in results:
            assert isinstance(result, dict)
            assert "label" in result
            assert "confidence" in result
            assert result["label"] in ["Real News", "Fake News"]


def test_empty_text():
    """Test prediction with empty text"""
    result = predict_texts("")

    if CI_ENVIRONMENT and "error" in result and "Model not loaded" in result["error"]:
        assert isinstance(result, dict)
        pytest.skip("Models not available in CI environment")
    else:
        assert isinstance(result, dict)
        assert "label" in result
        assert result["label"] in ["Real News", "Fake News"]


def test_model_availability():
    """Test that model loading works when models are available"""
    result = predict_texts("Test news content")

    if CI_ENVIRONMENT and "error" in result and "Model not loaded" in result["error"]:
        # In CI, models might not be available, which is expected
        assert isinstance(result, dict)
        pytest.skip("Models not available in CI environment - this is expected")
    else:
        # When models are available, should not have loading errors
        assert "error" not in result or result.get("error") != "Model not loaded"
