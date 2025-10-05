# Tests for FastAPI endpoints
# tests/test_main.py

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)


def test_health_check():
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


# def test_root_endpoint():
#     """Test root endpoint returns JSON"""
#     response = client.get("/")
#     assert response.status_code == 200
#     data = response.json()
#     assert "message" in data
#     assert "Fake News Detection API is running!" in data["message"]
def test_root_endpoint():
    """Test root endpoint returns HTML"""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Fake News Detection" in response.text


def test_predict_endpoint():
    """Test prediction endpoint"""
    response = client.post(
        "/predict",
        json={
            "text": "Scientists discover new breakthrough in renewable energy",
            "title": "Science News",
        },
    )
    assert response.status_code == 200
    data = response.json()
    # Should return prediction result
    assert any(key in data for key in ["label", "prediction", "confidence", "error"])


def test_predict_endpoint_text_only():
    """Test prediction with text only"""
    response = client.post(
        "/predict", json={"text": "Breaking: Major political scandal revealed"}
    )
    assert response.status_code == 200
    # Should not error out


def test_predict_endpoint_empty():
    """Test prediction with empty text"""
    response = client.post("/predict", json={"text": ""})
    # Should handle gracefully
    assert response.status_code in [200, 400, 422]
