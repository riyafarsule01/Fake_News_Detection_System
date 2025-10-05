#!/bin/bash
echo " Starting Fake News Detection API..."

# Check if models exist
if [ ! -f "models/best_model.pkl" ]; then
    echo " Models not found! Running setup..."
    ./scripts/setup.sh
fi

# Start the API
echo " Starting FastAPI server..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 3000