#!/bin/bash
echo " Setting up Fake News Detection Project..."

# Install Python dependencies
echo " Installing dependencies..."
pip install -r requirements.txt

# Download NLTK data
echo " Downloading NLTK data..."
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Train model if not exists
echo " Training ML model..."
if [ ! -f "models/best_model.pkl" ]; then
    python src/train.py
    echo " Model trained successfully!"
else
    echo " Model already exists!"
fi

echo " Setup complete! Run 'uvicorn app.main:app --host 0.0.0.0 --port 3000' to start the API"