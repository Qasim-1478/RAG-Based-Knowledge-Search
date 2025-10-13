#!/bin/bash

# RAG System Startup Script
echo "🚀 Starting RAG Knowledge Base System"
echo "======================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Check if Ollama is running
echo "🔍 Checking Ollama status..."
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "⚠️  Ollama is not running. Starting Ollama service..."
    echo "   Please run 'ollama serve' in another terminal"
    echo "   Then run 'python setup_model.py' to download the model"
    echo ""
    echo "🚀 Starting Flask application anyway..."
else
    echo "✅ Ollama is running"
    
    # Check if model is available
    echo "🤖 Checking model availability..."
    if python -c "
import requests
try:
    response = requests.get('http://localhost:11434/api/tags')
    models = [model['name'] for model in response.json().get('models', [])]
    if 'deepseek-r1:1.5b' in models:
        print('✅ Model is available')
        exit(0)
    else:
        print('❌ Model not found')
        exit(1)
except:
    print('❌ Cannot connect to Ollama')
    exit(1)
"; then
        echo "✅ Model is ready"
    else
        echo "⚠️  Model not found. Please run: python setup_model.py"
    fi
fi

echo ""
echo "🌐 Starting Flask application..."
echo "   Access the system at: http://localhost:5000"
echo "   Press Ctrl+C to stop"
echo ""

# Start Flask application
python app.py
