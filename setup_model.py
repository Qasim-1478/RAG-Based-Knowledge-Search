#!/usr/bin/env python3
"""
Script to download and setup the model using Ollama.
This ensures the model is available locally for offline RAG operations.
"""

import subprocess
import sys
import time
import requests

def check_ollama_installed():
    """Check if Ollama is installed and running."""
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Ollama is installed: {result.stdout.strip()}")
            return True
        else:
            print("❌ Ollama is not properly installed")
            return False
    except FileNotFoundError:
        print("❌ Ollama is not installed")
        return False

def check_ollama_running():
    """Check if Ollama service is running."""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            print("✅ Ollama service is running")
            return True
    except requests.exceptions.RequestException:
        pass
    
    print("❌ Ollama service is not running")
    print("Please start Ollama service: ollama serve")
    return False

def download_model():
    """Download the model."""
    model_name = "gemma3:1b"
    
    print(f"🚀 Starting download of {model_name}...")
    print("This may take several minutes depending on your internet connection...")
    
    try:
        # Start the download process
        process = subprocess.Popen(
            ['ollama', 'pull', model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Print progress in real-time
        for line in iter(process.stdout.readline, ''):
            print(line.strip())
        
        process.wait()
        
        if process.returncode == 0:
            print(f"✅ Successfully downloaded {model_name}")
            return True
        else:
            print(f"❌ Failed to download {model_name}")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False

def verify_model():
    """Verify the model is available and working."""
    model_name = "gemma3:1b"
    
    print(f"🔍 Verifying {model_name} is working...")
    
    try:
        # Test the model with a simple query
        test_payload = {
            "model": model_name,
            "prompt": "Hello, how are you?",
            "stream": False
        }
        
        response = requests.post(
            'http://localhost:11434/api/generate',
            json=test_payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Model is working correctly!")
            print(f"Test response: {result.get('response', 'No response')[:100]}...")
            return True
        else:
            print(f"❌ Model verification failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error verifying model: {e}")
        return False

def list_models():
    """List all available models."""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=10)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print("\n📋 Available models:")
            for model in models:
                print(f"  - {model['name']} (size: {model.get('size', 'unknown')})")
            return models
        else:
            print("❌ Failed to list models")
            return []
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return []

def main():
    """Main setup function."""
    print("🤖 Ollama Model Setup Script")
    print("=" * 50)
    
    # Check prerequisites
    if not check_ollama_installed():
        print("\n📥 Please install Ollama first:")
        print("Visit: https://ollama.ai/download")
        sys.exit(1)
    
    if not check_ollama_running():
        print("\n🔄 Please start Ollama service:")
        print("Run: ollama serve")
        sys.exit(1)
    
    # List current models
    current_models = list_models()
    model_name = "gemma3:1b"
    
    # Check if model already exists
    model_exists = any(model['name'] == model_name for model in current_models)
    
    if model_exists:
        print(f"✅ {model_name} is already installed!")
        if verify_model():
            print("🎉 Setup complete! The model is ready to use.")
            sys.exit(0)
        else:
            print("⚠️  Model exists but verification failed. Re-downloading...")
    
    # Download the model
    if download_model():
        if verify_model():
            print("\n🎉 Setup complete! The model is ready to use.")
            print(f"✅ {model_name} is installed and working")
        else:
            print("\n⚠️  Model downloaded but verification failed")
            sys.exit(1)
    else:
        print("\n❌ Setup failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
