#!/usr/bin/env python3
"""
Test script to verify the RAG system is working correctly.
"""

import os
import sys
import tempfile
import requests
import time
from document_processor import DocumentProcessor
from rag_system import RAGSystem

def test_document_processor():
    """Test document processing functionality."""
    print("🧪 Testing Document Processor...")
    
    processor = DocumentProcessor()
    
    # Create a test document
    test_content = """
    This is a test document about artificial intelligence.
    AI is a field of computer science that focuses on creating intelligent machines.
    Machine learning is a subset of AI that enables computers to learn without being explicitly programmed.
    Deep learning is a subset of machine learning that uses neural networks with multiple layers.
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        temp_file = f.name
    
    try:
        # Test processing
        result = processor.process_document(temp_file)
        
        if result:
            print("✅ Document processing successful")
            print(f"   - Filename: {result['filename']}")
            print(f"   - Chunks: {result['chunk_count']}")
            print(f"   - Words: {result['word_count']}")
            return result
        else:
            print("❌ Document processing failed")
            return None
            
    finally:
        # Clean up
        os.unlink(temp_file)

def test_rag_system():
    """Test RAG system functionality."""
    print("\n🧪 Testing RAG System...")
    
    try:
        rag = RAGSystem()
        
        # Check model status
        model_status = rag.check_model_status()
        print(f"📊 Model Status: {'✅ Available' if model_status else '❌ Not Available'}")
        
        if not model_status:
            print("⚠️  Model not available. Please run setup_model.py first.")
            return False
        
        # Test document processing
        test_doc = test_document_processor()
        if not test_doc:
            return False
        
        # Add document to RAG system
        print("📚 Adding document to knowledge base...")
        added_count = rag.add_documents([test_doc])
        print(f"✅ Added {added_count} chunks to knowledge base")
        
        # Test query
        print("🔍 Testing query...")
        query = "What is artificial intelligence?"
        relevant_docs = rag.retrieve_documents(query, max_results=2)
        
        if relevant_docs:
            print(f"✅ Retrieved {len(relevant_docs)} relevant documents")
            for i, doc in enumerate(relevant_docs):
                print(f"   {i+1}. Score: {doc['score']:.3f}, Content: {doc['content'][:50]}...")
        else:
            print("❌ No relevant documents retrieved")
            return False
        
        # Test answer generation
        print("🤖 Testing answer generation...")
        answer = rag.generate_answer(query, relevant_docs)
        
        if answer and not answer.startswith("Error"):
            print("✅ Answer generated successfully")
            print(f"   Answer: {answer[:100]}...")
            return True
        else:
            print(f"❌ Answer generation failed: {answer}")
            return False
            
    except Exception as e:
        print(f"❌ RAG system test failed: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints."""
    print("\n🧪 Testing API Endpoints...")
    
    base_url = "http://localhost:5000"
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/api/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Health endpoint working")
            print(f"   Status: {data['status']}")
            print(f"   Model Status: {data['model_status']}")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to API: {e}")
        print("   Make sure the Flask app is running (python app.py)")
        return False
    
    # Test documents endpoint
    try:
        response = requests.get(f"{base_url}/api/documents", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Documents endpoint working")
            print(f"   Total documents: {data['total_count']}")
        else:
            print(f"❌ Documents endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Documents endpoint error: {e}")
    
    return True

def main():
    """Run all tests."""
    print("🚀 RAG System Test Suite")
    print("=" * 50)
    
    # Test 1: Document Processor
    test_document_processor()
    
    # Test 2: RAG System
    rag_success = test_rag_system()
    
    # Test 3: API Endpoints (if Flask app is running)
    api_success = test_api_endpoints()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 50)
    print(f"RAG System: {'✅ PASS' if rag_success else '❌ FAIL'}")
    print(f"API Endpoints: {'✅ PASS' if api_success else '❌ FAIL'}")
    
    if rag_success and api_success:
        print("\n🎉 All tests passed! The RAG system is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
