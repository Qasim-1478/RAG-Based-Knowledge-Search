"""
Flask-based RAG (Retrieval-Augmented Generation) System
Knowledge-base Search Engine with offline LLM support
"""

import os
import json
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime

from document_processor import DocumentProcessor
from rag_system import RAGSystem
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Initialize components
doc_processor = DocumentProcessor()
rag_system = RAGSystem()

@app.route('/')
def index():
    """Serve the main frontend page."""
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_status': rag_system.check_model_status()
    })

@app.route('/api/upload', methods=['POST'])
def upload_documents():
    """Upload and process documents."""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        if not files or all(file.filename == '' for file in files):
            return jsonify({'error': 'No files selected'}), 400
        
        uploaded_files = []
        processed_documents = []
        
        for file in files:
            if file and file.filename:
                # Secure the filename
                filename = secure_filename(file.filename)
                
                # Generate unique filename to avoid conflicts
                unique_filename = f"{uuid.uuid4()}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                
                # Save file
                file.save(filepath)
                uploaded_files.append({
                    'original_name': filename,
                    'saved_path': filepath,
                    'size': os.path.getsize(filepath)
                })
                
                # Process document
                try:
                    document_data = doc_processor.process_document(filepath)
                    if document_data:
                        processed_documents.append(document_data)
                        logger.info(f"Processed document: {filename}")
                    else:
                        logger.warning(f"Failed to process document: {filename}")
                except Exception as e:
                    logger.error(f"Error processing {filename}: {str(e)}")
                    continue
        
        # Add documents to RAG system
        if processed_documents:
            success_count = rag_system.add_documents(processed_documents)
            logger.info(f"Added {success_count} documents to knowledge base")
        
        return jsonify({
            'message': f'Successfully processed {len(processed_documents)} documents',
            'uploaded_files': uploaded_files,
            'processed_count': len(processed_documents),
            'total_documents': rag_system.get_document_count()
        })
        
    except Exception as e:
        logger.error(f"Error in upload endpoint: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/query', methods=['POST'])
def query_documents():
    """Query the knowledge base and get synthesized answers."""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        # Optional parameters
        max_results = data.get('max_results', 5)
        include_sources = data.get('include_sources', True)
        
        logger.info(f"Processing query: {query}")
        
        # Get relevant documents
        relevant_docs = rag_system.retrieve_documents(query, max_results=max_results)
        
        if not relevant_docs:
            return jsonify({
                'query': query,
                'answer': 'No relevant documents found for your query.',
                'sources': [],
                'document_count': rag_system.get_document_count()
            })
        
        # Generate answer using LLM
        answer = rag_system.generate_answer(query, relevant_docs)
        
        response_data = {
            'query': query,
            'answer': answer,
            'document_count': rag_system.get_document_count()
        }
        
        if include_sources:
            response_data['sources'] = [
                {
                    'content': doc['content'][:200] + '...' if len(doc['content']) > 200 else doc['content'],
                    'filename': doc.get('filename', 'Unknown'),
                    'relevance_score': doc.get('score', 0)
                }
                for doc in relevant_docs
            ]
        
        logger.info(f"Generated answer for query: {query[:50]}...")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in query endpoint: {str(e)}")
        return jsonify({'error': f'Query failed: {str(e)}'}), 500

@app.route('/api/documents')
def list_documents():
    """List all documents in the knowledge base."""
    try:
        documents = rag_system.get_all_documents()
        return jsonify({
            'documents': documents,
            'total_count': len(documents)
        })
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        return jsonify({'error': f'Failed to list documents: {str(e)}'}), 500

@app.route('/api/clear', methods=['POST'])
def clear_documents():
    """Clear all documents from the knowledge base."""
    try:
        count = rag_system.clear_documents()
        return jsonify({
            'message': f'Cleared {count} documents from knowledge base',
            'cleared_count': count
        })
    except Exception as e:
        logger.error(f"Error clearing documents: {str(e)}")
        return jsonify({'error': f'Failed to clear documents: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500

def create_upload_folder():
    """Create upload folder if it doesn't exist."""
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
        logger.info(f"Created upload folder: {app.config['UPLOAD_FOLDER']}")

if __name__ == '__main__':
    # Create necessary directories
    create_upload_folder()
    
    # Initialize RAG system
    logger.info("Initializing RAG system...")
    
    # Check if model is available
    if not rag_system.check_model_status():
        logger.warning("Model is not available. Please run setup_model.py first.")
    
    logger.info("Starting Flask application...")
    app.run(debug=app.config['DEBUG'])
