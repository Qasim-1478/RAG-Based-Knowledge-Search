"""
RAG (Retrieval-Augmented Generation) system implementation.
Handles document storage, retrieval, and answer generation using Ollama.
"""

import os
import json
import logging
import requests
from typing import List, Dict, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from datetime import datetime
from config import Config

logger = logging.getLogger(__name__)

class RAGSystem:
    """Retrieval-Augmented Generation system."""
    
    def __init__(self):
        self.embedding_model = None
        self.vector_index = None
        self.documents = []
        self.embeddings = None
        self.ollama_base_url = Config.OLLAMA_BASE_URL
        self.ollama_model = Config.OLLAMA_MODEL
        
        # Initialize embedding model
        self._initialize_embedding_model()
        
        # Load or create vector database
        self._initialize_vector_db()
    
    def _initialize_embedding_model(self):
        """Initialize the sentence transformer model for embeddings."""
        try:
            logger.info(f"Loading embedding model: {Config.EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _initialize_vector_db(self):
        """Initialize or load the vector database."""
        try:
            db_path = Config.VECTOR_DB_PATH
            
            # Create directory if it doesn't exist
            os.makedirs(db_path, exist_ok=True)
            
            # Load existing data if available
            index_path = os.path.join(db_path, 'vector_index.faiss')
            docs_path = os.path.join(db_path, 'documents.pkl')
            embeddings_path = os.path.join(db_path, 'embeddings.pkl')
            
            if (os.path.exists(index_path) and 
                os.path.exists(docs_path) and 
                os.path.exists(embeddings_path)):
                
                logger.info("Loading existing vector database...")
                self.vector_index = faiss.read_index(index_path)
                
                with open(docs_path, 'rb') as f:
                    self.documents = pickle.load(f)
                
                with open(embeddings_path, 'rb') as f:
                    self.embeddings = pickle.load(f)
                
                logger.info(f"Loaded {len(self.documents)} documents from vector database")
            else:
                # Create new vector index
                dimension = Config.EMBEDDING_DIMENSION
                self.vector_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
                self.documents = []
                self.embeddings = np.array([]).reshape(0, dimension)
                logger.info("Created new vector database")
                
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}")
            # Fallback to in-memory only
            dimension = Config.EMBEDDING_DIMENSION
            self.vector_index = faiss.IndexFlatIP(dimension)
            self.documents = []
            self.embeddings = np.array([]).reshape(0, dimension)
    
    def _save_vector_db(self):
        """Save the vector database to disk."""
        try:
            db_path = Config.VECTOR_DB_PATH
            os.makedirs(db_path, exist_ok=True)
            
            # Save vector index
            index_path = os.path.join(db_path, 'vector_index.faiss')
            faiss.write_index(self.vector_index, index_path)
            
            # Save documents
            docs_path = os.path.join(db_path, 'documents.pkl')
            with open(docs_path, 'wb') as f:
                pickle.dump(self.documents, f)
            
            # Save embeddings
            embeddings_path = os.path.join(db_path, 'embeddings.pkl')
            with open(embeddings_path, 'wb') as f:
                pickle.dump(self.embeddings, f)
            
            logger.info("Vector database saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save vector database: {e}")
    
    def add_documents(self, documents: List[Dict]) -> int:
        """Add documents to the knowledge base."""
        try:
            added_count = 0
            
            for doc in documents:
                if not doc or 'chunks' not in doc:
                    continue
                
                # Generate embeddings for each chunk
                chunk_embeddings = self.embedding_model.encode(doc['chunks'])
                
                # Normalize embeddings for cosine similarity
                chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
                
                # Add each chunk as a separate document
                for i, chunk in enumerate(doc['chunks']):
                    chunk_doc = {
                        'content': chunk,
                        'filename': doc['filename'],
                        'chunk_index': i,
                        'total_chunks': len(doc['chunks']),
                        'metadata': {
                            'file_type': doc.get('file_type', 'unknown'),
                            'file_size': doc.get('file_size', 0),
                            'word_count': doc.get('word_count', 0),
                            'added_at': datetime.now().isoformat()
                        }
                    }
                    
                    self.documents.append(chunk_doc)
                    
                    # Add embedding to vector index
                    if len(self.embeddings) == 0:
                        self.embeddings = chunk_embeddings[i:i+1]
                    else:
                        self.embeddings = np.vstack([self.embeddings, chunk_embeddings[i:i+1]])
                    
                    self.vector_index.add(chunk_embeddings[i:i+1])
                    added_count += 1
            
            # Save updated database
            self._save_vector_db()
            
            logger.info(f"Added {added_count} document chunks to knowledge base")
            return added_count
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return 0
    
    def retrieve_documents(self, query: str, max_results: int = None) -> List[Dict]:
        """Retrieve relevant documents for a query."""
        try:
            if max_results is None:
                max_results = Config.MAX_RETRIEVAL_RESULTS
            
            if len(self.documents) == 0:
                logger.warning("No documents in knowledge base")
                return []
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Search for similar documents
            scores, indices = self.vector_index.search(query_embedding, min(max_results, len(self.documents)))
            
            # Retrieve relevant documents
            relevant_docs = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx].copy()
                    doc['score'] = float(score)
                    relevant_docs.append(doc)
            
            logger.info(f"Retrieved {len(relevant_docs)} relevant documents for query")
            return relevant_docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def generate_answer(self, query: str, relevant_docs: List[Dict]) -> str:
        """Generate an answer using the LLM."""
        try:
            # Check if model is available
            if not self.check_model_status():
                return "Error: Language model is not available. Please ensure Ollama is running and the model is installed."
            
            # Prepare context from relevant documents
            context_parts = []
            for doc in relevant_docs:
                context_parts.append(f"Document: {doc.get('filename', 'Unknown')}\n{doc['content']}")
            
            #context = "\n\n".join(context_parts)
            context = "\n\n---\n\n".join([f"Filename: {d['filename']}\nContent: {d['content']}" for d in relevant_docs])

            
            # Create the prompt
            prompt = prompt = f"""Using these documents, answer the user's question succinctly and accurately. 
If the answer cannot be found in the provided documents, say so clearly. Do not write gibberish.

Instructions:
- Answer only based on the documents.
- If the answer is not in the documents, say "I could not find the answer in the provided documents."
- Do not make up information or write gibberish.
- Answer in clear English.

Documents:
{context}

User Question:
{query}

Answer:"""

            
            # Call Ollama API
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_tokens": 1000
                }
            }
            
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', 'No response generated').strip()
                
                # Clean up the answer
                if answer.startswith('Answer:'):
                    answer = answer[7:].strip()
                
                logger.info("Successfully generated answer using LLM")
                return answer
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return f"Error generating answer: API returned status {response.status_code}"
                
        except requests.exceptions.Timeout:
            logger.error("Timeout while calling Ollama API")
            return "Error: Request timed out while generating answer."
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    def check_model_status(self) -> bool:
        """Check if the Ollama model is available."""
        try:
            # Check if Ollama service is running
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False
            
            # Check if our model is available
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            return self.ollama_model in model_names
            
        except Exception as e:
            logger.error(f"Error checking model status: {e}")
            return False
    
    def get_document_count(self) -> int:
        """Get the total number of document chunks."""
        return len(self.documents)
    
    def get_all_documents(self) -> List[Dict]:
        """Get all documents with metadata."""
        return [
            {
                'filename': doc['filename'],
                'chunk_index': doc['chunk_index'],
                'total_chunks': doc['total_chunks'],
                'content_preview': doc['content'][:100] + '...' if len(doc['content']) > 100 else doc['content'],
                'metadata': doc['metadata']
            }
            for doc in self.documents
        ]
    
    def clear_documents(self) -> int:
        """Clear all documents from the knowledge base."""
        try:
            count = len(self.documents)
            
            # Reset everything
            dimension = Config.EMBEDDING_DIMENSION
            self.vector_index = faiss.IndexFlatIP(dimension)
            self.documents = []
            self.embeddings = np.array([]).reshape(0, dimension)
            
            # Save empty database
            self._save_vector_db()
            
            logger.info(f"Cleared {count} documents from knowledge base")
            return count
            
        except Exception as e:
            logger.error(f"Error clearing documents: {e}")
            return 0
