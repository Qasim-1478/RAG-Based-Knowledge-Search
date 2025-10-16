# RAG Knowledge Base Search Engine

A Flask-based Retrieval-Augmented Generation (RAG) system that enables intelligent search across documents using the gemma3:1b
 model from Ollama. This system can work completely offline and provides a modern web interface for document upload, search, and answer synthesis.

## 🚀 Features

- **Document Processing**: Supports PDF, TXT, and Markdown files
- **Intelligent Retrieval**: Uses sentence transformers for semantic document retrieval
- **LLM Integration**: Powered by gemma3:1b model via Ollama (offline capable)
- **Modern UI**: Bootstrap-based responsive web interface
- **Vector Database**: Efficient document storage using FAISS
- **Real-time Status**: Model availability and health monitoring
- **Query History**: Local storage of recent queries and answers

## 📋 Prerequisites

Before setting up the system, ensure you have:

1. **Python 3.8+** installed
2. **Ollama** installed and running
3. **Internet connection** for initial model download (subsequent use is offline)

### Installing Ollama

Visit [https://ollama.ai/download](https://ollama.ai/download) and follow the installation instructions for your operating system.

## 🛠️ Installation & Setup

### 1. Clone and Navigate to Project

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate 
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the LLM Model

Run the model setup script to download and configure the gemma3:1b model:

```bash
python setup_model.py
```

This script will:
- Check if Ollama is installed and running
- Download the gemma3:1b model
- Verify the model is working correctly
- Display progress and status information

### 5. Start Ollama Service

Make sure Ollama is running:

```bash
ollama serve
```

Keep this running in a separate terminal.

### 6. Launch the Application

```bash
python app.py
```

The application will be available at localhost.



## 🔧 Configuration

The system can be configured via environment variables or by modifying `config.py`:

### Key Settings

- `OLLAMA_BASE_URL`: Ollama API endpoint (default: http://localhost:11434)
- `OLLAMA_MODEL`: Model name (default: gemma3:1b)
- `CHUNK_SIZE`: Document chunk size (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `MAX_RETRIEVAL_RESULTS`: Max documents to retrieve (default: 5)

### Environment Variables

Create a `.env` file to override defaults:

```env
FLASK_DEBUG=True
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:1b
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```





## 🧠 How It Works

### 1. Document Processing
- Documents are uploaded and processed by `DocumentProcessor`
- Text is extracted from PDFs, TXT, and MD files
- Content is chunked with configurable overlap

### 2. Embedding Generation
- Each document chunk is converted to embeddings using sentence-transformers
- Embeddings are stored in a FAISS vector index for efficient similarity search

### 3. Query Processing
- User queries are converted to embeddings
- Similar document chunks are retrieved using cosine similarity
- Retrieved chunks are ranked by relevance score

### 4. Answer Generation
- Relevant document chunks are passed as context to the LLM
- The gemma3:1b model generates answers using the prompt template
- Sources and relevance scores are returned to the user

## 🔧 Troubleshooting

### Model Not Available
```bash
# Check if Ollama is running
ollama list

# Restart Ollama service
ollama serve

# Re-run model setup
python setup_model.py
```

### Upload Issues
- Check file size (max 16MB)
- Ensure file format is supported (PDF, TXT, MD)
- Verify uploads directory permissions

### Performance Issues
- Reduce `CHUNK_SIZE` for smaller chunks
- Adjust `MAX_RETRIEVAL_RESULTS` for fewer retrieved documents
- Monitor system resources during processing

## 🚀 Advanced Usage

### Custom Models
To use a different Ollama model, update the configuration:

```python
# In config.py or .env
OLLAMA_MODEL=your-preferred-model:tag
```

### Batch Processing
For processing multiple documents programmatically:

```python
from document_processor import DocumentProcessor
from rag_system import RAGSystem

processor = DocumentProcessor()
rag = RAGSystem()

# Process multiple files
for file_path in file_paths:
    doc = processor.process_document(file_path)
    if doc:
        rag.add_documents([doc])
```

### Custom Prompts
Modify the prompt template in `rag_system.py` for different answer styles:

```python
prompt = f"""Your custom prompt template here.
Documents: {context}
Question: {query}
Answer:"""
```

**Note**: This is designed to work offline after initial setup. The gemma3:1b  model will be downloaded once and cached locally for subsequent use.
