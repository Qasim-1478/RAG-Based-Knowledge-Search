"""
Document processing module for handling text and PDF files.
"""

import os
import logging
from typing import List, Dict, Optional
import PyPDF2
from config import Config

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles processing of various document formats."""
    
    def __init__(self):
        self.supported_extensions = Config.ALLOWED_EXTENSIONS
    
    def is_supported_file(self, filename: str) -> bool:
        """Check if file extension is supported."""
        return ('.' in filename and 
                filename.rsplit('.', 1)[1].lower() in self.supported_extensions)
    
    def extract_text_from_pdf(self, filepath: str) -> str:
        """Extract text content from PDF file."""
        try:
            text_content = []
            
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_content.append(page_text)
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                        continue
            
            return '\n'.join(text_content)
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF {filepath}: {e}")
            return ""
    
    def extract_text_from_txt(self, filepath: str) -> str:
        """Extract text content from text file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(filepath, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                logger.error(f"Error reading text file {filepath}: {e}")
                return ""
        except Exception as e:
            logger.error(f"Error reading text file {filepath}: {e}")
            return ""
    
    def extract_text_from_md(self, filepath: str) -> str:
        """Extract text content from markdown file."""
        return self.extract_text_from_txt(filepath)  # Same as text file
    
    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split text into overlapping chunks."""
        if chunk_size is None:
            chunk_size = Config.CHUNK_SIZE
        if overlap is None:
            overlap = Config.CHUNK_OVERLAP
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence or word boundary
            if end < len(text):
                # Look for sentence boundary first
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
                else:
                    # Look for word boundary
                    word_end = text.rfind(' ', start, end)
                    if word_end > start + chunk_size // 2:
                        end = word_end
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            
            # Prevent infinite loop
            if start >= len(text):
                break
        
        return chunks
    
    def process_document(self, filepath: str) -> Optional[Dict]:
        """Process a document and return structured data."""
        try:
            if not os.path.exists(filepath):
                logger.error(f"File not found: {filepath}")
                return None
            
            filename = os.path.basename(filepath)
            
            if not self.is_supported_file(filename):
                logger.warning(f"Unsupported file type: {filename}")
                return None
            
            # Extract text based on file type
            file_extension = filename.rsplit('.', 1)[1].lower()
            
            if file_extension == 'pdf':
                text_content = self.extract_text_from_pdf(filepath)
            elif file_extension in ['txt', 'md']:
                text_content = self.extract_text_from_txt(filepath)
            else:
                logger.warning(f"Unknown file type: {file_extension}")
                return None
            
            if not text_content.strip():
                logger.warning(f"No text content extracted from: {filename}")
                return None
            
            # Chunk the text
            chunks = self.chunk_text(text_content)
            
            # Create document metadata
            document_data = {
                'filename': filename,
                'filepath': filepath,
                'file_size': os.path.getsize(filepath),
                'file_type': file_extension,
                'full_text': text_content,
                'chunks': chunks,
                'chunk_count': len(chunks),
                'word_count': len(text_content.split())
            }
            
            logger.info(f"Successfully processed {filename}: {len(chunks)} chunks, {document_data['word_count']} words")
            return document_data
            
        except Exception as e:
            logger.error(f"Error processing document {filepath}: {e}")
            return None
    
    def get_file_info(self, filepath: str) -> Dict:
        """Get basic file information."""
        try:
            if not os.path.exists(filepath):
                return {'error': 'File not found'}
            
            stat = os.stat(filepath)
            return {
                'filename': os.path.basename(filepath),
                'filepath': filepath,
                'size': stat.st_size,
                'modified': stat.st_mtime,
                'is_supported': self.is_supported_file(os.path.basename(filepath))
            }
        except Exception as e:
            return {'error': str(e)}
