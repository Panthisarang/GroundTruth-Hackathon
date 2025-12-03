import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, 
    TOP_K_RESULTS, POLICIES_DIR
)
from db import db
from llm_client import llm_client
from pii_masker import mask_pii

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Document:
    """A document that can be split into chunks and embedded."""
    
    def __init__(self, content: str, metadata: Optional[Dict] = None):
        self.content = content
        self.metadata = metadata or {}
        self.chunks = []
    
    def split_into_chunks(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List['DocumentChunk']:
        """Split the document into overlapping chunks."""
        if not self.content:
            return []
        
        # Simple sentence splitting (can be enhanced with better NLP)
        sentences = re.split(r'(?<=[.!?])\s+', self.content)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_length = len(sentence.split())
            
            # If adding this sentence would exceed the chunk size, finalize the current chunk
            if current_chunk and current_length + sentence_length > chunk_size:
                chunk_text = ' '.join(current_chunk)
                chunks.append(DocumentChunk(chunk_text, self.metadata.copy()))
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - overlap)
                current_chunk = current_chunk[overlap_start:]
                current_length = sum(len(s.split()) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add the last chunk if not empty
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(DocumentChunk(chunk_text, self.metadata.copy()))
        
        self.chunks = chunks
        return chunks

class DocumentChunk:
    """A chunk of a document with its embedding."""
    
    def __init__(self, content: str, metadata: Dict):
        self.content = content
        self.metadata = metadata
        self.embedding = None
    
    def to_dict(self) -> Dict:
        """Convert to a dictionary for storage."""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "embedding": self.embedding,
            "created_at": datetime.utcnow()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DocumentChunk':
        """Create from a dictionary."""
        chunk = cls(data["content"], data.get("metadata", {}))
        chunk.embedding = data.get("embedding")
        return chunk

class RAGPipeline:
    """Retrieval Augmented Generation pipeline."""
    
    def __init__(self):
        self.embedding_model = None
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
    
    def _get_embedding_model(self):
        """Lazy load the embedding model."""
        if self.embedding_model is None:
            logger.info("Loading embedding model...")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            # Update embedding dim based on actual model
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"Loaded embedding model: {EMBEDDING_MODEL} (dim={self.embedding_dim})")
        return self.embedding_model
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for the given text."""
        if not text:
            return [0.0] * self.embedding_dim
            
        # First try to get from cache
        cache_key = f"embedding:{hashlib.md5(text.encode()).hexdigest()}"
        cached = db.embeddings.find_one({"cache_key": cache_key})
        
        if cached and "embedding" in cached:
            return cached["embedding"]
        
        # If not in cache, generate embedding
        model = self._get_embedding_model()
        embedding = model.encode(text, convert_to_numpy=True).tolist()
        
        # Cache the embedding
        db.embeddings.update_one(
            {"cache_key": cache_key},
            {
                "$set": {
                    "text": text,
                    "embedding": embedding,
                    "created_at": datetime.utcnow()
                }
            },
            upsert=True
        )
        
        return embedding
    
    def process_document(self, file_path: str, metadata: Optional[Dict] = None) -> List[DocumentChunk]:
        """Process a document file and return its chunks with embeddings."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        # Extract text based on file type
        if file_path.lower().endswith('.pdf'):
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            text = "\n".join(page.extract_text() for page in reader.pages)
        elif file_path.lower().endswith(('.txt', '.md')):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        
        # Create document and split into chunks
        doc = Document(text, metadata or {})
        chunks = doc.split_into_chunks()
        
        # Generate embeddings for each chunk
        for chunk in chunks:
            chunk.embedding = self.get_embedding(chunk.content)
        
        return chunks
    
    def index_document(self, file_path: str, metadata: Optional[Dict] = None) -> List[str]:
        """Index a document and store its chunks in the database."""
        chunks = self.process_document(file_path, metadata)
        doc_ids = []
        
        for chunk in chunks:
            doc_id = str(db.embeddings.insert_one({
                "content": chunk.content,
                "embedding": chunk.embedding,
                "metadata": {
                    "source": file_path,
                    "chunk_index": len(doc_ids),
                    **chunk.metadata
                },
                "created_at": datetime.utcnow()
            }).inserted_id)
            
            doc_ids.append(doc_id)
        
        logger.info(f"Indexed {len(doc_ids)} chunks from {file_path}")
        return doc_ids
    
    def search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Dict]:
        """Search for relevant document chunks based on the query."""
        # Get query embedding
        query_embedding = self.get_embedding(query)
        
        # Search for similar chunks
        similar_chunks = db.search_similar_embeddings(query_embedding, top_k=top_k)
        
        # Format results
        results = []
        for chunk in similar_chunks:
            results.append({
                "content": chunk["content"],
                "metadata": chunk.get("metadata", {}),
                "score": chunk.get("score", 0.0)
            })
        
        return results
    
    def generate_response(self, query: str, context: Optional[str] = None, **kwargs) -> str:
        """Generate a response using the LLM with RAG."""
        # Get relevant context using RAG
        relevant_chunks = self.search(query)
        
        # Format context from relevant chunks
        context_text = "\n\n".join(
            f"Source: {chunk.get('metadata', {}).get('source', 'Unknown')}\n"
            f"Content: {chunk['content']}"
            for chunk in relevant_chunks
        )
        
        # Prepare messages for the LLM
        messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant that provides accurate information based on the provided context. 
                If you don't know the answer, say so. Don't make up information."""
            },
            {
                "role": "user",
                "content": f"""Context information is below.
                ---------------------
                {context_text}
                ---------------------
                Given the context information and not prior knowledge, answer the query.
                Query: {query}"""
            }
        ]
        
        # Call the LLM
        try:
            response = llm_client.call_llm(messages, **kwargs)
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error while processing your request."

# Create a singleton instance
rag_pipeline = RAGPipeline()

def process_and_index_documents(directory: str) -> List[str]:
    """Process and index all documents in the given directory."""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logger.warning(f"Created directory: {directory}")
        return []
    
    doc_ids = []
    supported_extensions = ('.pdf', '.txt', '.md')
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(supported_extensions):
                file_path = os.path.join(root, file)
                try:
                    ids = rag_pipeline.index_document(
                        file_path,
                        metadata={"source_type": "document", "filename": file}
                    )
                    doc_ids.extend(ids)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
    
    logger.info(f"Indexed {len(doc_ids)} chunks from {directory}")
    return doc_ids
