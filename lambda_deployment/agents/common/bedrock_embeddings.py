"""
Bedrock Embeddings Wrapper

Provides a compatible interface for Bedrock embeddings that matches OllamaEmbeddings.
Uses efficient batching with ThreadPoolExecutor for parallel processing.
"""

import boto3
import json
import logging
import time
import math
import multiprocessing
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class BedrockEmbeddings:
    """Wrapper for AWS Bedrock embeddings with efficient batching."""
    
    def __init__(self, model_id: str = "cohere.embed-v4:0", region: str = "us-east-1", embedding_dim: int = 256, batch_size: int = 96):
        """
        Initialize Bedrock embeddings client.
        
        Args:
            model_id: Bedrock embedding model ID
            region: AWS region
            embedding_dim: Embedding dimension
            batch_size: Number of texts to embed per API call
        """
        self.model_id = model_id
        self.region = region
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.client = boto3.client('bedrock-runtime', region_name=region)
        
        # Calculate optimal thread count
        cpu_count = multiprocessing.cpu_count()
        self.max_workers = max(1, min(10, math.ceil(cpu_count / 2)))
        
        logger.info(f"Initialized Bedrock embeddings: {model_id} in {region} (dim={embedding_dim}, batch={batch_size}, workers={self.max_workers})")
    
    def _embed_batch(self, start_idx: int, batch: List[str]) -> tuple:
        """
        Embed a single batch of texts using Bedrock API.
        
        Args:
            start_idx: Starting index in the original list
            batch: List of texts to embed
            
        Returns:
            Tuple of (start_idx, embeddings)
        """
        payload = {
            "input_type": "search_document",
            "texts": batch,
            "embedding_types": ["int8"],
            "output_dimension": self.embedding_dim,
            "truncate": "RIGHT"
        }
        
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                body=json.dumps(payload)
            )
            
            body_bytes = response['body'].read()
            result = json.loads(body_bytes)
            
            # Extract embeddings
            embeddings = result.get("embeddings")
            if embeddings is None:
                logger.warning(f"Batch {start_idx}: no embeddings field in response")
                batch_embeddings = [[0.0] * self.embedding_dim for _ in batch]
            else:
                # Handle both dict and list formats
                if isinstance(embeddings, dict):
                    batch_embeddings = embeddings.get("int8", embeddings.get("float", []))
                else:
                    batch_embeddings = embeddings
                
                # Validate shape
                if not batch_embeddings or len(batch_embeddings) != len(batch):
                    logger.warning(f"Batch {start_idx}: unexpected embedding shape")
                    batch_embeddings = [[0.0] * self.embedding_dim for _ in batch]
            
            return start_idx, batch_embeddings
            
        except Exception as e:
            logger.error(f"Batch {start_idx} failed: {e}")
            batch_embeddings = [[0.0] * self.embedding_dim for _ in batch]
            return start_idx, batch_embeddings
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents using efficient batching and parallel processing.
        
        Args:
            texts: List of text documents to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        start_time = time.time()
        logger.info(f"Embedding {len(texts)} documents with Bedrock (batch_size={self.batch_size}, workers={self.max_workers})")
        
        # Initialize results array
        all_embeddings = [None] * len(texts)
        
        # Create batches
        batches = [(i, texts[i:i + self.batch_size]) for i in range(0, len(texts), self.batch_size)]
        total_batches = len(batches)
        completed_batches = 0
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._embed_batch, start, batch) for start, batch in batches]
            
            for future in as_completed(futures):
                start_idx, batch_embeddings = future.result()
                all_embeddings[start_idx:start_idx + len(batch_embeddings)] = batch_embeddings
                completed_batches += 1
                
                if completed_batches % 5 == 0 or completed_batches == total_batches:
                    processed = min(completed_batches * self.batch_size, len(texts))
                    logger.info(f"Progress: {completed_batches}/{total_batches} batches ({processed}/{len(texts)} texts)")
        
        elapsed = time.time() - start_time
        logger.info(f"Embedded {len(all_embeddings)} documents in {elapsed:.2f}s using {self.max_workers} workers")
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        payload = {
            "input_type": "search_query",
            "texts": [text],
            "embedding_types": ["int8"],
            "output_dimension": self.embedding_dim,
            "truncate": "RIGHT"
        }
        
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                body=json.dumps(payload)
            )
            
            body_bytes = response['body'].read()
            result = json.loads(body_bytes)
            
            embeddings = result.get("embeddings")
            if embeddings is None:
                logger.warning(f"No embedding returned for query")
                return [0.0] * self.embedding_dim
            
            # Handle both dict and list formats
            if isinstance(embeddings, dict):
                embedding = embeddings.get("int8", embeddings.get("float", []))[0]
            else:
                embedding = embeddings[0]
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            return [0.0] * self.embedding_dim
