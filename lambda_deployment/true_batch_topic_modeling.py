#!/usr/bin/env python3
"""
TRUE Batch Topic Modeling - Using Ollama for embeddings and BERTopic for clustering.

Uses all-minilm:22m model from Ollama for embeddings, then BERTopic for topic extraction.
"""

import asyncio
import json
import time
import logging
from typing import List, Dict, Any
import numpy as np
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import BERTopic components
try:
    from bertopic import BERTopic
    import umap
    import hdbscan
    from sklearn.feature_extraction.text import CountVectorizer
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    logger.error("BERTopic not available. Install with: pip install bertopic umap-learn hdbscan")


class OllamaEmbeddings:
    """Custom embedding class that uses Ollama API for embeddings."""
    
    def __init__(self, endpoint: str = "http://192.168.10.68:11434", model: str = "all-minilm:22m"):
        """
        Initialize Ollama embeddings.
        
        Args:
            endpoint: Ollama server endpoint
            model: Embedding model name (all-minilm:22m)
        """
        self.endpoint = endpoint
        self.model = model
        self.logger = logging.getLogger(f"{__name__}.OllamaEmbeddings")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents using Ollama.
        
        Args:
            texts: List of text documents to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        self.logger.info(f"🔢 Embedding {len(texts)} documents using {self.model}...")
        start_time = time.time()
        
        for i, text in enumerate(texts):
            if i % 10 == 0:
                self.logger.info(f"   Progress: {i}/{len(texts)} documents embedded")
            
            try:
                # Call Ollama embeddings API
                response = requests.post(
                    f"{self.endpoint}/api/embeddings",
                    json={
                        "model": self.model,
                        "prompt": text
                    },
                    timeout=30
                )
                response.raise_for_status()
                
                result = response.json()
                embedding = result.get("embedding", [])
                
                if not embedding:
                    self.logger.warning(f"Empty embedding for document {i}, using zeros")
                    # Use zero vector as fallback
                    embedding = [0.0] * 384  # all-minilm produces 384-dim embeddings
                
                embeddings.append(embedding)
                
            except Exception as e:
                self.logger.error(f"Failed to embed document {i}: {e}")
                # Use zero vector as fallback
                embeddings.append([0.0] * 384)
        
        elapsed = time.time() - start_time
        self.logger.info(f"✅ Embedded {len(texts)} documents in {elapsed:.1f}s ({elapsed/len(texts):.2f}s per doc)")
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        return self.embed_documents([text])[0]
    
    def encode(self, texts, show_progress_bar: bool = False, batch_size: int = 32):
        """
        Encode texts (compatibility method for BERTopic/SentenceTransformer interface).
        
        Args:
            texts: Text or list of texts to encode
            show_progress_bar: Whether to show progress (ignored)
            batch_size: Batch size (ignored, we process all at once)
            
        Returns:
            Numpy array of embeddings
        """
        import numpy as np
        
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        # Get embeddings
        embeddings = self.embed_documents(texts)
        
        # Convert to numpy array
        return np.array(embeddings)


async def true_batch_topic_modeling(
    texts: List[str],
    num_topics: int = 5,
    min_topic_size: int = 3,
    ollama_endpoint: str = None,
    embedding_model: str = None,
    agent_config = None
) -> Dict[str, Any]:
    """
    Extract topics from a collection of texts using TRUE batch processing.
    
    Supports both Ollama and Bedrock for embeddings.
    
    Args:
        texts: List of text documents
        num_topics: Target number of topics (approximate)
        min_topic_size: Minimum documents per topic
        ollama_endpoint: Ollama server endpoint (for Ollama provider)
        embedding_model: Embedding model name (for Ollama provider)
        agent_config: AgentConfig object (alternative to individual params)
        
    Returns:
        Dictionary with topics, keywords, and metadata
    """
    start_time = time.time()
    
    if not BERTOPIC_AVAILABLE:
        return {
            "error": "BERTopic not available",
            "topics": [],
            "status": "failed"
        }
    
    if not texts or len(texts) < min_topic_size:
        return {
            "error": f"Need at least {min_topic_size} documents for topic modeling",
            "topics": [],
            "status": "insufficient_data"
        }
    
    logger.info(f"🎯 TRUE BATCH TOPIC MODELING: {len(texts)} documents")
    
    try:
        # Step 1: Get embeddings (TRUE BATCH - all at once)
        embedding_start = time.time()
        
        if agent_config:
            from agents.common.agent_config import get_embedding_model
            embedder = get_embedding_model(agent_config)
            provider = agent_config.embedding_provider
            logger.info(f"   Using {provider.upper()} embeddings: {agent_config.embedding_model if provider == 'ollama' else agent_config.bedrock_embedding_model_id}")
        else:
            # Fallback to Ollama
            embedder = OllamaEmbeddings(endpoint=ollama_endpoint, model=embedding_model)
            logger.info(f"   Using Ollama embeddings: {embedding_model}")
        
        embeddings = embedder.embed_documents(texts)
        embeddings_array = np.array(embeddings)
        embedding_time = time.time() - embedding_start
        
        logger.info(f"✅ Embeddings complete: {embeddings_array.shape} in {embedding_time:.1f}s")
        
        # Step 2: Initialize BERTopic with pre-computed embeddings
        logger.info(f"🔍 Initializing BERTopic with min_topic_size={min_topic_size}...")
        
        # Configure UMAP for dimensionality reduction
        umap_model = umap.UMAP(
            n_neighbors=min(15, len(texts) - 1),
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )
        
        # Configure HDBSCAN for clustering
        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=min_topic_size,
            min_samples=1,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        # Configure CountVectorizer for keyword extraction
        vectorizer_model = CountVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            min_df=1
        )
        
        # Initialize BERTopic
        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            top_n_words=10,
            nr_topics=num_topics if num_topics > 0 else "auto",
            calculate_probabilities=False,  # Faster without probabilities
            verbose=False
        )
        
        # Step 3: Fit BERTopic on pre-computed embeddings
        logger.info(f"🔬 Fitting BERTopic model...")
        fit_start = time.time()
        
        topics, _ = topic_model.fit_transform(texts, embeddings_array)
        
        fit_time = time.time() - fit_start
        logger.info(f"✅ BERTopic fitted in {fit_time:.1f}s")
        
        # Step 4: Extract topic information
        logger.info(f"📊 Extracting topic information...")
        topic_info = topic_model.get_topic_info()
        
        # Remove outlier topic (-1)
        topic_info = topic_info[topic_info['Topic'] != -1]
        
        # Build topics list
        topics_list = []
        for _, row in topic_info.iterrows():
            topic_id = int(row['Topic'])
            topic_count = int(row['Count'])
            
            # Get topic keywords
            topic_words = topic_model.get_topic(topic_id)
            keywords = [word for word, score in topic_words[:10]] if topic_words else []
            
            # Get representative documents
            try:
                repr_docs = topic_model.get_representative_docs(topic_id)
                representative_docs = repr_docs[:3] if repr_docs else []
            except:
                representative_docs = []
            
            # Generate topic name from top keywords
            topic_name = "_".join(keywords[:3]) if keywords else f"topic_{topic_id}"
            
            topics_list.append({
                "topic_id": topic_id,
                "name": topic_name,
                "keywords": keywords,
                "count": topic_count,
                "representative_docs": representative_docs,
                "percentage": (topic_count / len(texts)) * 100
            })
        
        # Sort by count (most common first)
        topics_list.sort(key=lambda x: x['count'], reverse=True)
        
        total_time = time.time() - start_time
        
        result = {
            "status": "success",
            "topics": topics_list,
            "total_documents": len(texts),
            "topics_found": len(topics_list),
            "outliers": int((np.array(topics) == -1).sum()),
            "processing_time_ms": total_time * 1000,
            "embedding_time_ms": embedding_time * 1000,
            "clustering_time_ms": fit_time * 1000,
            "embedding_model": embedding_model,
            "embedding_dimensions": embeddings_array.shape[1]
        }
        
        logger.info(f"✅ Topic modeling complete:")
        logger.info(f"   Topics found: {len(topics_list)}")
        logger.info(f"   Total time: {total_time:.1f}s")
        logger.info(f"   - Embeddings: {embedding_time:.1f}s")
        logger.info(f"   - Clustering: {fit_time:.1f}s")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Topic modeling failed: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "error": str(e),
            "topics": [],
            "status": "failed",
            "processing_time_ms": (time.time() - start_time) * 1000
        }


async def test_true_batch_topic_modeling():
    """Test TRUE batch topic modeling."""
    
    # Test documents about different topics
    test_texts = [
        # Politics topic
        "Trump announces new policy on immigration",
        "Biden administration proposes healthcare reform",
        "Congress debates infrastructure bill",
        "Supreme Court ruling on voting rights",
        "Presidential election campaign heats up",
        
        # Technology topic
        "New AI model breaks performance records",
        "Tech company announces breakthrough in quantum computing",
        "Cybersecurity experts warn of new threats",
        "Social media platform faces privacy concerns",
        "Smartphone manufacturer releases latest device",
        
        # Health topic
        "New vaccine shows promising results in trials",
        "Study links diet to heart disease risk",
        "Mental health awareness campaign launches",
        "Hospital implements innovative treatment protocol",
        "Researchers discover potential cancer therapy",
        
        # Climate topic
        "Scientists report record temperatures this year",
        "Renewable energy adoption accelerates globally",
        "Climate summit reaches historic agreement",
        "Extreme weather events increase in frequency",
        "Carbon emissions reduction targets announced",
        
        # Economy topic
        "Stock market reaches new highs",
        "Federal Reserve adjusts interest rates",
        "Unemployment rate drops to lowest level",
        "Inflation concerns impact consumer spending",
        "Economic growth exceeds expectations"
    ]
    
    print("\n" + "="*80)
    print("Testing TRUE Batch Topic Modeling with Ollama Embeddings")
    print("="*80)
    
    result = await true_batch_topic_modeling(
        texts=test_texts,
        num_topics=5,
        min_topic_size=3,
        ollama_endpoint="http://192.168.10.68:11434",
        embedding_model="all-minilm:22m"
    )
    
    print("\n📊 Results:")
    print(f"Status: {result['status']}")
    print(f"Topics found: {result.get('topics_found', 0)}")
    print(f"Total documents: {result.get('total_documents', 0)}")
    print(f"Processing time: {result.get('processing_time_ms', 0):.0f}ms")
    
    if result.get('topics'):
        print("\n🎯 Topics:")
        for i, topic in enumerate(result['topics'], 1):
            print(f"\n{i}. {topic['name']}")
            print(f"   Documents: {topic['count']} ({topic['percentage']:.1f}%)")
            print(f"   Keywords: {', '.join(topic['keywords'][:5])}")
            if topic['representative_docs']:
                print(f"   Example: {topic['representative_docs'][0][:80]}...")
    
    # Save results
    with open('topic_modeling_test_results.json', 'w') as f:
        json.dump(result, f, indent=2)
    print("\n💾 Results saved to topic_modeling_test_results.json")


if __name__ == "__main__":
    asyncio.run(test_true_batch_topic_modeling())
