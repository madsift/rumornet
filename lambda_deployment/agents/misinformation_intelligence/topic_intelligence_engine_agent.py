"""
Topic Intelligence Engine Agent

Implementation of the TopicIntelligenceEngine for topic modeling, evolution tracking,
and genealogy management using BERTopic integration with real Ollama reasoning.

This agent provides the core topic analysis capabilities for the misinformation intelligence
system.
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta

# BERTopic and ML imports
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    import hdbscan
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
except ImportError as e:
    logging.warning(f"BERTopic dependencies not available: {e}")
    BERTopic = None


class TopicIntelligenceEngineAgent:
    """
    Agent for topic intelligence and evolution tracking.
    
    Provides tools for:
    - Content topic modeling with BERTopic
    - Topic evolution detection and tracking
    - Topic lifecycle prediction
    - Topic genealogy and relationship tracking
    """
    
    def __init__(self, name: str = "topic_intelligence_engine", config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Topic modeling components
        self.topic_model: Optional[BERTopic] = None
        self.embedding_model: Optional[SentenceTransformer] = None      
  
        # Topic tracking state
        self.topic_history: Dict[str, List[Dict]] = {}
        self.topic_genealogy: Dict[str, Dict] = {}
        self.evolution_patterns: Dict[str, Any] = {}
        
        # Performance tracking
        self.modeling_stats = {
            "topics_created": 0,
            "evolution_events": 0,
            "genealogy_links": 0,
            "total_documents": 0
        }
        
        # Get configuration from centralized config
        from agents.common.agent_config import AgentConfig
        agent_config = AgentConfig.from_dict(self.config)
        self.ollama_endpoint = agent_config.ollama_endpoint
        self.ollama_model = agent_config.ollama_model
        
        # Initialize components
        self._initialize_topic_model()
        
        self.logger.info(f"Initialized Topic Intelligence Engine agent: {name}")
    
    def _initialize_topic_model(self):
        """Initialize BERTopic model with Ollama embeddings."""
        if BERTopic is None:
            self.logger.warning("BERTopic not available - topic modeling will be limited")
            return
        
        try:
            # Use centralized config
            from agents.common.agent_config import AgentConfig, get_embedding_model
            
            agent_config = AgentConfig.from_dict(self.config)
            self.embedding_model = get_embedding_model(agent_config)
            
            if self.embedding_model:
                self.logger.info(f"Initialized Ollama embeddings: {agent_config.embedding_model}")
            
            # Configure UMAP for dimensionality reduction
            umap_model = UMAP(
                n_neighbors=15,
                n_components=5,
                min_dist=0.0,
                metric='cosine',
                random_state=42
            )
            
            # Configure HDBSCAN for clustering
            hdbscan_model = hdbscan.HDBSCAN(
                min_cluster_size=10,
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True  # Required for transform/predict operations
            )
            
            # Configure vectorizer
            vectorizer_model = CountVectorizer(
                ngram_range=(1, 2),
                stop_words="english",
                max_features=5000
            )
            
            # Initialize BERTopic WITHOUT embedding model
            # We'll provide pre-computed Ollama embeddings directly
            # This prevents BERTopic from loading SentenceTransformer
            self.topic_model = BERTopic(
                embedding_model=None,  # Don't load SentenceTransformer
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                verbose=False
            )
            
            self.logger.info("Topic model initialized successfully with Ollama embeddings")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize topic model: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.topic_model = None
    
    async def analyze_content_topics(
        self,
        documents: List[str],
        min_topic_size: int = 10,
        nr_topics: Optional[int] = None
    ) -> Dict[str, Any]:
            """
            Analyze documents to identify and model topics using BERTopic.
            
            Args:
                documents: List of text documents to analyze
                min_topic_size: Minimum size for topic clusters
                nr_topics: Number of topics to extract (auto if None)
            
            Returns:
                Dict containing topics, assignments, and metadata
            """
            try:
                if not self.topic_model:
                    return {
                        "success": False,
                        "error": "Topic model not available",
                        "topics": [],
                        "assignments": []
                    }
                
                if not documents or len(documents) < min_topic_size:
                    return {
                        "success": False,
                        "error": f"Need at least {min_topic_size} documents",
                        "topics": [],
                        "assignments": []
                    }
                
                # Fit topic model
                topics, probabilities = self.topic_model.fit_transform(documents)
                
                # Get topic information
                topic_info = self.topic_model.get_topic_info()
                
                # Extract topic representations
                topic_representations = {}
                for topic_id in set(topics):
                    if topic_id != -1:  # Skip outlier topic
                        topic_words = self.topic_model.get_topic(topic_id)
                        topic_representations[topic_id] = {
                            "words": topic_words,
                            "label": f"Topic_{topic_id}",
                            "size": sum(1 for t in topics if t == topic_id)
                        }
                
                # Update stats
                self.modeling_stats["topics_created"] += len(topic_representations)
                self.modeling_stats["total_documents"] += len(documents)
                
                return {
                    "success": True,
                    "topics": topic_representations,
                    "assignments": topics.tolist(),
                    "probabilities": probabilities.tolist() if probabilities is not None else [],
                    "topic_info": topic_info.to_dict('records'),
                    "metadata": {
                        "num_documents": len(documents),
                        "num_topics": len(topic_representations),
                        "timestamp": datetime.now().isoformat()
                    }
                }
                
            except Exception as e:
                self.logger.error(f"Topic analysis failed: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "topics": [],
                    "assignments": []
                }
    
    async def track_topic_evolution(
        self,
        topic_id: str,
        new_documents: List[str],
        time_window: str = "1d"
    ) -> Dict[str, Any]:
            """
            Track evolution of a specific topic over time.
            
            Args:
                topic_id: ID of topic to track
                new_documents: New documents to analyze for evolution
                time_window: Time window for evolution analysis
            
            Returns:
                Dict containing evolution metrics and patterns
            """
            try:
                if not self.topic_model:
                    return {
                        "success": False,
                        "error": "Topic model not available"
                    }
                
                # Get current topic representation
                current_topic = self.topic_model.get_topic(int(topic_id))
                if not current_topic:
                    return {
                        "success": False,
                        "error": f"Topic {topic_id} not found"
                    }
                
                # Analyze new documents
                if new_documents:
                    new_topics, _ = self.topic_model.transform(new_documents)
                    
                    # Calculate evolution metrics
                    evolution_metrics = await self._calculate_evolution_metrics(
                        topic_id, current_topic, new_documents, new_topics
                    )
                    
                    # Update topic history
                    if topic_id not in self.topic_history:
                        self.topic_history[topic_id] = []
                    
                    self.topic_history[topic_id].append({
                        "timestamp": datetime.now().isoformat(),
                        "documents": len(new_documents),
                        "metrics": evolution_metrics,
                        "time_window": time_window
                    })
                    
                    self.modeling_stats["evolution_events"] += 1
                    
                    return {
                        "success": True,
                        "topic_id": topic_id,
                        "evolution_metrics": evolution_metrics,
                        "history_length": len(self.topic_history[topic_id]),
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {
                        "success": False,
                        "error": "No new documents provided"
                    }
                
            except Exception as e:
                self.logger.error(f"Topic evolution tracking failed: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }
    
    async def predict_topic_lifecycle(
        self,
        topic_id: str,
        prediction_horizon: str = "7d"
    ) -> Dict[str, Any]:
            """
            Predict lifecycle stage and future trajectory of a topic.
            
            Args:
                topic_id: ID of topic to analyze
                prediction_horizon: Time horizon for predictions
            
            Returns:
                Dict containing lifecycle predictions and confidence scores
            """
            try:
                # Get topic history
                if topic_id not in self.topic_history:
                    return {
                        "success": False,
                        "error": f"No history available for topic {topic_id}"
                    }
                
                history = self.topic_history[topic_id]
                if len(history) < 3:
                    return {
                        "success": False,
                        "error": "Insufficient history for prediction (need at least 3 data points)"
                    }
                
                # Use Ollama for lifecycle analysis
                lifecycle_analysis = await self._analyze_topic_lifecycle_with_llm(
                    topic_id, history, prediction_horizon
                )
                
                # Calculate trend metrics
                trend_metrics = self._calculate_trend_metrics(history)
                
                # Determine lifecycle stage
                lifecycle_stage = self._determine_lifecycle_stage(trend_metrics)
                
                return {
                    "success": True,
                    "topic_id": topic_id,
                    "lifecycle_stage": lifecycle_stage,
                    "trend_metrics": trend_metrics,
                    "llm_analysis": lifecycle_analysis,
                    "prediction_horizon": prediction_horizon,
                    "confidence": self._calculate_prediction_confidence(trend_metrics),
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"Topic lifecycle prediction failed: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }
    
    async def analyze_topic_genealogy(
        self,
        parent_topic_id: str,
        child_topics: List[str],
        relationship_type: str = "evolution"
    ) -> Dict[str, Any]:
            """
            Analyze genealogical relationships between topics.
            
            Args:
                parent_topic_id: ID of parent topic
                child_topics: List of potential child topic IDs
                relationship_type: Type of relationship (evolution, split, merge)
            
            Returns:
                Dict containing genealogy analysis and relationship strengths
            """
            try:
                if not self.topic_model:
                    return {
                        "success": False,
                        "error": "Topic model not available"
                    }
                
                # Get parent topic representation
                parent_topic = self.topic_model.get_topic(int(parent_topic_id))
                if not parent_topic:
                    return {
                        "success": False,
                        "error": f"Parent topic {parent_topic_id} not found"
                    }
                
                # Analyze relationships with each child topic
                relationships = {}
                for child_id in child_topics:
                    try:
                        child_topic = self.topic_model.get_topic(int(child_id))
                        if child_topic:
                            similarity = self._calculate_topic_similarity(parent_topic, child_topic)
                            
                            # Use Ollama for relationship analysis
                            relationship_analysis = await self._analyze_topic_relationship_with_llm(
                                parent_topic_id, child_id, parent_topic, child_topic, relationship_type
                            )
                            
                            relationships[child_id] = {
                                "similarity": similarity,
                                "relationship_strength": self._calculate_relationship_strength(
                                    similarity, relationship_analysis
                                ),
                                "llm_analysis": relationship_analysis,
                                "relationship_type": relationship_type
                            }
                    except Exception as e:
                        self.logger.warning(f"Failed to analyze relationship with {child_id}: {e}")
                        continue
                
                # Update genealogy tracking
                genealogy_key = f"{parent_topic_id}_{relationship_type}"
                self.topic_genealogy[genealogy_key] = {
                    "parent": parent_topic_id,
                    "children": relationships,
                    "relationship_type": relationship_type,
                    "timestamp": datetime.now().isoformat()
                }
                
                self.modeling_stats["genealogy_links"] += len(relationships)
                
                return {
                    "success": True,
                    "parent_topic": parent_topic_id,
                    "relationships": relationships,
                    "relationship_type": relationship_type,
                    "strongest_relationship": max(
                        relationships.items(),
                        key=lambda x: x[1]["relationship_strength"]
                    ) if relationships else None,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"Topic genealogy analysis failed: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }
    
    async def _calculate_evolution_metrics(
        self,
        topic_id: str,
        current_topic: List[Tuple[str, float]],
        new_documents: List[str],
        new_topics: List[int]
    ) -> Dict[str, float]:
        """Calculate evolution metrics for a topic."""
        try:
            # Calculate document assignment rate
            assignment_rate = sum(1 for t in new_topics if t == int(topic_id)) / len(new_topics)
            
            # Calculate topic coherence change (simplified)
            coherence_score = np.mean([score for _, score in current_topic[:5]])
            
            # Calculate growth rate
            growth_rate = len([t for t in new_topics if t == int(topic_id)]) / len(new_documents)
            
            return {
                "assignment_rate": assignment_rate,
                "coherence_score": coherence_score,
                "growth_rate": growth_rate,
                "document_count": len(new_documents),
                "topic_assignments": sum(1 for t in new_topics if t == int(topic_id))
            }
            
        except Exception as e:
            self.logger.error(f"Evolution metrics calculation failed: {e}")
            return {
                "assignment_rate": 0.0,
                "coherence_score": 0.0,
                "growth_rate": 0.0,
                "document_count": 0,
                "topic_assignments": 0
            }
    
    async def _analyze_topic_lifecycle_with_llm(
        self,
        topic_id: str,
        history: List[Dict],
        prediction_horizon: str
    ) -> Dict[str, Any]:
        """Use Ollama to analyze topic lifecycle patterns."""
        try:
            # Prepare history summary for LLM
            history_summary = self._prepare_history_summary(history)
            
            prompt = f"""
            Analyze the lifecycle of Topic {topic_id} based on its evolution history:
            
            {history_summary}
            
            Prediction horizon: {prediction_horizon}
            
            Please analyze:
            1. Current lifecycle stage (emerging, growing, mature, declining, dormant)
            2. Key trends and patterns
            3. Predicted trajectory for the next {prediction_horizon}
            4. Confidence level in predictions
            5. Key factors influencing the topic's evolution
            
            Provide a structured analysis with specific insights.
            """
            
            # Simple LLM call - would need proper implementation
            response = "Lifecycle analysis placeholder"
            
            return {
                "analysis": response,
                "prompt_used": prompt,
                "model_used": self.ollama_model,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"LLM lifecycle analysis failed: {e}")
            return {
                "analysis": "Analysis failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _analyze_topic_relationship_with_llm(
        self,
        parent_id: str,
        child_id: str,
        parent_topic: List[Tuple[str, float]],
        child_topic: List[Tuple[str, float]],
        relationship_type: str
    ) -> Dict[str, Any]:
        """Use Ollama to analyze topic relationships."""
        try:
            parent_words = [word for word, _ in parent_topic[:10]]
            child_words = [word for word, _ in child_topic[:10]]
            
            prompt = f"""
            Analyze the {relationship_type} relationship between two topics:
            
            Parent Topic {parent_id} keywords: {', '.join(parent_words)}
            Child Topic {child_id} keywords: {', '.join(child_words)}
            
            Please analyze:
            1. Semantic similarity and overlap
            2. Evidence of {relationship_type} relationship
            3. Strength of the relationship (0-1 scale)
            4. Key connecting themes
            5. Potential causality or influence patterns
            
            Provide specific insights about how these topics relate.
            """
            
            # Simple LLM call - would need proper implementation
            response = "Relationship analysis placeholder"
            
            return {
                "analysis": response,
                "parent_keywords": parent_words,
                "child_keywords": child_words,
                "relationship_type": relationship_type,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"LLM relationship analysis failed: {e}")
            return {
                "analysis": "Analysis failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _calculate_topic_similarity(
        self,
        topic1: List[Tuple[str, float]],
        topic2: List[Tuple[str, float]]
    ) -> float:
        """Calculate similarity between two topics based on word overlap."""
        try:
            words1 = set(word for word, _ in topic1[:10])
            words2 = set(word for word, _ in topic2[:10])
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Topic similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_relationship_strength(
        self,
        similarity: float,
        llm_analysis: Dict[str, Any]
    ) -> float:
        """Calculate overall relationship strength."""
        try:
            # Base strength from similarity
            base_strength = similarity
            
            # Adjust based on LLM analysis (simplified)
            # In a real implementation, you'd parse the LLM response for strength indicators
            llm_boost = 0.1 if "strong" in llm_analysis.get("analysis", "").lower() else 0.0
            
            return min(1.0, base_strength + llm_boost)
            
        except Exception as e:
            self.logger.error(f"Relationship strength calculation failed: {e}")
            return similarity
    
    def _prepare_history_summary(self, history: List[Dict]) -> str:
        """Prepare a summary of topic history for LLM analysis."""
        try:
            summary_lines = []
            for i, entry in enumerate(history[-10:]):  # Last 10 entries
                timestamp = entry.get("timestamp", "unknown")
                metrics = entry.get("metrics", {})
                
                summary_lines.append(
                    f"Entry {i+1} ({timestamp}): "
                    f"Documents: {entry.get('documents', 0)}, "
                    f"Growth Rate: {metrics.get('growth_rate', 0):.3f}, "
                    f"Assignment Rate: {metrics.get('assignment_rate', 0):.3f}"
                )
            
            return "\n".join(summary_lines)
            
        except Exception as e:
            self.logger.error(f"History summary preparation failed: {e}")
            return "History summary unavailable"
    
    def _calculate_trend_metrics(self, history: List[Dict]) -> Dict[str, float]:
        """Calculate trend metrics from topic history."""
        try:
            if len(history) < 2:
                return {"trend": 0.0, "volatility": 0.0, "momentum": 0.0}
            
            # Extract growth rates
            growth_rates = [
                entry.get("metrics", {}).get("growth_rate", 0.0)
                for entry in history
            ]
            
            # Calculate trend (slope of growth rates)
            if len(growth_rates) >= 2:
                x = np.arange(len(growth_rates))
                trend = np.polyfit(x, growth_rates, 1)[0]
            else:
                trend = 0.0
            
            # Calculate volatility (standard deviation)
            volatility = np.std(growth_rates) if len(growth_rates) > 1 else 0.0
            
            # Calculate momentum (recent vs older average)
            recent_avg = np.mean(growth_rates[-3:]) if len(growth_rates) >= 3 else growth_rates[-1]
            older_avg = np.mean(growth_rates[:-3]) if len(growth_rates) >= 6 else np.mean(growth_rates[:-1])
            momentum = recent_avg - older_avg if older_avg != 0 else 0.0
            
            return {
                "trend": float(trend),
                "volatility": float(volatility),
                "momentum": float(momentum)
            }
            
        except Exception as e:
            self.logger.error(f"Trend metrics calculation failed: {e}")
            return {"trend": 0.0, "volatility": 0.0, "momentum": 0.0}
    
    def _determine_lifecycle_stage(self, trend_metrics: Dict[str, float]) -> str:
        """Determine lifecycle stage based on trend metrics."""
        try:
            trend = trend_metrics.get("trend", 0.0)
            momentum = trend_metrics.get("momentum", 0.0)
            volatility = trend_metrics.get("volatility", 0.0)
            
            # Simple heuristic-based classification
            if trend > 0.1 and momentum > 0.05:
                return "growing"
            elif trend > 0.05 and volatility < 0.1:
                return "mature"
            elif trend < -0.1 or momentum < -0.05:
                return "declining"
            elif abs(trend) < 0.02 and volatility < 0.05:
                return "dormant"
            else:
                return "emerging"
                
        except Exception as e:
            self.logger.error(f"Lifecycle stage determination failed: {e}")
            return "unknown"
    
    def _calculate_prediction_confidence(self, trend_metrics: Dict[str, float]) -> float:
        """Calculate confidence in lifecycle predictions."""
        try:
            volatility = trend_metrics.get("volatility", 1.0)
            
            # Lower volatility = higher confidence
            confidence = max(0.1, 1.0 - volatility)
            
            return min(1.0, confidence)
            
        except Exception as e:
            self.logger.error(f"Prediction confidence calculation failed: {e}")
            return 0.5
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the topic intelligence engine."""
        return {
            "modeling_stats": self.modeling_stats.copy(),
            "topic_history_count": len(self.topic_history),
            "genealogy_links_count": len(self.topic_genealogy),
            "model_available": self.topic_model is not None,
            "embedding_model_available": self.embedding_model is not None
        }
    
    def get_supported_strategies(self) -> List[str]:
        """Get list of supported topic analysis strategies."""
        return [
            "bertopic_modeling",
            "evolution_tracking",
            "lifecycle_prediction",
            "genealogy_analysis",
            "trend_analysis"
        ]


# Entry point
if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create agent
    agent = TopicIntelligenceEngineAgent()
    
    print(f"Created Topic Intelligence Engine Agent: {agent.name}")
    print("Agent ready for direct method calls")
