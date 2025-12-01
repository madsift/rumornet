"""
CoordinationDetectorMCPAgent - Agent for detecting coordinated inauthentic behavior.

Detects coordinated behavior through temporal synchronization, content coordination,
network coordination, and behavioral pattern analysis using real Ollama integration.
"""

import asyncio
import logging
import time
import hashlib
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from statistics import mean, stdev

try:
    from fastmcp import FastMCP
    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False

from .core.data_models import (
    CoordinationIndicators, AlertRiskLevel, AlertType,
    TopicIntelligence, NetworkNode, NetworkEdge
)


@dataclass
class PostingEvent:
    """Posting event for coordination analysis."""
    user_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    content_hash: str = ""
    content_text: str = ""
    platform: str = ""
    topic_id: str = ""
    engagement_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "content_hash": self.content_hash,
            "content_text": self.content_text,
            "platform": self.platform,
            "topic_id": self.topic_id,
            "engagement_score": self.engagement_score,
            "metadata": self.metadata
        }


@dataclass
class SynchronizationScore:
    """Temporal synchronization analysis result."""
    user_pair: Tuple[str, str]
    synchronization_score: float = 0.0
    temporal_correlation: float = 0.0
    posting_interval_similarity: float = 0.0
    burst_synchronization: float = 0.0
    confidence: float = 0.0    

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "user_pair": list(self.user_pair),
            "synchronization_score": self.synchronization_score,
            "temporal_correlation": self.temporal_correlation,
            "posting_interval_similarity": self.posting_interval_similarity,
            "burst_synchronization": self.burst_synchronization,
            "confidence": self.confidence
        }


@dataclass
class BotNetworkAnalysis:
    """Bot network detection analysis result."""
    network_id: str = ""
    suspected_bots: List[str] = field(default_factory=list)
    coordination_score: float = 0.0
    bot_likelihood_scores: Dict[str, float] = field(default_factory=dict)
    network_centrality: float = 0.0
    behavioral_similarity: float = 0.0
    temporal_patterns: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "network_id": self.network_id,
            "suspected_bots": self.suspected_bots,
            "coordination_score": self.coordination_score,
            "bot_likelihood_scores": self.bot_likelihood_scores,
            "network_centrality": self.network_centrality,
            "behavioral_similarity": self.behavioral_similarity,
            "temporal_patterns": self.temporal_patterns,
            "confidence": self.confidence
        }


@dataclass
class ContentCoordinationScore:
    """Content coordination analysis result."""
    content_cluster_id: str = ""
    coordinated_users: List[str] = field(default_factory=list)
    content_similarity_score: float = 0.0
    semantic_similarity: float = 0.0
    structural_similarity: float = 0.0
    timing_coordination: float = 0.0
    amplification_pattern: float = 0.0
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content_cluster_id": self.content_cluster_id,
            "coordinated_users": self.coordinated_users,
            "content_similarity_score": self.content_similarity_score,
            "semantic_similarity": self.semantic_similarity,
            "structural_similarity": self.structural_similarity,
            "timing_coordination": self.timing_coordination,
            "amplification_pattern": self.amplification_pattern,
            "confidence": self.confidence
        }


@dataclass
class CoordinationDetectionResult:
    """Comprehensive coordination detection result."""
    topic_id: str = ""
    coordination_type: str = "unknown"  # temporal, content, network, behavioral
    coordination_indicators: Optional[CoordinationIndicators] = None
    synchronization_scores: List[SynchronizationScore] = field(default_factory=list)
    bot_networks: List[BotNetworkAnalysis] = field(default_factory=list)
    content_coordination: List[ContentCoordinationScore] = field(default_factory=list)
    overall_confidence: float = 0.0
    risk_level: AlertRiskLevel = AlertRiskLevel.LOW
    evidence: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "topic_id": self.topic_id,
            "coordination_type": self.coordination_type,
            "coordination_indicators": self.coordination_indicators.to_dict() if self.coordination_indicators else None,
            "synchronization_scores": [s.to_dict() for s in self.synchronization_scores],
            "bot_networks": [b.to_dict() for b in self.bot_networks],
            "content_coordination": [c.to_dict() for c in self.content_coordination],
            "overall_confidence": self.overall_confidence,
            "risk_level": self.risk_level.value,
            "evidence": self.evidence,
            "recommendations": self.recommendations,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class CoordinationDetectorConfig:
    """Configuration for CoordinationDetectorAgent."""
    agent_name: str = "coordination_detector_agent"
    
    # Detection thresholds
    synchronization_threshold: float = 0.7
    bot_likelihood_threshold: float = 0.6
    content_similarity_threshold: float = 0.8
    coordination_confidence_threshold: float = 0.75
    
    # Analysis parameters
    temporal_window_hours: int = 24
    min_events_for_analysis: int = 5
    max_user_pairs_analysis: int = 1000
    content_similarity_window_minutes: int = 60
    
    # Bot detection parameters
    bot_behavior_indicators: List[str] = field(default_factory=lambda: [
        "high_posting_frequency",
        "uniform_intervals",
        "content_repetition",
        "network_centrality",
        "engagement_anomalies"
    ])


class CoordinationDetectorAgent:
    """
    Agent for detecting coordinated inauthentic behavior.
    
    Provides comprehensive coordination detection through:
    - Temporal synchronization analysis
    - Content coordination detection
    - Bot network identification
    - Behavioral pattern analysis
    """
    
    def __init__(self, name: str = "agent", config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.config: CoordinationDetectorConfig = config
        
        # Analysis caches
        self.user_behavior_cache: Dict[str, Dict[str, Any]] = {}
        self.content_similarity_cache: Dict[str, float] = {}
        self.synchronization_cache: Dict[Tuple[str, str], SynchronizationScore] = {}
        
        # Performance tracking
        self.analysis_stats = {
            "coordination_detections": 0,
            "bot_networks_identified": 0,
            "synchronization_analyses": 0,
            "content_coordination_analyses": 0
        }
    def get_supported_features(self) -> List[str]:
        """Get list of supported coordination detection features."""
        return [
            "temporal_synchronization_analysis",
            "content_coordination_detection", 
            "bot_network_identification",
            "behavioral_pattern_analysis",
            "ollama_integration",
            "multi_modal_coordination_detection",
            "real_time_analysis",
            "confidence_scoring"
        ]
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get coordination analysis statistics."""
        stats = self.get_performance_stats()
        stats.update(self.analysis_stats)
        
        # Add cache statistics
        stats["cache_statistics"] = {
            "user_behavior_cache_size": len(self.user_behavior_cache),
            "content_similarity_cache_size": len(self.content_similarity_cache),
            "synchronization_cache_size": len(self.synchronization_cache)
        }
        
        return stats


# Main execution for testing
if __name__ == "__main__":
    import asyncio
    
    async def test_coordination_detector():
        """Test the coordination detector agent."""
        
        print("Testing CoordinationDetectorMCPAgent...")
        
        # Create agent
        config = CoordinationDetectorConfig()
        agent = CoordinationDetectorAgent(config)
        
        print(f"Created {agent.name} agent")
        print("Agent ready for direct method calls")
        
        try:
            # Initialize agent
            await agent.initialize()
            print(f"✓ Agent initialized: {agent.config.agent_name}")
            
            # Test data
            test_events = [
                {
                    "user_id": "user1",
                    "timestamp": "2024-01-01T10:00:00Z",
                    "content": "This is a test message about vaccines",
                    "platform": "twitter"
                },
                {
                    "user_id": "user2", 
                    "timestamp": "2024-01-01T10:01:00Z",
                    "content": "This is a test message about vaccines",
                    "platform": "twitter"
                },
                {
                    "user_id": "user3",
                    "timestamp": "2024-01-01T10:02:00Z", 
                    "content": "Similar message about vaccine safety",
                    "platform": "facebook"
                }
            ]
            
            print(f"✓ Test data prepared: {len(test_events)} events")
            
            # Test coordination detection
            print("\nTesting coordination detection...")
            result = await agent.detect_coordination(test_events, "test_topic")
            
            if result["success"]:
                data = result["data"]
                print(f"✓ Coordination detection completed")
                print(f"  - Coordination type: {data.get('coordination_type', 'unknown')}")
                print(f"  - Overall confidence: {data.get('overall_confidence', 0.0):.3f}")
                print(f"  - Risk level: {data.get('risk_level', 'unknown')}")
                print(f"  - Evidence count: {len(data.get('evidence', []))}")
            else:
                print(f"✗ Coordination detection failed: {result.get('error', {}).get('message', 'Unknown error')}")
            
            # Test performance stats
            stats = agent.get_analysis_statistics()
            print(f"\n✓ Performance stats: {stats['coordination_detections']} detections processed")
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
        
        finally:
            await agent.cleanup()
            print("✓ Agent cleanup completed")
    
    # Run test
    asyncio.run(test_coordination_detector())
