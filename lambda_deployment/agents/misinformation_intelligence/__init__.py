"""
Misinformation Intelligence Agents

This package contains FastMCP agents for comprehensive misinformation detection
and analysis, exported from the existing misinformation_intelligence backend.

Implemented agents (5 consolidated from 7):
- BaseMisinformationAgent: Foundation class with Ollama integration and circuit breakers
- TopicIntelligenceEngineMCPAgent: Topic modeling and evolution tracking (BERTopic)
- TopicAwareClaimMatcherMCPAgent: Topic-aware claim matching with 90% search reduction
- TopicGuidedEvidenceRetrieverMCPAgent: Evidence retrieval with Tavily integration
- CoordinationDetectorMCPAgent: Bot detection and coordination analysis
- TopicSocialAnalysisMCPAgent: Unified social analysis (network, credibility, early warning)
  * Consolidates: TopicNetworkAnalyzerMCPAgent, TopicCredibilityEngineMCPAgent, TopicBasedEarlyWarningMCPAgent
"""

__version__ = "1.0.0"
__author__ = "Misinformation Intelligence Team"

# Base agent for common functionality
from .base_misinformation_agent import BaseMisinformationAgent

# Configuration and data models
from .config.agent_config import BaseMisinformationAgentConfig
from .core.data_models import (
    AlertRiskLevel, AlertType, TopicIntelligence, 
    IntelligenceResponse, ClaimMatchResult, EvidencePackage,
    TopicCredibilityScore, EarlyWarningAlert
)

__all__ = [
    "BaseMisinformationAgent",
    "BaseMisinformationAgentConfig",
    "AlertRiskLevel",
    "AlertType", 
    "TopicIntelligence",
    "IntelligenceResponse",
    "ClaimMatchResult",
    "EvidencePackage",
    "TopicCredibilityScore",
    "EarlyWarningAlert"
]
