"""
Multi-Agent Orchestration Framework - Agents Module

This module provides agent implementations for the orchestration framework.
"""

from .evidence_gatherer_agent import EvidenceGathererAgent
from .pattern_detector_agent import PatternDetectorAgent

# Note: MultilingualKGReasoningAgent and other agents are imported directly where needed
# to avoid initialization issues during module import

__all__ = [
    "EvidenceGathererAgent",
    "PatternDetectorAgent",
]
