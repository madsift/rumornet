"""
KG Reasoning Agent - ReasoningStrategy Enum

This file exports the ReasoningStrategy enum used by multilingual reasoning agents.
The full KGReasoningMCPAgent class has been removed during MCP cleanup.
"""

from enum import Enum


class ReasoningStrategy(Enum):
    """Supported reasoning strategies."""
    COT = "chain_of_thought"
    TOT = "tree_of_thought" 
    GOT = "graph_of_thought"
    HYBRID = "hybrid_cot_tot"


__all__ = ["ReasoningStrategy"]
