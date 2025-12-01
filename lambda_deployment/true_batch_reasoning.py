#!/usr/bin/env python3
"""
TRUE Batch Reasoning - Send ALL claims to Ollama in ONE request.

Based on the kg_building_agent.py pattern that works with Bedrock/Ollama.
"""

import json
import time
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from strands import Agent
#from strands.models.ollama import OllamaModel


class SingleClaimReasoning(BaseModel):
    """Reasoning result for a single claim."""
    claim_id: int = Field(description="The numeric ID of the claim (1-based)")
    verdict: bool = Field(description="True if claim is true, False if false")
    confidence: float = Field(description="Confidence score 0.0-1.0")
    detected_language: str = Field(description="Language code (e.g., 'en', 'es')")
    reasoning: str = Field(description="Brief reasoning for the verdict")


class BatchReasoningResult(BaseModel):
    """Results for a batch of claims."""
    results: List[SingleClaimReasoning]


async def true_batch_reason(
    claims: List[str],
    ollama_endpoint: str = None,
    ollama_model: str = None,
    agent_config = None
) -> List[Dict[str, Any]]:
    """
    Send ALL claims to LLM in ONE request for TRUE batch processing.
    
    Supports both Ollama and Bedrock providers.
    
    Args:
        claims: List of claim texts to analyze
        ollama_endpoint: Ollama server endpoint (for Ollama provider)
        ollama_model: Model to use (for Ollama provider)
        agent_config: AgentConfig object (alternative to individual params)
        
    Returns:
        List of reasoning results, one per claim
    """
    start_time = time.time()
    
    if not claims:
        return []
    
    # Determine provider and initialize appropriate model
    if agent_config:
        from agents.common.agent_config import get_llm_agent
        agent = get_llm_agent(agent_config)
        provider = agent_config.llm_provider
        print(f"🚀 TRUE BATCH: Sending {len(claims)} claims to {provider.upper()} in ONE request...")
    else:
        # Fallback requires agent_config
        raise ValueError("agent_config is required for batch reasoning")
    
    # Build ONE prompt with ALL claims (like kg_building_agent.py does)
    prompt_claims = "\n".join([f"Claim {i+1}: \"{claim}\"" for i, claim in enumerate(claims)])
    
    prompt = (
        f"For each of the following {len(claims)} claims, determine:\n"
        "1. Is it TRUE or FALSE?\n"
        "2. Your confidence (0.0 to 1.0)\n"
        "3. The language (e.g., 'en', 'es', 'fr')\n"
        "4. Brief reasoning\n\n"
        "Match the output JSON schema exactly, providing a result for each claim_id.\n\n"
        f"{prompt_claims}"
    )
    
    try:
        # ONE call with structured_output (like kg_building_agent.py)
        llm_start = time.time()
        
        agent_response = agent.structured_output(BatchReasoningResult, prompt)
        llm_time = (time.time() - llm_start) * 1000
        batch_result = agent_response
        
        # Check if the agent returned a string instead of a parsed object (like kg_building_agent.py)
        if isinstance(batch_result, str):
            try:
                # Parse the JSON string into a Python dictionary
                parsed_data = json.loads(batch_result)
                # Create the Pydantic model from the parsed dictionary
                batch_result = BatchReasoningResult(**parsed_data)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"⚠️ Failed to parse agent's string response: {e}")
                batch_result = BatchReasoningResult(results=[])
        
        # Convert to standard format
        results = []
        for res in batch_result.results:
            claim_index = res.claim_id - 1  # Convert 1-based to 0-based
            if 0 <= claim_index < len(claims):
                results.append({
                    "verdict": res.verdict,
                    "confidence": res.confidence,
                    "detected_language": res.detected_language,
                    "reasoning_chain": [{"step": "1", "reasoning": res.reasoning}],
                    "execution_time_ms": llm_time / len(claims)  # Distribute time
                })
        
        total_time = (time.time() - start_time) * 1000
        avg_per_claim = total_time / len(claims)
        
        print(f"✅ TRUE BATCH complete: {len(results)} claims in {total_time:.0f}ms ({avg_per_claim:.0f}ms per claim)")
        
        return results
        
    except Exception as e:
        print(f"❌ TRUE BATCH failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Return empty results
        return [
            {
                "verdict": None,
                "confidence": 0.0,
                "detected_language": "unknown",
                "reasoning_chain": [],
                "error": str(e),
                "execution_time_ms": 0
            }
            for _ in claims
        ]


# Test function
async def test_true_batch():
    """Test TRUE batch reasoning."""
    test_claims = [
        "The Earth is flat",
        "Water boils at 100°C at sea level",
        "The moon is made of cheese",
        "Python is a programming language",
        "Vaccines cause autism"
    ]
    
    print("\n" + "="*80)
    print("Testing TRUE Batch Reasoning")
    print("="*80)
    
    results = await true_batch_reason(test_claims)
    
    print("\nResults:")
    for i, (claim, result) in enumerate(zip(test_claims, results), 1):
        print(f"\n{i}. {claim}")
        print(f"   Verdict: {result['verdict']}, Confidence: {result['confidence']:.2f}")
        print(f"   Language: {result['detected_language']}")
        if result.get('reasoning_chain'):
            print(f"   Reasoning: {result['reasoning_chain'][0]['reasoning'][:100]}...")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_true_batch())
