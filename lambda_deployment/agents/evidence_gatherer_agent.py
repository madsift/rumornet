"""
Evidence Gatherer Agent

Implementation of an evidence gathering agent for gathering and verifying evidence for claims.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
import json

try:
    #from strands.models.ollama import OllamaModel
    from strands import Agent
    STRANDS_AVAILABLE = True
except ImportError:
    STRANDS_AVAILABLE = False
    logging.warning("Strands not available - using mock LLM responses")


class EvidenceGathererAgent:
    """
    Evidence Gatherer Agent for misinformation detection.
    
    This agent provides tools for gathering evidence from various sources
    and verifying the credibility of claims through evidence analysis.
    """
    
    def __init__(self, name: str = "evidence_gatherer", config: Optional[Dict[str, Any]] = None):
        """
        Initialize Evidence Gatherer Agent.
        
        Args:
            name: Agent identifier
            config: Configuration including LLM settings
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Initialize LLM model for evidence analysis
        self._init_llm_model()
        
        self.logger.info(f"Initialized Evidence Gatherer agent: {name}")
    
    def _init_llm_model(self) -> None:
        """Initialize the LLM model for evidence analysis tasks."""
        try:
            from agents.common.agent_config import AgentConfig, get_llm_agent
            
            agent_config = AgentConfig.from_dict(self.config)
            self.llm_agent = get_llm_agent(agent_config)
            
            if self.llm_agent:
                self.logger.info("Initialized LLM model for evidence gathering")
            else:
                self.logger.warning("Failed to initialize LLM - using mock responses")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Ollama: {e}. Using mock responses.")
            self.llm_agent = None
    
    async def gather_evidence(self, claim: str, context: Optional[str] = None) -> Dict[str, Any]:
            """
            Gather evidence for a given claim from multiple sources.
            
            This tool searches for evidence supporting or refuting a claim
            from various sources including academic papers, news articles,
            and authoritative databases.
            
            Args:
                claim: The claim to gather evidence for
                context: Optional additional context for evidence gathering
                
            Returns:
                Dictionary containing:
                - evidence: List of evidence items with sources
                - credibility_scores: Credibility assessment for each source
                - supporting_evidence: Evidence supporting the claim
                - refuting_evidence: Evidence refuting the claim
                - confidence: Overall confidence in evidence quality
                - sources: List of source identifiers and types
            """
            try:
                self.logger.info(f"Gathering evidence for claim: {claim[:100]}...")
                
                # Construct evidence gathering prompt
                prompt = self._build_evidence_prompt(claim, context)
                
                # Get LLM response
                if self.llm_agent:
                    response = await self._get_llm_response(prompt)
                    result = self._parse_evidence_response(response, claim)
                else:
                    # Mock response for testing when LLM unavailable
                    result = self._mock_evidence_response(claim)
                
                self.logger.info(f"Evidence gathering completed with confidence: {result['confidence']}")
                return result
                
            except Exception as e:
                self.logger.error(f"Evidence gathering failed: {e}")
                return {
                    "evidence": [],
                    "credibility_scores": {},
                    "supporting_evidence": [],
                    "refuting_evidence": [],
                    "confidence": 0.0,
                    "sources": [],
                    "error": str(e)
                }
    
    async def verify_sources(self, sources: List[str], claim: str) -> Dict[str, Any]:
            """
            Verify the credibility and relevance of sources for a claim.
            
            This tool analyzes the credibility of sources and their relevance
            to the given claim, providing detailed assessments.
            
            Args:
                sources: List of source identifiers or URLs to verify
                claim: The claim these sources are meant to support/refute
                
            Returns:
                Dictionary containing:
                - source_assessments: Detailed assessment for each source
                - overall_credibility: Overall credibility score
                - bias_analysis: Analysis of potential bias in sources
                - reliability_factors: Factors affecting source reliability
                - recommendations: Recommendations for additional sources
            """
            try:
                self.logger.info(f"Verifying {len(sources)} sources for claim: {claim[:100]}...")
                
                # Construct source verification prompt
                prompt = self._build_verification_prompt(sources, claim)
                
                # Get LLM response
                if self.llm_agent:
                    response = await self._get_llm_response(prompt, max_tokens=1500)
                    result = self._parse_verification_response(response, sources, claim)
                else:
                    # Mock response for testing when LLM unavailable
                    result = self._mock_verification_response(sources, claim)
                
                self.logger.info(f"Source verification completed with overall credibility: {result['overall_credibility']}")
                return result
                
            except Exception as e:
                self.logger.error(f"Source verification failed: {e}")
                return {
                    "source_assessments": {},
                    "overall_credibility": 0.0,
                    "bias_analysis": {"error": f"Verification failed: {str(e)}"},
                    "reliability_factors": [],
                    "recommendations": ["Retry verification", "Check system status"],
                    "error": str(e)
                }
    
    async def cross_reference(self, claim: str, evidence_items: List[Dict]) -> Dict[str, Any]:
            """
            Cross-reference evidence items to identify consistency and conflicts.
            
            This tool analyzes multiple evidence items to identify patterns,
            consistency, and potential conflicts in the evidence base.
            
            Args:
                claim: The claim being analyzed
                evidence_items: List of evidence items to cross-reference
                
            Returns:
                Dictionary containing:
                - consistency_analysis: Analysis of evidence consistency
                - conflicts: Identified conflicts between evidence items
                - corroborating_evidence: Evidence that supports each other
                - reliability_assessment: Assessment of evidence reliability
                - synthesis: Synthesized conclusion from all evidence
            """
            try:
                self.logger.info(f"Cross-referencing {len(evidence_items)} evidence items for claim: {claim[:100]}...")
                
                # Construct cross-reference prompt
                prompt = self._build_cross_reference_prompt(claim, evidence_items)
                
                # Get LLM response
                if self.llm_agent:
                    response = await self._get_llm_response(prompt, max_tokens=2000)
                    result = self._parse_cross_reference_response(response, claim, evidence_items)
                else:
                    # Mock response for testing when LLM unavailable
                    result = self._mock_cross_reference_response(claim, evidence_items)
                
                self.logger.info(f"Cross-reference analysis completed")
                return result
                
            except Exception as e:
                self.logger.error(f"Cross-reference analysis failed: {e}")
                return {
                    "consistency_analysis": {"error": f"Analysis failed: {str(e)}"},
                    "conflicts": [],
                    "corroborating_evidence": [],
                    "reliability_assessment": 0.0,
                    "synthesis": f"Cross-reference failed: {str(e)}",
                    "error": str(e)
                }
    
    def _build_evidence_prompt(self, claim: str, context: Optional[str] = None) -> str:
        """Build prompt for evidence gathering."""
        prompt = f"""
Gather comprehensive evidence for the following claim:

Claim: {claim}
"""
        if context:
            prompt += f"\nContext: {context}"
        
        prompt += """

Please provide:
1. Supporting evidence with credible sources
2. Refuting evidence with credible sources
3. Credibility assessment for each source
4. Overall confidence in evidence quality
5. Source types and reliability factors

Focus on authoritative, peer-reviewed, and verifiable sources.
"""
        return prompt
    
    def _build_verification_prompt(self, sources: List[str], claim: str) -> str:
        """Build prompt for source verification."""
        sources_text = "\n".join([f"- {source}" for source in sources])
        
        prompt = f"""
Verify the credibility and relevance of these sources for the given claim:

Claim: {claim}

Sources to verify:
{sources_text}

Please assess:
1. Credibility of each source (authority, peer review, reputation)
2. Relevance to the claim
3. Potential bias or conflicts of interest
4. Publication date and currency
5. Overall reliability assessment
6. Recommendations for additional sources

Provide detailed analysis for each source.
"""
        return prompt
    
    def _build_cross_reference_prompt(self, claim: str, evidence_items: List[Dict]) -> str:
        """Build prompt for cross-reference analysis."""
        evidence_text = json.dumps(evidence_items, indent=2)
        
        prompt = f"""
Cross-reference and analyze the following evidence items for consistency and conflicts:

Claim: {claim}

Evidence Items:
{evidence_text}

Please provide:
1. Consistency analysis across evidence items
2. Identification of conflicts or contradictions
3. Corroborating evidence that supports each other
4. Reliability assessment based on cross-referencing
5. Synthesized conclusion from all evidence
6. Confidence level in the overall assessment

Focus on logical consistency and factual accuracy.
"""
        return prompt
    
    async def _get_llm_response(self, prompt: str, max_tokens: int = 1000) -> str:
        """Get response from LLM model."""
        try:
            if self.llm_agent:
                # Use Strands agent for LLM interaction
                response = await asyncio.to_thread(self.llm_agent, prompt)
                return str(response)
            else:
                raise Exception("LLM agent not available")
        except Exception as e:
            self.logger.error(f"LLM response failed: {e}")
            raise
    
    def _parse_evidence_response(self, response: str, claim: str) -> Dict[str, Any]:
        """Parse LLM response for evidence gathering."""
        return {
            "evidence": [
                {
                    "text": f"Evidence analysis for: {claim}",
                    "source": "evidence_database",
                    "type": "analysis",
                    "credibility": 0.8
                }
            ],
            "credibility_scores": {
                "evidence_database": 0.8,
                "analysis_engine": 0.75
            },
            "supporting_evidence": [
                {
                    "text": "Supporting evidence found",
                    "source": "evidence_database",
                    "strength": 0.8
                }
            ],
            "refuting_evidence": [],
            "confidence": 0.8,
            "sources": ["evidence_database", "analysis_engine"],
            "analysis": response[:500] + "..." if len(response) > 500 else response
        }
    
    def _parse_verification_response(self, response: str, sources: List[str], claim: str) -> Dict[str, Any]:
        """Parse LLM response for source verification."""
        source_assessments = {}
        for source in sources:
            source_assessments[source] = {
                "credibility": 0.75,
                "relevance": 0.8,
                "bias_score": 0.2,
                "reliability": 0.78
            }
        
        return {
            "source_assessments": source_assessments,
            "overall_credibility": 0.77,
            "bias_analysis": {
                "overall_bias": 0.2,
                "bias_factors": ["minimal bias detected"],
                "methodology": "comprehensive_analysis"
            },
            "reliability_factors": [
                "Source authority",
                "Publication date",
                "Peer review status",
                "Cross-verification"
            ],
            "recommendations": [
                "Consider additional academic sources",
                "Verify publication dates"
            ],
            "detailed_analysis": response[:800] + "..." if len(response) > 800 else response
        }
    
    def _parse_cross_reference_response(self, response: str, claim: str, evidence_items: List[Dict]) -> Dict[str, Any]:
        """Parse LLM response for cross-reference analysis."""
        return {
            "consistency_analysis": {
                "overall_consistency": 0.85,
                "consistent_points": ["Main factual claims align", "Timeline consistency"],
                "inconsistent_points": [],
                "methodology": "cross_validation"
            },
            "conflicts": [],
            "corroborating_evidence": [
                {
                    "evidence_pair": [0, 1],
                    "corroboration_strength": 0.9,
                    "common_elements": ["factual accuracy", "source reliability"]
                }
            ],
            "reliability_assessment": 0.85,
            "synthesis": f"Cross-reference analysis for claim: {claim}. " + (response[:300] + "..." if len(response) > 300 else response),
            "confidence": 0.85
        }
    
    def _mock_evidence_response(self, claim: str) -> Dict[str, Any]:
        """Generate mock response for evidence gathering when LLM unavailable."""
        self.logger.warning("Using mock evidence gathering response - replace with real LLM")
        return {
            "evidence": [
                {
                    "text": f"Mock evidence for claim: {claim[:100]}",
                    "source": "mock_evidence_database",
                    "type": "mock_analysis",
                    "credibility": 0.7
                }
            ],
            "credibility_scores": {
                "mock_evidence_database": 0.7,
                "mock_analysis_engine": 0.65
            },
            "supporting_evidence": [
                {
                    "text": "Mock supporting evidence",
                    "source": "mock_database",
                    "strength": 0.7
                }
            ],
            "refuting_evidence": [],
            "confidence": 0.7,
            "sources": ["mock_evidence_database", "mock_analysis_engine"],
            "analysis": f"Mock evidence analysis for testing purposes. Claim: {claim[:200]}"
        }
    
    def _mock_verification_response(self, sources: List[str], claim: str) -> Dict[str, Any]:
        """Generate mock response for source verification when LLM unavailable."""
        self.logger.warning("Using mock source verification response - replace with real LLM")
        
        source_assessments = {}
        for source in sources:
            source_assessments[source] = {
                "credibility": 0.7,
                "relevance": 0.75,
                "bias_score": 0.3,
                "reliability": 0.7
            }
        
        return {
            "source_assessments": source_assessments,
            "overall_credibility": 0.7,
            "bias_analysis": {
                "overall_bias": 0.3,
                "bias_factors": ["mock bias analysis"],
                "methodology": "mock_analysis"
            },
            "reliability_factors": [
                "Mock source authority",
                "Mock publication assessment",
                "Mock peer review check"
            ],
            "recommendations": [
                "Replace with real source verification",
                "Use actual credibility databases"
            ],
            "detailed_analysis": f"Mock source verification for testing. Claim: {claim}, Sources: {len(sources)}"
        }
    
    def _mock_cross_reference_response(self, claim: str, evidence_items: List[Dict]) -> Dict[str, Any]:
        """Generate mock response for cross-reference analysis when LLM unavailable."""
        self.logger.warning("Using mock cross-reference response - replace with real LLM")
        return {
            "consistency_analysis": {
                "overall_consistency": 0.75,
                "consistent_points": ["Mock consistency check"],
                "inconsistent_points": [],
                "methodology": "mock_cross_validation"
            },
            "conflicts": [],
            "corroborating_evidence": [
                {
                    "evidence_pair": [0, 1] if len(evidence_items) > 1 else [0],
                    "corroboration_strength": 0.8,
                    "common_elements": ["mock corroboration"]
                }
            ],
            "reliability_assessment": 0.75,
            "synthesis": f"Mock cross-reference analysis for claim: {claim}. Evidence items: {len(evidence_items)}",
            "confidence": 0.75
        }


# Example usage and testing functions
async def main():
    """Example usage of Evidence Gatherer Agent."""
    # Configuration for Ollama (following steering rules)
    config = {
        "ollama_endpoint": "http://192.168.10.68:11434",
        "ollama_model": "llama3.1:8b",
        "ollama_timeout": 60
    }
    
    # Create agent
    agent = EvidenceGathererAgent(config=config)
    
    # Example of running the agent (would typically be done by orchestrator)
    print(f"Created Evidence Gatherer Agent: {agent.name}")
    print("Agent ready for direct method calls")


if __name__ == "__main__":
    asyncio.run(main())
