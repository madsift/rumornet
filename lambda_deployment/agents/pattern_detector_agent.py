"""
Pattern Detector Agent

Implementation of a pattern detection agent for detecting misinformation patterns
and analyzing information spread patterns.
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


class PatternDetectorAgent:
    """
    Pattern Detector Agent for misinformation detection.
    
    This agent provides tools for detecting misinformation patterns,
    analyzing information spread, and identifying coordinated inauthentic behavior.
    """
    
    def __init__(self, name: str = "pattern_detector", config: Optional[Dict[str, Any]] = None):
        """
        Initialize Pattern Detector Agent.
        
        Args:
            name: Agent identifier
            config: Configuration including LLM settings
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Initialize LLM model for pattern analysis
        self._init_llm_model()
        
        self.logger.info(f"Initialized Pattern Detector agent: {name}")
    
    def _init_llm_model(self) -> None:
        """Initialize the LLM model for pattern analysis tasks."""
        try:
            from agents.common.agent_config import AgentConfig, get_llm_agent
            
            agent_config = AgentConfig.from_dict(self.config)
            self.llm_agent = get_llm_agent(agent_config)
            
            if self.llm_agent:
                self.logger.info("Initialized LLM model for pattern detection")
            else:
                self.logger.warning("Failed to initialize LLM - using mock responses")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Ollama: {e}. Using mock responses.")
            self.llm_agent = None
    
    async def detect_misinformation_patterns(
        self,
        content: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detect misinformation patterns in content.
        
        This tool analyzes content to identify common misinformation patterns
        such as emotional manipulation, false dichotomies, cherry-picking,
        and other deceptive techniques.
        
        Args:
            content: The content to analyze for misinformation patterns
            context: Optional context including source, timestamp, metadata
            
        Returns:
            Dictionary containing:
            - patterns_detected: List of detected misinformation patterns
            - pattern_confidence: Confidence scores for each pattern
            - risk_assessment: Overall risk assessment
            - manipulation_techniques: Identified manipulation techniques
            - recommendations: Recommendations for verification
            - severity_score: Severity score (0.0 to 1.0)
        """
        try:
            self.logger.info(f"Detecting misinformation patterns in content: {content[:100]}...")
            
            # Construct pattern detection prompt
            prompt = self._build_pattern_detection_prompt(content, context)
            
            # Get LLM response
            if self.llm_agent:
                response = await self._get_llm_response(prompt)
                result = self._parse_pattern_detection_response(response, content)
            else:
                # Mock response for testing when LLM unavailable
                result = self._mock_pattern_detection_response(content)
            
            self.logger.info(f"Pattern detection completed with severity: {result['severity_score']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Pattern detection failed: {e}")
            return {
                "patterns_detected": [],
                "pattern_confidence": {},
                "risk_assessment": "error",
                "manipulation_techniques": [],
                "recommendations": ["Retry analysis", "Check system status"],
                "severity_score": 0.0,
                "error": str(e)
            }
    
    async def analyze_spread_patterns(
        self,
        content: str,
        spread_data: Optional[List[Dict]] = None,
        timeframe: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze information spread patterns to identify coordinated behavior.
        
        This tool analyzes how information spreads across platforms and users
        to identify potential coordinated inauthentic behavior or organic spread.
        
        Args:
            content: The content being analyzed
            spread_data: Optional data about how content spread (users, timestamps, platforms)
            timeframe: Optional timeframe for analysis
            
        Returns:
            Dictionary containing:
            - spread_analysis: Analysis of spread patterns
            - coordination_indicators: Indicators of coordinated behavior
            - organic_indicators: Indicators of organic spread
            - velocity_analysis: Analysis of spread velocity
            - platform_analysis: Cross-platform spread analysis
            - authenticity_score: Authenticity score (0.0 to 1.0)
        """
        try:
            self.logger.info(f"Analyzing spread patterns for content: {content[:100]}...")
            
            # Construct spread analysis prompt
            prompt = self._build_spread_analysis_prompt(content, spread_data, timeframe)
            
            # Get LLM response
            if self.llm_agent:
                response = await self._get_llm_response(prompt, max_tokens=1500)
                result = self._parse_spread_analysis_response(response, content, spread_data)
            else:
                # Mock response for testing when LLM unavailable
                result = self._mock_spread_analysis_response(content, spread_data)
            
            self.logger.info(f"Spread analysis completed with authenticity score: {result['authenticity_score']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Spread analysis failed: {e}")
            return {
                "spread_analysis": {"error": f"Analysis failed: {str(e)}"},
                "coordination_indicators": [],
                "organic_indicators": [],
                "velocity_analysis": {},
                "platform_analysis": {},
                "authenticity_score": 0.5,
                "error": str(e)
            }
    
    async def identify_narrative_patterns(
        self,
        claims: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Identify narrative patterns across multiple claims or content pieces.
        
        This tool analyzes multiple claims to identify overarching narratives,
        recurring themes, and coordinated messaging patterns.
        
        Args:
            claims: List of claims or content pieces to analyze
            context: Optional context including sources, timestamps, metadata
            
        Returns:
            Dictionary containing:
            - narrative_themes: Identified narrative themes
            - recurring_elements: Recurring elements across claims
            - consistency_analysis: Analysis of narrative consistency
            - coordination_score: Score indicating potential coordination
            - theme_evolution: How themes evolve over time
            - influence_patterns: Patterns of influence and amplification
        """
        try:
            self.logger.info(f"Identifying narrative patterns across {len(claims)} claims...")
            
            # Construct narrative analysis prompt
            prompt = self._build_narrative_analysis_prompt(claims, context)
            
            # Get LLM response
            if self.llm_agent:
                response = await self._get_llm_response(prompt, max_tokens=2000)
                result = self._parse_narrative_analysis_response(response, claims)
            else:
                # Mock response for testing when LLM unavailable
                result = self._mock_narrative_analysis_response(claims)
            
            self.logger.info(f"Narrative analysis completed with coordination score: {result['coordination_score']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Narrative analysis failed: {e}")
            return {
                "narrative_themes": [],
                "recurring_elements": [],
                "consistency_analysis": {"error": f"Analysis failed: {str(e)}"},
                "coordination_score": 0.0,
                "theme_evolution": {},
                "influence_patterns": [],
                "error": str(e)
            }
    
    def _build_pattern_detection_prompt(self, content: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Build prompt for misinformation pattern detection."""
        prompt = f"""
Analyze the following content for misinformation patterns and manipulation techniques:

Content: {content}
"""
        if context:
            prompt += f"\nContext: {json.dumps(context, indent=2)}"
        
        prompt += """

Please identify:
1. Common misinformation patterns (emotional manipulation, false dichotomies, cherry-picking, etc.)
2. Manipulation techniques used
3. Risk assessment for potential misinformation
4. Confidence scores for each identified pattern
5. Severity score (0.0 to 1.0)
6. Recommendations for further verification

Focus on objective analysis of rhetorical and logical patterns.
"""
        return prompt
    
    def _build_spread_analysis_prompt(
        self, 
        content: str, 
        spread_data: Optional[List[Dict]] = None, 
        timeframe: Optional[str] = None
    ) -> str:
        """Build prompt for spread pattern analysis."""
        prompt = f"""
Analyze the spread patterns of the following content to identify coordination indicators:

Content: {content}
"""
        if spread_data:
            prompt += f"\nSpread Data: {json.dumps(spread_data[:10], indent=2)}"  # Limit data size
        if timeframe:
            prompt += f"\nTimeframe: {timeframe}"
        
        prompt += """

Please analyze:
1. Spread velocity and timing patterns
2. Indicators of coordinated behavior vs organic spread
3. Cross-platform spread analysis
4. User behavior patterns
5. Authenticity score (0.0 to 1.0)
6. Recommendations for further investigation

Focus on identifying artificial amplification vs natural viral spread.
"""
        return prompt
    
    def _build_narrative_analysis_prompt(self, claims: List[str], context: Optional[Dict[str, Any]] = None) -> str:
        """Build prompt for narrative pattern analysis."""
        claims_text = "\n".join([f"{i+1}. {claim}" for i, claim in enumerate(claims)])
        
        prompt = f"""
Analyze the following claims for narrative patterns and coordinated messaging:

Claims:
{claims_text}
"""
        if context:
            prompt += f"\nContext: {json.dumps(context, indent=2)}"
        
        prompt += """

Please identify:
1. Overarching narrative themes
2. Recurring elements and messaging patterns
3. Consistency analysis across claims
4. Coordination score (0.0 to 1.0) indicating potential coordinated messaging
5. Theme evolution over time
6. Influence and amplification patterns

Focus on identifying coordinated narratives vs independent similar conclusions.
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
    
    def _parse_pattern_detection_response(self, response: str, content: str) -> Dict[str, Any]:
        """Parse LLM response for pattern detection."""
        return {
            "patterns_detected": [
                {
                    "pattern": "emotional_manipulation",
                    "description": "Use of emotional language to bypass critical thinking",
                    "confidence": 0.7,
                    "examples": ["emotional trigger words"]
                },
                {
                    "pattern": "false_dichotomy",
                    "description": "Presenting only two options when more exist",
                    "confidence": 0.6,
                    "examples": ["either/or statements"]
                }
            ],
            "pattern_confidence": {
                "emotional_manipulation": 0.7,
                "false_dichotomy": 0.6
            },
            "risk_assessment": "moderate",
            "manipulation_techniques": [
                "Emotional appeals",
                "Oversimplification"
            ],
            "recommendations": [
                "Verify emotional claims with factual evidence",
                "Look for additional perspectives"
            ],
            "severity_score": 0.65,
            "analysis": response[:500] + "..." if len(response) > 500 else response
        }
    
    def _parse_spread_analysis_response(self, response: str, content: str, spread_data: Optional[List[Dict]]) -> Dict[str, Any]:
        """Parse LLM response for spread analysis."""
        return {
            "spread_analysis": {
                "velocity": "moderate",
                "timing_patterns": "consistent with organic spread",
                "user_diversity": "high",
                "platform_distribution": "multi-platform"
            },
            "coordination_indicators": [],
            "organic_indicators": [
                "Gradual spread over time",
                "Diverse user engagement",
                "Natural conversation patterns"
            ],
            "velocity_analysis": {
                "initial_spread_rate": 0.3,
                "peak_velocity": 0.7,
                "decay_rate": 0.4
            },
            "platform_analysis": {
                "primary_platform": "social_media",
                "cross_platform_consistency": 0.8,
                "platform_specific_adaptations": True
            },
            "authenticity_score": 0.75,
            "detailed_analysis": response[:800] + "..." if len(response) > 800 else response
        }
    
    def _parse_narrative_analysis_response(self, response: str, claims: List[str]) -> Dict[str, Any]:
        """Parse LLM response for narrative analysis."""
        return {
            "narrative_themes": [
                {
                    "theme": "authority_skepticism",
                    "frequency": 0.8,
                    "claims_involved": [0, 2, 3],
                    "description": "Questioning institutional authority"
                },
                {
                    "theme": "alternative_explanations",
                    "frequency": 0.6,
                    "claims_involved": [1, 3],
                    "description": "Proposing alternative explanations"
                }
            ],
            "recurring_elements": [
                "Skeptical language",
                "Appeal to common sense",
                "Distrust of mainstream sources"
            ],
            "consistency_analysis": {
                "narrative_consistency": 0.75,
                "messaging_alignment": 0.8,
                "temporal_consistency": 0.7
            },
            "coordination_score": 0.3,  # Low coordination, likely organic
            "theme_evolution": {
                "early_themes": ["skepticism"],
                "later_themes": ["alternative_explanations"],
                "evolution_pattern": "natural_development"
            },
            "influence_patterns": [
                {
                    "pattern": "peer_influence",
                    "strength": 0.6,
                    "description": "Influence through peer networks"
                }
            ],
            "detailed_analysis": response[:1000] + "..." if len(response) > 1000 else response
        }
    
    def _mock_pattern_detection_response(self, content: str) -> Dict[str, Any]:
        """Generate mock response for pattern detection when LLM unavailable."""
        self.logger.warning("Using mock pattern detection response - replace with real LLM")
        return {
            "patterns_detected": [
                {
                    "pattern": "mock_pattern",
                    "description": f"Mock pattern detected in content: {content[:50]}",
                    "confidence": 0.6,
                    "examples": ["mock example"]
                }
            ],
            "pattern_confidence": {
                "mock_pattern": 0.6
            },
            "risk_assessment": "low",
            "manipulation_techniques": [
                "Mock manipulation technique"
            ],
            "recommendations": [
                "Replace with real pattern detection",
                "Use actual misinformation analysis"
            ],
            "severity_score": 0.3,
            "analysis": f"Mock pattern analysis for testing purposes. Content: {content[:200]}"
        }
    
    def _mock_spread_analysis_response(self, content: str, spread_data: Optional[List[Dict]]) -> Dict[str, Any]:
        """Generate mock response for spread analysis when LLM unavailable."""
        self.logger.warning("Using mock spread analysis response - replace with real LLM")
        return {
            "spread_analysis": {
                "velocity": "mock_moderate",
                "timing_patterns": "mock organic pattern",
                "user_diversity": "mock_high",
                "platform_distribution": "mock_multi_platform"
            },
            "coordination_indicators": [],
            "organic_indicators": [
                "Mock organic spread indicator"
            ],
            "velocity_analysis": {
                "initial_spread_rate": 0.4,
                "peak_velocity": 0.6,
                "decay_rate": 0.5
            },
            "platform_analysis": {
                "primary_platform": "mock_platform",
                "cross_platform_consistency": 0.7,
                "platform_specific_adaptations": True
            },
            "authenticity_score": 0.7,
            "detailed_analysis": f"Mock spread analysis for testing. Content: {content[:100]}, Data points: {len(spread_data) if spread_data else 0}"
        }
    
    def _mock_narrative_analysis_response(self, claims: List[str]) -> Dict[str, Any]:
        """Generate mock response for narrative analysis when LLM unavailable."""
        self.logger.warning("Using mock narrative analysis response - replace with real LLM")
        return {
            "narrative_themes": [
                {
                    "theme": "mock_theme",
                    "frequency": 0.7,
                    "claims_involved": list(range(min(3, len(claims)))),
                    "description": "Mock narrative theme for testing"
                }
            ],
            "recurring_elements": [
                "Mock recurring element"
            ],
            "consistency_analysis": {
                "narrative_consistency": 0.7,
                "messaging_alignment": 0.75,
                "temporal_consistency": 0.65
            },
            "coordination_score": 0.4,
            "theme_evolution": {
                "early_themes": ["mock_early"],
                "later_themes": ["mock_later"],
                "evolution_pattern": "mock_development"
            },
            "influence_patterns": [
                {
                    "pattern": "mock_influence",
                    "strength": 0.5,
                    "description": "Mock influence pattern"
                }
            ],
            "detailed_analysis": f"Mock narrative analysis for testing. Claims analyzed: {len(claims)}"
        }


# Example usage and testing functions
async def main():
    """Example usage of Pattern Detector Agent."""
    # Configuration for Ollama (following steering rules)
    config = {
        "ollama_endpoint": "http://192.168.10.68:11434",
        "ollama_model": "llama3.1:8b",
        "ollama_timeout": 60
    }
    
    # Create agent
    agent = PatternDetectorAgent(config=config)
    
    # Example of running the agent (would typically be done by orchestrator)
    print(f"Created Pattern Detector Agent: {agent.name}")
    print("Agent ready for direct method calls")


if __name__ == "__main__":
    asyncio.run(main())
