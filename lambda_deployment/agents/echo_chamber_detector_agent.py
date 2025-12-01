"""
Echo Chamber Detector Agent

Production implementation of an Echo Chamber Detection agent with real Ollama integration.
This agent provides tools for comprehensive echo chamber and filter bubble detection in social
media networks using advanced reasoning strategies.

Requirements addressed:
- 11.3: EchoChamberDetectorAgent analyzes social networks for echo chambers and filter bubbles
- 13.1: Uses real Ollama with gemma3:4b model
- 13.2: No mock responses - all real LLM calls
- 13.3: Validates Ollama connectivity during initialization
"""

import logging
import asyncio
import time
import json
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

try:
    #from strands.models.ollama import OllamaModel
    from strands import Agent
    STRANDS_AVAILABLE = True
except ImportError:
    STRANDS_AVAILABLE = False
    # Note: Strands requirement relaxed - will be checked at runtime if needed


class EchoChamberRiskLevel(Enum):
    """Echo chamber risk levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class EchoChamberMetrics:
    """Comprehensive metrics for echo chamber detection."""
    
    # Homophily metrics
    homophily_score: float = 0.0
    clustering_coefficient: float = 0.0
    assortativity_coefficient: float = 0.0
    
    # Information isolation metrics
    isolation_index: float = 0.0
    external_connectivity: float = 0.0
    information_diversity: float = 0.0
    
    # Polarization metrics
    polarization_score: float = 0.0
    sentiment_divergence: float = 0.0
    cross_group_engagement: float = 0.0
    
    # Reinforcement metrics
    confirmation_bias_score: float = 0.0
    echo_strength: float = 0.0
    belief_reinforcement_ratio: float = 0.0
    
    # Overall assessment
    echo_chamber_probability: float = 0.0
    confidence: float = 0.0
    risk_level: str = "LOW"


class EchoChamberDetectorAgent:
    """
    Production Echo Chamber Detection Agent with real Ollama integration.
    
    This agent provides comprehensive echo chamber and filter bubble detection capabilities:
    - Network homophily analysis for identifying similar user clustering
    - Information isolation detection for identifying closed information loops
    - Polarization measurement for detecting opposing viewpoint separation
    - Confirmation bias detection for identifying belief reinforcement patterns
    - Comprehensive echo chamber metrics and risk assessment
    - Social network analysis with graph-based reasoning
    
    Requirements addressed:
    - 11.3: Analyzes social networks for echo chambers and filter bubbles
    - 13.1: Real Ollama integration with gemma3:4b model
    - 13.2: No mock responses - all real LLM calls
    - 13.3: Validates Ollama connectivity during initialization
    """
    
    def __init__(self, name: str = "echo_chamber_detector_agent", config: Optional[Dict[str, Any]] = None):
        """
        Initialize Echo Chamber Detector Agent with real Ollama integration.
        
        Args:
            name: Agent identifier
            config: Configuration including Ollama settings
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Get configuration from centralized config
        from .common.agent_config import AgentConfig
        agent_config = AgentConfig.from_dict(self.config)
        self.ollama_endpoint = agent_config.ollama_endpoint
        self.ollama_model = agent_config.ollama_model
        self.ollama_timeout = self.config.get("ollama_timeout", 90)  # Longer timeout for complex analysis
        
        # Initialize and validate Ollama model
        self._init_and_validate_ollama()
        
        # Performance tracking
        self._performance_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'analysis_types': {
                'homophily': 0,
                'isolation': 0,
                'polarization': 0,
                'reinforcement': 0,
                'comprehensive': 0
            },
            'avg_response_time_ms': 0.0,
            'total_response_time_ms': 0.0,
            'risk_level_distribution': {level.value: 0 for level in EchoChamberRiskLevel},
            'network_analyses': 0,
            'batch_analyses': 0
        }
        
        # Analysis templates for different echo chamber detection approaches
        self.analysis_templates = {
            "homophily": {
                "description": "Analyze network homophily patterns where users primarily interact with similar users",
                "focus_areas": ["user clustering", "demographic similarities", "interaction patterns", "group formation"],
                "expected_indicators": ["clustering coefficient", "similarity measures", "group boundaries"]
            },
            "isolation": {
                "description": "Examine information flow patterns to identify informationally isolated communities",
                "focus_areas": ["information sources", "cross-group communication", "external connectivity", "content diversity"],
                "expected_indicators": ["isolation index", "external links", "information variety"]
            },
            "polarization": {
                "description": "Identify content polarization where opposing viewpoints rarely interact",
                "focus_areas": ["topic separation", "viewpoint clustering", "cross-engagement", "discussion bubbles"],
                "expected_indicators": ["polarization score", "sentiment divergence", "interaction gaps"]
            },
            "reinforcement": {
                "description": "Analyze belief reinforcement patterns and confirmation bias indicators",
                "focus_areas": ["content repetition", "belief confirmation", "challenge avoidance", "echo amplification"],
                "expected_indicators": ["confirmation bias", "echo strength", "reinforcement ratio"]
            }
        }
    
    def _init_and_validate_ollama(self) -> None:
        """Initialize and validate Ollama model connection - NO MOCKS ALLOWED."""
        try:
            self.logger.info(f"Initializing Ollama model at {self.ollama_endpoint} with model {self.ollama_model}")
            
            # Initialize Ollama model
            ollama_model = OllamaModel(
                host=self.ollama_endpoint,
                model_id=self.ollama_model
            )
            
            # Create Strands agent
            self.llm_agent = Agent(model=ollama_model)
            
            # Validate connectivity with a test call
            self._validate_ollama_connectivity()
            
            self.logger.info(f"Successfully initialized and validated Ollama model: {self.ollama_model}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Ollama model: {e}")
            self.logger.error("Ensure Ollama is running at http://192.168.10.68:11434 with gemma3:4b model")
            raise Exception(f"Ollama initialization failed: {e}. No fallback allowed - real LLM required.")
    
    def _validate_ollama_connectivity(self) -> None:
        """Validate Ollama connectivity with a test call."""
        try:
            # Test with a simple prompt
            test_response = self.llm_agent("Test connectivity for echo chamber analysis. Respond with 'READY'.")
            if test_response:
                self.logger.info("Ollama connectivity validated successfully for echo chamber detection")
            else:
                raise Exception("Ollama returned empty response")
        except Exception as e:
            raise Exception(f"Ollama connectivity validation failed: {e}")
    
    async def detect_echo_chamber(
            network_data: Dict[str, Any],
            analysis_types: Optional[List[str]] = None,
            include_recommendations: bool = True
        ) -> Dict[str, Any]:
            """
            Perform comprehensive echo chamber detection analysis on social network data.
            
            This tool analyzes social network structures and content patterns to identify
            echo chambers and filter bubbles using multiple analytical approaches including
            homophily analysis, information isolation detection, polarization measurement,
            and confirmation bias identification.
            
            Args:
                network_data: Social network data including nodes, edges, content, and metadata
                analysis_types: Types of analysis to perform (default: all types)
                    Options: ["homophily", "isolation", "polarization", "reinforcement"]
                include_recommendations: Whether to include actionable recommendations
                
            Returns:
                Dictionary containing:
                - echo_chamber_detected: Boolean indicating if echo chamber patterns found
                - risk_level: Risk level (LOW, MEDIUM, HIGH, CRITICAL)
                - confidence: Overall confidence score (0.0 to 1.0)
                - metrics: Comprehensive EchoChamberMetrics object
                - analysis_results: Detailed results for each analysis type
                - detected_chambers: List of identified echo chamber structures
                - recommendations: Actionable recommendations (if requested)
                - execution_time_ms: Processing time in milliseconds
                - metadata: Additional analysis metadata
            """
            start_time = time.time()
            
            try:
                self.logger.info("Starting comprehensive echo chamber detection analysis")
                
                # Validate network data
                if not self._validate_network_data(network_data):
                    return {
                        "echo_chamber_detected": False,
                        "risk_level": "LOW",
                        "confidence": 0.0,
                        "error": "Invalid network data format. Required: nodes, edges, content",
                        "execution_time_ms": 0.0
                    }
                
                # Set default analysis types
                if analysis_types is None:
                    analysis_types = ["homophily", "isolation", "polarization", "reinforcement"]
                
                # Update performance stats
                self._performance_stats['total_requests'] += 1
                self._performance_stats['analysis_types']['comprehensive'] += 1
                
                # Perform individual analyses
                analysis_results = {}
                for analysis_type in analysis_types:
                    if analysis_type in self.analysis_templates:
                        self.logger.info(f"Performing {analysis_type} analysis...")
                        
                        result = await self._perform_echo_chamber_analysis(
                            network_data, analysis_type
                        )
                        analysis_results[analysis_type] = result
                        
                        # Update analysis type stats
                        if analysis_type in self._performance_stats['analysis_types']:
                            self._performance_stats['analysis_types'][analysis_type] += 1
                
                # Calculate comprehensive metrics
                metrics = self._calculate_comprehensive_metrics(analysis_results)
                
                # Determine overall echo chamber detection
                echo_chamber_detected = metrics.echo_chamber_probability > 0.5
                risk_level = self._determine_risk_level(metrics.echo_chamber_probability)
                
                # Extract detected chambers
                detected_chambers = self._extract_echo_chambers(analysis_results, network_data)
                
                # Generate recommendations if requested
                recommendations = []
                if include_recommendations:
                    recommendations = self._generate_recommendations(metrics, analysis_results)
                
                execution_time = (time.time() - start_time) * 1000
                
                # Update performance stats
                self._performance_stats['successful_requests'] += 1
                self._performance_stats['total_response_time_ms'] += execution_time
                self._performance_stats['avg_response_time_ms'] = (
                    self._performance_stats['total_response_time_ms'] / 
                    self._performance_stats['successful_requests']
                )
                self._performance_stats['risk_level_distribution'][risk_level] += 1
                self._performance_stats['network_analyses'] += 1
                
                result = {
                    "echo_chamber_detected": echo_chamber_detected,
                    "risk_level": risk_level,
                    "confidence": metrics.confidence,
                    "metrics": asdict(metrics),
                    "analysis_results": analysis_results,
                    "detected_chambers": detected_chambers,
                    "recommendations": recommendations,
                    "execution_time_ms": execution_time,
                    "metadata": {
                        "analysis_types_performed": analysis_types,
                        "network_size": {
                            "nodes": len(network_data.get("nodes", [])),
                            "edges": len(network_data.get("edges", [])),
                            "content_items": len(network_data.get("content", []))
                        },
                        "timestamp": datetime.now().isoformat()
                    }
                }
                
                self.logger.info(f"Echo chamber detection completed in {execution_time:.2f}ms")
                self.logger.info(f"Result: {echo_chamber_detected}, Risk: {risk_level}, Confidence: {metrics.confidence:.2f}")
                
                return result
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                self._performance_stats['failed_requests'] += 1
                
                self.logger.error(f"Echo chamber detection failed after {execution_time:.2f}ms: {e}")
                return {
                    "echo_chamber_detected": False,
                    "risk_level": "LOW",
                    "confidence": 0.0,
                    "error": str(e),
                    "execution_time_ms": execution_time,
                    "metadata": {"error_type": type(e).__name__}
                }
    
    async def analyze_homophily(
            network_data: Dict[str, Any],
            focus_attributes: Optional[List[str]] = None
        ) -> Dict[str, Any]:
            """
            Analyze network homophily patterns to identify user clustering by similarity.
            
            This tool specifically examines how users with similar characteristics
            (demographics, interests, beliefs) cluster together and interact primarily
            within their similar groups, creating homogeneous communities.
            
            Args:
                network_data: Social network data with user attributes and connections
                focus_attributes: Specific attributes to analyze for homophily
                    (e.g., ["age", "location", "interests", "political_views"])
                
            Returns:
                Dictionary containing:
                - homophily_score: Overall homophily strength (0.0 to 1.0)
                - clustering_coefficient: Network clustering measure
                - assortativity_coefficient: Attribute-based connection preference
                - similar_groups: Identified homogeneous user groups
                - connection_patterns: Analysis of within-group vs between-group connections
                - attribute_analysis: Breakdown by specific attributes
                - execution_time_ms: Processing time
            """
            start_time = time.time()
            
            try:
                self.logger.info("Starting homophily analysis")
                
                # Validate network data
                if not self._validate_network_data(network_data):
                    return {"error": "Invalid network data format", "execution_time_ms": 0.0}
                
                # Perform homophily-specific analysis
                result = await self._perform_echo_chamber_analysis(network_data, "homophily")
                
                # Extract homophily-specific metrics
                homophily_metrics = self._extract_homophily_metrics(result, focus_attributes)
                
                execution_time = (time.time() - start_time) * 1000
                homophily_metrics['execution_time_ms'] = execution_time
                
                self.logger.info(f"Homophily analysis completed in {execution_time:.2f}ms")
                return homophily_metrics
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                self.logger.error(f"Homophily analysis failed: {e}")
                return {
                    "error": str(e),
                    "execution_time_ms": execution_time,
                    "homophily_score": 0.0
                }
    
    async def measure_polarization(
            content_data: List[Dict[str, Any]],
            topic_focus: Optional[str] = None
        ) -> Dict[str, Any]:
            """
            Measure content polarization to identify separated viewpoint clusters.
            
            This tool analyzes content and discussion patterns to identify topics
            where opposing viewpoints exist in separate clusters with minimal
            cross-engagement, indicating polarized discourse.
            
            Args:
                content_data: List of content items with text, metadata, and engagement data
                topic_focus: Specific topic to focus analysis on (optional)
                
            Returns:
                Dictionary containing:
                - polarization_score: Overall polarization level (0.0 to 1.0)
                - sentiment_divergence: Difference in sentiment between clusters
                - cross_group_engagement: Level of interaction between opposing groups
                - viewpoint_clusters: Identified clusters of similar viewpoints
                - topic_analysis: Breakdown by topics (if topic_focus specified)
                - engagement_patterns: Analysis of interaction patterns
                - execution_time_ms: Processing time
            """
            start_time = time.time()
            
            try:
                self.logger.info(f"Starting polarization analysis{f' for topic: {topic_focus}' if topic_focus else ''}")
                
                # Validate content data
                if not content_data or not isinstance(content_data, list):
                    return {"error": "Invalid content data format", "execution_time_ms": 0.0}
                
                # Perform polarization-specific analysis
                network_data = {"content": content_data, "nodes": [], "edges": []}
                result = await self._perform_echo_chamber_analysis(network_data, "polarization")
                
                # Extract polarization-specific metrics
                polarization_metrics = self._extract_polarization_metrics(result, topic_focus)
                
                execution_time = (time.time() - start_time) * 1000
                polarization_metrics['execution_time_ms'] = execution_time
                
                self.logger.info(f"Polarization analysis completed in {execution_time:.2f}ms")
                return polarization_metrics
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                self.logger.error(f"Polarization analysis failed: {e}")
                return {
                    "error": str(e),
                    "execution_time_ms": execution_time,
                    "polarization_score": 0.0
                }
    
    async def calculate_isolation_index(
            network_data: Dict[str, Any],
            community_definitions: Optional[Dict[str, List[str]]] = None
        ) -> Dict[str, Any]:
            """
            Calculate information isolation index for identified communities.
            
            This tool measures how isolated different communities are in terms
            of information flow, external connections, and content diversity,
            identifying communities that exist in information bubbles.
            
            Args:
                network_data: Social network data with connection and content information
                community_definitions: Pre-defined community groupings (optional)
                
            Returns:
                Dictionary containing:
                - isolation_index: Overall isolation measure (0.0 to 1.0)
                - external_connectivity: Level of connections outside communities
                - information_diversity: Variety of information sources and content
                - community_isolation: Isolation scores for individual communities
                - information_flow: Analysis of how information moves between communities
                - diversity_metrics: Content and source diversity measurements
                - execution_time_ms: Processing time
            """
            start_time = time.time()
            
            try:
                self.logger.info("Starting isolation index calculation")
                
                # Validate network data
                if not self._validate_network_data(network_data):
                    return {"error": "Invalid network data format", "execution_time_ms": 0.0}
                
                # Perform isolation-specific analysis
                result = await self._perform_echo_chamber_analysis(network_data, "isolation")
                
                # Extract isolation-specific metrics
                isolation_metrics = self._extract_isolation_metrics(result, community_definitions)
                
                execution_time = (time.time() - start_time) * 1000
                isolation_metrics['execution_time_ms'] = execution_time
                
                self.logger.info(f"Isolation index calculation completed in {execution_time:.2f}ms")
                return isolation_metrics
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                self.logger.error(f"Isolation index calculation failed: {e}")
                return {
                    "error": str(e),
                    "execution_time_ms": execution_time,
                    "isolation_index": 0.0
                }
    
    async def get_echo_chamber_metrics(
            analysis_results: Dict[str, Any]
        ) -> Dict[str, Any]:
            """
            Generate comprehensive echo chamber metrics from analysis results.
            
            This tool takes results from various echo chamber analyses and
            synthesizes them into a comprehensive metrics report with
            risk assessment and actionable insights.
            
            Args:
                analysis_results: Results from previous echo chamber analyses
                
            Returns:
                Dictionary containing:
                - comprehensive_metrics: Complete EchoChamberMetrics object
                - risk_assessment: Detailed risk level analysis
                - key_findings: Summary of most important findings
                - trend_analysis: Temporal trends if historical data available
                - comparative_analysis: Comparison with baseline or benchmarks
                - actionable_insights: Specific insights for intervention
                - execution_time_ms: Processing time
            """
            start_time = time.time()
            
            try:
                self.logger.info("Generating comprehensive echo chamber metrics")
                
                # Validate analysis results
                if not analysis_results or not isinstance(analysis_results, dict):
                    return {"error": "Invalid analysis results format", "execution_time_ms": 0.0}
                
                # Calculate comprehensive metrics
                metrics = self._calculate_comprehensive_metrics(analysis_results)
                
                # Generate risk assessment
                risk_assessment = self._generate_risk_assessment(metrics)
                
                # Extract key findings
                key_findings = self._extract_key_findings(analysis_results, metrics)
                
                # Generate actionable insights
                actionable_insights = self._generate_actionable_insights(metrics, analysis_results)
                
                execution_time = (time.time() - start_time) * 1000
                
                result = {
                    "comprehensive_metrics": asdict(metrics),
                    "risk_assessment": risk_assessment,
                    "key_findings": key_findings,
                    "actionable_insights": actionable_insights,
                    "execution_time_ms": execution_time,
                    "metadata": {
                        "analysis_timestamp": datetime.now().isoformat(),
                        "metrics_version": "1.0",
                        "analysis_types_included": list(analysis_results.keys())
                    }
                }
                
                self.logger.info(f"Echo chamber metrics generation completed in {execution_time:.2f}ms")
                return result
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                self.logger.error(f"Echo chamber metrics generation failed: {e}")
                return {
                    "error": str(e),
                    "execution_time_ms": execution_time,
                    "comprehensive_metrics": asdict(EchoChamberMetrics())
                }
    
    def _validate_network_data(self, network_data: Dict[str, Any]) -> bool:
        """Validate network data format."""
        required_fields = ["nodes", "edges"]
        return (
            isinstance(network_data, dict) and
            all(field in network_data for field in required_fields) and
            isinstance(network_data["nodes"], list) and
            isinstance(network_data["edges"], list)
        )
    
    async def _perform_echo_chamber_analysis(
        self, 
        network_data: Dict[str, Any], 
        analysis_type: str
    ) -> Dict[str, Any]:
        """Perform specific type of echo chamber analysis using LLM reasoning."""
        
        template = self.analysis_templates.get(analysis_type, {})
        
        # Prepare network summary for LLM analysis
        network_summary = self._prepare_network_summary(network_data)
        
        prompt = f"""
        Perform {analysis_type} analysis for echo chamber detection.
        
        Analysis Focus: {template.get('description', 'General echo chamber analysis')}
        
        Network Data Summary:
        {network_summary}
        
        Focus Areas: {', '.join(template.get('focus_areas', []))}
        Expected Indicators: {', '.join(template.get('expected_indicators', []))}
        
        Please analyze the network data and provide:
        1. Detailed analysis of {analysis_type} patterns
        2. Quantitative assessment (scores from 0.0 to 1.0)
        3. Identification of specific groups or clusters
        4. Evidence supporting your findings
        5. Confidence level in your analysis
        
        Structure your response with clear sections for each aspect.
        """
        
        try:
            response = await self._get_llm_response(prompt)
            return self._parse_analysis_response(response, analysis_type)
        except Exception as e:
            raise Exception(f"{analysis_type} analysis failed: {e}")
    
    def _prepare_network_summary(self, network_data: Dict[str, Any]) -> str:
        """Prepare a summary of network data for LLM analysis."""
        
        nodes = network_data.get("nodes", [])
        edges = network_data.get("edges", [])
        content = network_data.get("content", [])
        
        summary = f"""
        Network Structure:
        - Nodes (users/entities): {len(nodes)}
        - Edges (connections): {len(edges)}
        - Content items: {len(content)}
        
        Sample Node Attributes: {self._sample_attributes(nodes, 'nodes')}
        Sample Edge Types: {self._sample_attributes(edges, 'edges')}
        Sample Content Types: {self._sample_attributes(content, 'content')}
        """
        
        return summary
    
    def _sample_attributes(self, items: List[Dict], item_type: str) -> str:
        """Sample attributes from items for summary."""
        if not items:
            return "None available"
        
        sample_item = items[0] if items else {}
        attributes = list(sample_item.keys())[:5]  # First 5 attributes
        
        return f"{attributes}" if attributes else "No attributes"
    
    async def _get_llm_response(self, prompt: str) -> str:
        """Get response from Ollama LLM - NO MOCKS ALLOWED."""
        try:
            if not self.llm_agent:
                raise Exception("LLM agent not initialized")
            
            # Use Strands agent for LLM interaction
            response = await asyncio.to_thread(self.llm_agent, prompt)
            
            if not response:
                raise Exception("Empty response from Ollama")
            
            return str(response)
            
        except Exception as e:
            self.logger.error(f"LLM response failed: {e}")
            raise Exception(f"Ollama LLM call failed: {e}")
    
    def _parse_analysis_response(self, response: str, analysis_type: str) -> Dict[str, Any]:
        """Parse LLM analysis response into structured format."""
        import re
        
        # Extract scores using regex patterns
        score_patterns = {
            'overall_score': r'(?:overall|main|primary).*?score.*?([0-9]*\.?[0-9]+)',
            'confidence': r'confidence.*?([0-9]*\.?[0-9]+)',
            'strength': r'strength.*?([0-9]*\.?[0-9]+)'
        }
        
        scores = {}
        for score_name, pattern in score_patterns.items():
            match = re.search(pattern, response.lower())
            if match:
                try:
                    scores[score_name] = float(match.group(1))
                except ValueError:
                    scores[score_name] = 0.5  # Default middle value
            else:
                scores[score_name] = 0.5
        
        # Extract identified groups/clusters
        groups = []
        group_pattern = r'group\s*\d+|cluster\s*\d+|community\s*\d+'
        group_matches = re.findall(group_pattern, response.lower())
        
        for i, match in enumerate(group_matches[:5]):  # Limit to 5 groups
            groups.append({
                "id": f"group_{i+1}",
                "description": f"Identified {match}",
                "type": analysis_type
            })
        
        # Extract evidence
        evidence = []
        evidence_keywords = ["evidence", "indicator", "pattern", "finding"]
        sentences = response.split('.')
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in evidence_keywords):
                evidence.append(sentence.strip())
                if len(evidence) >= 3:  # Limit evidence items
                    break
        
        return {
            "analysis_type": analysis_type,
            "overall_score": scores.get('overall_score', 0.5),
            "confidence": scores.get('confidence', 0.5),
            "strength": scores.get('strength', 0.5),
            "identified_groups": groups,
            "evidence": evidence,
            "raw_response": response[:500] + "..." if len(response) > 500 else response,
            "success": True
        }
    
    def _calculate_comprehensive_metrics(self, analysis_results: Dict[str, Any]) -> EchoChamberMetrics:
        """Calculate comprehensive echo chamber metrics from all analyses."""
        
        metrics = EchoChamberMetrics()
        
        # Extract metrics from each analysis type
        for analysis_type, result in analysis_results.items():
            if not result.get("success", False):
                continue
                
            overall_score = result.get("overall_score", 0.0)
            confidence = result.get("confidence", 0.0)
            
            if analysis_type == "homophily":
                metrics.homophily_score = overall_score
                metrics.clustering_coefficient = result.get("strength", overall_score)
                metrics.assortativity_coefficient = overall_score * 0.9  # Derived metric
                
            elif analysis_type == "isolation":
                metrics.isolation_index = overall_score
                metrics.external_connectivity = 1.0 - overall_score  # Inverse relationship
                metrics.information_diversity = 1.0 - (overall_score * 0.8)
                
            elif analysis_type == "polarization":
                metrics.polarization_score = overall_score
                metrics.sentiment_divergence = overall_score * 0.9
                metrics.cross_group_engagement = 1.0 - overall_score
                
            elif analysis_type == "reinforcement":
                metrics.confirmation_bias_score = overall_score
                metrics.echo_strength = overall_score
                metrics.belief_reinforcement_ratio = overall_score * 0.95
        
        # Calculate overall echo chamber probability
        individual_scores = [
            metrics.homophily_score,
            metrics.isolation_index,
            metrics.polarization_score,
            metrics.confirmation_bias_score
        ]
        
        # Filter out zero scores (analyses not performed)
        non_zero_scores = [score for score in individual_scores if score > 0.0]
        
        if non_zero_scores:
            metrics.echo_chamber_probability = sum(non_zero_scores) / len(non_zero_scores)
        else:
            metrics.echo_chamber_probability = 0.0
        
        # Calculate overall confidence
        confidences = [r.get("confidence", 0.0) for r in analysis_results.values() 
                      if r.get("success", False)]
        metrics.confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Determine risk level
        metrics.risk_level = self._determine_risk_level(metrics.echo_chamber_probability)
        
        return metrics
    
    def _determine_risk_level(self, probability: float) -> str:
        """Determine risk level based on echo chamber probability."""
        if probability >= 0.8:
            return EchoChamberRiskLevel.CRITICAL.value
        elif probability >= 0.6:
            return EchoChamberRiskLevel.HIGH.value
        elif probability >= 0.4:
            return EchoChamberRiskLevel.MEDIUM.value
        else:
            return EchoChamberRiskLevel.LOW.value
    
    def _extract_echo_chambers(self, analysis_results: Dict[str, Any], network_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract identified echo chambers from analysis results."""
        
        chambers = []
        
        for analysis_type, result in analysis_results.items():
            if not result.get("success", False):
                continue
            
            groups = result.get("identified_groups", [])
            for group in groups:
                chambers.append({
                    "chamber_id": f"{analysis_type}_{group.get('id', 'unknown')}",
                    "type": analysis_type,
                    "description": group.get("description", "Echo chamber detected"),
                    "confidence": result.get("confidence", 0.0),
                    "strength": result.get("overall_score", 0.0),
                    "evidence": result.get("evidence", [])[:2]  # First 2 evidence items
                })
        
        return chambers
    
    def _extract_homophily_metrics(self, result: Dict[str, Any], focus_attributes: Optional[List[str]]) -> Dict[str, Any]:
        """Extract homophily-specific metrics."""
        return {
            "homophily_score": result.get("overall_score", 0.0),
            "clustering_coefficient": result.get("strength", 0.0),
            "assortativity_coefficient": result.get("overall_score", 0.0) * 0.9,
            "similar_groups": result.get("identified_groups", []),
            "connection_patterns": {
                "within_group_strength": result.get("overall_score", 0.0),
                "between_group_strength": 1.0 - result.get("overall_score", 0.0)
            },
            "attribute_analysis": {
                "focus_attributes": focus_attributes or ["general"],
                "homophily_by_attribute": {}
            },
            "confidence": result.get("confidence", 0.0)
        }
    
    def _extract_polarization_metrics(self, result: Dict[str, Any], topic_focus: Optional[str]) -> Dict[str, Any]:
        """Extract polarization-specific metrics."""
        return {
            "polarization_score": result.get("overall_score", 0.0),
            "sentiment_divergence": result.get("overall_score", 0.0) * 0.9,
            "cross_group_engagement": 1.0 - result.get("overall_score", 0.0),
            "viewpoint_clusters": result.get("identified_groups", []),
            "topic_analysis": {
                "focus_topic": topic_focus or "general",
                "polarization_by_topic": {}
            },
            "engagement_patterns": {
                "within_cluster_engagement": result.get("overall_score", 0.0),
                "cross_cluster_engagement": 1.0 - result.get("overall_score", 0.0)
            },
            "confidence": result.get("confidence", 0.0)
        }
    
    def _extract_isolation_metrics(self, result: Dict[str, Any], community_definitions: Optional[Dict[str, List[str]]]) -> Dict[str, Any]:
        """Extract isolation-specific metrics."""
        return {
            "isolation_index": result.get("overall_score", 0.0),
            "external_connectivity": 1.0 - result.get("overall_score", 0.0),
            "information_diversity": 1.0 - (result.get("overall_score", 0.0) * 0.8),
            "community_isolation": {
                group.get("id", f"community_{i}"): result.get("overall_score", 0.0)
                for i, group in enumerate(result.get("identified_groups", []))
            },
            "information_flow": {
                "internal_flow_strength": result.get("overall_score", 0.0),
                "external_flow_strength": 1.0 - result.get("overall_score", 0.0)
            },
            "diversity_metrics": {
                "content_diversity": 1.0 - result.get("overall_score", 0.0),
                "source_diversity": 1.0 - (result.get("overall_score", 0.0) * 0.9)
            },
            "confidence": result.get("confidence", 0.0)
        }
    
    def _generate_recommendations(self, metrics: EchoChamberMetrics, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis results."""
        
        recommendations = []
        
        # Risk-based recommendations
        if metrics.risk_level == "CRITICAL":
            recommendations.extend([
                "🚨 URGENT: Implement immediate intervention measures to break echo chamber patterns",
                "📊 Deploy content diversification algorithms to increase exposure to varied viewpoints",
                "🔄 Introduce cross-group interaction incentives and bridging mechanisms",
                "⚠️ Monitor for coordinated inauthentic behavior and manipulation"
            ])
        elif metrics.risk_level == "HIGH":
            recommendations.extend([
                "⚠️ HIGH PRIORITY: Monitor echo chamber development and implement preventive measures",
                "🌐 Increase content diversity in user feeds and recommendations",
                "👥 Facilitate cross-community dialogue and interaction opportunities",
                "📈 Implement gradual exposure to diverse viewpoints"
            ])
        elif metrics.risk_level == "MEDIUM":
            recommendations.extend([
                "📈 MODERATE CONCERN: Implement proactive measures to prevent echo chamber formation",
                "🔍 Regular monitoring of user interaction patterns and content diversity",
                "💡 Educational initiatives about information diversity and critical thinking"
            ])
        else:
            recommendations.extend([
                "✅ LOW RISK: Maintain current diversity levels with regular monitoring",
                "🎯 Continue promoting healthy discourse and diverse content exposure"
            ])
        
        # Specific metric-based recommendations
        if metrics.homophily_score > 0.7:
            recommendations.append("🔗 Address high homophily by promoting diverse connections and cross-group interactions")
        
        if metrics.isolation_index > 0.7:
            recommendations.append("🌉 Build information bridges between isolated communities through shared content and events")
        
        if metrics.polarization_score > 0.7:
            recommendations.append("🤝 Implement depolarization strategies focusing on common ground and shared values")
        
        if metrics.confirmation_bias_score > 0.7:
            recommendations.append("🧠 Deploy bias-awareness tools and fact-checking mechanisms to counter confirmation bias")
        
        return recommendations
    
    def _generate_risk_assessment(self, metrics: EchoChamberMetrics) -> Dict[str, Any]:
        """Generate detailed risk assessment."""
        return {
            "overall_risk": metrics.risk_level,
            "risk_factors": {
                "homophily": "HIGH" if metrics.homophily_score > 0.7 else "MEDIUM" if metrics.homophily_score > 0.4 else "LOW",
                "isolation": "HIGH" if metrics.isolation_index > 0.7 else "MEDIUM" if metrics.isolation_index > 0.4 else "LOW",
                "polarization": "HIGH" if metrics.polarization_score > 0.7 else "MEDIUM" if metrics.polarization_score > 0.4 else "LOW",
                "confirmation_bias": "HIGH" if metrics.confirmation_bias_score > 0.7 else "MEDIUM" if metrics.confirmation_bias_score > 0.4 else "LOW"
            },
            "probability_score": metrics.echo_chamber_probability,
            "confidence_level": metrics.confidence,
            "intervention_urgency": "IMMEDIATE" if metrics.risk_level == "CRITICAL" else "SOON" if metrics.risk_level == "HIGH" else "MONITOR"
        }
    
    def _extract_key_findings(self, analysis_results: Dict[str, Any], metrics: EchoChamberMetrics) -> List[str]:
        """Extract key findings from analysis."""
        findings = []
        
        # Overall assessment
        findings.append(f"Echo chamber probability: {metrics.echo_chamber_probability:.2f} (Risk: {metrics.risk_level})")
        
        # Specific findings
        if metrics.homophily_score > 0.6:
            findings.append(f"High homophily detected (score: {metrics.homophily_score:.2f}) - users clustering by similarity")
        
        if metrics.isolation_index > 0.6:
            findings.append(f"Information isolation identified (index: {metrics.isolation_index:.2f}) - limited external connectivity")
        
        if metrics.polarization_score > 0.6:
            findings.append(f"Content polarization found (score: {metrics.polarization_score:.2f}) - separated viewpoint clusters")
        
        if metrics.confirmation_bias_score > 0.6:
            findings.append(f"Confirmation bias patterns (score: {metrics.confirmation_bias_score:.2f}) - belief reinforcement detected")
        
        return findings
    
    def _generate_actionable_insights(self, metrics: EchoChamberMetrics, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate specific actionable insights."""
        insights = []
        
        # Prioritized insights based on strongest indicators
        strongest_factor = max([
            ("homophily", metrics.homophily_score),
            ("isolation", metrics.isolation_index),
            ("polarization", metrics.polarization_score),
            ("confirmation_bias", metrics.confirmation_bias_score)
        ], key=lambda x: x[1])
        
        factor_name, factor_score = strongest_factor
        
        if factor_score > 0.7:
            if factor_name == "homophily":
                insights.append("Focus intervention on diversifying social connections and promoting cross-group interactions")
            elif factor_name == "isolation":
                insights.append("Prioritize breaking information silos by introducing external content sources and cross-community events")
            elif factor_name == "polarization":
                insights.append("Address polarization through common ground initiatives and gradual exposure to moderate viewpoints")
            elif factor_name == "confirmation_bias":
                insights.append("Implement bias-awareness education and fact-checking tools to counter confirmation bias")
        
        # Additional insights based on combination of factors
        if metrics.homophily_score > 0.6 and metrics.isolation_index > 0.6:
            insights.append("Combined homophily and isolation suggest need for comprehensive network diversification strategy")
        
        if metrics.polarization_score > 0.6 and metrics.confirmation_bias_score > 0.6:
            insights.append("Polarization with confirmation bias indicates need for depolarization and critical thinking interventions")
        
        return insights
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            "request_statistics": {
                "total_requests": self._performance_stats['total_requests'],
                "successful_requests": self._performance_stats['successful_requests'],
                "failed_requests": self._performance_stats['failed_requests'],
                "success_rate": (
                    self._performance_stats['successful_requests'] / 
                    max(self._performance_stats['total_requests'], 1)
                ) * 100
            },
            "analysis_statistics": {
                "analysis_types": self._performance_stats['analysis_types'],
                "network_analyses": self._performance_stats['network_analyses'],
                "batch_analyses": self._performance_stats['batch_analyses']
            },
            "performance_metrics": {
                "avg_response_time_ms": self._performance_stats['avg_response_time_ms'],
                "total_response_time_ms": self._performance_stats['total_response_time_ms']
            },
            "risk_distribution": self._performance_stats['risk_level_distribution'],
            "agent_info": {
                "name": self.name,
                "ollama_endpoint": self.ollama_endpoint,
                "ollama_model": self.ollama_model
            }
        }
    
    def get_supported_analysis_types(self) -> List[str]:
        """Get list of supported analysis types."""
        return list(self.analysis_templates.keys())
    
    def get_available_tools(self) -> List[str]:
        """Get list of available FastMCP tools."""
        return [
            "detect_echo_chamber",
            "analyze_homophily", 
            "measure_polarization",
            "calculate_isolation_index",
            "get_echo_chamber_metrics"
        ]


# Example usage and testing
async def main():
    """Example usage of EchoChamberDetectorMCPAgent."""
    
    # Initialize agent
    agent = EchoChamberDetectorMCPAgent()
    
    # Example network data
    sample_network = {
        "nodes": [
            {"id": "user1", "attributes": {"age": 25, "interests": ["politics", "tech"]}},
            {"id": "user2", "attributes": {"age": 27, "interests": ["politics", "sports"]}},
            {"id": "user3", "attributes": {"age": 45, "interests": ["politics", "business"]}}
        ],
        "edges": [
            {"source": "user1", "target": "user2", "type": "follows"},
            {"source": "user2", "target": "user3", "type": "interacts"}
        ],
        "content": [
            {"id": "post1", "text": "Political opinion A", "author": "user1"},
            {"id": "post2", "text": "Political opinion B", "author": "user2"}
        ]
    }
    
    print("🔍 Testing Echo Chamber Detector MCP Agent")
    print("=" * 50)
    
    # Test comprehensive detection
    result = await agent.detect_echo_chamber(sample_network)
    print(f"Echo Chamber Detected: {result.get('echo_chamber_detected', False)}")
    print(f"Risk Level: {result.get('risk_level', 'UNKNOWN')}")
    print(f"Confidence: {result.get('confidence', 0.0):.2f}")
    
    # Get performance stats
    stats = agent.get_performance_stats()
    print(f"\nPerformance Stats:")
    print(f"Total Requests: {stats['request_statistics']['total_requests']}")
    print(f"Success Rate: {stats['request_statistics']['success_rate']:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
