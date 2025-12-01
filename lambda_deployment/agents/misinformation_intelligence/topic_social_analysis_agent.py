"""
Topic Social Analysis Agent - Unified social behavior analysis.

Consolidates functionality from:
- TopicNetworkAnalyzer (network analysis, communities, echo chambers)
- TopicCredibilityEngine (user credibility assessment)
- TopicBasedEarlyWarning (real-time monitoring, alerts)

Provides comprehensive social analysis with shared caching, integrated workflows,
and optimized performance.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from dataclasses import dataclass, field

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
from .core.data_models import (
    TopicIntelligence, NetworkNode, NetworkEdge, NetworkCommunity,
    TopicCommunityAnalysis, EchoChamber, BridgeInfluencer,
    EarlyWarningAlert, AlertType, AlertRiskLevel, VelocityMetrics,
    CoordinationIndicators
)


@dataclass
class TopicSocialAnalysisConfig:
    """Configuration for unified social analysis agent."""
    agent_name: str = "topic_social_analysis_agent"
    
    # Network analysis parameters
    min_community_size: int = 5
    echo_chamber_threshold: float = 0.8
    bridge_centrality_threshold: float = 0.6
    
    # Credibility parameters
    historical_accuracy_weight: float = 0.4
    network_authority_weight: float = 0.3
    coordination_risk_weight: float = 0.3
    
    # Early warning parameters
    velocity_spike_threshold: float = 5.0
    coordination_threshold: float = 0.8
    viral_potential_threshold: float = 0.7
    monitoring_window_hours: int = 6
    



class TopicSocialAnalysisAgent:
    """
    Unified agent for social behavior analysis.
    
    Provides comprehensive tools for:
    - Network analysis (communities, echo chambers, influence propagation)
    - Credibility assessment (user scoring, coordination risk)
    - Early warning (emergence detection, velocity spikes, alerts)
    """
    
    def __init__(self, name: str = "topic_social_analysis", config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Shared caches for performance optimization
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        self.network_cache: Dict[str, Any] = {}
        self.credibility_cache: Dict[Tuple[str, str], float] = {}  # (user_id, topic_id) -> score
        self.community_cache: Dict[str, List[NetworkCommunity]] = {}
        
        # Monitoring state
        self.monitored_topics: Dict[str, Dict[str, Any]] = {}
        self.recent_alerts: Dict[str, datetime] = {}
        
        # Performance tracking
        self.analysis_stats = {
            "network_analyses": 0,
            "credibility_assessments": 0,
            "early_warnings": 0,
            "communities_detected": 0,
            "echo_chambers_found": 0,
            "alerts_generated": 0
        }
        
        # Initialize Ollama for LLM reasoning
        self._initialize_ollama()
        
        self._initialize_ollama()
        
        self.logger.info(f"Initialized Topic Social Analysis agent: {name}")
    
    def _initialize_ollama(self):
        """Initialize Ollama client for LLM-powered analysis."""
        try:
            from agents.common.agent_config import AgentConfig, get_llm_agent
            
            agent_config = AgentConfig.from_dict(self.config)
            self.ollama_agent = get_llm_agent(agent_config)
            
            if self.ollama_agent:
                self.logger.info("Initialized Ollama for social analysis")
            else:
                self.logger.warning("Failed to initialize Ollama")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Ollama: {e}")
            self.ollama_agent = None
    
    # ============================================================
    # NETWORK ANALYSIS METHODS
    # ============================================================
    
    async def analyze_topic_communities(
        self,
        topic_id: str,
        network_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
            """
            Analyze communities around specific topics.
            
            Args:
                topic_id: ID of the topic to analyze
                network_data: Optional network graph data
                
            Returns:
                Community analysis with detected communities and metrics
            """
            start_time = time.time()
            
            try:
                # Validate input
                if not self._validate_input(topic_id):
                    return self._create_error_response("Invalid topic_id", "INVALID_INPUT")
                
                # Check cache
                cache_key = f"communities_{topic_id}"
                if cache_key in self.community_cache:
                    self.performance_stats["cache_hits"] += 1
                    return self._create_success_response(
                        {"communities": [c.to_dict() for c in self.community_cache[cache_key]],
                         "cached": True}
                    )
                
                # Perform community detection
                communities = await self._detect_communities(topic_id, network_data)
                
                # Cache results
                self.community_cache[cache_key] = communities
                
                # Update stats
                self.analysis_stats["network_analyses"] += 1
                self.analysis_stats["communities_detected"] += len(communities)
                
                processing_time = (time.time() - start_time) * 1000
                
                return self._create_success_response(
                    {
                        "topic_id": topic_id,
                        "communities": [c.to_dict() for c in communities],
                        "total_communities": len(communities),
                        "processing_time_ms": processing_time
                    }
                )
                
            except Exception as e:
                self.logger.error(f"Community analysis failed: {e}")
                return self._create_error_response(str(e), "ANALYSIS_ERROR")
        
        async def detect_echo_chambers(self, 
            topic_id: str,
            network_data: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            """
            Detect echo chambers within topic networks.
            
            Args:
                topic_id: ID of the topic to analyze
                network_data: Optional network graph data
                
            Returns:
                Echo chamber analysis with homophily scores and isolation metrics
            """
            start_time = time.time()
            
            try:
                # Detect echo chambers
                echo_chambers = await self._detect_echo_chambers_internal(topic_id, network_data)
                
                # Update stats
                self.analysis_stats["echo_chambers_found"] += len(echo_chambers)
                
                processing_time = (time.time() - start_time) * 1000
                
                return self._create_success_response(
                    {
                        "topic_id": topic_id,
                        "echo_chambers": [ec.to_dict() for ec in echo_chambers],
                        "total_echo_chambers": len(echo_chambers),
                        "processing_time_ms": processing_time
                    }
                )
                
            except Exception as e:
                self.logger.error(f"Echo chamber detection failed: {e}")
                return self._create_error_response(str(e), "DETECTION_ERROR")
        
        async def identify_bridge_influencers(self, 
            topic_id: str,
            network_data: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            """
            Identify bridge influencers connecting different communities.
            
            Args:
                topic_id: ID of the topic to analyze
                network_data: Optional network graph data
                
            Returns:
                List of bridge influencers with centrality scores
            """
            start_time = time.time()
            
            try:
                # Identify bridges
                bridges = await self._identify_bridges_internal(topic_id, network_data)
                
                processing_time = (time.time() - start_time) * 1000
                
                return self._create_success_response(
                    {
                        "topic_id": topic_id,
                        "bridge_influencers": [b.to_dict() for b in bridges],
                        "total_bridges": len(bridges),
                        "processing_time_ms": processing_time
                    }
                )
                
            except Exception as e:
                self.logger.error(f"Bridge identification failed: {e}")
                return self._create_error_response(str(e), "IDENTIFICATION_ERROR")
        
        # ============================================================
        # CREDIBILITY ASSESSMENT TOOLS
        # ============================================================
        
        async def calculate_topic_credibility(self, 
            user_id: str,
            topic_id: str,
            user_profile: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            """
            Calculate topic-specific credibility score for a user.
            
            Args:
                user_id: User identifier
                topic_id: Topic identifier
                user_profile: Optional user profile data
                
            Returns:
                Credibility score with explanation and contributing factors
            """
            start_time = time.time()
            
            try:
                # Check cache
                cache_key = (user_id, topic_id)
                if cache_key in self.credibility_cache:
                    self.performance_stats["cache_hits"] += 1
                    return self._create_success_response(
                        {"credibility_score": self.credibility_cache[cache_key], "cached": True}
                    )
                
                # Calculate credibility
                credibility_result = await self._calculate_credibility_internal(
                    user_id, topic_id, user_profile
                )
                
                # Cache result
                self.credibility_cache[cache_key] = credibility_result["score"]
                
                # Update stats
                self.analysis_stats["credibility_assessments"] += 1
                
                processing_time = (time.time() - start_time) * 1000
                credibility_result["processing_time_ms"] = processing_time
                
                return self._create_success_response(credibility_result)
                
            except Exception as e:
                self.logger.error(f"Credibility calculation failed: {e}")
                return self._create_error_response(str(e), "CREDIBILITY_ERROR")
        
        async def assess_coordination_risk(self, 
            user_id: str,
            topic_id: str,
            behavioral_data: Optional[List[Dict[str, Any]]] = None
        ) -> Dict[str, Any]:
            """
            Assess coordination risk for bot detection.
            
            Args:
                user_id: User identifier
                topic_id: Topic identifier
                behavioral_data: Optional behavioral history
                
            Returns:
                Coordination risk assessment with indicators
            """
            start_time = time.time()
            
            try:
                # Assess coordination risk
                risk_assessment = await self._assess_coordination_risk_internal(
                    user_id, topic_id, behavioral_data
                )
                
                processing_time = (time.time() - start_time) * 1000
                risk_assessment["processing_time_ms"] = processing_time
                
                return self._create_success_response(risk_assessment)
                
            except Exception as e:
                self.logger.error(f"Coordination risk assessment failed: {e}")
                return self._create_error_response(str(e), "RISK_ASSESSMENT_ERROR")
        
        # ============================================================
        # EARLY WARNING TOOLS
        # ============================================================
        
        async def analyze_topic_emergence(self, 
            topic_intelligence: Dict[str, Any]
        ) -> Dict[str, Any]:
            """
            Analyze topic for emergence patterns and generate alerts.
            
            Args:
                topic_intelligence: Topic intelligence data with velocity metrics
                
            Returns:
                Early warning alerts for emerging threats
            """
            start_time = time.time()
            
            try:
                # Analyze emergence
                alerts = await self._analyze_emergence_internal(topic_intelligence)
                
                # Update stats
                self.analysis_stats["early_warnings"] += 1
                self.analysis_stats["alerts_generated"] += len(alerts)
                
                processing_time = (time.time() - start_time) * 1000
                
                return self._create_success_response(
                    {
                        "alerts": [a.to_dict() for a in alerts],
                        "total_alerts": len(alerts),
                        "processing_time_ms": processing_time
                    }
                )
                
            except Exception as e:
                self.logger.error(f"Emergence analysis failed: {e}")
                return self._create_error_response(str(e), "EMERGENCE_ERROR")
        
        async def detect_velocity_spikes(self, 
            topic_id: str,
            velocity_data: Dict[str, Any]
        ) -> Dict[str, Any]:
            """
            Detect velocity spikes indicating viral spread.
            
            Args:
                topic_id: Topic identifier
                velocity_data: Velocity metrics over time
                
            Returns:
                Spike detection results with severity assessment
            """
            start_time = time.time()
            
            try:
                # Detect spikes
                spike_analysis = await self._detect_velocity_spikes_internal(
                    topic_id, velocity_data
                )
                
                processing_time = (time.time() - start_time) * 1000
                spike_analysis["processing_time_ms"] = processing_time
                
                return self._create_success_response(spike_analysis)
                
            except Exception as e:
                self.logger.error(f"Velocity spike detection failed: {e}")
                return self._create_error_response(str(e), "SPIKE_DETECTION_ERROR")
        
        async def predict_viral_potential(self, 
            topic_id: str,
            topic_intelligence: Dict[str, Any]
        ) -> Dict[str, Any]:
            """
            Predict viral potential of a topic using Ollama analysis.
            
            Args:
                topic_id: Topic identifier
                topic_intelligence: Topic intelligence with velocity and network data
                
            Returns:
                Viral potential prediction with confidence score
            """
            start_time = time.time()
            
            try:
                # Predict viral potential using Ollama
                prediction = await self._predict_viral_potential_internal(
                    topic_id, topic_intelligence
                )
                
                processing_time = (time.time() - start_time) * 1000
                prediction["processing_time_ms"] = processing_time
                
                return self._create_success_response(prediction)
                
            except Exception as e:
                self.logger.error(f"Viral potential prediction failed: {e}")
                return self._create_error_response(str(e), "PREDICTION_ERROR")
        
        # ============================================================
        # INTEGRATED WORKFLOW TOOLS
        # ============================================================
        
        async def comprehensive_social_analysis(self, 
            topic_id: str,
            topic_intelligence: Dict[str, Any],
            network_data: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            """
            Perform comprehensive social analysis combining all capabilities.
            
            This integrated workflow:
            1. Analyzes network communities
            2. Detects echo chambers
            3. Identifies bridge influencers
            4. Assesses credibility patterns
            5. Generates early warning alerts
            
            Args:
                topic_id: Topic identifier
                topic_intelligence: Topic intelligence data
                network_data: Optional network graph data
                
            Returns:
                Comprehensive analysis results from all components
            """
            start_time = time.time()
            
            try:
                results = {
                    "topic_id": topic_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # 1. Network analysis
                communities = await self._detect_communities(topic_id, network_data)
                results["communities"] = [c.to_dict() for c in communities]
                
                # 2. Echo chamber detection
                echo_chambers = await self._detect_echo_chambers_internal(topic_id, network_data)
                results["echo_chambers"] = [ec.to_dict() for ec in echo_chambers]
                
                # 3. Bridge influencers
                bridges = await self._identify_bridges_internal(topic_id, network_data)
                results["bridge_influencers"] = [b.to_dict() for b in bridges]
                
                # 4. Credibility patterns (aggregate)
                results["credibility_summary"] = await self._analyze_credibility_patterns(
                    topic_id, communities
                )
                
                # 5. Early warning alerts
                alerts = await self._analyze_emergence_internal(topic_intelligence)
                results["alerts"] = [a.to_dict() for a in alerts]
                
                # Overall risk assessment
                results["overall_risk_level"] = self._calculate_overall_risk(results)
                
                processing_time = (time.time() - start_time) * 1000
                results["processing_time_ms"] = processing_time
                
                return self._create_success_response(results)
                
            except Exception as e:
                self.logger.error(f"Comprehensive analysis failed: {e}")
                return self._create_error_response(str(e), "COMPREHENSIVE_ANALYSIS_ERROR")
    
    # ============================================================
    # INTERNAL IMPLEMENTATION METHODS
    # ============================================================
    
    async def _detect_communities(
        self, topic_id: str, network_data: Optional[Dict[str, Any]]
    ) -> List[NetworkCommunity]:
        """Internal community detection implementation using Louvain algorithm."""
        if not NETWORKX_AVAILABLE or not network_data:
            return []
        
        try:
            # Build NetworkX graph
            G = nx.Graph()
            
            # Add nodes
            for node in network_data.get("nodes", []):
                G.add_node(node["id"], **node)
            
            # Add edges
            for edge in network_data.get("edges", []):
                G.add_edge(edge["source"], edge["target"], weight=edge.get("weight", 1.0))
            
            if len(G.nodes()) < self.config.min_community_size:
                return []
            
            # Detect communities using NetworkX's greedy modularity algorithm
            # (alternative to Louvain when python-louvain not available)
            try:
                import community as community_louvain
                partition = community_louvain.best_partition(G)
            except ImportError:
                # Fallback to NetworkX's built-in community detection
                from networkx.algorithms import community as nx_community
                communities_generator = nx_community.greedy_modularity_communities(G)
                
                # Convert to partition format
                partition = {}
                for comm_id, comm_members in enumerate(communities_generator):
                    for member in comm_members:
                        partition[member] = comm_id
            
            # Group nodes by community
            communities_dict = defaultdict(list)
            for node, comm_id in partition.items():
                communities_dict[comm_id].append(node)
            
            # Build community objects
            communities = []
            for comm_id, members in communities_dict.items():
                if len(members) >= self.config.min_community_size:
                    # Calculate community metrics
                    subgraph = G.subgraph(members)
                    
                    # Create community as dict (NetworkCommunity dataclass may not match)
                    community = type('NetworkCommunity', (), {
                        'community_id': f"{topic_id}_comm_{comm_id}",
                        'members': members,
                        'size': len(members),
                        'density': nx.density(subgraph) if len(members) > 1 else 0.0,
                        'central_nodes': self._get_central_nodes(subgraph, top_k=5),
                        'to_dict': lambda self: {
                            'community_id': self.community_id,
                            'members': self.members,
                            'size': self.size,
                            'density': self.density,
                            'central_nodes': self.central_nodes
                        }
                    })()
                    communities.append(community)
            
            return communities
            
        except Exception as e:
            self.logger.error(f"Community detection failed: {e}")
            return []
    
    def _get_central_nodes(self, graph: 'nx.Graph', top_k: int = 5) -> List[str]:
        """Get most central nodes in a graph."""
        try:
            if len(graph.nodes()) == 0:
                return []
            
            # Calculate degree centrality
            centrality = nx.degree_centrality(graph)
            
            # Sort by centrality
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            
            return [node for node, _ in sorted_nodes[:top_k]]
            
        except Exception as e:
            self.logger.error(f"Central nodes calculation failed: {e}")
            return []
    
    async def _detect_echo_chambers_internal(
        self, topic_id: str, network_data: Optional[Dict[str, Any]]
    ) -> List[EchoChamber]:
        """Internal echo chamber detection implementation."""
        if not NETWORKX_AVAILABLE or not network_data:
            return []
        
        try:
            # First detect communities
            communities = await self._detect_communities(topic_id, network_data)
            
            # Build full graph
            G = nx.Graph()
            for node in network_data.get("nodes", []):
                G.add_node(node["id"], **node)
            for edge in network_data.get("edges", []):
                G.add_edge(edge["source"], edge["target"], weight=edge.get("weight", 1.0))
            
            echo_chambers = []
            
            for community in communities:
                # Calculate homophily (internal vs external connections)
                internal_edges = 0
                external_edges = 0
                
                for member in community.members:
                    if member not in G:
                        continue
                    
                    for neighbor in G.neighbors(member):
                        if neighbor in community.members:
                            internal_edges += 1
                        else:
                            external_edges += 1
                
                # Avoid division by zero
                total_edges = internal_edges + external_edges
                if total_edges == 0:
                    continue
                
                # Homophily score: ratio of internal to total connections
                homophily_score = internal_edges / total_edges if total_edges > 0 else 0
                
                # Echo chamber if homophily exceeds threshold
                if homophily_score >= self.config.echo_chamber_threshold:
                    # Calculate isolation (inverse of external connections)
                    isolation_score = 1.0 - (external_edges / total_edges) if total_edges > 0 else 1.0
                    
                    # Create echo chamber as simple object
                    echo_chamber = type('EchoChamber', (), {
                        'chamber_id': f"{topic_id}_echo_{community.community_id}",
                        'members': community.members,
                        'homophily_score': homophily_score,
                        'isolation_score': isolation_score,
                        'dominant_narratives': [],
                        'to_dict': lambda self: {
                            'chamber_id': self.chamber_id,
                            'members': self.members[:10],  # Limit for readability
                            'size': len(self.members),
                            'homophily_score': self.homophily_score,
                            'isolation_score': self.isolation_score
                        }
                    })()
                    echo_chambers.append(echo_chamber)
            
            return echo_chambers
            
        except Exception as e:
            self.logger.error(f"Echo chamber detection failed: {e}")
            return []
    
    async def _identify_bridges_internal(
        self, topic_id: str, network_data: Optional[Dict[str, Any]]
    ) -> List[BridgeInfluencer]:
        """Internal bridge influencer identification using betweenness centrality."""
        if not NETWORKX_AVAILABLE or not network_data:
            return []
        
        try:
            # Build graph
            G = nx.Graph()
            for node in network_data.get("nodes", []):
                G.add_node(node["id"], **node)
            for edge in network_data.get("edges", []):
                G.add_edge(edge["source"], edge["target"], weight=edge.get("weight", 1.0))
            
            if len(G.nodes()) < 3:
                return []
            
            # Calculate betweenness centrality (identifies bridges)
            betweenness = nx.betweenness_centrality(G)
            
            # Get communities for bridge detection
            communities = await self._detect_communities(topic_id, network_data)
            community_map = {}
            for comm in communities:
                for member in comm.members:
                    community_map[member] = comm.community_id
            
            bridges = []
            
            # Find nodes with high betweenness that connect different communities
            for node, centrality in betweenness.items():
                if centrality >= self.config.bridge_centrality_threshold:
                    # Find which communities this node connects
                    connected_communities = set()
                    if node in G:
                        for neighbor in G.neighbors(node):
                            if neighbor in community_map:
                                connected_communities.add(community_map[neighbor])
                    
                    # Bridge if connects 2+ communities
                    if len(connected_communities) >= 2:
                        # Create bridge influencer as simple object
                        bridge = type('BridgeInfluencer', (), {
                            'user_id': node,
                            'betweenness_centrality': centrality,
                            'communities_connected': list(connected_communities),
                            'bridge_strength': centrality * len(connected_communities),
                            'to_dict': lambda self: {
                                'user_id': self.user_id,
                                'betweenness_centrality': self.betweenness_centrality,
                                'communities_connected': self.communities_connected,
                                'bridge_strength': self.bridge_strength
                            }
                        })()
                        bridges.append(bridge)
            
            # Sort by bridge strength
            bridges.sort(key=lambda x: x.bridge_strength, reverse=True)
            
            return bridges[:20]  # Top 20 bridges
            
        except Exception as e:
            self.logger.error(f"Bridge identification failed: {e}")
            return []
    
    async def _calculate_credibility_internal(
        self, user_id: str, topic_id: str, user_profile: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Internal credibility calculation."""
        # Placeholder - implement actual credibility calculation
        return {
            "user_id": user_id,
            "topic_id": topic_id,
            "score": 0.75,
            "confidence": 0.8,
            "factors": {
                "historical_accuracy": 0.7,
                "network_authority": 0.8,
                "coordination_risk": 0.2
            }
        }
    
    async def _assess_coordination_risk_internal(
        self, user_id: str, topic_id: str, behavioral_data: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Internal coordination risk assessment."""
        # Placeholder - implement actual risk assessment
        return {
            "user_id": user_id,
            "topic_id": topic_id,
            "risk_score": 0.3,
            "indicators": [],
            "confidence": 0.7
        }
    
    async def _analyze_emergence_internal(
        self, topic_intelligence: Dict[str, Any]
    ) -> List[EarlyWarningAlert]:
        """Internal emergence analysis."""
        # Placeholder - implement actual emergence analysis
        return []
    
    async def _detect_velocity_spikes_internal(
        self, topic_id: str, velocity_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Internal velocity spike detection."""
        # Placeholder - implement actual spike detection
        return {
            "topic_id": topic_id,
            "spike_detected": False,
            "severity": "low",
            "confidence": 0.5
        }
    
    async def _predict_viral_potential_internal(
        self, topic_id: str, topic_intelligence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Internal viral potential prediction using Ollama."""
        try:
            # Check if Ollama is available
            if not self.ollama_agent:
                self.logger.warning("Ollama not available, using heuristic prediction")
                return self._heuristic_viral_prediction(topic_id, topic_intelligence)
            
            # Use Ollama for prediction
            velocity = topic_intelligence.get('velocity', 0)
            network_size = topic_intelligence.get('network_size', 0)
            engagement = topic_intelligence.get('engagement', 0)
            
            prompt = f"""Analyze the viral potential of this social media topic:

Topic ID: {topic_id}
Daily Velocity: {velocity:.1f} posts/day
Network Size: {network_size} users
Total Engagement: {engagement} interactions

Based on these metrics, predict:
1. Viral potential score (0.0-1.0)
2. Confidence in prediction (0.0-1.0)
3. Key factors driving virality
4. Risk assessment

Provide a brief analysis (2-3 sentences)."""
            
            response = await self._get_ollama_response(prompt)
            
            # Parse response for scores (simple heuristic if parsing fails)
            viral_score = self._extract_score_from_text(response, default=0.6)
            
            return {
                "topic_id": topic_id,
                "viral_potential": viral_score,
                "confidence": 0.75,
                "analysis": response,
                "key_factors": self._extract_factors(velocity, network_size, engagement)
            }
            
        except Exception as e:
            self.logger.error(f"Viral prediction failed: {e}")
            return self._heuristic_viral_prediction(topic_id, topic_intelligence)
    
    def _heuristic_viral_prediction(self, topic_id: str, topic_intelligence: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback heuristic viral prediction."""
        velocity = topic_intelligence.get('velocity', 0)
        network_size = topic_intelligence.get('network_size', 0)
        engagement = topic_intelligence.get('engagement', 0)
        
        # Simple heuristic: normalize and combine metrics
        velocity_score = min(velocity / 100, 1.0)  # Normalize to 0-1
        network_score = min(network_size / 1000, 1.0)
        engagement_score = min(engagement / 10000, 1.0)
        
        viral_potential = (velocity_score * 0.4 + network_score * 0.3 + engagement_score * 0.3)
        
        return {
            "topic_id": topic_id,
            "viral_potential": viral_potential,
            "confidence": 0.6,
            "analysis": f"Heuristic prediction based on velocity ({velocity:.1f}), network size ({network_size}), and engagement ({engagement})",
            "key_factors": self._extract_factors(velocity, network_size, engagement)
        }
    
    def _extract_score_from_text(self, text: str, default: float = 0.5) -> float:
        """Extract numerical score from LLM response."""
        import re
        # Look for patterns like "0.7" or "70%"
        matches = re.findall(r'(\d+\.?\d*)', text)
        if matches:
            score = float(matches[0])
            # Normalize if percentage
            if score > 1.0:
                score = score / 100.0
            return min(max(score, 0.0), 1.0)
        return default
    
    def _extract_factors(self, velocity: float, network_size: int, engagement: int) -> List[str]:
        """Extract key factors driving virality."""
        factors = []
        if velocity > 50:
            factors.append("high_velocity")
        if network_size > 500:
            factors.append("large_network")
        if engagement > 5000:
            factors.append("high_engagement")
        return factors if factors else ["moderate_activity"]
    
    async def _analyze_credibility_patterns(
        self, topic_id: str, communities: List[NetworkCommunity]
    ) -> Dict[str, Any]:
        """Analyze credibility patterns across communities."""
        # Placeholder - implement actual pattern analysis
        return {
            "average_credibility": 0.7,
            "high_credibility_users": 0,
            "low_credibility_users": 0,
            "coordination_risk_users": 0
        }
    
    def _calculate_overall_risk(self, analysis_results: Dict[str, Any]) -> str:
        """Calculate overall risk level from analysis results."""
        # Simple heuristic - implement more sophisticated logic
        alert_count = len(analysis_results.get("alerts", []))
        echo_chamber_count = len(analysis_results.get("echo_chambers", []))
        
        if alert_count > 5 or echo_chamber_count > 3:
            return "HIGH"
        elif alert_count > 2 or echo_chamber_count > 1:
            return "MEDIUM"
        else:
            return "LOW"
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = super().get_performance_stats()
        stats.update(self.analysis_stats)
        stats.update({
            "cache_sizes": {
                "user_profiles": len(self.user_profiles),
                "networks": len(self.network_cache),
                "credibility": len(self.credibility_cache),
                "communities": len(self.community_cache)
            }
        })
        return stats


# Entry point for FastMCP
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    agent = TopicSocialAnalysisAgent()
    
    print(f"Created {agent.name} agent")
    print("Agent ready for direct method calls")
    
    t
        agent.run()
    except KeyboardInterrupt:
        logging.info("Topic Social Analysis Agent stopped by user")
    except Exception as e:
        logging.error(f"Topic Social Analysis Agent failed: {e}")
        sys.exit(1)
