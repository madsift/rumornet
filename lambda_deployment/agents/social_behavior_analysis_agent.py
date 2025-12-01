"""
SocialBehaviorAnalysisAgent - Consolidated agent for social/user behavior analysis.

Consolidates functionality from:
- CoordinationDetectorAgent
- EchoChamberDetectorAgent  
- (Future: TopicSocialAnalysisAgent)

Provides unified interface for detecting coordinated behavior, echo chambers,
and social network patterns in misinformation campaigns.
"""

import asyncio
import logging
import time
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class SocialBehaviorConfig:
    """Configuration for social behavior analysis."""
    # Detection thresholds
    coordination_threshold: float = 0.7
    echo_chamber_threshold: float = 0.6
    bot_likelihood_threshold: float = 0.6
    
    # Analysis parameters
    min_users_for_analysis: int = 3
    temporal_window_hours: int = 24


class SocialBehaviorAnalysisAgent:
    """
    Consolidated agent for social and user behavior analysis.
    
    Provides tools for:
    - Coordination detection (temporal, content, network)
    - Echo chamber identification
    - Bot network detection
    - Social network analysis
    """
    
    def __init__(self, name: str = "social_behavior_analysis", config: Optional[Dict[str, Any]] = None):
        """Initialize social behavior analysis agent."""
        self.name = name
        self.config_dict = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.behavior_config = SocialBehaviorConfig()
        
        # Analysis caches
        self.user_behavior_cache: Dict[str, Dict[str, Any]] = {}
        self.coordination_cache: Dict[str, Any] = {}
        
        # Performance tracking
        self.analysis_stats = {
            "coordination_detections": 0,
            "echo_chamber_detections": 0,
            "bot_network_detections": 0,
            "total_analyses": 0
        }
    
    async def detect_coordination(
        self,
        events: List[Dict[str, Any]],
        analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
            """
            Detect coordinated behavior from posting events.
            
            Args:
                events: List of posting events with user_id, timestamp, content
                analysis_type: Type of analysis (temporal, content, network, comprehensive)
            
            Returns:
                Coordination detection results with confidence scores
            """
            start_time = time.time()
            
            try:
                if not events or len(events) < self.behavior_config.min_users_for_analysis:
                    return {
                        "success": False,
                        "error": f"Need at least {self.behavior_config.min_users_for_analysis} events",
                        "coordination_detected": False
                    }
                
                # Analyze coordination
                result = await self._detect_coordination_internal(events, analysis_type)
                
                self.analysis_stats["coordination_detections"] += 1
                self.analysis_stats["total_analyses"] += 1
                
                processing_time = (time.time() - start_time) * 1000
                result["processing_time_ms"] = processing_time
                result["success"] = True
                
                return result
                
            except Exception as e:
                self.logger.error(f"Coordination detection failed: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "coordination_detected": False
                }
    
    async def detect_echo_chamber(
        self,
        user_network: Dict[str, List[str]],
        user_content: Dict[str, List[str]]
    ) -> Dict[str, Any]:
            """
            Detect echo chambers in user networks.
            
            Args:
                user_network: Dictionary mapping user_id to list of connected users
                user_content: Dictionary mapping user_id to list of their content
            
            Returns:
                Echo chamber detection results with identified chambers
            """
            start_time = time.time()
            
            try:
                if not user_network or len(user_network) < self.behavior_config.min_users_for_analysis:
                    return {
                        "success": False,
                        "error": f"Need at least {self.behavior_config.min_users_for_analysis} users",
                        "echo_chamber_detected": False
                    }
                
                # Analyze echo chambers
                result = await self._detect_echo_chamber_internal(user_network, user_content)
                
                self.analysis_stats["echo_chamber_detections"] += 1
                self.analysis_stats["total_analyses"] += 1
                
                processing_time = (time.time() - start_time) * 1000
                result["processing_time_ms"] = processing_time
                result["success"] = True
                
                return result
                
            except Exception as e:
                self.logger.error(f"Echo chamber detection failed: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "echo_chamber_detected": False
                }
    
    async def detect_bot_network(
        self,
        user_ids: List[str],
        user_behavior_data: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
            """
            Detect bot networks from user behavior patterns.
            
            Args:
                user_ids: List of user IDs to analyze
                user_behavior_data: Optional behavioral data for each user
            
            Returns:
                Bot network detection results with likelihood scores
            """
            start_time = time.time()
            
            try:
                if not user_ids or len(user_ids) < 2:
                    return {
                        "success": False,
                        "error": "Need at least 2 users for bot network analysis",
                        "bot_network_detected": False
                    }
                
                # Analyze bot networks
                result = await self._detect_bot_network_internal(user_ids, user_behavior_data)
                
                self.analysis_stats["bot_network_detections"] += 1
                self.analysis_stats["total_analyses"] += 1
                
                processing_time = (time.time() - start_time) * 1000
                result["processing_time_ms"] = processing_time
                result["success"] = True
                
                return result
                
            except Exception as e:
                self.logger.error(f"Bot network detection failed: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "bot_network_detected": False
                }
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for social behavior analysis."""
        return {
            "success": True,
            "stats": self.analysis_stats.copy()
        }
    
    async def _detect_coordination_internal(
        self, 
        events: List[Dict[str, Any]], 
        analysis_type: str
    ) -> Dict[str, Any]:
        """Internal coordination detection logic."""
        
        # Group events by user
        user_events = defaultdict(list)
        for event in events:
            user_id = event.get("user_id", "unknown")
            user_events[user_id].append(event)
        
        # Calculate coordination scores
        coordination_score = 0.0
        evidence = []
        
        # Temporal analysis
        if analysis_type in ["temporal", "comprehensive"]:
            temporal_score = self._analyze_temporal_patterns(user_events)
            coordination_score = max(coordination_score, temporal_score)
            if temporal_score > self.behavior_config.coordination_threshold:
                evidence.append(f"High temporal coordination: {temporal_score:.2f}")
        
        # Content analysis
        if analysis_type in ["content", "comprehensive"]:
            content_score = self._analyze_content_similarity(events)
            coordination_score = max(coordination_score, content_score)
            if content_score > self.behavior_config.coordination_threshold:
                evidence.append(f"High content similarity: {content_score:.2f}")
        
        coordination_detected = coordination_score > self.behavior_config.coordination_threshold
        
        return {
            "coordination_detected": coordination_detected,
            "coordination_score": coordination_score,
            "coordination_type": analysis_type,
            "evidence": evidence,
            "users_analyzed": len(user_events),
            "events_analyzed": len(events)
        }
    
    async def _detect_echo_chamber_internal(
        self, 
        user_network: Dict[str, List[str]], 
        user_content: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Internal echo chamber detection logic."""
        
        # Calculate network density
        total_connections = sum(len(connections) for connections in user_network.values())
        max_connections = len(user_network) * (len(user_network) - 1)
        network_density = total_connections / max_connections if max_connections > 0 else 0
        
        # Calculate content homogeneity
        content_similarity = self._calculate_content_homogeneity(user_content)
        
        # Echo chamber score
        echo_score = (network_density * 0.5 + content_similarity * 0.5)
        echo_detected = echo_score > self.behavior_config.echo_chamber_threshold
        
        # Identify chambers (simplified clustering)
        chambers = []
        if echo_detected:
            chambers.append({
                "chamber_id": "chamber_1",
                "members": list(user_network.keys()),
                "density": network_density,
                "homogeneity": content_similarity
            })
        
        return {
            "echo_chamber_detected": echo_detected,
            "echo_chamber_score": echo_score,
            "network_density": network_density,
            "content_homogeneity": content_similarity,
            "chambers_identified": chambers,
            "users_analyzed": len(user_network)
        }
    
    async def _detect_bot_network_internal(
        self, 
        user_ids: List[str], 
        user_behavior_data: Optional[Dict[str, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Internal bot network detection logic."""
        
        bot_scores = {}
        suspected_bots = []
        
        for user_id in user_ids:
            # Calculate bot likelihood score
            bot_score = self._calculate_bot_likelihood(user_id, user_behavior_data)
            bot_scores[user_id] = bot_score
            
            if bot_score > self.behavior_config.bot_likelihood_threshold:
                suspected_bots.append(user_id)
        
        bot_network_detected = len(suspected_bots) >= 2
        
        return {
            "bot_network_detected": bot_network_detected,
            "suspected_bots": suspected_bots,
            "bot_likelihood_scores": bot_scores,
            "users_analyzed": len(user_ids),
            "network_coordination_score": len(suspected_bots) / len(user_ids) if user_ids else 0
        }
    
    def _analyze_temporal_patterns(self, user_events: Dict[str, List[Dict]]) -> float:
        """Analyze temporal coordination patterns."""
        if len(user_events) < 2:
            return 0.0
        
        # Simplified: Check if users post within similar time windows
        all_timestamps = []
        for events in user_events.values():
            for event in events:
                ts = event.get("timestamp")
                if isinstance(ts, str):
                    try:
                        ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                        all_timestamps.append(ts)
                    except:
                        pass
        
        if len(all_timestamps) < 2:
            return 0.0
        
        # Calculate temporal clustering (simplified)
        all_timestamps.sort()
        time_diffs = []
        for i in range(len(all_timestamps) - 1):
            diff = (all_timestamps[i+1] - all_timestamps[i]).total_seconds()
            time_diffs.append(diff)
        
        if not time_diffs:
            return 0.0
        
        # High coordination = many posts in short time windows
        avg_diff = sum(time_diffs) / len(time_diffs)
        coordination_score = 1.0 / (1.0 + avg_diff / 3600.0)  # Normalize by hours
        
        return min(1.0, coordination_score)
    
    def _analyze_content_similarity(self, events: List[Dict[str, Any]]) -> float:
        """Analyze content similarity across events."""
        contents = [e.get("content", "") for e in events if e.get("content")]
        
        if len(contents) < 2:
            return 0.0
        
        # Simplified: Calculate content hash similarity
        content_hashes = set()
        for content in contents:
            content_hash = hashlib.md5(content.encode()).hexdigest()
            content_hashes.add(content_hash)
        
        # High similarity = few unique hashes
        similarity = 1.0 - (len(content_hashes) / len(contents))
        return similarity
    
    def _calculate_content_homogeneity(self, user_content: Dict[str, List[str]]) -> float:
        """Calculate content homogeneity across users."""
        all_content = []
        for content_list in user_content.values():
            all_content.extend(content_list)
        
        if len(all_content) < 2:
            return 0.0
        
        # Simplified: Check for repeated content
        unique_content = set(all_content)
        homogeneity = 1.0 - (len(unique_content) / len(all_content))
        return homogeneity
    
    def _calculate_bot_likelihood(
        self, 
        user_id: str, 
        user_behavior_data: Optional[Dict[str, Dict[str, Any]]]
    ) -> float:
        """Calculate bot likelihood score for a user."""
        if not user_behavior_data or user_id not in user_behavior_data:
            return 0.5  # Unknown
        
        behavior = user_behavior_data[user_id]
        
        # Bot indicators (simplified)
        bot_score = 0.0
        
        # High posting frequency
        posting_freq = behavior.get("posting_frequency", 0)
        if posting_freq > 10:  # More than 10 posts per day
            bot_score += 0.3
        
        # Uniform intervals
        interval_variance = behavior.get("interval_variance", 1.0)
        if interval_variance < 0.1:  # Very uniform
            bot_score += 0.3
        
        # Content repetition
        content_uniqueness = behavior.get("content_uniqueness", 1.0)
        if content_uniqueness < 0.3:  # Lots of repetition
            bot_score += 0.4
        
        return min(1.0, bot_score)


# Factory function for easy instantiation
def create_social_behavior_agent(config: Optional[Dict[str, Any]] = None) -> SocialBehaviorAnalysisAgent:
    """Create and return a social behavior analysis agent."""
    return SocialBehaviorAnalysisAgent(config=config)
