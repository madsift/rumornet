#!/usr/bin/env python3
"""
Granular Misinformation Orchestrator - CONCURRENT VERSION

Provides ACTIONABLE, DETAILED intelligence with WHO-WHAT-WHEN tracking.
Preserves all metadata and provides pinpoint accuracy for misinformation detection.

**CONCURRENT EXECUTION**: This version runs agents in parallel for maximum performance.
Use this version with AWS Bedrock or other providers that support concurrent requests.
For single-GPU Ollama setups, use the sequential version instead.

Key Features:
- Post-level metadata preservation (post_id, user_id, timestamp)
- User-level aggregation (who posted what)
- Temporal analysis (when patterns emerged)
- Granular pattern attribution (specific text snippets)
- Cross-agent integration (combined intelligence)
- Actionable reporting (specific posts/users to investigate)
- **CONCURRENT agent execution** (4x-5x faster than sequential)

Performance:
- Sequential: ~40-60 seconds per post
- Concurrent: ~10-15 seconds per post
- Speedup: 4x-5x faster!

Usage:
    # Same interface as sequential version
    from granular_misinformation_orchestrator_concurrent import GranularMisinformationOrchestrator
    
    config = {"llm_provider": "bedrock", ...}
    orchestrator = GranularMisinformationOrchestrator(config=config)
    await orchestrator.initialize_agents()
    result = await orchestrator.analyze_post_with_metadata(post)
"""

import asyncio
import json
import logging
import time
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add path for agent imports
sys.path.append('.')

@dataclass
class PostMetadata:
    """Complete metadata for a social media post."""
    post_id: str
    user_id: str
    username: str
    timestamp: str
    platform: str
    subreddit: Optional[str] = None
    upvotes: int = 0
    comments: int = 0
    shares: int = 0
    text: str = ""
    text_length: int = 0
    
    def __post_init__(self):
        self.text_length = len(self.text)

@dataclass
class MisinformationAnalysis:
    """Detailed misinformation analysis for a single post."""
    verdict: Optional[bool]  # True=true, False=false, None=uncertain
    confidence: float
    detected_language: str
    reasoning_chain: List[str]
    patterns_detected: List[Dict[str, Any]]
    severity_score: float
    risk_level: str  # CRITICAL, HIGH, MODERATE, LOW
    specific_examples: List[Dict[str, Any]]  # Specific text snippets
    manipulation_tactics: List[str]
    execution_time_ms: float
    error: Optional[str] = None

@dataclass
class UserProfile:
    """Aggregated profile for a user across all their posts."""
    user_id: str
    username: str
    total_posts: int = 0
    misinformation_posts: int = 0
    high_confidence_misinfo: int = 0  # confidence > 0.8
    avg_confidence: float = 0.0
    patterns_used: List[str] = field(default_factory=list)
    manipulation_tactics: List[str] = field(default_factory=list)
    languages_used: List[str] = field(default_factory=list)
    posts: List[Dict[str, Any]] = field(default_factory=list)
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    echo_chambers: List[str] = field(default_factory=list)
    
    @property
    def misinformation_rate(self) -> float:
        """Percentage of posts containing misinformation."""
        return (self.misinformation_posts / self.total_posts * 100) if self.total_posts > 0 else 0.0

@dataclass
class TemporalPattern:
    """Pattern occurrence over time."""
    pattern_name: str
    occurrences: List[Dict[str, Any]] = field(default_factory=list)
    first_occurrence: Optional[str] = None
    last_occurrence: Optional[str] = None
    peak_time: Optional[str] = None
    users_involved: List[str] = field(default_factory=list)

class GranularMisinformationOrchestrator:
    """
    Orchestrator that preserves metadata and provides granular, actionable intelligence.
    
    **CONCURRENT VERSION**: This version runs agents in parallel for maximum performance.
    
    This orchestrator wraps existing agents while maintaining WHO-WHAT-WHEN tracking
    to provide specific, actionable intelligence instead of vague summaries.
    
    Key Differences from Sequential Version:
    - Agents run concurrently using asyncio.gather()
    - 4x-5x faster analysis (10-15s vs 40-60s per post)
    - Requires provider that supports concurrent requests (Bedrock, OpenAI)
    - Not suitable for single-GPU Ollama setups
    
    Performance:
    - Sequential: ~40-60 seconds per post
    - Concurrent: ~10-15 seconds per post
    - Speedup: 4x-5x faster!
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize orchestrator with configuration.
        
        Supports multiple LLM providers:
        - Ollama (local): {"llm_provider": "ollama", "ollama_endpoint": "...", "ollama_model": "..."}
        - Bedrock (AWS): {"llm_provider": "bedrock", "bedrock_region": "...", "bedrock_model_id": "..."}
        - Backward compatible: {"ollama_endpoint": "...", "ollama_model": "..."} (defaults to Ollama)
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.orchestrator")
        
        # Agent storage
        self.agents = {}
        
        # Results storage with full metadata
        self.post_analyses: List[Dict[str, Any]] = []
        self.user_profiles: Dict[str, UserProfile] = {}
        self.temporal_patterns: Dict[str, TemporalPattern] = {}
        self.topic_analyses: Dict[str, Any] = {}  # Topic-level analysis
        
        # Use AgentConfig for flexible provider support
        from agents.common.agent_config import AgentConfig
        self.agent_config = AgentConfig.from_dict(self.config)
        
        # Log provider information
        provider = self.agent_config.llm_provider
        self.logger.info("🎯 Granular Misinformation Orchestrator initialized")
        self.logger.info(f"🤖 LLM Provider: {provider}")
        
        if provider == "ollama":
            self.logger.info(f"📡 Ollama: {self.agent_config.ollama_model} at {self.agent_config.ollama_endpoint}")
        elif provider == "bedrock":
            self.logger.info(f"☁️ Bedrock: {self.agent_config.bedrock_model_id} in {self.agent_config.bedrock_region}")
        
        self.logger.info("📊 Focus: WHO said WHAT WHEN with actionable intelligence")
    
    async def initialize_agents(self):
        """
        Initialize agents with flexible provider configuration.
        
        Passes full config to agents so they can use AgentConfig to select
        the appropriate provider (Ollama, Bedrock, etc.).
        """
        self.logger.info("Initializing agents...")
        
        # Pass full config to agents - they will use AgentConfig.from_dict() to extract what they need
        agent_config = self.config.copy()
        
        # Add default response language for multilingual agent
        if "default_response_language" not in agent_config:
            agent_config["default_response_language"] = "auto"
        
        # Multilingual KG Reasoning Agent
        try:
            from agents.multilingual_kg_reasoning_agent import MultilingualKGReasoningAgent
            self.agents["reasoning"] = MultilingualKGReasoningAgent(config=agent_config)
            self.logger.info("✅ Reasoning agent initialized")
        except Exception as e:
            self.logger.error(f"❌ Reasoning agent failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        # Pattern Detector Agent
        try:
            from agents.pattern_detector_agent import PatternDetectorAgent
            self.agents["pattern"] = PatternDetectorAgent(config=agent_config)
            self.logger.info("✅ Pattern detector initialized")
        except Exception as e:
            self.logger.error(f"❌ Pattern detector failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        # Evidence Gatherer Agent
        try:
            from agents.evidence_gatherer_agent import EvidenceGathererAgent
            self.agents["evidence"] = EvidenceGathererAgent(config=agent_config)
            self.logger.info("✅ Evidence gatherer initialized")
        except Exception as e:
            self.logger.error(f"❌ Evidence gatherer failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        # Social Behavior Analysis Agent (Consolidated)
        try:
            from agents.social_behavior_analysis_agent import SocialBehaviorAnalysisAgent
            self.agents["social_behavior"] = SocialBehaviorAnalysisAgent(config=agent_config)
            self.logger.info("✅ Social behavior analysis agent initialized (consolidated)")
        except Exception as e:
            self.logger.error(f"❌ Social behavior analysis agent failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        self.logger.info(f"🎉 Initialized {len(self.agents)} agents: {list(self.agents.keys())}")
    
    async def analyze_post_with_metadata(self, post: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single post while preserving ALL metadata.
        
        This is the core function that ensures WHO-WHAT-WHEN tracking.
        
        Args:
            post: Post data with metadata (post_id, user_id, timestamp, text, etc.)
            
        Returns:
            Complete analysis with preserved metadata
        """
        start_time = time.time()
        
        # Extract and preserve metadata
        metadata = PostMetadata(
            post_id=post.get("submission_id", post.get("post_id", "unknown")),
            user_id=post.get("author_id", post.get("user_id", "unknown")),
            username=post.get("author", post.get("username", "unknown")),
            timestamp=post.get("created_utc", post.get("timestamp", datetime.now().isoformat())),
            platform=post.get("platform", "reddit"),
            subreddit=post.get("subreddit"),
            upvotes=post.get("score", post.get("upvotes", 0)),
            comments=post.get("num_comments", post.get("comments", 0)),
            shares=post.get("shares", 0),
            text=post.get("posts", post.get("text", "")).strip()
        )
        
        if not metadata.text:
            return {
                "metadata": asdict(metadata),
                "analysis": None,
                "error": "No text content",
                "processing_time_ms": 0
            }
        
        self.logger.info(f"📝 Analyzing post {metadata.post_id} by {metadata.username} at {metadata.timestamp}")
        self.logger.info(f"   Text preview: {metadata.text[:100]}...")
        
        # ========================================
        # CONCURRENT AGENT EXECUTION
        # ========================================
        # Run all agents in parallel for maximum performance
        # This is safe with Bedrock/OpenAI which support concurrent requests
        
        analysis_results = {}
        
        # Define async tasks for each agent
        async def run_reasoning():
            """Run reasoning analysis concurrently."""
            if "reasoning" not in self.agents:
                return None
            try:
                from agents.kg_reasoning_agent import ReasoningStrategy
                reasoning_agent = self.agents["reasoning"]
                result = await reasoning_agent._execute_multilingual_reasoning(
                    claim=metadata.text,
                    strategy=ReasoningStrategy.COT,
                    response_language="auto",
                    context=None
                )
                self.logger.info(f"   ✅ Reasoning: {result.get('verdict')} (confidence: {result.get('confidence', 0):.2f})")
                return result
            except Exception as e:
                self.logger.error(f"   ❌ Reasoning failed: {e}")
                return {"error": str(e)}
        
        async def run_pattern_detection():
            """Run pattern detection concurrently."""
            if "pattern" not in self.agents:
                self.logger.warning(f"   ⚠️ Pattern agent not initialized")
                return None
            try:
                pattern_agent = self.agents["pattern"]
                result = await pattern_agent.detect_misinformation_patterns(
                    content=metadata.text,
                    context={"post_id": metadata.post_id, "user_id": metadata.user_id}
                )
                patterns_found = len(result.get("patterns_detected", []))
                self.logger.info(f"   ✅ Patterns: {patterns_found} detected")
                return result
            except Exception as e:
                self.logger.error(f"   ❌ Pattern detection failed: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                return {"error": str(e)}
        
        async def run_topic_extraction():
            """Run topic extraction concurrently."""
            if "topic_modeling" not in self.agents:
                return None
            try:
                topic_agent = self.agents["topic_modeling"]
                result = await topic_agent._extract_topics_from_text(
                    text=metadata.text,
                    context={"post_id": metadata.post_id}
                )
                topics_found = result.get("topics", [])
                self.logger.info(f"   ✅ Topics: {len(topics_found)} extracted - {[t.get('name', 'unknown') for t in topics_found[:3]]}")
                return result
            except Exception as e:
                self.logger.error(f"   ❌ Topic extraction failed: {e}")
                return {"error": str(e), "topics": []}
        
        # Execute Phase 1: Core analysis agents in parallel
        self.logger.info(f"   🚀 Running core agents concurrently...")
        phase1_results = await asyncio.gather(
            run_reasoning(),
            run_pattern_detection(),
            run_topic_extraction(),
            return_exceptions=True  # Don't fail if one agent fails
        )
        
        # Unpack results
        reasoning_result, pattern_result, topic_result = phase1_results
        
        # Handle exceptions
        if isinstance(reasoning_result, Exception):
            self.logger.error(f"   ❌ Reasoning exception: {reasoning_result}")
            reasoning_result = {"error": str(reasoning_result)}
        if isinstance(pattern_result, Exception):
            self.logger.error(f"   ❌ Pattern exception: {pattern_result}")
            pattern_result = {"error": str(pattern_result)}
        if isinstance(topic_result, Exception):
            self.logger.error(f"   ❌ Topic exception: {topic_result}")
            topic_result = {"error": str(topic_result), "topics": []}
        
        # Store results
        if reasoning_result:
            analysis_results["reasoning"] = reasoning_result
        if pattern_result:
            analysis_results["patterns"] = pattern_result
        if topic_result:
            analysis_results["topics"] = topic_result
        
        # Phase 2: Evidence gathering (conditional on reasoning confidence)
        # These can also run concurrently if both conditions are met
        reasoning_confidence = analysis_results.get("reasoning", {}).get("confidence", 0)
        topics = analysis_results.get("topics", {}).get("topics", [])
        
        evidence_tasks = []
        
        # 4. Evidence Gathering (for high-confidence misinformation)
        async def run_evidence_gathering():
            """Run evidence gathering concurrently."""
            if "evidence" not in self.agents:
                return None
            try:
                evidence_agent = self.agents["evidence"]
                result = await evidence_agent.gather_evidence(
                    claim=metadata.text,
                    context=f"Post by {metadata.username} on {metadata.platform}"
                )
                self.logger.info(f"   ✅ Evidence gathered")
                return result
            except Exception as e:
                self.logger.error(f"   ❌ Evidence gathering failed: {e}")
                return {"error": str(e)}
        
        # 5. Topic-Guided Evidence
        async def run_topic_evidence():
            """Run topic-guided evidence retrieval concurrently."""
            if "topic_evidence" not in self.agents or not topics:
                return None
            try:
                topic_evidence_agent = self.agents["topic_evidence"]
                top_topic = topics[0].get("name", "") if topics else ""
                if not top_topic:
                    return None
                
                result = await topic_evidence_agent._retrieve_topic_evidence(
                    claim=metadata.text,
                    topic=top_topic,
                    context={"post_id": metadata.post_id}
                )
                self.logger.info(f"   ✅ Topic-guided evidence retrieved for topic: {top_topic}")
                return result
            except Exception as e:
                self.logger.error(f"   ❌ Topic-guided evidence failed: {e}")
                return {"error": str(e)}
        
        # Run evidence gathering tasks concurrently if applicable
        if reasoning_confidence > 0.7:
            self.logger.info(f"   🚀 Running evidence agents concurrently...")
            if "evidence" in self.agents:
                evidence_tasks.append(run_evidence_gathering())
            if "topic_evidence" in self.agents and topics:
                evidence_tasks.append(run_topic_evidence())
            
            if evidence_tasks:
                evidence_results = await asyncio.gather(*evidence_tasks, return_exceptions=True)
                
                # Process evidence results
                for i, result in enumerate(evidence_results):
                    if isinstance(result, Exception):
                        self.logger.error(f"   ❌ Evidence task {i} exception: {result}")
                    elif result:
                        if i == 0 and "evidence" in self.agents:
                            analysis_results["evidence"] = result
                        elif i == 1 and "topic_evidence" in self.agents:
                            analysis_results["topic_evidence"] = result
                        elif i == 0 and "topic_evidence" in self.agents:
                            analysis_results["topic_evidence"] = result
        
        # Create comprehensive analysis
        misinformation_analysis = self._create_misinformation_analysis(
            metadata, analysis_results
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Complete result with ALL metadata preserved
        result = {
            "metadata": asdict(metadata),
            "analysis": asdict(misinformation_analysis),
            "raw_agent_results": analysis_results,
            "processing_time_ms": processing_time,
            "analyzed_at": datetime.now().isoformat()
        }
        
        # Store for aggregation
        self.post_analyses.append(result)
        self._update_user_profile(metadata, misinformation_analysis)
        self._update_temporal_patterns(metadata, misinformation_analysis)
        
        self.logger.info(f"✅ Post {metadata.post_id} analyzed in {processing_time:.0f}ms")
        
        return result
    
    def _create_misinformation_analysis(
        self, 
        metadata: PostMetadata, 
        agent_results: Dict[str, Any]
    ) -> MisinformationAnalysis:
        """Create comprehensive misinformation analysis from agent results."""
        
        reasoning = agent_results.get("reasoning", {})
        patterns = agent_results.get("patterns", {})
        
        # Extract verdict and confidence
        verdict = reasoning.get("verdict")
        confidence = reasoning.get("confidence", 0.0)
        detected_language = reasoning.get("detected_language", "unknown")
        reasoning_chain = reasoning.get("reasoning_chain", [])
        
        # Extract patterns with specific examples
        patterns_detected = patterns.get("patterns_detected", [])
        severity_score = patterns.get("severity_score", 0.0)
        
        # Extract specific text examples from patterns
        specific_examples = self._extract_specific_examples(metadata.text, patterns_detected)
        
        # Determine risk level
        risk_level = self._determine_risk_level(verdict, confidence, severity_score)
        
        # Extract manipulation tactics
        manipulation_tactics = patterns.get("manipulation_techniques", [])
        
        return MisinformationAnalysis(
            verdict=verdict,
            confidence=confidence,
            detected_language=detected_language,
            reasoning_chain=reasoning_chain,
            patterns_detected=patterns_detected,
            severity_score=severity_score,
            risk_level=risk_level,
            specific_examples=specific_examples,
            manipulation_tactics=manipulation_tactics,
            execution_time_ms=reasoning.get("execution_time_ms", 0),
            error=reasoning.get("error")
        )
    
    def _extract_specific_examples(
        self, 
        text: str, 
        patterns: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract specific text snippets that triggered patterns."""
        examples = []
        
        for pattern in patterns:
            pattern_name = pattern.get("pattern", "unknown")
            confidence = pattern.get("confidence", 0.0)
            
            # Try to find specific examples in the text
            pattern_examples = pattern.get("examples", [])
            
            for example in pattern_examples:
                # Find the example in the original text
                if isinstance(example, str) and example.lower() in text.lower():
                    start_idx = text.lower().find(example.lower())
                    end_idx = start_idx + len(example)
                    
                    # Get context (50 chars before and after)
                    context_start = max(0, start_idx - 50)
                    context_end = min(len(text), end_idx + 50)
                    context = text[context_start:context_end]
                    
                    examples.append({
                        "pattern": pattern_name,
                        "text_snippet": example,
                        "context": context,
                        "start_char": start_idx,
                        "end_char": end_idx,
                        "confidence": confidence
                    })
        
        return examples
    
    def _determine_risk_level(
        self, 
        verdict: Optional[bool], 
        confidence: float, 
        severity: float
    ) -> str:
        """Determine risk level based on analysis results."""
        
        # CRITICAL: High confidence false claim with high severity
        if verdict == False and confidence > 0.85 and severity > 0.7:
            return "CRITICAL"
        
        # HIGH: High confidence false claim OR high severity
        if (verdict == False and confidence > 0.7) or severity > 0.6:
            return "HIGH"
        
        # MODERATE: Medium confidence false claim
        if verdict == False and confidence > 0.5:
            return "MODERATE"
        
        # LOW: Everything else
        return "LOW"
    
    def _update_user_profile(
        self, 
        metadata: PostMetadata, 
        analysis: MisinformationAnalysis
    ):
        """Update user profile with new post analysis."""
        
        user_id = metadata.user_id
        
        # Create profile if doesn't exist
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                username=metadata.username,
                first_seen=metadata.timestamp
            )
        
        profile = self.user_profiles[user_id]
        
        # Update counts
        profile.total_posts += 1
        profile.last_seen = metadata.timestamp
        
        # Track misinformation
        if analysis.verdict == False:
            profile.misinformation_posts += 1
            
            if analysis.confidence > 0.8:
                profile.high_confidence_misinfo += 1
            
            # Add post to profile
            profile.posts.append({
                "post_id": metadata.post_id,
                "timestamp": metadata.timestamp,
                "text_preview": metadata.text[:200],
                "verdict": analysis.verdict,
                "confidence": analysis.confidence,
                "risk_level": analysis.risk_level,
                "patterns": [p.get("pattern") for p in analysis.patterns_detected]
            })
        
        # Update average confidence
        total_confidence = profile.avg_confidence * (profile.total_posts - 1) + analysis.confidence
        profile.avg_confidence = total_confidence / profile.total_posts
        
        # Track patterns and tactics
        for pattern in analysis.patterns_detected:
            pattern_name = pattern.get("pattern", "unknown")
            if pattern_name not in profile.patterns_used:
                profile.patterns_used.append(pattern_name)
        
        for tactic in analysis.manipulation_tactics:
            if tactic not in profile.manipulation_tactics:
                profile.manipulation_tactics.append(tactic)
        
        # Track languages
        if analysis.detected_language not in profile.languages_used:
            profile.languages_used.append(analysis.detected_language)
    
    def _update_temporal_patterns(
        self, 
        metadata: PostMetadata, 
        analysis: MisinformationAnalysis
    ):
        """Track patterns over time."""
        
        for pattern_data in analysis.patterns_detected:
            pattern_name = pattern_data.get("pattern", "unknown")
            
            # Create pattern tracker if doesn't exist
            if pattern_name not in self.temporal_patterns:
                self.temporal_patterns[pattern_name] = TemporalPattern(
                    pattern_name=pattern_name,
                    first_occurrence=metadata.timestamp
                )
            
            pattern = self.temporal_patterns[pattern_name]
            
            # Add occurrence
            pattern.occurrences.append({
                "post_id": metadata.post_id,
                "user_id": metadata.user_id,
                "timestamp": metadata.timestamp,
                "confidence": pattern_data.get("confidence", 0.0)
            })
            
            # Update last occurrence
            pattern.last_occurrence = metadata.timestamp
            
            # Track users
            if metadata.user_id not in pattern.users_involved:
                pattern.users_involved.append(metadata.user_id)
    
    async def analyze_batch_with_metadata(
        self, 
        posts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple posts while preserving metadata (ONE AT A TIME).
        
        Args:
            posts: List of posts with metadata
            
        Returns:
            List of complete analyses with metadata
        """
        self.logger.info(f"🔥 Starting batch analysis of {len(posts)} posts")
        
        results = []
        for i, post in enumerate(posts, 1):
            self.logger.info(f"📊 Processing post {i}/{len(posts)}")
            
            try:
                result = await self.analyze_post_with_metadata(post)
                results.append(result)
            except Exception as e:
                self.logger.error(f"❌ Failed to analyze post {i}: {e}")
                results.append({
                    "metadata": {"post_id": post.get("submission_id", "unknown"), "error": str(e)},
                    "analysis": None,
                    "error": str(e)
                })
        
        self.logger.info(f"✅ Batch analysis complete: {len(results)} posts processed")
        
        # Run topic modeling on the entire collection
        if "topic_modeling" in self.agents and len(posts) > 5:
            self.logger.info(f"🔍 Running topic modeling on {len(posts)} posts...")
            try:
                topic_agent = self.agents["topic_modeling"]
                
                # Collect all post texts
                all_texts = [post.get("posts", post.get("text", "")).strip() for post in posts if post.get("posts", post.get("text", "")).strip()]
                
                if all_texts:
                    # Extract topics from the collection
                    topics_result = await topic_agent._extract_topics_from_collection(
                        texts=all_texts,
                        num_topics=5,
                        context={"total_posts": len(posts)}
                    )
                    
                    # Add topics to each result
                    topics = topics_result.get("topics", [])
                    self.logger.info(f"✅ Extracted {len(topics)} topics from collection")
                    
                    # Store topics for reporting
                    for topic in topics:
                        topic_name = topic.get("name", "unknown")
                        if topic_name not in self.topic_analyses:
                            self.topic_analyses[topic_name] = {
                                "posts": [],
                                "keywords": topic.get("keywords", []),
                                "representative_docs": topic.get("representative_docs", [])
                            }
                    
            except Exception as e:
                self.logger.error(f"❌ Topic modeling failed: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
        
        return results
    
    async def analyze_batch_true_batch(
        self, 
        posts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple posts using TRUE BATCH processing (send all to Ollama at once).
        
        This sends all posts in a single request to Ollama for batch inference.
        Much faster than processing one at a time.
        
        Args:
            posts: List of posts with metadata
            
        Returns:
            List of complete analyses with metadata
        """
        import time
        start_time = time.time()
        
        # Configure batch size (smaller batches for better reliability)
        CHUNK_SIZE = 32  # Process 64 posts at a time
        
        self.logger.info(f"🚀 CHUNKED BATCH: Processing {len(posts)} posts in chunks of {CHUNK_SIZE}")
        
        # Extract metadata for all posts
        all_metadata = []
        claims = []
        
        for post in posts:
            metadata = PostMetadata(
                post_id=post.get("submission_id", post.get("post_id", "unknown")),
                user_id=post.get("author_id", post.get("user_id", "unknown")),
                username=post.get("author", post.get("username", "unknown")),
                timestamp=post.get("created_utc", post.get("timestamp", datetime.now().isoformat())),
                platform=post.get("platform", "reddit"),
                subreddit=post.get("subreddit"),
                upvotes=post.get("score", post.get("upvotes", 0)),
                comments=post.get("num_comments", post.get("comments", 0)),
                shares=post.get("shares", 0),
                text=post.get("posts", post.get("text", "")).strip()
            )
            
            if metadata.text:
                all_metadata.append(metadata)
                claims.append(metadata.text)
        
        if not claims:
            return []
        
        # Process in chunks for better reliability
        batch_results = []
        
        if "reasoning" in self.agents:
            try:
                # Import the TRUE batch reasoning function
                from true_batch_reasoning import true_batch_reason
                
                # Process claims in chunks CONCURRENTLY
                total_chunks = (len(claims) + CHUNK_SIZE - 1) // CHUNK_SIZE
                self.logger.info(f"🚀 CONCURRENT: Processing {len(claims)} claims in {total_chunks} chunks simultaneously...")
                
                # Create all tasks
                tasks = []
                for chunk_idx in range(0, len(claims), CHUNK_SIZE):
                    chunk_claims = claims[chunk_idx:chunk_idx + CHUNK_SIZE]
                    chunk_num = (chunk_idx // CHUNK_SIZE) + 1
                    
                    self.logger.info(f"📦 Launching chunk {chunk_num}/{total_chunks} with {len(chunk_claims)} claims...")
                    
                    task = true_batch_reason(
                        claims=chunk_claims,
                        agent_config=self.agent_config
                    )
                    tasks.append(task)
                
                # Run all chunks concurrently
                self.logger.info(f"⚡ Running {len(tasks)} chunks in parallel...")
                chunk_results = await asyncio.gather(*tasks)
                
                # Flatten results
                batch_reasoning = []
                for chunk_result in chunk_results:
                    batch_reasoning.extend(chunk_result)
                
                self.logger.info(f"✅ All {total_chunks} chunks complete concurrently: {len(batch_reasoning)} results")
                
                # TRUE BATCH pattern detection - send ALL claims in ONE request
                batch_patterns = []
                if "pattern" in self.agents:
                    try:
                        # Import TRUE batch pattern detection
                        from true_batch_pattern_detection import true_batch_pattern_detection
                        
                        # Process patterns in chunks CONCURRENTLY
                        self.logger.info(f"🔍 CONCURRENT PATTERNS: Detecting patterns on {len(claims)} claims in {total_chunks} chunks simultaneously...")
                        
                        # Create all pattern tasks
                        pattern_tasks = []
                        for chunk_idx in range(0, len(claims), CHUNK_SIZE):
                            chunk_claims = claims[chunk_idx:chunk_idx + CHUNK_SIZE]
                            chunk_num = (chunk_idx // CHUNK_SIZE) + 1
                            
                            self.logger.info(f"📦 Launching pattern chunk {chunk_num}/{total_chunks} with {len(chunk_claims)} claims...")
                            
                            task = true_batch_pattern_detection(
                                claims=chunk_claims,
                                agent_config=self.agent_config
                            )
                            pattern_tasks.append(task)
                        
                        # Run all pattern chunks concurrently
                        self.logger.info(f"⚡ Running {len(pattern_tasks)} pattern chunks in parallel...")
                        pattern_chunk_results = await asyncio.gather(*pattern_tasks)
                        
                        # Flatten results
                        batch_patterns = []
                        for chunk_result in pattern_chunk_results:
                            batch_patterns.extend(chunk_result)
                        
                        self.logger.info(f"✅ All {total_chunks} pattern chunks complete concurrently: {len(batch_patterns)} results")
                    except Exception as e:
                        self.logger.error(f"❌ TRUE BATCH pattern detection failed: {e}")
                        import traceback
                        self.logger.error(traceback.format_exc())
                        batch_patterns = [{"patterns_detected": [], "severity_score": 0.0} for _ in claims]
                else:
                    batch_patterns = [{"patterns_detected": [], "severity_score": 0.0} for _ in claims]
                
                # Process each result with metadata
                for i, (metadata, reasoning_result, pattern_result) in enumerate(zip(all_metadata, batch_reasoning, batch_patterns)):
                    # Create comprehensive analysis
                    analysis = MisinformationAnalysis(
                        verdict=reasoning_result.get("verdict"),
                        confidence=reasoning_result.get("confidence", 0.0),
                        detected_language=reasoning_result.get("detected_language", "unknown"),
                        reasoning_chain=reasoning_result.get("reasoning_chain", []),
                        patterns_detected=pattern_result.get("patterns_detected", []),
                        severity_score=pattern_result.get("severity_score", 0.0),
                        risk_level=self._determine_risk_level(
                            reasoning_result.get("verdict"),
                            reasoning_result.get("confidence", 0.0),
                            pattern_result.get("severity_score", 0.0)
                        ),
                        specific_examples=self._extract_specific_examples(
                            metadata.text,
                            pattern_result.get("patterns_detected", [])
                        ),
                        manipulation_tactics=pattern_result.get("manipulation_techniques", []),
                        execution_time_ms=reasoning_result.get("execution_time_ms", 0)
                    )
                    
                    result = {
                        "metadata": asdict(metadata),
                        "analysis": asdict(analysis),
                        "raw_agent_results": {
                            "reasoning": reasoning_result,
                            "patterns": pattern_result
                        },
                        "processing_time_ms": reasoning_result.get("execution_time_ms", 0),
                        "analyzed_at": datetime.now().isoformat()
                    }
                    
                    # Store for aggregation
                    self.post_analyses.append(result)
                    self._update_user_profile(metadata, analysis)
                    self._update_temporal_patterns(metadata, analysis)
                    
                    batch_results.append(result)
                
            except Exception as e:
                import traceback
                self.logger.error(f"❌ Batch reasoning failed: {e}")
                self.logger.error(traceback.format_exc())
                # Fallback to individual processing
                return await self.analyze_batch_with_metadata(posts)
        
        batch_time = (time.time() - start_time) * 1000
        if batch_results:
            self.logger.info(f"✅ TRUE BATCH complete: {len(batch_results)} posts in {batch_time:.0f}ms ({batch_time/len(batch_results):.0f}ms per post)")
        else:
            self.logger.warning(f"⚠️ TRUE BATCH complete but no results: {batch_time:.0f}ms")
        
        # TRUE BATCH topic modeling on the entire batch
        if len(claims) > 5:
            self.logger.info(f"🔍 TRUE BATCH TOPIC MODELING on {len(claims)} posts...")
            try:
                # Import TRUE batch topic modeling
                from true_batch_topic_modeling import true_batch_topic_modeling
                
                # Extract topics using embeddings + BERTopic
                topics_result = await true_batch_topic_modeling(
                    texts=claims,
                    num_topics=5,
                    min_topic_size=max(3, len(claims) // 20),  # Dynamic min size
                    agent_config=self.agent_config
                )
                
                if topics_result.get("status") == "success":
                    topics = topics_result.get("topics", [])
                    self.logger.info(f"✅ Extracted {len(topics)} topics from batch")
                    self.logger.info(f"   Embedding time: {topics_result.get('embedding_time_ms', 0):.0f}ms")
                    self.logger.info(f"   Clustering time: {topics_result.get('clustering_time_ms', 0):.0f}ms")
                    
                    # Store topics for reporting
                    for topic in topics:
                        topic_name = topic.get("name", "unknown")
                        if topic_name not in self.topic_analyses:
                            self.topic_analyses[topic_name] = {
                                "posts": [],
                                "keywords": topic.get("keywords", []),
                                "count": topic.get("count", 0),
                                "percentage": topic.get("percentage", 0),
                                "representative_docs": topic.get("representative_docs", [])
                            }
                else:
                    self.logger.warning(f"⚠️  Topic modeling returned status: {topics_result.get('status')}")
                
            except Exception as e:
                self.logger.error(f"❌ TRUE BATCH topic modeling failed: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
        
        # Social Behavior Analysis on the batch
        if "social_behavior" in self.agents and len(batch_results) >= 3:
            try:
                self.logger.info(f"🔍 Analyzing social behavior patterns across {len(batch_results)} posts...")
                social_agent = self.agents["social_behavior"]
                
                # Prepare events for coordination detection
                events = []
                user_posts = defaultdict(list)
                
                for result in batch_results:
                    # Check if result has metadata key
                    if not isinstance(result, dict):
                        continue
                    
                    metadata = result.get("metadata")
                    if not metadata:
                        continue
                    
                    # Add to events for coordination detection
                    events.append({
                        "user_id": metadata.get("user_id", "unknown"),
                        "timestamp": metadata.get("timestamp", ""),
                        "content": metadata.get("text", ""),
                        "post_id": metadata.get("post_id", "unknown"),
                        "platform": metadata.get("platform", "unknown")
                    })
                    
                    # Collect user posts for echo chamber detection
                    user_id = metadata.get("user_id", "unknown")
                    user_posts[user_id].append(metadata.get("text", ""))
                
                # Only proceed if we have valid events
                if not events:
                    self.logger.warning("No valid events for social behavior analysis")
                    return batch_results
                
                # Detect coordination
                coordination_result = await social_agent._detect_coordination_internal(
                    events=events,
                    analysis_type="comprehensive"
                )
                
                # Store coordination results
                if coordination_result.get("coordination_detected"):
                    self.logger.info(f"⚠️  Coordination detected! Score: {coordination_result.get('coordination_score', 0):.2f}")
                
                # Detect echo chambers
                
                if len(user_posts) >= 3:
                    user_network = {user_id: list(user_posts.keys()) for user_id in user_posts.keys()}
                    
                    echo_result = await social_agent._detect_echo_chamber_internal(
                        user_network=user_network,
                        user_content=dict(user_posts)
                    )
                    
                    if echo_result.get("echo_chamber_detected"):
                        self.logger.info(f"⚠️  Echo chamber detected! Score: {echo_result.get('echo_chamber_score', 0):.2f}")
                
                self.logger.info(f"✅ Social behavior analysis complete")
                
            except Exception as e:
                self.logger.error(f"❌ Social behavior analysis failed: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
        
        return batch_results
    
    def generate_actionable_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive actionable intelligence report.
        
        This is what you need: WHO said WHAT WHEN with specific examples.
        """
        
        self.logger.info("📋 Generating actionable intelligence report...")
        
        # Filter high-risk posts
        high_risk_posts = [
            p for p in self.post_analyses
            if p.get("analysis", {}).get("risk_level") in ["CRITICAL", "HIGH"]
        ]
        
        # Sort by confidence (highest first)
        high_risk_posts.sort(
            key=lambda x: x.get("analysis", {}).get("confidence", 0),
            reverse=True
        )
        
        # Get top offenders
        top_offenders = sorted(
            self.user_profiles.values(),
            key=lambda x: x.misinformation_posts,
            reverse=True
        )[:10]
        
        # Get temporal trends
        temporal_trends = self._analyze_temporal_trends()
        
        # Analyze topics across all posts
        topic_analysis = self._analyze_topics()
        
        # Create detailed report
        report = {
            "executive_summary": {
                "total_posts_analyzed": len(self.post_analyses),
                "misinformation_detected": len([
                    p for p in self.post_analyses
                    if p.get("analysis", {}).get("verdict") == False
                ]),
                "high_risk_posts": len(high_risk_posts),
                "critical_posts": len([
                    p for p in self.post_analyses
                    if p.get("analysis", {}).get("risk_level") == "CRITICAL"
                ]),
                "unique_users": len(self.user_profiles),
                "users_posting_misinfo": len([
                    u for u in self.user_profiles.values()
                    if u.misinformation_posts > 0
                ]),
                "patterns_detected": len(self.temporal_patterns),
                "topics_identified": len(topic_analysis.get("topics", [])),
                "report_generated": datetime.now().isoformat()
            },
            
            "high_priority_posts": [
                {
                    "post_id": p["metadata"]["post_id"],
                    "user_id": p["metadata"]["user_id"],
                    "username": p["metadata"]["username"],
                    "timestamp": p["metadata"]["timestamp"],
                    "platform": p["metadata"]["platform"],
                    "subreddit": p["metadata"].get("subreddit"),
                    "text_preview": p["metadata"]["text"][:300],
                    "full_text_length": p["metadata"]["text_length"],
                    "engagement": {
                        "upvotes": p["metadata"]["upvotes"],
                        "comments": p["metadata"]["comments"],
                        "shares": p["metadata"]["shares"]
                    },
                    "analysis": {
                        "verdict": "MISINFORMATION" if p["analysis"]["verdict"] == False else "TRUE" if p["analysis"]["verdict"] == True else "UNCERTAIN",
                        "confidence": p["analysis"]["confidence"],
                        "risk_level": p["analysis"]["risk_level"],
                        "detected_language": p["analysis"]["detected_language"],
                        "patterns": [pat.get("pattern") for pat in p["analysis"]["patterns_detected"]],
                        "manipulation_tactics": p["analysis"]["manipulation_tactics"],
                        "specific_examples": p["analysis"]["specific_examples"][:3]  # Top 3 examples
                    },
                    "recommended_action": self._get_recommended_action(p["analysis"])
                }
                for p in high_risk_posts[:20]  # Top 20 high-risk posts
            ],
            
            "top_offenders": [
                {
                    "user_id": profile.user_id,
                    "username": profile.username,
                    "statistics": {
                        "total_posts": profile.total_posts,
                        "misinformation_posts": profile.misinformation_posts,
                        "high_confidence_misinfo": profile.high_confidence_misinfo,
                        "misinformation_rate": f"{profile.misinformation_rate:.1f}%",
                        "avg_confidence": profile.avg_confidence
                    },
                    "activity_period": {
                        "first_seen": profile.first_seen,
                        "last_seen": profile.last_seen
                    },
                    "patterns_used": profile.patterns_used,
                    "manipulation_tactics": profile.manipulation_tactics,
                    "languages": profile.languages_used,
                    "recent_posts": profile.posts[-5:],  # Last 5 posts
                    "recommended_action": self._get_user_action(profile)
                }
                for profile in top_offenders
            ],
            
            "temporal_analysis": temporal_trends,
            
            "topic_analysis": topic_analysis,
            
            "pattern_breakdown": [
                {
                    "pattern_name": pattern.pattern_name,
                    "total_occurrences": len(pattern.occurrences),
                    "unique_users": len(pattern.users_involved),
                    "first_seen": pattern.first_occurrence,
                    "last_seen": pattern.last_occurrence,
                    "recent_examples": [
                        {
                            "post_id": occ["post_id"],
                            "user_id": occ["user_id"],
                            "timestamp": occ["timestamp"],
                            "confidence": occ["confidence"]
                        }
                        for occ in pattern.occurrences[-5:]  # Last 5 occurrences
                    ]
                }
                for pattern in sorted(
                    self.temporal_patterns.values(),
                    key=lambda x: len(x.occurrences),
                    reverse=True
                )[:10]  # Top 10 patterns
            ]
        }
        
        return report
    
    def _get_recommended_action(self, analysis: Dict[str, Any]) -> str:
        """Get recommended action for a post based on risk level."""
        risk_level = analysis.get("risk_level", "LOW")
        confidence = analysis.get("confidence", 0)
        
        if risk_level == "CRITICAL":
            return "IMMEDIATE: Remove content, suspend account, alert moderation team"
        elif risk_level == "HIGH":
            return "URGENT: Add warning label, reduce algorithmic amplification, queue for review"
        elif risk_level == "MODERATE":
            return "REVIEW: Flag for fact-checking, monitor engagement"
        else:
            return "MONITOR: Track for pattern development"
    
    def _get_user_action(self, profile: UserProfile) -> str:
        """Get recommended action for a user based on their profile."""
        misinfo_rate = profile.misinformation_rate
        
        if misinfo_rate > 70 and profile.high_confidence_misinfo > 5:
            return "IMMEDIATE: Suspend account, investigate for coordinated behavior"
        elif misinfo_rate > 50:
            return "URGENT: Restrict posting, add warning to profile, monitor closely"
        elif misinfo_rate > 30:
            return "REVIEW: Add to watchlist, fact-check future posts"
        else:
            return "MONITOR: Track for pattern changes"
    
    def _analyze_topics(self) -> Dict[str, Any]:
        """Analyze topics across all posts with misinformation tracking."""
        
        if not self.post_analyses:
            return {"status": "no_data", "topics": []}
        
        # Use TRUE batch topic modeling results if available
        if self.topic_analyses:
            self.logger.info(f"📊 Using TRUE batch topic modeling results: {len(self.topic_analyses)} topics")
            
            # Enhance topics with misinformation analysis
            topics = []
            for topic_name, topic_data in self.topic_analyses.items():
                # Find posts that match this topic's keywords
                topic_keywords = set(topic_data.get("keywords", []))
                
                # Stats for this topic
                total_posts = 0
                misinformation_posts = 0
                high_confidence_misinfo = 0
                total_confidence = 0.0
                users = set()
                posts_list = []
                patterns_count = defaultdict(int)
                
                # Match posts to this topic based on keyword overlap
                for post in self.post_analyses:
                    post_text = post["metadata"]["text"].lower()
                    # Check if any topic keywords appear in post
                    if any(keyword.lower() in post_text for keyword in topic_keywords):
                        total_posts += 1
                        is_misinfo = post.get("analysis", {}).get("verdict") == False
                        confidence = post.get("analysis", {}).get("confidence", 0.0)
                        total_confidence += confidence
                        
                        if is_misinfo:
                            misinformation_posts += 1
                            if confidence > 0.8:
                                high_confidence_misinfo += 1
                        
                        users.add(post["metadata"]["user_id"])
                        
                        posts_list.append({
                            "post_id": post["metadata"]["post_id"],
                            "user_id": post["metadata"]["user_id"],
                            "timestamp": post["metadata"]["timestamp"],
                            "text_preview": post["metadata"]["text"][:150],
                            "verdict": "MISINFO" if is_misinfo else "TRUE",
                            "confidence": confidence
                        })
                        
                        # Track patterns
                        for pattern in post.get("analysis", {}).get("patterns_detected", []):
                            pattern_name = pattern.get("pattern", "unknown")
                            patterns_count[pattern_name] += 1
                
                if total_posts > 0:
                    avg_confidence = total_confidence / total_posts
                    misinfo_rate = (misinformation_posts / total_posts * 100)
                    
                    topics.append({
                        "topic_name": topic_name,
                        "total_posts": total_posts,
                        "misinformation_posts": misinformation_posts,
                        "high_confidence_misinfo": high_confidence_misinfo,
                        "misinformation_rate": f"{misinfo_rate:.1f}%",
                        "unique_users": len(users),
                        "avg_confidence": avg_confidence,
                        "keywords": topic_data.get("keywords", [])[:10],
                        "percentage": topic_data.get("percentage", 0),
                        "representative_docs": topic_data.get("representative_docs", []),
                        "top_patterns": sorted(
                            patterns_count.items(),
                            key=lambda x: x[1],
                            reverse=True
                        )[:5],
                        "recent_posts": posts_list[-5:],
                        "risk_assessment": self._assess_topic_risk(misinfo_rate, high_confidence_misinfo)
                    })
            
            # Sort by misinformation posts (highest first)
            topics.sort(key=lambda x: x["misinformation_posts"], reverse=True)
            
            return {
                "status": "active",
                "topics": topics,
                "summary": {
                    "total_topics": len(topics),
                    "high_risk_topics": len([t for t in topics if t["risk_assessment"] in ["CRITICAL", "HIGH"]]),
                    "topics_with_misinfo": len([t for t in topics if t["misinformation_posts"] > 0])
                }
            }
        
        # Fallback: no topic modeling results
        return {
            "status": "no_data",
            "topics": [],
            "summary": {
                "total_topics": 0,
                "high_risk_topics": 0,
                "topics_with_misinfo": 0
            }
        }
    
    def _assess_topic_risk(self, misinfo_rate: float, high_confidence_count: int) -> str:
        """Assess risk level for a topic based on misinformation rate."""
        if misinfo_rate > 75 and high_confidence_count > 2:
            return "CRITICAL"
        elif misinfo_rate > 50:
            return "HIGH"
        elif misinfo_rate > 25:
            return "MODERATE"
        else:
            return "LOW"
    
    def _analyze_temporal_trends(self) -> Dict[str, Any]:
        """Analyze how patterns evolved over time."""
        
        if not self.post_analyses:
            return {"status": "no_data"}
        
        # Group posts by time windows (daily)
        from collections import defaultdict
        daily_stats = defaultdict(lambda: {
            "total_posts": 0,
            "misinformation_posts": 0,
            "patterns": defaultdict(int),
            "users": set()
        })
        
        for post in self.post_analyses:
            timestamp = post["metadata"]["timestamp"]
            # Extract date (YYYY-MM-DD)
            try:
                if isinstance(timestamp, str):
                    date = timestamp.split("T")[0] if "T" in timestamp else timestamp.split()[0]
                else:
                    date = str(timestamp)[:10]
            except:
                date = "unknown"
            
            daily_stats[date]["total_posts"] += 1
            daily_stats[date]["users"].add(post["metadata"]["user_id"])
            
            if post.get("analysis", {}).get("verdict") == False:
                daily_stats[date]["misinformation_posts"] += 1
            
            # Track patterns
            for pattern in post.get("analysis", {}).get("patterns_detected", []):
                pattern_name = pattern.get("pattern", "unknown")
                daily_stats[date]["patterns"][pattern_name] += 1
        
        # Convert to list format
        trends = []
        for date in sorted(daily_stats.keys()):
            stats = daily_stats[date]
            trends.append({
                "date": date,
                "total_posts": stats["total_posts"],
                "misinformation_posts": stats["misinformation_posts"],
                "misinfo_rate": f"{(stats['misinformation_posts'] / stats['total_posts'] * 100):.1f}%" if stats["total_posts"] > 0 else "0%",
                "unique_users": len(stats["users"]),
                "top_patterns": sorted(
                    stats["patterns"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]  # Top 3 patterns per day
            })
        
        return {
            "status": "active",
            "daily_trends": trends,
            "trend_summary": {
                "total_days": len(trends),
                "peak_misinfo_day": max(trends, key=lambda x: x["misinformation_posts"])["date"] if trends else None,
                "avg_daily_posts": sum(t["total_posts"] for t in trends) / len(trends) if trends else 0
            }
        }
    
    def print_actionable_report(self, report: Dict[str, Any]):
        """Print actionable report in readable format."""
        
        print("\n" + "="*100)
        print("GRANULAR MISINFORMATION INTELLIGENCE REPORT")
        print("="*100)
        
        # Executive Summary
        summary = report["executive_summary"]
        print(f"\n📊 EXECUTIVE SUMMARY")
        print(f"   Total Posts Analyzed: {summary['total_posts_analyzed']}")
        print(f"   Misinformation Detected: {summary['misinformation_detected']} ({summary['misinformation_detected']/max(summary['total_posts_analyzed'],1)*100:.1f}%)")
        print(f"   High Risk Posts: {summary['high_risk_posts']}")
        print(f"   Critical Posts: {summary['critical_posts']}")
        print(f"   Unique Users: {summary['unique_users']}")
        print(f"   Users Posting Misinfo: {summary['users_posting_misinfo']}")
        print(f"   Patterns Detected: {summary['patterns_detected']}")
        print(f"   Topics Identified: {summary.get('topics_identified', 0)}")
        
        # High Priority Posts
        print(f"\n🚨 HIGH PRIORITY POSTS (Top 10)")
        print("="*100)
        for i, post in enumerate(report["high_priority_posts"][:10], 1):
            print(f"\n{i}. POST ID: {post['post_id']}")
            print(f"   User: {post['username']} (ID: {post['user_id']})")
            print(f"   Timestamp: {post['timestamp']}")
            print(f"   Platform: {post['platform']}" + (f" / {post['subreddit']}" if post.get('subreddit') else ""))
            print(f"   Text: {post['text_preview']}")
            print(f"   ")
            print(f"   Analysis:")
            print(f"      Verdict: {post['analysis']['verdict']}")
            print(f"      Confidence: {post['analysis']['confidence']:.2f}")
            print(f"      Risk Level: {post['analysis']['risk_level']}")
            print(f"      Language: {post['analysis']['detected_language']}")
            print(f"      Patterns: {', '.join(post['analysis']['patterns'])}")
            if post['analysis']['manipulation_tactics']:
                print(f"      Manipulation: {', '.join(post['analysis']['manipulation_tactics'])}")
            print(f"   ")
            print(f"   Engagement: {post['engagement']['upvotes']} upvotes, {post['engagement']['comments']} comments")
            print(f"   ")
            print(f"   ⚡ ACTION: {post['recommended_action']}")
            print(f"   {'-'*98}")
        
        # Top Offenders
        print(f"\n👤 TOP OFFENDERS (Users Posting Most Misinformation)")
        print("="*100)
        for i, user in enumerate(report["top_offenders"][:5], 1):
            print(f"\n{i}. USER: {user['username']} (ID: {user['user_id']})")
            stats = user['statistics']
            print(f"   Total Posts: {stats['total_posts']}")
            print(f"   Misinformation Posts: {stats['misinformation_posts']} ({stats['misinformation_rate']})")
            print(f"   High Confidence Misinfo: {stats['high_confidence_misinfo']}")
            print(f"   Active: {user['activity_period']['first_seen']} to {user['activity_period']['last_seen']}")
            print(f"   Patterns Used: {', '.join(user['patterns_used'][:5])}")
            print(f"   ")
            print(f"   ⚡ ACTION: {user['recommended_action']}")
            print(f"   {'-'*98}")
        
        # Topic Analysis
        topic_analysis = report.get("topic_analysis", {})
        if topic_analysis.get("status") == "active":
            print(f"\n📚 TOPIC ANALYSIS")
            print("="*100)
            summary = topic_analysis.get("summary", {})
            print(f"   Total Topics Identified: {summary.get('total_topics', 0)}")
            print(f"   High Risk Topics: {summary.get('high_risk_topics', 0)}")
            print(f"   Topics with Misinformation: {summary.get('topics_with_misinfo', 0)}")
            
            print(f"\n   Top Topics by Misinformation (Top 5):")
            for i, topic in enumerate(topic_analysis.get("topics", [])[:5], 1):
                print(f"\n   {i}. TOPIC: {topic['topic_name']}")
                print(f"      Total Posts: {topic['total_posts']}")
                print(f"      Misinformation Posts: {topic['misinformation_posts']} ({topic['misinformation_rate']})")
                print(f"      High Confidence Misinfo: {topic['high_confidence_misinfo']}")
                print(f"      Unique Users: {topic['unique_users']}")
                print(f"      Risk Assessment: {topic['risk_assessment']}")
                print(f"      Keywords: {', '.join(topic['keywords'][:5])}")
                if topic['top_patterns']:
                    top_patterns = [f"{p[0]} ({p[1]})" for p in topic['top_patterns'][:3]]
                    print(f"      Top Patterns: {', '.join(top_patterns)}")
                print(f"      Recent Posts:")
                for post in topic['recent_posts'][-3:]:  # Last 3 posts
                    print(f"         - {post['post_id']}: {post['text_preview'][:80]}... [{post['verdict']}, conf: {post['confidence']:.2f}]")
        
        # Temporal Trends
        temporal = report["temporal_analysis"]
        if temporal.get("status") == "active":
            print(f"\n📈 TEMPORAL TRENDS")
            print("="*100)
            print(f"   Analysis Period: {temporal['trend_summary']['total_days']} days")
            print(f"   Peak Misinformation Day: {temporal['trend_summary']['peak_misinfo_day']}")
            print(f"   Avg Daily Posts: {temporal['trend_summary']['avg_daily_posts']:.1f}")
            print(f"\n   Recent Daily Breakdown:")
            for day in temporal["daily_trends"][-7:]:  # Last 7 days
                print(f"      {day['date']}: {day['misinformation_posts']}/{day['total_posts']} misinfo ({day['misinfo_rate']}), {day['unique_users']} users")
        
        # Pattern Breakdown
        print(f"\n🔍 PATTERN BREAKDOWN (Top 5)")
        print("="*100)
        for i, pattern in enumerate(report["pattern_breakdown"][:5], 1):
            print(f"\n{i}. PATTERN: {pattern['pattern_name']}")
            print(f"   Total Occurrences: {pattern['total_occurrences']}")
            print(f"   Unique Users: {pattern['unique_users']}")
            print(f"   First Seen: {pattern['first_seen']}")
            print(f"   Last Seen: {pattern['last_seen']}")
        
        print("\n" + "="*100)
        print("END OF REPORT")
        print("="*100 + "\n")
    
    def save_report_json(self, report: Dict[str, Any], filename: str = "granular_misinfo_report.json"):
        """Save report to JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        self.logger.info(f"📄 Report saved to {filename}")


# Helper functions for agent internal methods
async def _detect_misinformation_patterns_internal(agent, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to call pattern detection internal logic."""
    try:
        # Build prompt
        prompt = agent._build_pattern_detection_prompt(content, context)
        # Get LLM response
        response = await agent._get_llm_response(prompt)
        # Parse response
        result = agent._parse_pattern_detection_response(response, content)
        return result
    except Exception as e:
        return {"error": str(e), "patterns_detected": [], "severity_score": 0.0}

async def _gather_evidence_internal(agent, claim: str, context: str) -> Dict[str, Any]:
    """Helper to call evidence gathering internal logic."""
    try:
        # Build prompt
        prompt = agent._build_evidence_prompt(claim, context)
        # Get LLM response
        response = await agent._get_llm_response(prompt)
        # Parse response
        result = agent._parse_evidence_response(response, claim)
        return result
    except Exception as e:
        return {"error": str(e), "evidence": [], "confidence": 0.0}

async def _extract_topics_from_text(agent, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to extract topics from a single text."""
    try:
        # Build prompt for topic extraction
        prompt = f"""Extract the main topics from this text. Identify 1-3 key topics.

Text: {text}

Return topics in this format:
Topic 1: [topic name]
Keywords: [keyword1, keyword2, keyword3]

Topic 2: [topic name]
Keywords: [keyword1, keyword2, keyword3]
"""
        # Get LLM response
        response = await agent._get_llm_response(prompt)
        
        # Parse topics from response
        topics = []
        lines = response.strip().split('\n')
        current_topic = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('Topic'):
                # Extract topic name
                parts = line.split(':', 1)
                if len(parts) > 1:
                    topic_name = parts[1].strip()
                    current_topic = {"name": topic_name, "keywords": []}
                    topics.append(current_topic)
            elif line.startswith('Keywords:') and current_topic:
                # Extract keywords
                parts = line.split(':', 1)
                if len(parts) > 1:
                    keywords_str = parts[1].strip()
                    keywords = [k.strip() for k in keywords_str.split(',')]
                    current_topic["keywords"] = keywords
        
        return {"topics": topics, "status": "success"}
    except Exception as e:
        return {"error": str(e), "topics": []}

async def _retrieve_topic_evidence(agent, claim: str, topic: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to retrieve evidence guided by topic."""
    try:
        # Build prompt for topic-guided evidence retrieval
        prompt = f"""Find evidence related to this claim within the context of the topic "{topic}".

Claim: {claim}
Topic: {topic}

Provide relevant evidence, sources, and confidence level.
"""
        # Get LLM response
        response = await agent._get_llm_response(prompt)
        
        return {
            "evidence": response,
            "topic": topic,
            "claim": claim,
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "evidence": "", "topic": topic}

async def _execute_batch_multilingual_reasoning(agent, claims: List[str], strategy: str, response_language: str, max_concurrent: int) -> List[Dict[str, Any]]:
    """Helper to execute batch multilingual reasoning."""
    try:
        from agents.kg_reasoning_agent import ReasoningStrategy
        
        # Convert strategy string to enum
        try:
            reasoning_strategy = ReasoningStrategy(strategy)
        except ValueError:
            reasoning_strategy = ReasoningStrategy.COT
        
        # Call the agent's batch method
        results = await agent._batch_multilingual_reason_internal(
            claims=claims,
            strategy=reasoning_strategy,
            response_language=response_language,
            max_concurrent=max_concurrent
        )
        
        return results
    except Exception as e:
        # Return error for each claim
        return [{"error": str(e), "verdict": None, "confidence": 0.0} for _ in claims]

async def _batch_multilingual_reason_internal(agent, claims: List[str], strategy, response_language: str, max_concurrent: int) -> List[Dict[str, Any]]:
    """Internal batch reasoning method."""
    results = []
    
    # Process all claims in a single batch
    for claim in claims:
        try:
            result = await agent._execute_multilingual_reasoning(
                claim=claim,
                strategy=strategy,
                response_language=response_language,
                context=None
            )
            results.append(result)
        except Exception as e:
            results.append({
                "error": str(e),
                "verdict": None,
                "confidence": 0.0,
                "detected_language": "unknown",
                "reasoning_chain": [],
                "execution_time_ms": 0
            })
    
    return results

async def _batch_multilingual_reason(agent, claims: List[str], strategy: str, response_language: str, max_concurrent: int) -> Dict[str, Any]:
    """Call the agent's batch processing with concurrency control."""
    import asyncio
    from agents.kg_reasoning_agent import ReasoningStrategy
    
    # Convert strategy string to enum
    try:
        reasoning_strategy = ReasoningStrategy(strategy)
    except ValueError:
        reasoning_strategy = ReasoningStrategy.COT
    
    # Process claims with controlled concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(claim):
        async with semaphore:
            return await agent._execute_multilingual_reasoning(
                claim=claim,
                strategy=reasoning_strategy,
                response_language=response_language,
                context=None
            )
    
    # Process all claims concurrently with semaphore control
    results = await asyncio.gather(*[process_with_semaphore(claim) for claim in claims], return_exceptions=True)
    
    # Handle exceptions
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append({
                "error": str(result),
                "verdict": None,
                "confidence": 0.0,
                "detected_language": "unknown",
                "reasoning_chain": [],
                "execution_time_ms": 0
            })
        else:
            processed_results.append(result)
    
    return {
        "results": processed_results,
        "total_claims": len(claims),
        "successful": len([r for r in processed_results if "error" not in r])
    }

async def _extract_topics_from_collection(agent, texts: List[str], num_topics: int, context: Dict[str, Any]) -> Dict[str, Any]:
    """Extract topics from a collection of texts using BERTopic."""
    try:
        # Use the agent's BERTopic model directly (it's called bertopic_model, not topic_model)
        if hasattr(agent, 'bertopic_model') and agent.bertopic_model:
            topics, probs = agent.bertopic_model.fit_transform(texts)
            
            # Get topic info
            topic_info = agent.bertopic_model.get_topic_info()
            
            # Format results
            topics_list = []
            for idx, row in topic_info.iterrows():
                if row['Topic'] != -1:  # Skip outlier topic
                    topic_keywords = agent.bertopic_model.get_topic(row['Topic'])
                    topics_list.append({
                        "id": int(row['Topic']),
                        "name": f"Topic_{row['Topic']}",
                        "keywords": [word for word, score in topic_keywords[:5]],
                        "count": int(row['Count'])
                    })
            
            return {"topics": topics_list, "total_documents": len(texts)}
        else:
            return {"topics": [], "error": "BERTopic model not initialized"}
    except Exception as e:
        agent.logger.error(f"Topic extraction from collection failed: {e}")
        import traceback
        agent.logger.error(traceback.format_exc())
        return {"topics": [], "error": str(e)}

# Monkey patch the agents to add internal methods
def patch_agents():
    """Add internal methods to agents for direct calling."""
    from agents.pattern_detector_agent import PatternDetectorAgent
    from agents.evidence_gatherer_agent import EvidenceGathererAgent
    
    PatternDetectorAgent._detect_misinformation_patterns_internal = _detect_misinformation_patterns_internal
    EvidenceGathererAgent._gather_evidence_internal = _gather_evidence_internal
    
    # Add reasoning agent batch methods
    try:
        from agents.multilingual_kg_reasoning_agent import MultilingualKGReasoningAgent
        MultilingualKGReasoningAgent._execute_batch_multilingual_reasoning = _execute_batch_multilingual_reasoning
        MultilingualKGReasoningAgent._batch_multilingual_reason_internal = _batch_multilingual_reason_internal
        MultilingualKGReasoningAgent._batch_multilingual_reason = _batch_multilingual_reason
    except ImportError:
        pass



# This is a library module - no demo code here
# Use test_granular_with_real_posts.py to test with real data from port 7000
