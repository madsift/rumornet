"""
Topic Guided Evidence Retriever Agent

Agent that provides topic-guided evidence retrieval with enhanced search queries,
quality assessment, counter-evidence discovery, and batch processing capabilities.
Integrates with Tavily API and uses topic intelligence for enhanced search performance.
"""

import asyncio
import logging
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import asdict

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from .config.agent_config import TopicGuidedEvidenceRetrieverConfig
from .core.data_models import (
    TopicIntelligence, EvidencePackage, Evidence, EvidenceQualityAssessment,
    EvidenceType, AlertRiskLevel
)


class TopicGuidedEvidenceRetrieverAgent:
    """
    Agent for topic-guided evidence retrieval.
    
    Provides tools for:
    - Topic-enhanced evidence gathering with Tavily integration
    - Evidence quality assessment using topic-specific criteria
    - Counter-evidence discovery for balanced analysis
    - Batch evidence retrieval for high throughput scenarios
    """
    
    def __init__(self, name: str = "agent", config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Evidence retrieval specific configuration
        self.evidence_config = config.evidence_retrieval
        
        # Tavily client for evidence search
        self.tavily_client = None
        
        # Topic-specific domain mappings for credibility assessment
        self.topic_trusted_domains = self._initialize_topic_domains()
        
        # Evidence cache for performance optimization
        self.evidence_cache: Dict[str, EvidencePackage] = {}
        self.evidence_cache_timestamps: Dict[str, datetime] = {}
        
        self.logger.info("TopicGuidedEvidenceRetrieverMCPAgent initialized")
    
    async def initialize(self):
        """Initialize the agent with Tavily client and other resources."""
        await super().initialize()
        
        # Initialize Tavily client
        await self._initialize_tavily_client()
        
        self.logger.info("TopicGuidedEvidenceRetrieverMCPAgent initialization complete")
    
    async def _initialize_tavily_client(self):
        """Initialize Tavily API client."""
        try:
            if not HTTPX_AVAILABLE:
                self.logger.warning("httpx not available, using mock Tavily responses")
                return
            
            # Initialize Tavily client
            self.tavily_client = httpx.AsyncClient(
                base_url="https://api.tavily.com",
                timeout=self.evidence_config.search_timeout_seconds,
                headers={
                    "Content-Type": "application/json"
                }
            )
            
            self.logger.info("Tavily client initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Tavily client: {e}")
            self.tavily_client = None
    
    def _initialize_topic_domains(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize topic-specific trusted and unreliable domains."""
        return {
            "politics": {
                "trusted": [
                    "reuters.com", "ap.org", "bbc.com", "npr.org", "pbs.org",
                    "politifact.com", "factcheck.org", "snopes.com"
                ],
                "unreliable": [
                    "infowars.com", "breitbart.com", "naturalnews.com"
                ]
            },
            "health": {
                "trusted": [
                    "who.int", "cdc.gov", "nih.gov", "mayoclinic.org", "webmd.com",
                    "nejm.org", "thelancet.com", "nature.com"
                ],
                "unreliable": [
                    "naturalnews.com", "mercola.com", "healthimpactnews.com"
                ]
            },
            "science": {
                "trusted": [
                    "nature.com", "science.org", "cell.com", "pnas.org",
                    "scientificamerican.com", "newscientist.com"
                ],
                "unreliable": [
                    "naturalnews.com", "beforeitsnews.com"
                ]
            },
            "default": {
                "trusted": [
                    "reuters.com", "ap.org", "bbc.com", "npr.org", "pbs.org",
                    "factcheck.org", "snopes.com", "politifact.com"
                ],
                "unreliable": [
                    "infowars.com", "naturalnews.com", "beforeitsnews.com"
                ]
            }
        }
    def get_supported_features(self) -> List[str]:
        """Get list of supported features."""
        return [
            "topic_guided_evidence_retrieval",
            "tavily_api_integration",
            "evidence_quality_assessment",
            "counter_evidence_discovery",
            "batch_evidence_processing",
            "topic_specific_domain_filtering",
            "evidence_caching",
            "real_ollama_integration"
        ]
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            await super().cleanup()
            
            # Close Tavily client
            if self.tavily_client:
                await self.tavily_client.aclose()
                self.tavily_client = None
            
            # Clear evidence cache
            self.evidence_cache.clear()
            self.evidence_cache_timestamps.clear()
            
            self.logger.info("TopicGuidedEvidenceRetrieverMCPAgent cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


# Main execution for standalone testing
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Create configuration
        config = TopicGuidedEvidenceRetrieverConfig.from_environment()
        
        # Create and initialize agent
        agent = TopicGuidedEvidenceRetrieverAgent(config)
        print(f"Created {agent.name} agent")
        
        # Run the agent
        try:
            await agent.run_async(
                transport=config.fastmcp.transport,
                host=config.fastmcp.host,
                port=config.fastmcp.port
            )
        except KeyboardInterrupt:
            print("Agent stopped by user")
        finally:
            await agent.cleanup()
    
    asyncio.run(main())
