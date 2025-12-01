"""
Orchestrator monitoring wrapper for the Agent Monitoring Dashboard.

This module wraps the GranularMisinformationOrchestrator with monitoring capabilities
to track agent execution status, collect performance metrics, and provide real-time updates.
"""

import asyncio
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict

from models.data_models import AgentStatus, ExecutionMetrics


class OrchestratorMonitor:
    """
    Monitor wrapper for GranularMisinformationOrchestrator.
    
    Tracks agent execution status, collects performance metrics,
    and provides real-time status updates during analysis.
    """
    
    def __init__(self, orchestrator):
        """
        Initialize the orchestrator monitor.
        
        Args:
            orchestrator: GranularMisinformationOrchestrator instance to monitor
        """
        self.orchestrator = orchestrator
        self.logger = logging.getLogger(f"{__name__}.monitor")
        
        # Agent status tracking
        self.agent_statuses: Dict[str, AgentStatus] = {}
        
        # Metrics tracking
        self.execution_history: List[Dict[str, Any]] = []
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0
        
        # Initialize agent statuses
        self._initialize_agent_statuses()
    
    def _initialize_agent_statuses(self):
        """Initialize status tracking for all known agents."""
        # Standard agents in the orchestrator
        agent_names = [
            "reasoning",
            "pattern",
            "evidence",
            "social_behavior",
            "topic_modeling",
            "topic_evidence",
            "coordination_detector",
            "echo_chamber_detector"
        ]
        
        for agent_name in agent_names:
            self.agent_statuses[agent_name] = AgentStatus(
                agent_name=agent_name,
                status="idle"
            )
    
    async def analyze_with_monitoring(
        self,
        posts: List[Dict[str, Any]],
        use_batch: bool = True
    ) -> Dict[str, Any]:
        """
        Execute analysis with real-time monitoring.
        
        Args:
            posts: List of posts to analyze
            use_batch: Whether to use batch processing (True) or sequential (False)
            
        Returns:
            Complete analysis results with monitoring data
        """
        start_time = time.time()
        self.total_executions += 1
        
        self.logger.info(f"Starting monitored analysis of {len(posts)} posts")
        
        # Reset all agent statuses to idle
        self._reset_all_statuses()
        
        try:
            # Initialize orchestrator agents if not already done
            if not self.orchestrator.agents:
                self.logger.info("Initializing orchestrator agents...")
                await self.orchestrator.initialize_agents()
            
            # Track which agents are available
            available_agents = list(self.orchestrator.agents.keys())
            self.logger.info(f"Available agents: {available_agents}")
            
            # Execute analysis based on mode
            if use_batch:
                results = await self._analyze_batch_with_monitoring(posts)
            else:
                results = await self._analyze_sequential_with_monitoring(posts)
            
            # Mark execution as successful
            self.successful_executions += 1
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Generate report
            report = self.orchestrator.generate_actionable_report()
            
            # Record execution
            execution_record = {
                "execution_id": f"exec_{int(time.time())}",
                "timestamp": datetime.now().isoformat(),
                "total_posts": len(posts),
                "execution_time_ms": execution_time_ms,
                "agent_statuses": {
                    name: status.__dict__.copy()
                    for name, status in self.agent_statuses.items()
                },
                "success": True
            }
            self.execution_history.append(execution_record)
            
            return {
                "status": "success",
                "results": results,
                "report": report,
                "execution_time_ms": execution_time_ms,
                "agent_statuses": self.get_all_statuses(),
                "metrics": self.get_metrics()
            }
            
        except Exception as e:
            self.failed_executions += 1
            execution_time_ms = (time.time() - start_time) * 1000
            
            self.logger.error(f"Analysis failed: {e}")
            
            # Mark all executing agents as failed
            for agent_name, status in self.agent_statuses.items():
                if status.status == "executing":
                    self._update_agent_status(
                        agent_name,
                        "failed",
                        error=str(e)
                    )
            
            # Record failed execution
            execution_record = {
                "execution_id": f"exec_{int(time.time())}",
                "timestamp": datetime.now().isoformat(),
                "total_posts": len(posts),
                "execution_time_ms": execution_time_ms,
                "agent_statuses": {
                    name: status.__dict__.copy()
                    for name, status in self.agent_statuses.items()
                },
                "success": False,
                "error": str(e)
            }
            self.execution_history.append(execution_record)
            
            return {
                "status": "error",
                "error": str(e),
                "execution_time_ms": execution_time_ms,
                "agent_statuses": self.get_all_statuses(),
                "metrics": self.get_metrics()
            }
    
    async def _analyze_batch_with_monitoring(
        self,
        posts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze posts in batch mode with monitoring."""
        self.logger.info("Using batch analysis mode")
        
        # Track reasoning agent
        if "reasoning" in self.orchestrator.agents:
            self._update_agent_status("reasoning", "executing")
        
        # Track pattern agent
        if "pattern" in self.orchestrator.agents:
            self._update_agent_status("pattern", "executing")
        
        # Execute batch analysis
        results = await self.orchestrator.analyze_batch_true_batch(posts)
        
        # Mark agents as completed
        if "reasoning" in self.orchestrator.agents:
            self._update_agent_status(
                "reasoning",
                "completed",
                posts_processed=len(posts)
            )
        
        if "pattern" in self.orchestrator.agents:
            self._update_agent_status(
                "pattern",
                "completed",
                posts_processed=len(posts)
            )
        
        # Track topic modeling if it ran
        if len(posts) > 5:
            self._update_agent_status("topic_modeling", "executing")
            # Topic modeling happens inside analyze_batch_true_batch
            self._update_agent_status(
                "topic_modeling",
                "completed",
                posts_processed=len(posts)
            )
        
        # Track social behavior analysis if it ran
        if "social_behavior" in self.orchestrator.agents and len(results) >= 3:
            self._update_agent_status("social_behavior", "executing")
            # Social behavior analysis happens inside analyze_batch_true_batch
            self._update_agent_status(
                "social_behavior",
                "completed",
                posts_processed=len(posts)
            )
        
        return results
    
    async def _analyze_sequential_with_monitoring(
        self,
        posts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze posts sequentially with monitoring."""
        self.logger.info("Using sequential analysis mode")
        
        results = []
        
        for i, post in enumerate(posts, 1):
            self.logger.info(f"Processing post {i}/{len(posts)}")
            
            # Track reasoning agent for this post
            if "reasoning" in self.orchestrator.agents:
                self._update_agent_status("reasoning", "executing")
            
            # Track pattern agent for this post
            if "pattern" in self.orchestrator.agents:
                self._update_agent_status("pattern", "executing")
            
            # Analyze single post
            result = await self.orchestrator.analyze_post_with_metadata(post)
            results.append(result)
            
            # Mark agents as completed for this post
            if "reasoning" in self.orchestrator.agents:
                self._update_agent_status(
                    "reasoning",
                    "completed",
                    posts_processed=1
                )
            
            if "pattern" in self.orchestrator.agents:
                self._update_agent_status(
                    "pattern",
                    "completed",
                    posts_processed=1
                )
        
        return results
    
    def _update_agent_status(
        self,
        agent_name: str,
        status: str,
        posts_processed: int = 0,
        error: Optional[str] = None
    ):
        """Update the status of a specific agent."""
        if agent_name not in self.agent_statuses:
            self.agent_statuses[agent_name] = AgentStatus(
                agent_name=agent_name,
                status=status
            )
        
        agent_status = self.agent_statuses[agent_name]
        
        # Update status
        if status == "executing":
            agent_status.status = "executing"
            agent_status.start_time = datetime.now()
            agent_status.end_time = None
            agent_status.execution_time_ms = 0.0
            agent_status.error = None
            
        elif status in ["completed", "failed"]:
            agent_status.status = status
            
            # Set end time if not already set
            if agent_status.end_time is None:
                agent_status.end_time = datetime.now()
            
            # Calculate execution time
            if agent_status.start_time and agent_status.end_time:
                time_delta = agent_status.end_time - agent_status.start_time
                agent_status.execution_time_ms = time_delta.total_seconds() * 1000
            
            # Update posts processed
            if posts_processed > 0:
                agent_status.posts_processed += posts_processed
            
            # Set error if failed
            if error:
                agent_status.error = error
        
        else:
            agent_status.status = status
    
    def _reset_all_statuses(self):
        """Reset all agent statuses to idle."""
        for agent_name in self.agent_statuses:
            self.agent_statuses[agent_name].status = "idle"
            self.agent_statuses[agent_name].start_time = None
            self.agent_statuses[agent_name].end_time = None
            self.agent_statuses[agent_name].execution_time_ms = 0.0
            self.agent_statuses[agent_name].posts_processed = 0
            self.agent_statuses[agent_name].error = None
    
    def get_agent_status(self, agent_name: str) -> Optional[AgentStatus]:
        """
        Get the status of a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            AgentStatus object or None if not found
        """
        return self.agent_statuses.get(agent_name)
    
    def get_all_statuses(self) -> Dict[str, AgentStatus]:
        """
        Get statuses of all agents.
        
        Returns:
            Dictionary mapping agent names to AgentStatus objects
        """
        return self.agent_statuses.copy()
    
    def get_metrics(self) -> ExecutionMetrics:
        """
        Get aggregated performance metrics.
        
        Returns:
            ExecutionMetrics object with performance data
        """
        # Calculate total execution time
        total_execution_time_ms = sum(
            status.execution_time_ms
            for status in self.agent_statuses.values()
        )
        
        # Calculate average execution time
        avg_execution_time_ms = (
            total_execution_time_ms / self.total_executions
            if self.total_executions > 0
            else 0.0
        )
        
        # Calculate posts per second
        total_posts_processed = sum(
            status.posts_processed
            for status in self.agent_statuses.values()
        )
        
        posts_per_second = (
            total_posts_processed / (total_execution_time_ms / 1000)
            if total_execution_time_ms > 0
            else 0.0
        )
        
        # Per-agent metrics
        agent_metrics = {}
        for agent_name, status in self.agent_statuses.items():
            agent_metrics[agent_name] = {
                "total_executions": self.total_executions,
                "posts_processed": status.posts_processed,
                "avg_execution_time_ms": status.execution_time_ms,
                "current_status": status.status,
                "last_error": status.error
            }
        
        return ExecutionMetrics(
            total_executions=self.total_executions,
            successful_executions=self.successful_executions,
            failed_executions=self.failed_executions,
            total_execution_time_ms=total_execution_time_ms,
            average_execution_time_ms=avg_execution_time_ms,
            posts_per_second=posts_per_second,
            agent_metrics=agent_metrics
        )
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """
        Get execution history.
        
        Returns:
            List of execution records
        """
        return self.execution_history.copy()
