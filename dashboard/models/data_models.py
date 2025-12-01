"""
Core data models for the Agent Monitoring Dashboard.

This module defines the data structures used throughout the dashboard
for tracking agent status, execution metrics, results, and configuration.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List


@dataclass
class AgentStatus:
    """Track individual agent status during execution.
    
    Attributes:
        agent_name: Name of the agent
        status: Current status (idle, executing, completed, failed)
        start_time: When the agent started execution
        end_time: When the agent completed execution
        execution_time_ms: Total execution time in milliseconds
        posts_processed: Number of posts processed by this agent
        error: Error message if the agent failed
        metadata: Additional metadata about the execution
    """
    agent_name: str
    status: str  # "idle", "executing", "completed", "failed"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time_ms: float = 0.0
    posts_processed: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_name": self.agent_name,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "execution_time_ms": self.execution_time_ms,
            "posts_processed": self.posts_processed,
            "error": self.error,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentStatus":
        """Create from dictionary."""
        return cls(
            agent_name=data["agent_name"],
            status=data["status"],
            start_time=datetime.fromisoformat(data["start_time"]) if data.get("start_time") else None,
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            execution_time_ms=data.get("execution_time_ms", 0.0),
            posts_processed=data.get("posts_processed", 0),
            error=data.get("error"),
            metadata=data.get("metadata", {})
        )


@dataclass
class ExecutionMetrics:
    """Aggregated execution metrics across all agents.
    
    Attributes:
        total_executions: Total number of executions
        successful_executions: Number of successful executions
        failed_executions: Number of failed executions
        total_execution_time_ms: Total execution time across all runs
        average_execution_time_ms: Average execution time per run
        posts_per_second: Throughput in posts per second
        agent_metrics: Per-agent metrics breakdown
    """
    total_executions: int
    successful_executions: int
    failed_executions: int
    total_execution_time_ms: float
    average_execution_time_ms: float
    posts_per_second: float
    agent_metrics: Dict[str, Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "total_execution_time_ms": self.total_execution_time_ms,
            "average_execution_time_ms": self.average_execution_time_ms,
            "posts_per_second": self.posts_per_second,
            "agent_metrics": self.agent_metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionMetrics":
        """Create from dictionary."""
        return cls(
            total_executions=data["total_executions"],
            successful_executions=data["successful_executions"],
            failed_executions=data["failed_executions"],
            total_execution_time_ms=data["total_execution_time_ms"],
            average_execution_time_ms=data["average_execution_time_ms"],
            posts_per_second=data["posts_per_second"],
            agent_metrics=data["agent_metrics"]
        )


@dataclass
class ExecutionResult:
    """Complete result of an analysis execution.
    
    Attributes:
        execution_id: Unique identifier for this execution
        timestamp: When the execution occurred
        total_posts: Total number of posts in the batch
        posts_analyzed: Number of posts successfully analyzed
        misinformation_detected: Number of posts flagged as misinformation
        high_risk_posts: Number of high-risk posts detected
        execution_time_ms: Total execution time in milliseconds
        agent_statuses: Status of each agent during execution
        full_report: Complete analysis report from orchestrator
        markdown_report: Generated markdown summary
    """
    execution_id: str
    timestamp: datetime
    total_posts: int
    posts_analyzed: int
    misinformation_detected: int
    high_risk_posts: int
    execution_time_ms: float
    agent_statuses: Dict[str, AgentStatus]
    full_report: Dict[str, Any]
    markdown_report: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "execution_id": self.execution_id,
            "timestamp": self.timestamp.isoformat(),
            "total_posts": self.total_posts,
            "posts_analyzed": self.posts_analyzed,
            "misinformation_detected": self.misinformation_detected,
            "high_risk_posts": self.high_risk_posts,
            "execution_time_ms": self.execution_time_ms,
            "agent_statuses": {name: status.to_dict() for name, status in self.agent_statuses.items()},
            "full_report": self.full_report,
            "markdown_report": self.markdown_report
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionResult":
        """Create from dictionary."""
        return cls(
            execution_id=data["execution_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            total_posts=data["total_posts"],
            posts_analyzed=data["posts_analyzed"],
            misinformation_detected=data["misinformation_detected"],
            high_risk_posts=data["high_risk_posts"],
            execution_time_ms=data["execution_time_ms"],
            agent_statuses={
                name: AgentStatus.from_dict(status_dict) 
                for name, status_dict in data["agent_statuses"].items()
            },
            full_report=data["full_report"],
            markdown_report=data["markdown_report"]
        )


@dataclass
class DashboardConfig:
    """Configuration settings for the dashboard.
    
    Attributes:
        ollama_endpoint: URL for Ollama API endpoint
        ollama_model: Model name to use for analysis
        auto_refresh_interval: Seconds between auto-refreshes
        max_history_items: Maximum number of history items to keep
        default_batch_size: Default batch size for processing
        enable_debug_mode: Whether to enable debug logging
    """
    ollama_endpoint: str
    ollama_model: str
    auto_refresh_interval: int  # seconds
    max_history_items: int
    default_batch_size: int
    enable_debug_mode: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "ollama_endpoint": self.ollama_endpoint,
            "ollama_model": self.ollama_model,
            "auto_refresh_interval": self.auto_refresh_interval,
            "max_history_items": self.max_history_items,
            "default_batch_size": self.default_batch_size,
            "enable_debug_mode": self.enable_debug_mode
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DashboardConfig":
        """Create from dictionary."""
        return cls(
            ollama_endpoint=data["ollama_endpoint"],
            ollama_model=data["ollama_model"],
            auto_refresh_interval=data["auto_refresh_interval"],
            max_history_items=data["max_history_items"],
            default_batch_size=data["default_batch_size"],
            enable_debug_mode=data["enable_debug_mode"]
        )
    
    @classmethod
    def default(cls) -> "DashboardConfig":
        """Create default configuration."""
        return cls(
            ollama_endpoint="http://192.168.10.68:11434",
            ollama_model="gemma3:4b",
            auto_refresh_interval=5,
            max_history_items=50,
            default_batch_size=10,
            enable_debug_mode=False
        )
