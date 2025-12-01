"""
State management for the Agent Monitoring Dashboard.

This module provides functions for initializing and managing Streamlit session state,
including agent statuses, execution history, and configuration.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from models.data_models import (
    AgentStatus,
    ExecutionMetrics,
    ExecutionResult,
    DashboardConfig
)


def initialize_session_state(session_state: Any) -> None:
    """Initialize session state variables if they don't exist.
    
    Args:
        session_state: Streamlit session state object (st.session_state)
    """
    # Configuration
    if "config" not in session_state:
        session_state.config = DashboardConfig.default()
    
    # Agent statuses
    if "agent_statuses" not in session_state:
        session_state.agent_statuses = {}
    
    # Execution history
    if "execution_history" not in session_state:
        session_state.execution_history = []
    
    # Current execution
    if "current_execution_id" not in session_state:
        session_state.current_execution_id = None
    
    # Current execution result
    if "current_result" not in session_state:
        session_state.current_result = None
    
    # Metrics
    if "metrics" not in session_state:
        session_state.metrics = None
    
    # Filters
    if "filters" not in session_state:
        session_state.filters = {
            "search_query": "",
            "risk_level": "all",
            "confidence_threshold": 0.0,
            "pattern_filter": "all"
        }
    
    # UI state
    if "selected_history_id" not in session_state:
        session_state.selected_history_id = None
    
    if "show_errors" not in session_state:
        session_state.show_errors = False
    
    if "error_log" not in session_state:
        session_state.error_log = []


def update_agent_status(
    session_state: Any,
    agent_name: str,
    status: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    execution_time_ms: float = 0.0,
    posts_processed: int = 0,
    error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Update the status of a specific agent.
    
    Args:
        session_state: Streamlit session state object
        agent_name: Name of the agent to update
        status: New status (idle, executing, completed, failed)
        start_time: Start time of execution
        end_time: End time of execution
        execution_time_ms: Execution time in milliseconds
        posts_processed: Number of posts processed
        error: Error message if failed
        metadata: Additional metadata
    """
    if agent_name not in session_state.agent_statuses:
        session_state.agent_statuses[agent_name] = AgentStatus(
            agent_name=agent_name,
            status=status,
            start_time=start_time,
            end_time=end_time,
            execution_time_ms=execution_time_ms,
            posts_processed=posts_processed,
            error=error,
            metadata=metadata or {}
        )
    else:
        agent_status = session_state.agent_statuses[agent_name]
        agent_status.status = status
        if start_time is not None:
            agent_status.start_time = start_time
        if end_time is not None:
            agent_status.end_time = end_time
        if execution_time_ms > 0:
            agent_status.execution_time_ms = execution_time_ms
        if posts_processed > 0:
            agent_status.posts_processed = posts_processed
        if error is not None:
            agent_status.error = error
        if metadata is not None:
            agent_status.metadata.update(metadata)


def get_agent_status(session_state: Any, agent_name: str) -> Optional[AgentStatus]:
    """Get the status of a specific agent.
    
    Args:
        session_state: Streamlit session state object
        agent_name: Name of the agent
        
    Returns:
        AgentStatus object or None if not found
    """
    return session_state.agent_statuses.get(agent_name)


def get_all_agent_statuses(session_state: Any) -> Dict[str, AgentStatus]:
    """Get statuses of all agents.
    
    Args:
        session_state: Streamlit session state object
        
    Returns:
        Dictionary mapping agent names to AgentStatus objects
    """
    return session_state.agent_statuses.copy()


def reset_agent_statuses(session_state: Any) -> None:
    """Reset all agent statuses to idle.
    
    Args:
        session_state: Streamlit session state object
    """
    for agent_name in session_state.agent_statuses:
        session_state.agent_statuses[agent_name].status = "idle"
        session_state.agent_statuses[agent_name].start_time = None
        session_state.agent_statuses[agent_name].end_time = None
        session_state.agent_statuses[agent_name].execution_time_ms = 0.0
        session_state.agent_statuses[agent_name].posts_processed = 0
        session_state.agent_statuses[agent_name].error = None


def add_execution_result(
    session_state: Any,
    result: ExecutionResult
) -> None:
    """Add an execution result to history.
    
    Args:
        session_state: Streamlit session state object
        result: ExecutionResult to add
    """
    session_state.execution_history.append(result)
    
    # Trim history if it exceeds max items
    max_items = session_state.config.max_history_items
    if len(session_state.execution_history) > max_items:
        session_state.execution_history = session_state.execution_history[-max_items:]


def get_execution_history(session_state: Any) -> List[ExecutionResult]:
    """Get execution history.
    
    Args:
        session_state: Streamlit session state object
        
    Returns:
        List of ExecutionResult objects
    """
    return session_state.execution_history.copy()


def get_execution_by_id(
    session_state: Any,
    execution_id: str
) -> Optional[ExecutionResult]:
    """Get a specific execution result by ID.
    
    Args:
        session_state: Streamlit session state object
        execution_id: ID of the execution to retrieve
        
    Returns:
        ExecutionResult or None if not found
    """
    for result in session_state.execution_history:
        if result.execution_id == execution_id:
            return result
    return None


def clear_execution_history(session_state: Any) -> None:
    """Clear all execution history.
    
    Args:
        session_state: Streamlit session state object
    """
    session_state.execution_history = []
    session_state.selected_history_id = None


def update_config(
    session_state: Any,
    config: DashboardConfig
) -> None:
    """Update dashboard configuration.
    
    Args:
        session_state: Streamlit session state object
        config: New configuration
    """
    session_state.config = config


def get_config(session_state: Any) -> DashboardConfig:
    """Get current dashboard configuration.
    
    Args:
        session_state: Streamlit session state object
        
    Returns:
        Current DashboardConfig
    """
    return session_state.config


def update_filters(
    session_state: Any,
    search_query: Optional[str] = None,
    risk_level: Optional[str] = None,
    confidence_threshold: Optional[float] = None,
    pattern_filter: Optional[str] = None
) -> None:
    """Update result filters.
    
    Args:
        session_state: Streamlit session state object
        search_query: Search query for post/user IDs
        risk_level: Risk level filter (all, high, medium, low)
        confidence_threshold: Minimum confidence threshold
        pattern_filter: Pattern type filter
    """
    if search_query is not None:
        session_state.filters["search_query"] = search_query
    if risk_level is not None:
        session_state.filters["risk_level"] = risk_level
    if confidence_threshold is not None:
        session_state.filters["confidence_threshold"] = confidence_threshold
    if pattern_filter is not None:
        session_state.filters["pattern_filter"] = pattern_filter


def get_filters(session_state: Any) -> Dict[str, Any]:
    """Get current filters.
    
    Args:
        session_state: Streamlit session state object
        
    Returns:
        Dictionary of current filter values
    """
    return session_state.filters.copy()


def log_error(
    session_state: Any,
    component: str,
    error_message: str,
    stack_trace: Optional[str] = None
) -> None:
    """Log an error to the error log.
    
    Args:
        session_state: Streamlit session state object
        component: Name of the component where error occurred
        error_message: Error message
        stack_trace: Optional stack trace
    """
    error_entry = {
        "timestamp": datetime.now(),
        "component": component,
        "message": error_message,
        "stack_trace": stack_trace
    }
    session_state.error_log.append(error_entry)


def get_error_log(session_state: Any) -> List[Dict[str, Any]]:
    """Get the error log.
    
    Args:
        session_state: Streamlit session state object
        
    Returns:
        List of error entries
    """
    return session_state.error_log.copy()


def clear_error_log(session_state: Any) -> None:
    """Clear the error log.
    
    Args:
        session_state: Streamlit session state object
    """
    session_state.error_log = []


def generate_execution_id() -> str:
    """Generate a unique execution ID.
    
    Returns:
        Unique execution ID string
    """
    return str(uuid.uuid4())


def start_new_execution(session_state: Any) -> str:
    """Start a new execution and return its ID.
    
    Args:
        session_state: Streamlit session state object
        
    Returns:
        New execution ID
    """
    execution_id = generate_execution_id()
    session_state.current_execution_id = execution_id
    reset_agent_statuses(session_state)
    return execution_id


def set_current_result(
    session_state: Any,
    result: ExecutionResult
) -> None:
    """Set the current execution result.
    
    Args:
        session_state: Streamlit session state object
        result: ExecutionResult to set as current
    """
    session_state.current_result = result


def get_current_result(session_state: Any) -> Optional[ExecutionResult]:
    """Get the current execution result.
    
    Args:
        session_state: Streamlit session state object
        
    Returns:
        Current ExecutionResult or None
    """
    return session_state.current_result
