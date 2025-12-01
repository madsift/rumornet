"""
Property-based tests for agent status consistency.

**Feature: agent-monitoring-dashboard, Property 1: Agent status consistency**
**Validates: Requirements 1.2, 1.3**

This module tests that agent status updates maintain consistency:
- When an agent transitions from "executing" to "completed", execution time > 0
- When an agent transitions from "executing" to "completed", end_time > start_time
"""

import pytest
from hypothesis import given, strategies as st, settings
from datetime import datetime, timedelta
from unittest.mock import Mock

from dashboard.core.orchestrator_monitor import OrchestratorMonitor
from dashboard.models.data_models import AgentStatus


# Mock orchestrator for testing
class MockOrchestrator:
    """Mock GranularMisinformationOrchestrator for testing."""
    
    def __init__(self):
        self.agents = {}
        self.post_analyses = []
        self.user_profiles = {}
        self.temporal_patterns = {}
        self.topic_analyses = {}
    
    async def initialize_agents(self):
        """Mock agent initialization."""
        self.agents = {
            "reasoning": Mock(),
            "pattern": Mock(),
            "evidence": Mock()
        }
    
    async def analyze_batch_true_batch(self, posts):
        """Mock batch analysis."""
        return [{"metadata": {"post_id": f"post_{i}"}, "analysis": {}} for i in range(len(posts))]
    
    async def analyze_post_with_metadata(self, post):
        """Mock single post analysis."""
        return {"metadata": {"post_id": post.get("post_id", "unknown")}, "analysis": {}}
    
    def generate_actionable_report(self):
        """Mock report generation."""
        return {"executive_summary": {}}


@given(
    execution_time_ms=st.floats(min_value=1.0, max_value=100000.0)
)
@settings(max_examples=100)
def test_completed_agent_has_positive_execution_time(execution_time_ms):
    """
    Property 1: Agent status consistency (execution time)
    
    For any agent that completes execution, the execution time must be > 0.
    This ensures that when an agent transitions to "completed", we have valid timing data.
    """
    monitor = OrchestratorMonitor(MockOrchestrator())
    
    agent_name = "test_agent"
    
    # Simulate agent execution
    start_time = datetime.now()
    monitor._update_agent_status(agent_name, "executing")
    
    # Set start time manually for testing
    monitor.agent_statuses[agent_name].start_time = start_time
    
    # Complete the agent
    end_time = start_time + timedelta(milliseconds=execution_time_ms)
    monitor.agent_statuses[agent_name].end_time = end_time
    monitor._update_agent_status(agent_name, "completed")
    
    # Verify execution time is positive
    status = monitor.get_agent_status(agent_name)
    assert status is not None
    assert status.status == "completed"
    assert status.execution_time_ms > 0, \
        f"Execution time must be > 0 for completed agent, got {status.execution_time_ms}"


@given(
    time_delta_ms=st.integers(min_value=1, max_value=100000)
)
@settings(max_examples=100)
def test_completed_agent_end_time_after_start_time(time_delta_ms):
    """
    Property 1: Agent status consistency (time ordering)
    
    For any agent that completes execution, end_time must be after start_time.
    This ensures temporal consistency in agent execution tracking.
    """
    monitor = OrchestratorMonitor(MockOrchestrator())
    
    agent_name = "test_agent"
    
    # Simulate agent execution with specific timing
    start_time = datetime.now()
    end_time = start_time + timedelta(milliseconds=time_delta_ms)
    
    # Start execution
    monitor._update_agent_status(agent_name, "executing")
    monitor.agent_statuses[agent_name].start_time = start_time
    
    # Complete execution
    monitor.agent_statuses[agent_name].end_time = end_time
    monitor._update_agent_status(agent_name, "completed")
    
    # Verify time ordering
    status = monitor.get_agent_status(agent_name)
    assert status is not None
    assert status.start_time is not None
    assert status.end_time is not None
    assert status.end_time > status.start_time, \
        f"End time must be after start time: {status.start_time} -> {status.end_time}"


@given(
    agent_name=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
    posts_processed=st.integers(min_value=1, max_value=1000)
)
@settings(max_examples=100)
def test_agent_posts_processed_tracking(agent_name, posts_processed):
    """
    Property 1: Agent status consistency (posts tracking)
    
    For any agent execution, the posts_processed count must accurately reflect
    the number of posts the agent has processed.
    """
    monitor = OrchestratorMonitor(MockOrchestrator())
    
    # Start execution
    monitor._update_agent_status(agent_name, "executing")
    
    # Complete with posts processed
    monitor._update_agent_status(
        agent_name,
        "completed",
        posts_processed=posts_processed
    )
    
    # Verify posts processed
    status = monitor.get_agent_status(agent_name)
    assert status is not None
    assert status.posts_processed == posts_processed, \
        f"Posts processed should be {posts_processed}, got {status.posts_processed}"


@given(
    num_agents=st.integers(min_value=1, max_value=10),
    execution_time_ms=st.floats(min_value=1.0, max_value=10000.0)
)
@settings(max_examples=100)
def test_multiple_agents_status_consistency(num_agents, execution_time_ms):
    """
    Property 1: Agent status consistency (multiple agents)
    
    For any set of agents executing concurrently, each agent's status
    must be independently consistent with timing constraints.
    """
    monitor = OrchestratorMonitor(MockOrchestrator())
    
    agent_names = [f"agent_{i}" for i in range(num_agents)]
    
    # Start all agents
    start_times = {}
    for agent_name in agent_names:
        start_time = datetime.now()
        monitor._update_agent_status(agent_name, "executing")
        monitor.agent_statuses[agent_name].start_time = start_time
        start_times[agent_name] = start_time
    
    # Complete all agents
    for agent_name in agent_names:
        end_time = start_times[agent_name] + timedelta(milliseconds=execution_time_ms)
        monitor.agent_statuses[agent_name].end_time = end_time
        monitor._update_agent_status(agent_name, "completed")
    
    # Verify all agents have consistent status
    for agent_name in agent_names:
        status = monitor.get_agent_status(agent_name)
        assert status is not None
        assert status.status == "completed"
        assert status.execution_time_ms > 0
        assert status.end_time > status.start_time


@given(
    agent_name=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')))
)
@settings(max_examples=100)
def test_failed_agent_preserves_timing_data(agent_name):
    """
    Property 1: Agent status consistency (failure case)
    
    For any agent that fails during execution, timing data must still be
    consistent (execution time > 0, end_time > start_time).
    """
    monitor = OrchestratorMonitor(MockOrchestrator())
    
    # Start execution
    start_time = datetime.now()
    monitor._update_agent_status(agent_name, "executing")
    monitor.agent_statuses[agent_name].start_time = start_time
    
    # Fail the agent
    end_time = start_time + timedelta(milliseconds=100)
    monitor.agent_statuses[agent_name].end_time = end_time
    monitor._update_agent_status(agent_name, "failed", error="Test error")
    
    # Verify timing consistency even for failed agent
    status = monitor.get_agent_status(agent_name)
    assert status is not None
    assert status.status == "failed"
    assert status.error == "Test error"
    assert status.execution_time_ms > 0
    assert status.end_time > status.start_time


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
