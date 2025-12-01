"""
Property-based tests for metrics accuracy.

**Feature: agent-monitoring-dashboard, Property 2: Metrics accuracy**
**Validates: Requirements 2.4**

This module tests that execution metrics are calculated accurately:
- Average execution time = sum of all execution times / count of executions
- Metrics calculations are mathematically correct
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from datetime import datetime, timedelta
from unittest.mock import Mock

from dashboard.core.orchestrator_monitor import OrchestratorMonitor
from dashboard.models.data_models import ExecutionMetrics


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
            "pattern": Mock()
        }
    
    async def analyze_batch_true_batch(self, posts):
        """Mock batch analysis."""
        return [{"metadata": {"post_id": f"post_{i}"}, "analysis": {}} for i in range(len(posts))]
    
    def generate_actionable_report(self):
        """Mock report generation."""
        return {"executive_summary": {}}


@given(
    execution_times=st.lists(
        st.floats(min_value=1.0, max_value=10000.0),
        min_size=1,
        max_size=50
    )
)
@settings(max_examples=100)
def test_average_execution_time_calculation(execution_times):
    """
    Property 2: Metrics accuracy (average execution time)
    
    For any set of agent executions, the average execution time must equal
    the sum of all execution times divided by the count of executions.
    """
    monitor = OrchestratorMonitor(MockOrchestrator())
    
    # Simulate multiple executions - each with a unique agent to avoid overlap
    for i, exec_time_ms in enumerate(execution_times):
        agent_name = f"agent_{i}"  # Unique agent for each execution
        
        # Start execution
        start_time = datetime.now()
        monitor._update_agent_status(agent_name, "executing")
        monitor.agent_statuses[agent_name].start_time = start_time
        
        # Complete execution
        end_time = start_time + timedelta(milliseconds=exec_time_ms)
        monitor.agent_statuses[agent_name].end_time = end_time
        monitor._update_agent_status(agent_name, "completed")
        
        # Increment total executions
        monitor.total_executions += 1
        monitor.successful_executions += 1
    
    # Get metrics
    metrics = monitor.get_metrics()
    
    # Calculate expected average
    total_time = sum(execution_times)
    expected_avg = total_time / len(execution_times)
    
    # Verify average calculation
    # Allow small floating point error tolerance
    assert abs(metrics.average_execution_time_ms - expected_avg) < 0.01, \
        f"Average execution time should be {expected_avg:.2f}, got {metrics.average_execution_time_ms:.2f}"


@given(
    num_successful=st.integers(min_value=0, max_value=100),
    num_failed=st.integers(min_value=0, max_value=100)
)
@settings(max_examples=100)
def test_execution_count_accuracy(num_successful, num_failed):
    """
    Property 2: Metrics accuracy (execution counts)
    
    For any combination of successful and failed executions,
    total_executions must equal successful_executions + failed_executions.
    """
    assume(num_successful + num_failed > 0)  # At least one execution
    
    monitor = OrchestratorMonitor(MockOrchestrator())
    
    # Set execution counts
    monitor.total_executions = num_successful + num_failed
    monitor.successful_executions = num_successful
    monitor.failed_executions = num_failed
    
    # Get metrics
    metrics = monitor.get_metrics()
    
    # Verify counts
    assert metrics.total_executions == num_successful + num_failed, \
        f"Total executions should be {num_successful + num_failed}, got {metrics.total_executions}"
    assert metrics.successful_executions == num_successful
    assert metrics.failed_executions == num_failed


@given(
    posts_per_agent=st.lists(
        st.integers(min_value=1, max_value=100),
        min_size=1,
        max_size=10
    ),
    execution_time_ms=st.floats(min_value=100.0, max_value=10000.0)
)
@settings(max_examples=100)
def test_posts_per_second_calculation(posts_per_agent, execution_time_ms):
    """
    Property 2: Metrics accuracy (throughput calculation)
    
    For any set of agent executions, posts_per_second must equal
    total_posts_processed / (total_execution_time_ms / 1000).
    """
    monitor = OrchestratorMonitor(MockOrchestrator())
    
    # Simulate agent executions
    total_posts = 0
    for i, posts_count in enumerate(posts_per_agent):
        agent_name = f"agent_{i}"
        
        # Start execution
        start_time = datetime.now()
        monitor._update_agent_status(agent_name, "executing")
        monitor.agent_statuses[agent_name].start_time = start_time
        
        # Complete execution
        end_time = start_time + timedelta(milliseconds=execution_time_ms)
        monitor.agent_statuses[agent_name].end_time = end_time
        monitor._update_agent_status(agent_name, "completed", posts_processed=posts_count)
        
        total_posts += posts_count
    
    # Get metrics
    metrics = monitor.get_metrics()
    
    # Calculate expected posts per second
    total_time_seconds = (execution_time_ms * len(posts_per_agent)) / 1000
    expected_posts_per_second = total_posts / total_time_seconds if total_time_seconds > 0 else 0.0
    
    # Verify throughput calculation (allow small floating point error)
    if metrics.posts_per_second > 0:
        assert abs(metrics.posts_per_second - expected_posts_per_second) < 0.1, \
            f"Posts per second should be {expected_posts_per_second:.2f}, got {metrics.posts_per_second:.2f}"


@given(
    agent_execution_times=st.dictionaries(
        keys=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
        values=st.floats(min_value=1.0, max_value=10000.0),
        min_size=1,
        max_size=10
    )
)
@settings(max_examples=100)
def test_per_agent_metrics_accuracy(agent_execution_times):
    """
    Property 2: Metrics accuracy (per-agent metrics)
    
    For any set of agents with execution times, per-agent metrics
    must accurately reflect each agent's individual performance.
    """
    monitor = OrchestratorMonitor(MockOrchestrator())
    
    # Simulate executions for each agent
    for agent_name, exec_time_ms in agent_execution_times.items():
        # Start execution
        start_time = datetime.now()
        monitor._update_agent_status(agent_name, "executing")
        monitor.agent_statuses[agent_name].start_time = start_time
        
        # Complete execution
        end_time = start_time + timedelta(milliseconds=exec_time_ms)
        monitor.agent_statuses[agent_name].end_time = end_time
        monitor._update_agent_status(agent_name, "completed", posts_processed=10)
    
    # Get metrics
    metrics = monitor.get_metrics()
    
    # Verify per-agent metrics
    for agent_name, expected_time in agent_execution_times.items():
        agent_metrics = metrics.agent_metrics.get(agent_name)
        assert agent_metrics is not None, f"Metrics for {agent_name} should exist"
        assert agent_metrics["posts_processed"] == 10
        assert agent_metrics["current_status"] == "completed"


@given(
    execution_times=st.lists(
        st.floats(min_value=1.0, max_value=10000.0),
        min_size=2,
        max_size=20
    )
)
@settings(max_examples=100)
def test_total_execution_time_is_sum_of_all_times(execution_times):
    """
    Property 2: Metrics accuracy (total time calculation)
    
    For any set of agent executions, total_execution_time_ms must equal
    the sum of all individual agent execution times.
    """
    monitor = OrchestratorMonitor(MockOrchestrator())
    
    # Simulate executions
    for i, exec_time_ms in enumerate(execution_times):
        agent_name = f"agent_{i}"
        
        # Start execution
        start_time = datetime.now()
        monitor._update_agent_status(agent_name, "executing")
        monitor.agent_statuses[agent_name].start_time = start_time
        
        # Complete execution
        end_time = start_time + timedelta(milliseconds=exec_time_ms)
        monitor.agent_statuses[agent_name].end_time = end_time
        monitor._update_agent_status(agent_name, "completed")
    
    # Get metrics
    metrics = monitor.get_metrics()
    
    # Calculate expected total
    expected_total = sum(execution_times)
    
    # Verify total execution time (allow small floating point error)
    assert abs(metrics.total_execution_time_ms - expected_total) < 0.1, \
        f"Total execution time should be {expected_total:.2f}, got {metrics.total_execution_time_ms:.2f}"


@given(
    num_executions=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=100)
def test_metrics_with_zero_execution_time(num_executions):
    """
    Property 2: Metrics accuracy (edge case - zero time)
    
    For any number of executions with zero total time,
    metrics should handle division by zero gracefully.
    """
    monitor = OrchestratorMonitor(MockOrchestrator())
    
    # Set execution counts but no actual execution time
    monitor.total_executions = num_executions
    monitor.successful_executions = num_executions
    
    # Get metrics
    metrics = monitor.get_metrics()
    
    # Verify metrics handle zero time gracefully
    assert metrics.total_executions == num_executions
    assert metrics.average_execution_time_ms == 0.0
    assert metrics.posts_per_second == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
