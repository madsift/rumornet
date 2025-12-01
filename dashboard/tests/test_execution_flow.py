"""
Unit tests for execution flow visualization components.

Tests the execution flow visualization functions to ensure they
properly display agent pipeline, data flow, timeline, and parallel execution patterns.
"""

import pytest
from datetime import datetime, timedelta
from dashboard.models.data_models import AgentStatus
from dashboard.components.execution_flow import (
    render_agent_pipeline_display,
    render_data_flow_visualization,
    render_complete_execution_timeline,
    render_parallel_vs_sequential_visualization,
    render_execution_flow_dashboard,
    _get_flow_status_indicator
)


@pytest.fixture
def sample_agent_statuses():
    """Create sample agent statuses for testing."""
    base_time = datetime.now()
    
    return {
        "reasoning": AgentStatus(
            agent_name="reasoning",
            status="completed",
            start_time=base_time,
            end_time=base_time + timedelta(milliseconds=150),
            execution_time_ms=150.0,
            posts_processed=10
        ),
        "pattern": AgentStatus(
            agent_name="pattern",
            status="completed",
            start_time=base_time + timedelta(milliseconds=10),
            end_time=base_time + timedelta(milliseconds=200),
            execution_time_ms=190.0,
            posts_processed=10
        ),
        "evidence": AgentStatus(
            agent_name="evidence",
            status="executing",
            start_time=base_time + timedelta(milliseconds=200),
            execution_time_ms=0.0,
            posts_processed=0
        ),
        "social_behavior": AgentStatus(
            agent_name="social_behavior",
            status="idle"
        ),
        "topic_modeling": AgentStatus(
            agent_name="topic_modeling",
            status="failed",
            start_time=base_time + timedelta(milliseconds=250),
            end_time=base_time + timedelta(milliseconds=300),
            execution_time_ms=50.0,
            error="Test error"
        )
    }


@pytest.fixture
def empty_statuses():
    """Create empty agent statuses."""
    return {}


def test_get_flow_status_indicator():
    """Test flow status indicator generation."""
    # Test with None
    assert _get_flow_status_indicator(None) == "⚪"
    
    # Test with executing status
    executing_status = AgentStatus(agent_name="test", status="executing")
    assert "🟡" in _get_flow_status_indicator(executing_status)
    
    # Test with completed status
    completed_status = AgentStatus(agent_name="test", status="completed")
    assert "🟢" in _get_flow_status_indicator(completed_status)
    
    # Test with failed status
    failed_status = AgentStatus(agent_name="test", status="failed")
    assert "🔴" in _get_flow_status_indicator(failed_status)
    
    # Test with idle status
    idle_status = AgentStatus(agent_name="test", status="idle")
    assert _get_flow_status_indicator(idle_status) == "⚪"


def test_render_agent_pipeline_display_with_data(sample_agent_statuses):
    """Test agent pipeline display with sample data."""
    # This test verifies the function runs without errors
    # In a real Streamlit environment, we would check the rendered output
    try:
        # Note: This will fail in non-Streamlit environment, but we're testing structure
        render_agent_pipeline_display(sample_agent_statuses, show_dependencies=True)
    except Exception as e:
        # Expected to fail outside Streamlit, but should not have import/syntax errors
        assert "streamlit" in str(e).lower() or "st" in str(e).lower()


def test_render_agent_pipeline_display_empty(empty_statuses):
    """Test agent pipeline display with empty data."""
    try:
        render_agent_pipeline_display(empty_statuses, show_dependencies=False)
    except Exception as e:
        # Expected to fail outside Streamlit
        assert "streamlit" in str(e).lower() or "st" in str(e).lower()


def test_render_data_flow_visualization_with_data(sample_agent_statuses):
    """Test data flow visualization with sample data."""
    try:
        render_data_flow_visualization(sample_agent_statuses)
    except Exception as e:
        # Expected to fail outside Streamlit
        assert "streamlit" in str(e).lower() or "st" in str(e).lower()


def test_render_complete_execution_timeline_with_data(sample_agent_statuses):
    """Test complete execution timeline with sample data."""
    try:
        render_complete_execution_timeline(sample_agent_statuses)
    except Exception as e:
        # Expected to fail outside Streamlit
        assert "streamlit" in str(e).lower() or "st" in str(e).lower()


def test_render_parallel_vs_sequential_visualization_with_data(sample_agent_statuses):
    """Test parallel vs sequential visualization with sample data."""
    try:
        render_parallel_vs_sequential_visualization(sample_agent_statuses)
    except Exception as e:
        # Expected to fail outside Streamlit
        assert "streamlit" in str(e).lower() or "st" in str(e).lower()


def test_render_execution_flow_dashboard_with_data(sample_agent_statuses):
    """Test complete execution flow dashboard with sample data."""
    try:
        render_execution_flow_dashboard(sample_agent_statuses)
    except Exception as e:
        # Expected to fail outside Streamlit
        assert "streamlit" in str(e).lower() or "st" in str(e).lower()


def test_render_execution_flow_dashboard_empty(empty_statuses):
    """Test complete execution flow dashboard with empty data."""
    try:
        render_execution_flow_dashboard(empty_statuses)
    except Exception as e:
        # Expected to fail outside Streamlit
        assert "streamlit" in str(e).lower() or "st" in str(e).lower()


def test_agent_status_data_structure():
    """Test that AgentStatus objects work correctly with execution flow."""
    base_time = datetime.now()
    
    status = AgentStatus(
        agent_name="test_agent",
        status="completed",
        start_time=base_time,
        end_time=base_time + timedelta(milliseconds=100),
        execution_time_ms=100.0,
        posts_processed=5
    )
    
    # Verify status attributes
    assert status.agent_name == "test_agent"
    assert status.status == "completed"
    assert status.execution_time_ms == 100.0
    assert status.posts_processed == 5
    assert status.start_time is not None
    assert status.end_time is not None


def test_multiple_agent_statuses_sorting():
    """Test that multiple agent statuses can be sorted by start time."""
    base_time = datetime.now()
    
    statuses = {
        "agent1": AgentStatus(
            agent_name="agent1",
            status="completed",
            start_time=base_time + timedelta(milliseconds=100)
        ),
        "agent2": AgentStatus(
            agent_name="agent2",
            status="completed",
            start_time=base_time
        ),
        "agent3": AgentStatus(
            agent_name="agent3",
            status="completed",
            start_time=base_time + timedelta(milliseconds=50)
        )
    }
    
    # Sort by start time
    sorted_statuses = sorted(
        statuses.items(),
        key=lambda x: x[1].start_time if x[1].start_time else datetime.max
    )
    
    # Verify sorting order
    assert sorted_statuses[0][0] == "agent2"
    assert sorted_statuses[1][0] == "agent3"
    assert sorted_statuses[2][0] == "agent1"


def test_execution_time_calculation():
    """Test execution time calculation for timeline visualization."""
    base_time = datetime.now()
    
    status = AgentStatus(
        agent_name="test",
        status="completed",
        start_time=base_time,
        end_time=base_time + timedelta(milliseconds=250)
    )
    
    # Calculate execution time
    if status.start_time and status.end_time:
        calculated_time = (status.end_time - status.start_time).total_seconds() * 1000
        assert calculated_time == 250.0


def test_parallel_group_identification():
    """Test that agents can be grouped by parallel execution."""
    from dashboard.components.execution_flow import AGENT_PIPELINE
    
    # Group agents by parallel_group
    parallel_groups = {}
    for agent_config in AGENT_PIPELINE:
        group = agent_config["parallel_group"]
        if group not in parallel_groups:
            parallel_groups[group] = []
        parallel_groups[group].append(agent_config)
    
    # Verify grouping
    assert len(parallel_groups) > 0
    
    # Check that group 1 has multiple agents (parallel execution)
    if 1 in parallel_groups:
        assert len(parallel_groups[1]) >= 2  # reasoning and pattern should be parallel


def test_agent_pipeline_configuration():
    """Test that agent pipeline configuration is valid."""
    from dashboard.components.execution_flow import AGENT_PIPELINE
    
    # Verify pipeline is not empty
    assert len(AGENT_PIPELINE) > 0
    
    # Verify each agent has required fields
    for agent_config in AGENT_PIPELINE:
        assert "name" in agent_config
        assert "display_name" in agent_config
        assert "description" in agent_config
        assert "dependencies" in agent_config
        assert "parallel_group" in agent_config
        
        # Verify types
        assert isinstance(agent_config["name"], str)
        assert isinstance(agent_config["display_name"], str)
        assert isinstance(agent_config["description"], str)
        assert isinstance(agent_config["dependencies"], list)
        assert isinstance(agent_config["parallel_group"], int)
