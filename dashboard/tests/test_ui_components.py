"""
Unit tests for UI components.

Tests that UI components can be instantiated and called without errors.
Note: Full UI testing requires Streamlit runtime, so these tests verify
basic functionality and data handling.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from dashboard.models.data_models import AgentStatus, ExecutionMetrics
from dashboard.components.ui_components import (
    render_agent_status_card,
    render_metrics_dashboard,
    render_results_table,
    render_execution_timeline,
    render_progress_bar,
    render_error_panel,
    render_summary_cards,
    render_filter_panel,
    render_agent_grid
)


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit functions for testing."""
    with patch('dashboard.components.ui_components.st') as mock_st:
        # Create mock column objects that support context manager protocol
        def create_mock_column():
            mock_col = MagicMock()
            mock_col.__enter__ = Mock(return_value=mock_col)
            mock_col.__exit__ = Mock(return_value=False)
            return mock_col
        
        # Mock container
        mock_container = MagicMock()
        mock_container.__enter__ = Mock(return_value=mock_container)
        mock_container.__exit__ = Mock(return_value=False)
        mock_st.container = Mock(return_value=mock_container)
        
        # Mock columns - return list of mock columns
        def mock_columns_func(num_cols):
            # Handle both integer and list inputs
            if isinstance(num_cols, list):
                count = len(num_cols)
            else:
                count = num_cols
            return [create_mock_column() for _ in range(count)]
        
        mock_st.columns = Mock(side_effect=mock_columns_func)
        
        # Mock other Streamlit functions
        mock_st.markdown = Mock()
        mock_st.metric = Mock()
        mock_st.error = Mock()
        mock_st.divider = Mock()
        mock_st.subheader = Mock()
        mock_st.dataframe = Mock()
        mock_st.info = Mock()
        mock_st.caption = Mock()
        mock_st.text = Mock()
        mock_st.progress = Mock()
        mock_st.success = Mock()
        
        # Mock expander
        mock_expander = MagicMock()
        mock_expander.__enter__ = Mock(return_value=mock_expander)
        mock_expander.__exit__ = Mock(return_value=False)
        mock_st.expander = Mock(return_value=mock_expander)
        
        mock_st.code = Mock()
        mock_st.json = Mock()
        mock_st.text_input = Mock(return_value="")
        mock_st.multiselect = Mock(return_value=[])
        mock_st.slider = Mock(return_value=0.5)
        mock_st.button = Mock(return_value=False)
        mock_st.rerun = Mock()
        mock_st.warning = Mock()
        
        yield mock_st


def test_render_agent_status_card_idle(mock_streamlit):
    """Test rendering agent status card for idle agent."""
    status = AgentStatus(
        agent_name="test_agent",
        status="idle"
    )
    
    # Should not raise any exceptions
    render_agent_status_card(status)
    
    # Verify Streamlit functions were called
    assert mock_streamlit.markdown.called
    assert mock_streamlit.divider.called


def test_render_agent_status_card_executing(mock_streamlit):
    """Test rendering agent status card for executing agent."""
    status = AgentStatus(
        agent_name="test_agent",
        status="executing",
        start_time=datetime.now(),
        posts_processed=5
    )
    
    render_agent_status_card(status)
    
    assert mock_streamlit.markdown.called
    assert mock_streamlit.metric.called


def test_render_agent_status_card_completed(mock_streamlit):
    """Test rendering agent status card for completed agent."""
    status = AgentStatus(
        agent_name="test_agent",
        status="completed",
        start_time=datetime.now(),
        end_time=datetime.now(),
        execution_time_ms=1500.0,
        posts_processed=10
    )
    
    render_agent_status_card(status)
    
    assert mock_streamlit.markdown.called
    assert mock_streamlit.metric.called


def test_render_agent_status_card_failed(mock_streamlit):
    """Test rendering agent status card for failed agent."""
    status = AgentStatus(
        agent_name="test_agent",
        status="failed",
        error="Test error message"
    )
    
    render_agent_status_card(status)
    
    assert mock_streamlit.error.called


def test_render_metrics_dashboard(mock_streamlit):
    """Test rendering metrics dashboard."""
    metrics = ExecutionMetrics(
        total_executions=10,
        successful_executions=8,
        failed_executions=2,
        total_execution_time_ms=15000.0,
        average_execution_time_ms=1500.0,
        posts_per_second=2.5,
        agent_metrics={
            "agent1": {
                "current_status": "completed",
                "posts_processed": 50,
                "avg_execution_time_ms": 1200.0,
                "last_error": None
            }
        }
    )
    
    render_metrics_dashboard(metrics)
    
    assert mock_streamlit.subheader.called
    assert mock_streamlit.metric.called
    assert mock_streamlit.dataframe.called


def test_render_results_table_empty(mock_streamlit):
    """Test rendering results table with no results."""
    render_results_table([])
    
    assert mock_streamlit.info.called


def test_render_results_table_with_results(mock_streamlit):
    """Test rendering results table with results."""
    results = [
        {
            "metadata": {
                "post_id": "post1",
                "username": "user1"
            },
            "analysis": {
                "verdict": False,
                "confidence": 0.85,
                "risk_level": "HIGH",
                "patterns_detected": ["pattern1", "pattern2"]
            }
        }
    ]
    
    render_results_table(results)
    
    assert mock_streamlit.dataframe.called


def test_render_execution_timeline_empty(mock_streamlit):
    """Test rendering execution timeline with no data."""
    render_execution_timeline({})
    
    assert mock_streamlit.info.called


def test_render_execution_timeline_with_agents(mock_streamlit):
    """Test rendering execution timeline with agents."""
    statuses = {
        "agent1": AgentStatus(
            agent_name="agent1",
            status="completed",
            start_time=datetime.now(),
            execution_time_ms=1000.0
        ),
        "agent2": AgentStatus(
            agent_name="agent2",
            status="executing",
            start_time=datetime.now()
        )
    }
    
    render_execution_timeline(statuses)
    
    assert mock_streamlit.markdown.called
    assert mock_streamlit.text.called


def test_render_progress_bar(mock_streamlit):
    """Test rendering progress bar."""
    render_progress_bar(current=50, total=100, label="Processing")
    
    assert mock_streamlit.progress.called
    assert mock_streamlit.metric.called


def test_render_progress_bar_invalid(mock_streamlit):
    """Test rendering progress bar with invalid values."""
    render_progress_bar(current=10, total=0)
    
    assert mock_streamlit.warning.called


def test_render_error_panel_no_errors(mock_streamlit):
    """Test rendering error panel with no errors."""
    render_error_panel([])
    
    assert mock_streamlit.success.called


def test_render_error_panel_with_errors(mock_streamlit):
    """Test rendering error panel with errors."""
    errors = [
        {
            "component": "test_component",
            "timestamp": "2024-01-01T12:00:00",
            "level": "ERROR",
            "message": "Test error",
            "stack_trace": "Traceback...",
            "context": {"key": "value"}
        }
    ]
    
    render_error_panel(errors)
    
    assert mock_streamlit.error.called
    assert mock_streamlit.code.called


def test_render_summary_cards(mock_streamlit):
    """Test rendering summary cards."""
    summary = {
        "total_posts_analyzed": 100,
        "misinformation_detected": 25,
        "high_risk_posts": 10,
        "critical_posts": 3,
        "unique_users": 50,
        "users_posting_misinfo": 15,
        "patterns_detected": 8,
        "topics_identified": 5
    }
    
    render_summary_cards(summary)
    
    assert mock_streamlit.metric.called


def test_render_filter_panel(mock_streamlit):
    """Test rendering filter panel."""
    filters = render_filter_panel()
    
    assert isinstance(filters, dict)
    assert "search" in filters
    assert "risk_levels" in filters
    assert "confidence_min" in filters
    assert "patterns" in filters


def test_render_agent_grid_empty(mock_streamlit):
    """Test rendering agent grid with no agents."""
    render_agent_grid({})
    
    assert mock_streamlit.info.called


def test_render_agent_grid_with_agents(mock_streamlit):
    """Test rendering agent grid with agents."""
    statuses = {
        "agent1": AgentStatus(agent_name="agent1", status="completed"),
        "agent2": AgentStatus(agent_name="agent2", status="executing"),
        "agent3": AgentStatus(agent_name="agent3", status="idle")
    }
    
    render_agent_grid(statuses, columns=2)
    
    assert mock_streamlit.markdown.called
    assert mock_streamlit.caption.called
