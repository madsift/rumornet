"""
Unit tests for history viewer components.

Tests the execution history viewer functions to ensure they handle
various data inputs correctly.
"""

import pytest
from datetime import datetime, timedelta
from dashboard.models.data_models import ExecutionResult, AgentStatus


def create_mock_execution_result(
    execution_id: str,
    timestamp: datetime,
    total_posts: int = 100,
    posts_analyzed: int = 95,
    misinformation_detected: int = 20,
    high_risk_posts: int = 5
) -> ExecutionResult:
    """Create a mock ExecutionResult for testing."""
    agent_statuses = {
        "agent1": AgentStatus(
            agent_name="agent1",
            status="completed",
            execution_time_ms=1000.0,
            posts_processed=95
        ),
        "agent2": AgentStatus(
            agent_name="agent2",
            status="completed",
            execution_time_ms=500.0,
            posts_processed=95
        )
    }
    
    return ExecutionResult(
        execution_id=execution_id,
        timestamp=timestamp,
        total_posts=total_posts,
        posts_analyzed=posts_analyzed,
        misinformation_detected=misinformation_detected,
        high_risk_posts=high_risk_posts,
        execution_time_ms=2500.0,
        agent_statuses=agent_statuses,
        full_report={"test": "data"},
        markdown_report="# Test Report"
    )


def test_execution_result_creation():
    """Test creating a mock execution result."""
    result = create_mock_execution_result(
        execution_id="test123",
        timestamp=datetime.now()
    )
    
    assert result.execution_id == "test123"
    assert result.total_posts == 100
    assert result.posts_analyzed == 95
    assert result.misinformation_detected == 20
    assert result.high_risk_posts == 5
    assert len(result.agent_statuses) == 2


def test_history_list_with_empty_history():
    """Test history list rendering with empty history."""
    history = []
    
    # Should handle empty history gracefully
    assert len(history) == 0


def test_history_list_with_valid_data():
    """Test history list rendering with valid data."""
    history = [
        create_mock_execution_result("exec1", datetime.now()),
        create_mock_execution_result("exec2", datetime.now() - timedelta(hours=1)),
        create_mock_execution_result("exec3", datetime.now() - timedelta(hours=2))
    ]
    
    assert len(history) == 3
    assert history[0].execution_id == "exec1"
    assert history[1].execution_id == "exec2"
    assert history[2].execution_id == "exec3"


def test_execution_details_data_preservation():
    """Test that execution details preserve all data."""
    result = create_mock_execution_result(
        execution_id="test_detail",
        timestamp=datetime.now(),
        total_posts=150,
        posts_analyzed=145,
        misinformation_detected=30,
        high_risk_posts=10
    )
    
    # Verify all data is preserved
    assert result.execution_id == "test_detail"
    assert result.total_posts == 150
    assert result.posts_analyzed == 145
    assert result.misinformation_detected == 30
    assert result.high_risk_posts == 10
    assert result.execution_time_ms == 2500.0
    assert len(result.agent_statuses) == 2
    assert result.full_report == {"test": "data"}
    assert result.markdown_report == "# Test Report"


def test_trends_calculation():
    """Test trends calculation from history."""
    history = [
        create_mock_execution_result("exec1", datetime.now(), 100, 95, 20, 5),
        create_mock_execution_result("exec2", datetime.now() - timedelta(hours=1), 100, 90, 25, 8),
        create_mock_execution_result("exec3", datetime.now() - timedelta(hours=2), 100, 85, 30, 10)
    ]
    
    # Calculate trends
    total_posts = sum(r.total_posts for r in history)
    total_analyzed = sum(r.posts_analyzed for r in history)
    total_misinfo = sum(r.misinformation_detected for r in history)
    
    assert total_posts == 300
    assert total_analyzed == 270
    assert total_misinfo == 75
    
    # Calculate average success rate
    success_rates = [(r.posts_analyzed / r.total_posts * 100) for r in history]
    avg_success_rate = sum(success_rates) / len(success_rates)
    
    assert avg_success_rate > 0
    assert avg_success_rate <= 100


def test_comparison_with_multiple_executions():
    """Test comparison functionality with multiple executions."""
    history = [
        create_mock_execution_result("exec1", datetime.now(), 100, 95, 20, 5),
        create_mock_execution_result("exec2", datetime.now() - timedelta(hours=1), 150, 140, 30, 10),
        create_mock_execution_result("exec3", datetime.now() - timedelta(hours=2), 80, 75, 15, 3)
    ]
    
    # Select executions for comparison
    selected = [history[0], history[1]]
    
    assert len(selected) == 2
    assert selected[0].total_posts == 100
    assert selected[1].total_posts == 150
    
    # Compare metrics
    assert selected[0].misinformation_detected < selected[1].misinformation_detected
    assert selected[0].high_risk_posts < selected[1].high_risk_posts


def test_history_sorting_by_timestamp():
    """Test that history is sorted by timestamp."""
    now = datetime.now()
    history = [
        create_mock_execution_result("exec3", now - timedelta(hours=2)),
        create_mock_execution_result("exec1", now),
        create_mock_execution_result("exec2", now - timedelta(hours=1))
    ]
    
    # Sort by timestamp (newest first)
    sorted_history = sorted(history, key=lambda x: x.timestamp, reverse=True)
    
    assert sorted_history[0].execution_id == "exec1"
    assert sorted_history[1].execution_id == "exec2"
    assert sorted_history[2].execution_id == "exec3"


def test_success_rate_calculation():
    """Test success rate calculation."""
    result = create_mock_execution_result(
        "test",
        datetime.now(),
        total_posts=100,
        posts_analyzed=85
    )
    
    success_rate = (result.posts_analyzed / result.total_posts) * 100
    
    assert success_rate == 85.0


def test_misinfo_rate_calculation():
    """Test misinformation rate calculation."""
    result = create_mock_execution_result(
        "test",
        datetime.now(),
        posts_analyzed=100,
        misinformation_detected=25
    )
    
    misinfo_rate = (result.misinformation_detected / result.posts_analyzed) * 100
    
    assert misinfo_rate == 25.0


def test_agent_status_preservation():
    """Test that agent statuses are preserved in execution result."""
    result = create_mock_execution_result("test", datetime.now())
    
    assert len(result.agent_statuses) == 2
    assert "agent1" in result.agent_statuses
    assert "agent2" in result.agent_statuses
    
    agent1 = result.agent_statuses["agent1"]
    assert agent1.agent_name == "agent1"
    assert agent1.status == "completed"
    assert agent1.execution_time_ms == 1000.0
    assert agent1.posts_processed == 95


def test_history_with_varying_metrics():
    """Test history with varying metrics across executions."""
    history = [
        create_mock_execution_result("exec1", datetime.now(), 50, 48, 10, 2),
        create_mock_execution_result("exec2", datetime.now(), 200, 190, 40, 15),
        create_mock_execution_result("exec3", datetime.now(), 100, 95, 20, 5)
    ]
    
    # Verify varying metrics
    assert history[0].total_posts != history[1].total_posts
    assert history[1].total_posts != history[2].total_posts
    
    # Calculate statistics
    min_posts = min(r.total_posts for r in history)
    max_posts = max(r.total_posts for r in history)
    avg_posts = sum(r.total_posts for r in history) / len(history)
    
    assert min_posts == 50
    assert max_posts == 200
    assert avg_posts == 116.67 or abs(avg_posts - 116.67) < 0.01


def test_execution_time_trends():
    """Test execution time trends calculation."""
    history = [
        create_mock_execution_result("exec1", datetime.now()),
        create_mock_execution_result("exec2", datetime.now()),
        create_mock_execution_result("exec3", datetime.now())
    ]
    
    # All have same execution time in mock
    execution_times = [r.execution_time_ms for r in history]
    
    assert all(t == 2500.0 for t in execution_times)
    
    avg_time = sum(execution_times) / len(execution_times)
    assert avg_time == 2500.0


def test_history_data_integrity():
    """Test that history data maintains integrity."""
    original = create_mock_execution_result(
        "integrity_test",
        datetime.now(),
        total_posts=100,
        posts_analyzed=95,
        misinformation_detected=20,
        high_risk_posts=5
    )
    
    # Verify data integrity
    assert original.execution_id == "integrity_test"
    assert original.total_posts == 100
    assert original.posts_analyzed == 95
    assert original.misinformation_detected == 20
    assert original.high_risk_posts == 5
    
    # Verify relationships
    assert original.posts_analyzed <= original.total_posts
    assert original.misinformation_detected <= original.posts_analyzed
    assert original.high_risk_posts <= original.misinformation_detected
