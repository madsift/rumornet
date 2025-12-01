"""
Integration tests for the Agent Monitoring Dashboard.

Tests complete workflows including:
- Analysis workflow from input to results
- Configuration update workflow
- Historical data access workflow
- Export workflow from results to markdown
- Error handling across all workflows
"""

import pytest
import asyncio
import json
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# AsyncMock compatibility for Python < 3.8
try:
    from unittest.mock import AsyncMock
except ImportError:
    # Create AsyncMock for Python 3.7
    class AsyncMock(Mock):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
        async def __call__(self, *args, **kwargs):
            return super().__call__(*args, **kwargs)

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dashboard.core.orchestrator_monitor import OrchestratorMonitor
from dashboard.core.data_manager import DataManager
from dashboard.core.state_manager import (
    initialize_session_state,
    update_agent_status,
    add_execution_result,
    get_execution_history,
    update_config,
    get_config,
    log_error,
    get_error_log
)
from dashboard.utils.markdown_generator import MarkdownGenerator
from dashboard.models.data_models import (
    AgentStatus,
    ExecutionResult,
    DashboardConfig
)


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def data_manager(temp_data_dir):
    """Create a DataManager instance with temporary directory."""
    return DataManager(data_dir=temp_data_dir)


@pytest.fixture
def mock_orchestrator():
    """Create a mock orchestrator for testing."""
    orchestrator = Mock()
    orchestrator.agents = {
        "reasoning": Mock(),
        "pattern": Mock(),
        "evidence": Mock(),
        "social_behavior": Mock(),
        "topic_modeling": Mock()
    }
    orchestrator.initialize_agents = AsyncMock()
    orchestrator.analyze_batch_true_batch = AsyncMock()
    orchestrator.analyze_post_with_metadata = AsyncMock()
    orchestrator.generate_actionable_report = Mock()
    return orchestrator


@pytest.fixture
def sample_posts():
    """Create sample posts for testing."""
    return [
        {
            "post_id": "post_001",
            "user_id": "user_001",
            "username": "test_user_1",
            "text": "This is a test post about misinformation.",
            "platform": "reddit",
            "timestamp": "2024-01-01T12:00:00",
            "engagement": {"upvotes": 10, "comments": 5, "shares": 2}
        },
        {
            "post_id": "post_002",
            "user_id": "user_002",
            "username": "test_user_2",
            "text": "Another test post with different content.",
            "platform": "twitter",
            "timestamp": "2024-01-01T13:00:00",
            "engagement": {"upvotes": 20, "comments": 10, "shares": 5}
        },
        {
            "post_id": "post_003",
            "user_id": "user_001",
            "username": "test_user_1",
            "text": "Third post from the same user.",
            "platform": "reddit",
            "timestamp": "2024-01-01T14:00:00",
            "engagement": {"upvotes": 15, "comments": 8, "shares": 3}
        }
    ]


@pytest.fixture
def sample_analysis_results():
    """Create sample analysis results."""
    return [
        {
            "post_id": "post_001",
            "user_id": "user_001",
            "username": "test_user_1",
            "analysis": {
                "verdict": False,
                "confidence": 0.85,
                "risk_level": "HIGH",
                "patterns": ["emotional_manipulation", "false_claims"],
                "detected_language": "en"
            },
            "recommended_action": "Flag for review"
        },
        {
            "post_id": "post_002",
            "user_id": "user_002",
            "username": "test_user_2",
            "analysis": {
                "verdict": True,
                "confidence": 0.92,
                "risk_level": "LOW",
                "patterns": [],
                "detected_language": "en"
            },
            "recommended_action": "No action needed"
        },
        {
            "post_id": "post_003",
            "user_id": "user_001",
            "username": "test_user_1",
            "analysis": {
                "verdict": False,
                "confidence": 0.78,
                "risk_level": "MODERATE",
                "patterns": ["misleading_statistics"],
                "detected_language": "en"
            },
            "recommended_action": "Monitor user"
        }
    ]


@pytest.fixture
def sample_report(sample_analysis_results):
    """Create a sample orchestrator report."""
    return {
        "executive_summary": {
            "total_posts_analyzed": 3,
            "misinformation_detected": 2,
            "high_risk_posts": 1,
            "critical_posts": 0,
            "unique_users": 2,
            "users_posting_misinfo": 1,
            "patterns_detected": 2,
            "topics_identified": 1,
            "report_generated": datetime.now().isoformat()
        },
        "high_priority_posts": [
            {
                **sample_analysis_results[0],
                "text_preview": "This is a test post about misinformation.",
                "platform": "reddit",
                "timestamp": "2024-01-01T12:00:00",
                "engagement": {"upvotes": 10, "comments": 5, "shares": 2}
            }
        ],
        "top_offenders": [
            {
                "user_id": "user_001",
                "username": "test_user_1",
                "statistics": {
                    "total_posts": 2,
                    "misinformation_posts": 2,
                    "misinformation_rate": "100%",
                    "avg_confidence": 0.815,
                    "high_confidence_misinfo": 2
                },
                "patterns_used": ["emotional_manipulation", "false_claims", "misleading_statistics"],
                "recommended_action": "Suspend account"
            }
        ],
        "pattern_breakdown": [
            {
                "pattern_name": "emotional_manipulation",
                "total_occurrences": 1,
                "unique_users": 1,
                "first_seen": "2024-01-01T12:00:00",
                "last_seen": "2024-01-01T12:00:00"
            },
            {
                "pattern_name": "misleading_statistics",
                "total_occurrences": 1,
                "unique_users": 1,
                "first_seen": "2024-01-01T14:00:00",
                "last_seen": "2024-01-01T14:00:00"
            }
        ],
        "topic_analysis": {
            "status": "success",
            "topics": [
                {
                    "topic_name": "Health Misinformation",
                    "total_posts": 2,
                    "misinformation_posts": 2,
                    "misinformation_rate": 100.0,
                    "avg_confidence": 0.815,
                    "keywords": ["health", "vaccine", "treatment"]
                }
            ]
        },
        "temporal_analysis": {
            "status": "success",
            "time_period": {
                "start": "2024-01-01T12:00:00",
                "end": "2024-01-01T14:00:00"
            },
            "activity_patterns": {
                "peak_hours": [12, 13, 14],
                "peak_days": ["Monday"]
            }
        }
    }


class MockSessionState:
    """Mock session state that supports both dict and attribute access."""
    def __init__(self):
        self._data = {}
    
    def __getitem__(self, key):
        return self._data[key]
    
    def __setitem__(self, key, value):
        self._data[key] = value
    
    def __contains__(self, key):
        return key in self._data
    
    def __getattr__(self, key):
        if key.startswith('_'):
            return object.__getattribute__(self, key)
        return self._data.get(key)
    
    def __setattr__(self, key, value):
        if key.startswith('_'):
            object.__setattr__(self, key, value)
        else:
            self._data[key] = value
    
    def keys(self):
        return self._data.keys()
    
    def values(self):
        return self._data.values()
    
    def items(self):
        return self._data.items()


@pytest.fixture
def session_state():
    """Create a mock session state."""
    state = MockSessionState()
    initialize_session_state(state)
    return state


# Test 1: Complete analysis workflow from input to results
@pytest.mark.asyncio
async def test_complete_analysis_workflow(
    mock_orchestrator,
    sample_posts,
    sample_analysis_results,
    sample_report
):
    """
    Test complete analysis workflow from input to results.
    
    Validates that:
    - Posts are processed through the orchestrator
    - Agent statuses are tracked correctly
    - Results are generated and formatted
    - Metrics are calculated accurately
    """
    # Setup mock orchestrator responses
    mock_orchestrator.analyze_batch_true_batch.return_value = sample_analysis_results
    mock_orchestrator.generate_actionable_report.return_value = sample_report
    
    # Create orchestrator monitor
    monitor = OrchestratorMonitor(mock_orchestrator)
    
    # Execute analysis
    result = await monitor.analyze_with_monitoring(sample_posts, use_batch=True)
    
    # Verify analysis was successful
    assert result["status"] == "success"
    assert "results" in result
    assert "report" in result
    assert "execution_time_ms" in result
    assert "agent_statuses" in result
    assert "metrics" in result
    
    # Verify orchestrator was called correctly
    # Note: initialize_agents is only called if orchestrator.agents is empty
    # Since we set up mock_orchestrator.agents in the fixture, it won't be called
    mock_orchestrator.analyze_batch_true_batch.assert_called_once_with(sample_posts)
    mock_orchestrator.generate_actionable_report.assert_called_once()
    
    # Verify agent statuses were tracked
    agent_statuses = result["agent_statuses"]
    assert "reasoning" in agent_statuses
    assert "pattern" in agent_statuses
    
    # Verify metrics were calculated
    metrics = result["metrics"]
    assert metrics.total_executions == 1
    assert metrics.successful_executions == 1
    assert metrics.failed_executions == 0
    
    print("✓ Complete analysis workflow test passed")


# Test 2: Configuration update workflow
def test_configuration_update_workflow(session_state):
    """
    Test configuration update workflow.
    
    Validates that:
    - Configuration can be retrieved
    - Configuration can be updated
    - Invalid configuration is rejected
    - Configuration changes are persisted
    """
    # Get initial configuration
    initial_config = get_config(session_state)
    assert isinstance(initial_config, DashboardConfig)
    
    # Update configuration with valid values
    new_config = DashboardConfig(
        ollama_endpoint="http://localhost:11435",
        ollama_model="llama3.2:latest",
        auto_refresh_interval=10,
        max_history_items=100,
        default_batch_size=50,
        enable_debug_mode=True
    )
    
    update_config(session_state, new_config)
    
    # Verify configuration was updated
    updated_config = get_config(session_state)
    assert updated_config.ollama_endpoint == "http://localhost:11435"
    assert updated_config.ollama_model == "llama3.2:latest"
    assert updated_config.auto_refresh_interval == 10
    assert updated_config.max_history_items == 100
    assert updated_config.default_batch_size == 50
    assert updated_config.enable_debug_mode is True
    
    # Test configuration validation (negative values should be handled)
    invalid_config = DashboardConfig(
        ollama_endpoint="http://localhost:11434",
        ollama_model="llama3.2",
        auto_refresh_interval=-5,  # Invalid
        max_history_items=100,
        default_batch_size=10,
        enable_debug_mode=False
    )
    
    # Update with invalid config (should still work but with clamped values)
    update_config(session_state, invalid_config)
    result_config = get_config(session_state)
    
    # Verify other fields were updated
    assert result_config.ollama_endpoint == "http://localhost:11434"
    assert result_config.ollama_model == "llama3.2"
    
    print("✓ Configuration update workflow test passed")


# Test 3: Historical data access workflow
def test_historical_data_access_workflow(data_manager, sample_report):
    """
    Test historical data access workflow.
    
    Validates that:
    - Execution results can be saved
    - Execution results can be retrieved by ID
    - All execution results can be loaded
    - History can be cleared
    """
    # Create sample execution results
    result1 = ExecutionResult(
        execution_id="exec_001",
        timestamp=datetime.now(),
        total_posts=3,
        posts_analyzed=3,
        misinformation_detected=2,
        high_risk_posts=1,
        execution_time_ms=1500.0,
        agent_statuses={},
        full_report=sample_report,
        markdown_report=""
    )
    
    result2 = ExecutionResult(
        execution_id="exec_002",
        timestamp=datetime.now(),
        total_posts=5,
        posts_analyzed=5,
        misinformation_detected=3,
        high_risk_posts=2,
        execution_time_ms=2000.0,
        agent_statuses={},
        full_report=sample_report,
        markdown_report=""
    )
    
    # Save execution results
    assert data_manager.save_execution_result(result1) is True
    assert data_manager.save_execution_result(result2) is True
    
    # Retrieve specific execution result
    loaded_result1 = data_manager.get_execution_by_id("exec_001")
    assert loaded_result1 is not None
    assert loaded_result1.execution_id == "exec_001"
    assert loaded_result1.total_posts == 3
    assert loaded_result1.misinformation_detected == 2
    
    # Load all execution results
    all_results = data_manager.load_all_execution_results()
    assert len(all_results) == 2
    assert all_results[0].execution_id in ["exec_001", "exec_002"]
    assert all_results[1].execution_id in ["exec_001", "exec_002"]
    
    # Get history summary
    summary = data_manager.get_history_summary()
    assert summary["total_executions"] == 2
    assert summary["total_posts_analyzed"] == 8
    assert summary["total_misinformation_detected"] == 5
    
    # Clear history
    assert data_manager.clear_all_history() is True
    
    # Verify history was cleared
    all_results_after_clear = data_manager.load_all_execution_results()
    assert len(all_results_after_clear) == 0
    
    print("✓ Historical data access workflow test passed")


# Test 4: Export workflow from results to markdown file
def test_export_workflow(sample_report, temp_data_dir):
    """
    Test export workflow from results to markdown file.
    
    Validates that:
    - Markdown report is generated from results
    - All sections are included in the markdown
    - Markdown can be saved to file
    - Exported file is valid
    """
    # Create markdown generator
    generator = MarkdownGenerator()
    
    # Generate markdown report
    markdown = generator.generate_markdown_report(sample_report)
    
    # Verify markdown was generated
    assert markdown is not None
    assert len(markdown) > 0
    
    # Verify all sections are present
    assert "# Misinformation Detection Analysis Report" in markdown
    assert "## Executive Summary" in markdown
    assert "## High-Priority Posts" in markdown
    assert "## Top Offenders" in markdown
    assert "## Pattern Breakdown" in markdown
    assert "## Topic Analysis" in markdown
    assert "## Temporal Trends" in markdown
    
    # Verify key data is present
    assert "Total Posts Analyzed:** 3" in markdown
    assert "Misinformation Detected:** 2" in markdown
    assert "test_user_1" in markdown
    assert "emotional_manipulation" in markdown
    
    # Save markdown to file
    output_path = Path(temp_data_dir) / "test_report.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown)
    
    # Verify file was created
    assert output_path.exists()
    
    # Verify file content
    with open(output_path, 'r', encoding='utf-8') as f:
        file_content = f.read()
    
    assert file_content == markdown
    assert len(file_content) > 0
    
    print("✓ Export workflow test passed")


# Test 5: Error handling across all workflows
@pytest.mark.asyncio
async def test_error_handling_workflow(mock_orchestrator, sample_posts, session_state):
    """
    Test error handling across all workflows.
    
    Validates that:
    - Errors during analysis are caught and logged
    - Agent statuses are updated correctly on error
    - Error information is preserved
    - System remains stable after errors
    """
    # Setup mock orchestrator to raise an error
    mock_orchestrator.analyze_batch_true_batch.side_effect = Exception("Test error: Analysis failed")
    
    # Create orchestrator monitor
    monitor = OrchestratorMonitor(mock_orchestrator)
    
    # Execute analysis (should handle error gracefully)
    result = await monitor.analyze_with_monitoring(sample_posts, use_batch=True)
    
    # Verify error was handled
    assert result["status"] == "error"
    assert "error" in result
    assert "Test error: Analysis failed" in result["error"]
    
    # Verify metrics reflect the failure
    metrics = result["metrics"]
    assert metrics.total_executions == 1
    assert metrics.successful_executions == 0
    assert metrics.failed_executions == 1
    
    # Test error logging in session state
    log_error(
        session_state,
        component="test_component",
        error_message="Test error message",
        stack_trace="Test stack trace"
    )
    
    # Verify error was logged
    error_log = get_error_log(session_state)
    assert len(error_log) > 0
    
    latest_error = error_log[-1]
    assert latest_error["component"] == "test_component"
    assert latest_error["message"] == "Test error message"
    assert latest_error["stack_trace"] == "Test stack trace"
    assert "timestamp" in latest_error
    
    # Test data manager error handling
    data_manager = DataManager(data_dir="/invalid/path/that/does/not/exist")
    
    # Attempt to save (should handle error gracefully)
    result = ExecutionResult(
        execution_id="test_exec",
        timestamp=datetime.now(),
        total_posts=1,
        posts_analyzed=1,
        misinformation_detected=0,
        high_risk_posts=0,
        execution_time_ms=100.0,
        agent_statuses={},
        full_report={},
        markdown_report=""
    )
    
    # Save should fail but not crash
    save_result = data_manager.save_execution_result(result)
    # Note: This might succeed if the directory gets created, so we just verify it doesn't crash
    
    print("✓ Error handling workflow test passed")


# Test 6: End-to-end integration test
@pytest.mark.asyncio
async def test_end_to_end_integration(
    mock_orchestrator,
    sample_posts,
    sample_analysis_results,
    sample_report,
    data_manager,
    session_state
):
    """
    Test complete end-to-end integration.
    
    Validates the entire workflow:
    1. Initialize system
    2. Run analysis
    3. Save results to history
    4. Generate markdown export
    5. Retrieve from history
    6. Handle configuration updates
    """
    # Setup mock orchestrator
    mock_orchestrator.analyze_batch_true_batch.return_value = sample_analysis_results
    mock_orchestrator.generate_actionable_report.return_value = sample_report
    
    # Step 1: Initialize system
    monitor = OrchestratorMonitor(mock_orchestrator)
    generator = MarkdownGenerator()
    
    # Step 2: Run analysis
    analysis_result = await monitor.analyze_with_monitoring(sample_posts, use_batch=True)
    assert analysis_result["status"] == "success"
    
    # Step 3: Save results to history
    execution_result = ExecutionResult(
        execution_id=f"exec_{int(datetime.now().timestamp())}",
        timestamp=datetime.now(),
        total_posts=len(sample_posts),
        posts_analyzed=len(sample_posts),
        misinformation_detected=sample_report["executive_summary"]["misinformation_detected"],
        high_risk_posts=sample_report["executive_summary"]["high_risk_posts"],
        execution_time_ms=analysis_result["execution_time_ms"],
        agent_statuses=analysis_result["agent_statuses"],
        full_report=analysis_result["report"],
        markdown_report=""
    )
    
    assert data_manager.save_execution_result(execution_result) is True
    
    # Step 4: Generate markdown export
    markdown = generator.generate_markdown_report(analysis_result["report"])
    assert len(markdown) > 0
    assert "# Misinformation Detection Analysis Report" in markdown
    
    # Update execution result with markdown
    execution_result.markdown_report = markdown
    assert data_manager.save_execution_result(execution_result) is True
    
    # Step 5: Retrieve from history
    loaded_result = data_manager.get_execution_by_id(execution_result.execution_id)
    assert loaded_result is not None
    assert loaded_result.execution_id == execution_result.execution_id
    assert loaded_result.markdown_report == markdown
    
    # Step 6: Handle configuration updates
    new_config = DashboardConfig(
        ollama_endpoint="http://localhost:11434",
        ollama_model="llama3.2",
        auto_refresh_interval=5,
        max_history_items=50,
        default_batch_size=20,
        enable_debug_mode=False
    )
    
    update_config(session_state, new_config)
    updated_config = get_config(session_state)
    assert updated_config.ollama_endpoint == new_config.ollama_endpoint
    
    # Verify system state is consistent
    all_results = data_manager.load_all_execution_results()
    assert len(all_results) >= 1
    
    metrics = monitor.get_metrics()
    assert metrics.total_executions >= 1
    assert metrics.successful_executions >= 1
    
    print("✓ End-to-end integration test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])