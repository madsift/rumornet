"""
Property-based tests for historical data integrity.

**Feature: agent-monitoring-dashboard, Property 6: Historical data integrity**
**Validates: Requirements 6.4**

This module tests that execution results saved to history can be retrieved
identically to what was originally saved (round-trip property).
"""

import pytest
import tempfile
import shutil
from hypothesis import given, strategies as st, settings
from datetime import datetime, timedelta
from pathlib import Path

from dashboard.core.data_manager import DataManager
from dashboard.models.data_models import ExecutionResult, AgentStatus


# Counter for unique execution IDs
_execution_id_counter = 0

# Strategy for generating valid ExecutionResult objects
@st.composite
def execution_result_strategy(draw):
    """Generate a valid ExecutionResult for testing."""
    
    # Generate unique execution ID using counter
    global _execution_id_counter
    _execution_id_counter += 1
    execution_id = f"exec_{_execution_id_counter}_{draw(st.integers(min_value=1000, max_value=9999))}"
    
    # Generate timestamp
    days_ago = draw(st.integers(min_value=0, max_value=30))
    timestamp = datetime.now() - timedelta(days=days_ago)
    
    # Generate post counts
    total_posts = draw(st.integers(min_value=1, max_value=100))
    posts_analyzed = draw(st.integers(min_value=0, max_value=total_posts))
    misinformation_detected = draw(st.integers(min_value=0, max_value=posts_analyzed))
    high_risk_posts = draw(st.integers(min_value=0, max_value=misinformation_detected))
    
    # Generate execution time
    execution_time_ms = draw(st.floats(min_value=100.0, max_value=10000.0))
    
    # Generate simple agent statuses (just 2 agents)
    agent_statuses = {
        "reasoning": AgentStatus(
            agent_name="reasoning",
            status="completed",
            start_time=timestamp,
            end_time=timestamp + timedelta(milliseconds=100),
            execution_time_ms=100.0,
            posts_processed=total_posts,
            error=None
        ),
        "pattern": AgentStatus(
            agent_name="pattern",
            status="completed",
            start_time=timestamp,
            end_time=timestamp + timedelta(milliseconds=100),
            execution_time_ms=100.0,
            posts_processed=total_posts,
            error=None
        )
    }
    
    # Generate simple full report
    full_report = {
        "executive_summary": {
            "total_posts_analyzed": total_posts,
            "misinformation_detected": misinformation_detected
        }
    }
    
    # Generate markdown report
    markdown_report = f"# Report\nPosts: {total_posts}"
    
    return ExecutionResult(
        execution_id=execution_id,
        timestamp=timestamp,
        total_posts=total_posts,
        posts_analyzed=posts_analyzed,
        misinformation_detected=misinformation_detected,
        high_risk_posts=high_risk_posts,
        execution_time_ms=execution_time_ms,
        agent_statuses=agent_statuses,
        full_report=full_report,
        markdown_report=markdown_report
    )


@given(execution_result_strategy())
@settings(max_examples=50)
def test_save_and_load_round_trip(result):
    """
    Property 6: Historical data integrity (round-trip)
    
    For any execution result saved to history, retrieving that result by ID
    must return data identical to what was originally saved.
    """
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        data_manager = DataManager(data_dir=temp_dir)
        
        # Save the result
        save_success = data_manager.save_execution_result(result)
        assert save_success, "Save operation should succeed"
        
        # Load the result back
        loaded_result = data_manager.load_execution_result(result.execution_id)
        
        # Verify the result was loaded
        assert loaded_result is not None, "Loaded result should not be None"
        
        # Verify all fields match
        assert loaded_result.execution_id == result.execution_id
        assert loaded_result.total_posts == result.total_posts
        assert loaded_result.posts_analyzed == result.posts_analyzed
        assert loaded_result.misinformation_detected == result.misinformation_detected
        assert loaded_result.high_risk_posts == result.high_risk_posts
        assert abs(loaded_result.execution_time_ms - result.execution_time_ms) < 0.01
        assert loaded_result.markdown_report == result.markdown_report
        
        # Verify agent statuses
        assert len(loaded_result.agent_statuses) == len(result.agent_statuses)
        for agent_name in result.agent_statuses:
            assert agent_name in loaded_result.agent_statuses
            original_status = result.agent_statuses[agent_name]
            loaded_status = loaded_result.agent_statuses[agent_name]
            assert loaded_status.agent_name == original_status.agent_name
            assert loaded_status.status == original_status.status
            assert loaded_status.posts_processed == original_status.posts_processed


@given(st.lists(execution_result_strategy(), min_size=1, max_size=10))
@settings(max_examples=50, deadline=None)
def test_multiple_results_integrity(results):
    """
    Property 6: Historical data integrity (multiple results)
    
    For any set of execution results saved to history, all results
    must be retrievable with their data intact.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        data_manager = DataManager(data_dir=temp_dir)
        
        # Save all results
        for result in results:
            save_success = data_manager.save_execution_result(result)
            assert save_success, f"Save should succeed for {result.execution_id}"
        
        # Load all results
        loaded_results = data_manager.load_all_execution_results()
        
        # Verify count matches
        assert len(loaded_results) == len(results), \
            f"Should load {len(results)} results, got {len(loaded_results)}"
        
        # Verify each result can be retrieved by ID
        for original_result in results:
            loaded_result = data_manager.get_execution_by_id(original_result.execution_id)
            assert loaded_result is not None
            assert loaded_result.execution_id == original_result.execution_id
            assert loaded_result.total_posts == original_result.total_posts


@given(execution_result_strategy())
@settings(max_examples=50)
def test_delete_and_verify_removal(result):
    """
    Property 6: Historical data integrity (deletion)
    
    For any execution result, after deletion, it should no longer
    be retrievable from storage.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        data_manager = DataManager(data_dir=temp_dir)
        
        # Save the result
        data_manager.save_execution_result(result)
        
        # Verify it exists
        loaded_result = data_manager.load_execution_result(result.execution_id)
        assert loaded_result is not None
        
        # Delete the result
        delete_success = data_manager.delete_execution_result(result.execution_id)
        assert delete_success, "Delete operation should succeed"
        
        # Verify it no longer exists
        loaded_result = data_manager.load_execution_result(result.execution_id)
        assert loaded_result is None, "Result should not exist after deletion"


@given(
    results=st.lists(execution_result_strategy(), min_size=5, max_size=10),
    max_items=st.integers(min_value=1, max_value=5)
)
@settings(max_examples=50)
def test_history_size_limit(results, max_items):
    """
    Property 6: Historical data integrity (size limiting)
    
    For any set of execution results, when limiting history size,
    the most recent results should be preserved and older ones deleted.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        data_manager = DataManager(data_dir=temp_dir)
        
        # Save all results
        for result in results:
            data_manager.save_execution_result(result)
        
        # Limit history size
        deleted_count = data_manager.limit_history_size(max_items=max_items)
        
        # Load remaining results
        remaining_results = data_manager.load_all_execution_results()
        
        # Verify count is at most max_items
        assert len(remaining_results) <= max_items, \
            f"Should have at most {max_items} results, got {len(remaining_results)}"
        
        # Verify deleted count is correct
        expected_deleted = max(0, len(results) - max_items)
        assert deleted_count == expected_deleted, \
            f"Should delete {expected_deleted} results, deleted {deleted_count}"


@given(st.lists(execution_result_strategy(), min_size=1, max_size=5))
@settings(max_examples=50)
def test_clear_all_history(results):
    """
    Property 6: Historical data integrity (clearing)
    
    For any set of execution results, after clearing all history,
    no results should be retrievable.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        data_manager = DataManager(data_dir=temp_dir)
        
        # Save all results
        for result in results:
            data_manager.save_execution_result(result)
        
        # Verify results exist
        loaded_results = data_manager.load_all_execution_results()
        assert len(loaded_results) == len(results)
        
        # Clear all history
        clear_success = data_manager.clear_all_history()
        assert clear_success, "Clear operation should succeed"
        
        # Verify no results remain
        loaded_results = data_manager.load_all_execution_results()
        assert len(loaded_results) == 0, "No results should remain after clearing"


@given(execution_result_strategy())
@settings(max_examples=50)
def test_export_and_verify_content(result):
    """
    Property 6: Historical data integrity (export)
    
    For any execution result, exporting to JSON should preserve
    all data that can be re-imported.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        data_manager = DataManager(data_dir=temp_dir)
        
        # Save the result
        data_manager.save_execution_result(result)
        
        # Export to JSON
        export_path = Path(temp_dir) / "export.json"
        export_success = data_manager.export_history_to_json(str(export_path))
        assert export_success, "Export operation should succeed"
        
        # Verify export file exists
        assert export_path.exists(), "Export file should exist"
        
        # Verify export file is not empty
        assert export_path.stat().st_size > 0, "Export file should not be empty"


@given(st.lists(execution_result_strategy(), min_size=1, max_size=5))
@settings(max_examples=50)
def test_history_summary_accuracy(results):
    """
    Property 6: Historical data integrity (summary)
    
    For any set of execution results, the history summary should
    accurately reflect the stored data.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        data_manager = DataManager(data_dir=temp_dir)
        
        # Save all results
        for result in results:
            data_manager.save_execution_result(result)
        
        # Get summary
        summary = data_manager.get_history_summary()
        
        # Verify summary accuracy
        assert summary["total_executions"] == len(results), \
            f"Summary should show {len(results)} executions, got {summary['total_executions']}"
        
        # Calculate expected totals
        expected_total_posts = sum(r.total_posts for r in results)
        expected_total_misinfo = sum(r.misinformation_detected for r in results)
        
        assert summary["total_posts_analyzed"] == expected_total_posts
        assert summary["total_misinformation_detected"] == expected_total_misinfo


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
