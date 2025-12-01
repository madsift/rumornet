"""
Property-based tests for agent status transitions.

**Feature: agent-monitoring-dashboard, Property 10: Status update ordering**
**Validates: Requirements 1.1, 1.2, 1.3, 1.4**

This module tests that agent status transitions follow the correct order:
idle → executing → (completed | failed)
"""

import pytest
from hypothesis import given, strategies as st, settings
from datetime import datetime, timedelta
from typing import List, Tuple

from dashboard.models.data_models import AgentStatus
from dashboard.core.state_manager import (
    initialize_session_state,
    update_agent_status,
    get_agent_status
)


# Mock session state for testing
class MockSessionState:
    """Mock Streamlit session state for testing."""
    
    def __init__(self):
        self.agent_statuses = {}
        self.config = None
        self.execution_history = []
        self.current_execution_id = None
        self.current_result = None
        self.metrics = None
        self.filters = {}
        self.selected_history_id = None
        self.show_errors = False
        self.error_log = []
    
    def __contains__(self, key):
        return hasattr(self, key)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)


# Strategy for generating valid status transitions
@st.composite
def status_transition_sequence(draw):
    """Generate a valid sequence of status transitions.
    
    Valid sequences:
    - idle → executing → completed
    - idle → executing → failed
    - idle (stays idle)
    - executing → completed
    - executing → failed
    """
    agent_name = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))))
    
    # Choose a transition path
    path = draw(st.sampled_from([
        ["idle", "executing", "completed"],
        ["idle", "executing", "failed"],
        ["idle"],
        ["executing", "completed"],
        ["executing", "failed"]
    ]))
    
    # Generate timestamps for each transition
    base_time = datetime.now()
    transitions = []
    
    for i, status in enumerate(path):
        if status == "idle":
            transitions.append((status, None, None, 0.0))
        elif status == "executing":
            start_time = base_time + timedelta(milliseconds=i * 100)
            transitions.append((status, start_time, None, 0.0))
        elif status in ["completed", "failed"]:
            # Get the start time from the previous executing state
            start_time = None
            for prev_status, prev_start, _, _ in reversed(transitions):
                if prev_status == "executing":
                    start_time = prev_start
                    break
            
            if start_time is None:
                start_time = base_time
            
            end_time = start_time + timedelta(milliseconds=draw(st.integers(min_value=1, max_value=10000)))
            execution_time_ms = (end_time - start_time).total_seconds() * 1000
            transitions.append((status, start_time, end_time, execution_time_ms))
    
    return agent_name, transitions


@given(status_transition_sequence())
@settings(max_examples=100)
def test_status_update_ordering(transition_data):
    """
    Property 10: Status update ordering
    
    For any agent execution sequence, status updates must occur in the correct order:
    idle → executing → (completed | failed)
    
    This test verifies that:
    1. Status transitions follow valid paths
    2. When transitioning to completed/failed, execution time is > 0
    3. When transitioning to completed/failed, end_time > start_time
    """
    agent_name, transitions = transition_data
    
    # Create mock session state
    session_state = MockSessionState()
    initialize_session_state(session_state)
    
    # Apply all transitions
    for status, start_time, end_time, execution_time_ms in transitions:
        update_agent_status(
            session_state,
            agent_name=agent_name,
            status=status,
            start_time=start_time,
            end_time=end_time,
            execution_time_ms=execution_time_ms
        )
    
    # Get final agent status
    final_status = get_agent_status(session_state, agent_name)
    
    assert final_status is not None, "Agent status should exist after updates"
    assert final_status.agent_name == agent_name
    
    # Verify final state properties
    if final_status.status in ["completed", "failed"]:
        # When an agent completes or fails, execution time must be > 0
        assert final_status.execution_time_ms > 0, \
            f"Execution time must be > 0 for {final_status.status} status, got {final_status.execution_time_ms}"
        
        # End time must be after start time
        if final_status.start_time and final_status.end_time:
            assert final_status.end_time > final_status.start_time, \
                f"End time must be after start time: {final_status.start_time} -> {final_status.end_time}"
    
    elif final_status.status == "executing":
        # When executing, start time should be set
        assert final_status.start_time is not None, \
            "Start time must be set for executing status"
    
    # Verify the status is one of the valid states
    assert final_status.status in ["idle", "executing", "completed", "failed"], \
        f"Status must be one of the valid states, got {final_status.status}"


@given(
    agent_name=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
    execution_time_ms=st.floats(min_value=1.0, max_value=100000.0)
)
@settings(max_examples=100)
def test_completed_status_requires_positive_execution_time(agent_name, execution_time_ms):
    """
    Test that completed status always has positive execution time.
    
    For any agent that completes execution, the execution time must be > 0.
    """
    session_state = MockSessionState()
    initialize_session_state(session_state)
    
    start_time = datetime.now()
    end_time = start_time + timedelta(milliseconds=execution_time_ms)
    
    # Transition to executing
    update_agent_status(
        session_state,
        agent_name=agent_name,
        status="executing",
        start_time=start_time
    )
    
    # Transition to completed
    update_agent_status(
        session_state,
        agent_name=agent_name,
        status="completed",
        end_time=end_time,
        execution_time_ms=execution_time_ms
    )
    
    final_status = get_agent_status(session_state, agent_name)
    
    assert final_status.status == "completed"
    assert final_status.execution_time_ms > 0
    assert final_status.end_time > final_status.start_time


@given(
    agent_name=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
    time_delta_ms=st.integers(min_value=1, max_value=100000)
)
@settings(max_examples=100)
def test_end_time_after_start_time(agent_name, time_delta_ms):
    """
    Test that end time is always after start time for completed/failed agents.
    
    For any agent execution, if both start_time and end_time are set,
    end_time must be after start_time.
    """
    session_state = MockSessionState()
    initialize_session_state(session_state)
    
    start_time = datetime.now()
    end_time = start_time + timedelta(milliseconds=time_delta_ms)
    execution_time_ms = time_delta_ms
    
    # Transition through valid states
    update_agent_status(
        session_state,
        agent_name=agent_name,
        status="executing",
        start_time=start_time
    )
    
    update_agent_status(
        session_state,
        agent_name=agent_name,
        status="completed",
        end_time=end_time,
        execution_time_ms=execution_time_ms
    )
    
    final_status = get_agent_status(session_state, agent_name)
    
    assert final_status.start_time is not None
    assert final_status.end_time is not None
    assert final_status.end_time > final_status.start_time


@given(
    agent_name=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')))
)
@settings(max_examples=100)
def test_idle_to_executing_transition(agent_name):
    """
    Test that transitioning from idle to executing sets start_time.
    
    For any agent, when transitioning to executing status, start_time must be set.
    """
    session_state = MockSessionState()
    initialize_session_state(session_state)
    
    start_time = datetime.now()
    
    # Transition to executing
    update_agent_status(
        session_state,
        agent_name=agent_name,
        status="executing",
        start_time=start_time
    )
    
    final_status = get_agent_status(session_state, agent_name)
    
    assert final_status.status == "executing"
    assert final_status.start_time is not None
    assert final_status.end_time is None  # Should not be set yet


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
