"""
Unit tests for main dashboard application.

Tests the main dashboard initialization, state management,
and component integration.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dashboard.models.data_models import (
    AgentStatus,
    ExecutionResult,
    DashboardConfig
)


class MockSessionState:
    """Mock Streamlit session state for testing."""
    
    def __init__(self):
        self.data = {}
    
    def __contains__(self, key):
        return key in self.data
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, value):
        self.data[key] = value
    
    def __getattr__(self, key):
        if key == 'data':
            return object.__getattribute__(self, 'data')
        return self.data.get(key)
    
    def __setattr__(self, key, value):
        if key == 'data':
            object.__setattr__(self, key, value)
        else:
            self.data[key] = value
    
    def get(self, key, default=None):
        return self.data.get(key, default)
    
    def keys(self):
        return self.data.keys()


@pytest.fixture
def mock_session_state():
    """Create a mock session state."""
    return MockSessionState()


@pytest.fixture
def sample_config():
    """Create a sample dashboard configuration."""
    return DashboardConfig.default()


@pytest.fixture
def sample_agent_status():
    """Create a sample agent status."""
    return AgentStatus(
        agent_name="test_agent",
        status="completed",
        start_time=datetime.now(),
        execution_time_ms=100.0,
        posts_processed=5
    )


def test_dashboard_config_default():
    """Test default dashboard configuration."""
    config = DashboardConfig.default()
    
    assert config.ollama_endpoint == "http://localhost:11434"
    assert config.ollama_model == "llama2"
    assert config.auto_refresh_interval == 5
    assert config.max_history_items == 50
    assert config.default_batch_size == 10
    assert config.enable_debug_mode == False


def test_dashboard_config_to_dict():
    """Test configuration serialization."""
    config = DashboardConfig.default()
    config_dict = config.to_dict()
    
    assert isinstance(config_dict, dict)
    assert "ollama_endpoint" in config_dict
    assert "ollama_model" in config_dict
    assert "auto_refresh_interval" in config_dict


def test_dashboard_config_from_dict():
    """Test configuration deserialization."""
    config_dict = {
        "ollama_endpoint": "http://test:11434",
        "ollama_model": "test_model",
        "auto_refresh_interval": 10,
        "max_history_items": 100,
        "default_batch_size": 20,
        "enable_debug_mode": True
    }
    
    config = DashboardConfig.from_dict(config_dict)
    
    assert config.ollama_endpoint == "http://test:11434"
    assert config.ollama_model == "test_model"
    assert config.auto_refresh_interval == 10
    assert config.max_history_items == 100
    assert config.default_batch_size == 20
    assert config.enable_debug_mode == True


def test_session_state_initialization(mock_session_state):
    """Test session state initialization."""
    from dashboard.core.state_manager import initialize_session_state
    
    initialize_session_state(mock_session_state)
    
    # Check that all required keys are initialized
    assert "config" in mock_session_state
    assert "agent_statuses" in mock_session_state
    assert "execution_history" in mock_session_state
    assert "current_execution_id" in mock_session_state
    assert "current_result" in mock_session_state
    assert "metrics" in mock_session_state
    assert "filters" in mock_session_state
    assert "selected_history_id" in mock_session_state
    assert "show_errors" in mock_session_state
    assert "error_log" in mock_session_state


def test_agent_status_update(mock_session_state, sample_agent_status):
    """Test updating agent status in session state."""
    from dashboard.core.state_manager import (
        initialize_session_state,
        update_agent_status,
        get_agent_status
    )
    
    initialize_session_state(mock_session_state)
    
    # Update agent status
    update_agent_status(
        mock_session_state,
        agent_name="test_agent",
        status="executing",
        start_time=datetime.now(),
        posts_processed=5
    )
    
    # Retrieve and verify
    status = get_agent_status(mock_session_state, "test_agent")
    
    assert status is not None
    assert status.agent_name == "test_agent"
    assert status.status == "executing"
    assert status.posts_processed == 5


def test_execution_history_management(mock_session_state):
    """Test execution history management."""
    from dashboard.core.state_manager import (
        initialize_session_state,
        add_execution_result,
        get_execution_history,
        clear_execution_history
    )
    
    initialize_session_state(mock_session_state)
    
    # Create sample execution result
    result = ExecutionResult(
        execution_id="test_123",
        timestamp=datetime.now(),
        total_posts=10,
        posts_analyzed=10,
        misinformation_detected=3,
        high_risk_posts=1,
        execution_time_ms=1000.0,
        agent_statuses={},
        full_report={},
        markdown_report=""
    )
    
    # Add to history
    add_execution_result(mock_session_state, result)
    
    # Retrieve history
    history = get_execution_history(mock_session_state)
    
    assert len(history) == 1
    assert history[0].execution_id == "test_123"
    
    # Clear history
    clear_execution_history(mock_session_state)
    history = get_execution_history(mock_session_state)
    
    assert len(history) == 0


def test_error_logging(mock_session_state):
    """Test error logging functionality."""
    from dashboard.core.state_manager import (
        initialize_session_state,
        log_error,
        get_error_log,
        clear_error_log
    )
    
    initialize_session_state(mock_session_state)
    
    # Log an error
    log_error(
        mock_session_state,
        component="test_component",
        error_message="Test error message",
        stack_trace="Test stack trace"
    )
    
    # Retrieve error log
    error_log = get_error_log(mock_session_state)
    
    assert len(error_log) == 1
    assert error_log[0]["component"] == "test_component"
    assert error_log[0]["message"] == "Test error message"
    assert error_log[0]["stack_trace"] == "Test stack trace"
    
    # Clear error log
    clear_error_log(mock_session_state)
    error_log = get_error_log(mock_session_state)
    
    assert len(error_log) == 0


def test_filter_management(mock_session_state):
    """Test filter management."""
    from dashboard.core.state_manager import (
        initialize_session_state,
        update_filters,
        get_filters
    )
    
    initialize_session_state(mock_session_state)
    
    # Update filters
    update_filters(
        mock_session_state,
        search_query="test query",
        risk_level="high",
        confidence_threshold=0.8,
        pattern_filter="test_pattern"
    )
    
    # Retrieve filters
    filters = get_filters(mock_session_state)
    
    assert filters["search_query"] == "test query"
    assert filters["risk_level"] == "high"
    assert filters["confidence_threshold"] == 0.8
    assert filters["pattern_filter"] == "test_pattern"


def test_execution_id_generation():
    """Test execution ID generation."""
    from dashboard.core.state_manager import generate_execution_id
    
    id1 = generate_execution_id()
    id2 = generate_execution_id()
    
    # IDs should be unique
    assert id1 != id2
    
    # IDs should be strings
    assert isinstance(id1, str)
    assert isinstance(id2, str)
    
    # IDs should not be empty
    assert len(id1) > 0
    assert len(id2) > 0


def test_start_new_execution(mock_session_state):
    """Test starting a new execution."""
    from dashboard.core.state_manager import (
        initialize_session_state,
        start_new_execution,
        update_agent_status,
        get_all_agent_statuses
    )
    
    initialize_session_state(mock_session_state)
    
    # Add some agent statuses
    update_agent_status(
        mock_session_state,
        agent_name="agent1",
        status="completed",
        posts_processed=5
    )
    
    # Start new execution
    execution_id = start_new_execution(mock_session_state)
    
    # Verify execution ID is set
    assert mock_session_state.current_execution_id == execution_id
    
    # Verify agent statuses are reset
    statuses = get_all_agent_statuses(mock_session_state)
    for status in statuses.values():
        assert status.status == "idle"
        assert status.posts_processed == 0


def test_config_update(mock_session_state):
    """Test configuration update."""
    from dashboard.core.state_manager import (
        initialize_session_state,
        update_config,
        get_config
    )
    
    initialize_session_state(mock_session_state)
    
    # Create new config
    new_config = DashboardConfig(
        ollama_endpoint="http://new:11434",
        ollama_model="new_model",
        auto_refresh_interval=15,
        max_history_items=200,
        default_batch_size=30,
        enable_debug_mode=True
    )
    
    # Update config
    update_config(mock_session_state, new_config)
    
    # Retrieve and verify
    config = get_config(mock_session_state)
    
    assert config.ollama_endpoint == "http://new:11434"
    assert config.ollama_model == "new_model"
    assert config.auto_refresh_interval == 15
    assert config.max_history_items == 200
    assert config.default_batch_size == 30
    assert config.enable_debug_mode == True


def test_multiple_agent_statuses(mock_session_state):
    """Test managing multiple agent statuses."""
    from dashboard.core.state_manager import (
        initialize_session_state,
        update_agent_status,
        get_all_agent_statuses
    )
    
    initialize_session_state(mock_session_state)
    
    # Add multiple agents
    agents = ["agent1", "agent2", "agent3"]
    
    for agent in agents:
        update_agent_status(
            mock_session_state,
            agent_name=agent,
            status="completed",
            posts_processed=10
        )
    
    # Retrieve all statuses
    statuses = get_all_agent_statuses(mock_session_state)
    
    assert len(statuses) == 3
    
    for agent in agents:
        assert agent in statuses
        assert statuses[agent].status == "completed"
        assert statuses[agent].posts_processed == 10


def test_execution_result_serialization():
    """Test execution result serialization."""
    result = ExecutionResult(
        execution_id="test_123",
        timestamp=datetime.now(),
        total_posts=10,
        posts_analyzed=10,
        misinformation_detected=3,
        high_risk_posts=1,
        execution_time_ms=1000.0,
        agent_statuses={},
        full_report={"test": "data"},
        markdown_report="# Test Report"
    )
    
    # Convert to dict
    result_dict = result.to_dict()
    
    assert isinstance(result_dict, dict)
    assert result_dict["execution_id"] == "test_123"
    assert result_dict["total_posts"] == 10
    assert result_dict["misinformation_detected"] == 3
    
    # Convert back from dict
    restored_result = ExecutionResult.from_dict(result_dict)
    
    assert restored_result.execution_id == result.execution_id
    assert restored_result.total_posts == result.total_posts
    assert restored_result.misinformation_detected == result.misinformation_detected
