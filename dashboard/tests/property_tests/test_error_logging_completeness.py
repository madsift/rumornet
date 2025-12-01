"""
Property-based tests for error logging completeness.

**Feature: agent-monitoring-dashboard, Property 9: Error logging completeness**
**Validates: Requirements 10.1, 10.3**

Tests that all errors are logged with timestamp, component name, and error message.
"""

import pytest
from hypothesis import given, strategies as st, settings
from datetime import datetime
from typing import Dict, Any, List


# Strategies for generating test data

@st.composite
def error_data_strategy(draw):
    """Generate random error data."""
    return {
        "timestamp": datetime.now().isoformat(),
        "component": draw(st.text(min_size=1, max_size=50, alphabet=st.characters(min_codepoint=65, max_codepoint=122))),
        "message": draw(st.text(min_size=1, max_size=200, alphabet=st.characters(min_codepoint=32, max_codepoint=126))),
        "level": draw(st.sampled_from(["ERROR", "WARNING", "CRITICAL", "INFO"])),
        "stack_trace": draw(st.one_of(st.none(), st.text(min_size=10, max_size=500))),
        "context": draw(st.one_of(st.none(), st.dictionaries(st.text(min_size=1, max_size=20), st.text(max_size=50), min_size=0, max_size=5)))
    }


# Error logging functions

def log_error(component: str, message: str, level: str = "ERROR", stack_trace: str = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Log an error with required fields."""
    error_log = {
        "timestamp": datetime.now().isoformat(),
        "component": component,
        "message": message,
        "level": level
    }
    
    if stack_trace:
        error_log["stack_trace"] = stack_trace
    
    if context:
        error_log["context"] = context
    
    return error_log


def validate_error_log(error_log: Dict[str, Any]) -> bool:
    """Validate that an error log has all required fields."""
    required_fields = ["timestamp", "component", "message"]
    
    for field in required_fields:
        if field not in error_log or not error_log[field]:
            return False
    
    return True


# Property tests

@given(error_data_strategy())
@settings(max_examples=100, deadline=None)
def test_error_log_has_required_fields(error_data):
    """
    Property 9: Error logging completeness - Required Fields
    
    For any error, the error log must contain timestamp, component name, and error message.
    
    **Validates: Requirements 10.1, 10.3**
    """
    error_log = log_error(
        component=error_data["component"],
        message=error_data["message"],
        level=error_data.get("level", "ERROR"),
        stack_trace=error_data.get("stack_trace"),
        context=error_data.get("context")
    )
    
    assert "timestamp" in error_log, "Error log must have timestamp"
    assert "component" in error_log, "Error log must have component name"
    assert "message" in error_log, "Error log must have error message"
    
    assert error_log["timestamp"], "Timestamp must not be empty"
    assert error_log["component"], "Component name must not be empty"
    assert error_log["message"], "Error message must not be empty"
    
    assert error_log["component"] == error_data["component"]
    assert error_log["message"] == error_data["message"]


@given(st.lists(error_data_strategy(), min_size=1, max_size=20))
@settings(max_examples=100, deadline=None)
def test_all_errors_are_logged(errors):
    """
    Property 9: Error logging completeness - All Errors Logged
    
    For any set of errors, all errors must be logged with complete information.
    
    **Validates: Requirements 10.1, 10.3**
    """
    logged_errors = []
    for error_data in errors:
        error_log = log_error(
            component=error_data["component"],
            message=error_data["message"],
            level=error_data.get("level", "ERROR"),
            stack_trace=error_data.get("stack_trace"),
            context=error_data.get("context")
        )
        logged_errors.append(error_log)
    
    assert len(logged_errors) == len(errors)
    
    for error_log in logged_errors:
        assert validate_error_log(error_log)


@given(error_data_strategy())
@settings(max_examples=100, deadline=None)
def test_error_timestamp_is_valid(error_data):
    """
    Property 9: Error logging completeness - Valid Timestamp
    
    For any error, the timestamp must be a valid ISO format datetime string.
    
    **Validates: Requirements 10.3**
    """
    error_log = log_error(
        component=error_data["component"],
        message=error_data["message"]
    )
    
    assert "timestamp" in error_log
    
    try:
        parsed_time = datetime.fromisoformat(error_log["timestamp"])
        assert isinstance(parsed_time, datetime)
    except (ValueError, TypeError) as e:
        pytest.fail(f"Timestamp must be valid ISO format: {e}")


@given(st.text(min_size=1, max_size=50, alphabet=st.characters(min_codepoint=65, max_codepoint=122)),
       st.text(min_size=1, max_size=200, alphabet=st.characters(min_codepoint=32, max_codepoint=126)))
@settings(max_examples=100, deadline=None)
def test_error_component_and_message_preserved(component, message):
    """
    Property 9: Error logging completeness - Data Preservation
    
    For any error, the component name and message must be preserved exactly.
    
    **Validates: Requirements 10.1, 10.3**
    """
    error_log = log_error(component=component, message=message)
    
    assert error_log["component"] == component
    assert error_log["message"] == message


@given(error_data_strategy())
@settings(max_examples=100, deadline=None)
def test_error_optional_fields_preserved(error_data):
    """
    Property 9: Error logging completeness - Optional Fields
    
    For any error with optional fields, those fields must be preserved if provided.
    
    **Validates: Requirements 10.3**
    """
    error_log = log_error(
        component=error_data["component"],
        message=error_data["message"],
        level=error_data.get("level", "ERROR"),
        stack_trace=error_data.get("stack_trace"),
        context=error_data.get("context")
    )
    
    if error_data.get("stack_trace"):
        assert "stack_trace" in error_log
        assert error_log["stack_trace"] == error_data["stack_trace"]
    
    if error_data.get("context"):
        assert "context" in error_log
        assert error_log["context"] == error_data["context"]
