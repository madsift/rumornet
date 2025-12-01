"""
Unit tests for error logging functionality.

Tests the error logging components to ensure they properly log,
store, and display errors with all required information.
"""

import pytest
from datetime import datetime
from dashboard.components.error_logging import ErrorLogger


def test_error_logger_initialization():
    """Test ErrorLogger initialization."""
    logger = ErrorLogger()
    
    assert logger is not None
    assert logger.errors == []
    assert logger.get_error_count() == 0


def test_log_error_basic():
    """Test basic error logging."""
    logger = ErrorLogger()
    
    error_log = logger.log_error(
        component="TestComponent",
        message="Test error message"
    )
    
    assert error_log is not None
    assert error_log["component"] == "TestComponent"
    assert error_log["message"] == "Test error message"
    assert error_log["level"] == "ERROR"
    assert "timestamp" in error_log
    
    # Verify error was added to logger
    assert logger.get_error_count() == 1


def test_log_error_with_all_fields():
    """Test error logging with all optional fields."""
    logger = ErrorLogger()
    
    context = {"user_id": "123", "action": "test"}
    stack_trace = "Traceback (most recent call last):\n  File test.py"
    
    error_log = logger.log_error(
        component="TestComponent",
        message="Test error",
        level="CRITICAL",
        stack_trace=stack_trace,
        context=context
    )
    
    assert error_log["component"] == "TestComponent"
    assert error_log["message"] == "Test error"
    assert error_log["level"] == "CRITICAL"
    assert error_log["stack_trace"] == stack_trace
    assert error_log["context"] == context


def test_log_exception():
    """Test logging an exception."""
    logger = ErrorLogger()
    
    try:
        raise ValueError("Test exception")
    except ValueError as e:
        error_log = logger.log_exception(
            component="TestComponent",
            exception=e
        )
    
    assert error_log is not None
    assert "ValueError" in error_log["message"]
    assert "Test exception" in error_log["message"]
    assert "stack_trace" in error_log
    assert logger.get_error_count() == 1


def test_get_errors_no_filter():
    """Test getting all errors without filtering."""
    logger = ErrorLogger()
    
    logger.log_error("Component1", "Error 1")
    logger.log_error("Component2", "Error 2")
    logger.log_error("Component3", "Error 3")
    
    errors = logger.get_errors()
    
    assert len(errors) == 3


def test_get_errors_filter_by_level():
    """Test filtering errors by level."""
    logger = ErrorLogger()
    
    logger.log_error("Component1", "Error 1", level="ERROR")
    logger.log_error("Component2", "Warning 1", level="WARNING")
    logger.log_error("Component3", "Critical 1", level="CRITICAL")
    logger.log_error("Component4", "Error 2", level="ERROR")
    
    errors = logger.get_errors(level="ERROR")
    
    assert len(errors) == 2
    assert all(e["level"] == "ERROR" for e in errors)


def test_get_errors_filter_by_component():
    """Test filtering errors by component."""
    logger = ErrorLogger()
    
    logger.log_error("ComponentA", "Error 1")
    logger.log_error("ComponentB", "Error 2")
    logger.log_error("ComponentA", "Error 3")
    logger.log_error("ComponentC", "Error 4")
    
    errors = logger.get_errors(component="ComponentA")
    
    assert len(errors) == 2
    assert all(e["component"] == "ComponentA" for e in errors)


def test_get_errors_with_limit():
    """Test limiting number of returned errors."""
    logger = ErrorLogger()
    
    for i in range(10):
        logger.log_error(f"Component{i}", f"Error {i}")
    
    errors = logger.get_errors(limit=5)
    
    assert len(errors) == 5
    # Should return the last 5 errors
    assert errors[0]["message"] == "Error 5"
    assert errors[4]["message"] == "Error 9"


def test_get_error_count():
    """Test getting error count."""
    logger = ErrorLogger()
    
    logger.log_error("Component1", "Error 1", level="ERROR")
    logger.log_error("Component2", "Warning 1", level="WARNING")
    logger.log_error("Component3", "Error 2", level="ERROR")
    
    assert logger.get_error_count() == 3
    assert logger.get_error_count("ERROR") == 2
    assert logger.get_error_count("WARNING") == 1
    assert logger.get_error_count("CRITICAL") == 0


def test_clear_errors():
    """Test clearing all errors."""
    logger = ErrorLogger()
    
    logger.log_error("Component1", "Error 1")
    logger.log_error("Component2", "Error 2")
    
    assert logger.get_error_count() == 2
    
    logger.clear_errors()
    
    assert logger.get_error_count() == 0
    assert logger.get_errors() == []


def test_error_timestamp_format():
    """Test that error timestamps are in ISO format."""
    logger = ErrorLogger()
    
    error_log = logger.log_error("TestComponent", "Test error")
    
    # Verify timestamp can be parsed
    try:
        parsed_time = datetime.fromisoformat(error_log["timestamp"])
        assert isinstance(parsed_time, datetime)
    except ValueError:
        pytest.fail("Timestamp is not in valid ISO format")


def test_multiple_errors_same_component():
    """Test logging multiple errors from the same component."""
    logger = ErrorLogger()
    
    logger.log_error("Component1", "Error 1")
    logger.log_error("Component1", "Error 2")
    logger.log_error("Component1", "Error 3")
    
    errors = logger.get_errors(component="Component1")
    
    assert len(errors) == 3
    assert all(e["component"] == "Component1" for e in errors)


def test_error_levels():
    """Test different error levels."""
    logger = ErrorLogger()
    
    logger.log_error("Component1", "Info message", level="INFO")
    logger.log_error("Component2", "Warning message", level="WARNING")
    logger.log_error("Component3", "Error message", level="ERROR")
    logger.log_error("Component4", "Critical message", level="CRITICAL")
    
    assert logger.get_error_count("INFO") == 1
    assert logger.get_error_count("WARNING") == 1
    assert logger.get_error_count("ERROR") == 1
    assert logger.get_error_count("CRITICAL") == 1


def test_error_context_preservation():
    """Test that error context is preserved."""
    logger = ErrorLogger()
    
    context = {
        "user_id": "user123",
        "action": "submit_form",
        "data": {"field1": "value1"}
    }
    
    error_log = logger.log_error(
        "FormComponent",
        "Validation failed",
        context=context
    )
    
    assert error_log["context"] == context
    assert error_log["context"]["user_id"] == "user123"
    assert error_log["context"]["action"] == "submit_form"


def test_stack_trace_preservation():
    """Test that stack traces are preserved."""
    logger = ErrorLogger()
    
    stack_trace = """Traceback (most recent call last):
  File "test.py", line 10, in <module>
    raise ValueError("Test error")
ValueError: Test error"""
    
    error_log = logger.log_error(
        "TestComponent",
        "Test error",
        stack_trace=stack_trace
    )
    
    assert error_log["stack_trace"] == stack_trace
    assert "Traceback" in error_log["stack_trace"]
    assert "ValueError" in error_log["stack_trace"]


def test_error_ordering():
    """Test that errors are stored in chronological order."""
    logger = ErrorLogger()
    
    logger.log_error("Component1", "Error 1")
    logger.log_error("Component2", "Error 2")
    logger.log_error("Component3", "Error 3")
    
    errors = logger.get_errors()
    
    # Errors should be in the order they were logged
    assert errors[0]["message"] == "Error 1"
    assert errors[1]["message"] == "Error 2"
    assert errors[2]["message"] == "Error 3"


def test_combined_filters():
    """Test using multiple filters together."""
    logger = ErrorLogger()
    
    logger.log_error("ComponentA", "Error 1", level="ERROR")
    logger.log_error("ComponentA", "Warning 1", level="WARNING")
    logger.log_error("ComponentB", "Error 2", level="ERROR")
    logger.log_error("ComponentA", "Error 3", level="ERROR")
    
    # Filter by both component and level
    errors = logger.get_errors(component="ComponentA", level="ERROR")
    
    assert len(errors) == 2
    assert all(e["component"] == "ComponentA" for e in errors)
    assert all(e["level"] == "ERROR" for e in errors)


def test_empty_logger():
    """Test operations on empty logger."""
    logger = ErrorLogger()
    
    assert logger.get_error_count() == 0
    assert logger.get_errors() == []
    assert logger.get_errors(level="ERROR") == []
    assert logger.get_errors(component="Any") == []
    
    # Clear should not raise error
    logger.clear_errors()
    assert logger.get_error_count() == 0
