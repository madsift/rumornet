"""
Property-based tests for configuration validation.

**Feature: agent-monitoring-dashboard, Property 7: Configuration validation**
**Validates: Requirements 7.2, 7.5**

Tests that invalid configuration values are rejected and valid values are accepted.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume

from dashboard.models.data_models import DashboardConfig
from dashboard.utils.config_manager import ConfigManager


# Strategies for generating test data

@st.composite
def valid_url_strategy(draw):
    """Generate valid URLs."""
    scheme = draw(st.sampled_from(["http", "https"]))
    host = draw(st.sampled_from(["localhost", "127.0.0.1", "example.com"]))
    port = draw(st.integers(min_value=1024, max_value=65535))
    return f"{scheme}://{host}:{port}"


@st.composite
def invalid_url_strategy(draw):
    """Generate invalid URLs."""
    return draw(st.sampled_from([
        "",  # Empty
        "not-a-url",  # No scheme
        "://localhost",  # No scheme
        "http://",  # No host
        "ftp://localhost:8080",  # Wrong scheme (but still valid URL format)
        "   ",  # Whitespace only
    ]))


@st.composite
def valid_model_name_strategy(draw):
    """Generate valid model names."""
    return draw(st.text(
        alphabet=st.characters(whitelist_categories=('Ll', 'Lu', 'Nd'), whitelist_characters='-_'),
        min_size=2,
        max_size=50
    ))


@st.composite
def invalid_model_name_strategy(draw):
    """Generate invalid model names."""
    return draw(st.sampled_from([
        "",  # Empty
        "a",  # Too short
        " ",  # Whitespace only
    ]))


@st.composite
def valid_config_strategy(draw):
    """Generate valid configuration."""
    return DashboardConfig(
        ollama_endpoint=draw(valid_url_strategy()),
        ollama_model=draw(valid_model_name_strategy()),
        auto_refresh_interval=draw(st.integers(min_value=1, max_value=300)),
        max_history_items=draw(st.integers(min_value=1, max_value=1000)),
        default_batch_size=draw(st.integers(min_value=1, max_value=1000)),
        enable_debug_mode=draw(st.booleans())
    )


@st.composite
def invalid_config_strategy(draw):
    """Generate invalid configuration."""
    # Choose which field to make invalid
    invalid_field = draw(st.sampled_from([
        "ollama_endpoint",
        "ollama_model",
        "auto_refresh_interval",
        "max_history_items",
        "default_batch_size"
    ]))
    
    config_dict = {
        "ollama_endpoint": draw(valid_url_strategy()),
        "ollama_model": draw(valid_model_name_strategy()),
        "auto_refresh_interval": draw(st.integers(min_value=1, max_value=300)),
        "max_history_items": draw(st.integers(min_value=1, max_value=1000)),
        "default_batch_size": draw(st.integers(min_value=1, max_value=1000)),
        "enable_debug_mode": draw(st.booleans())
    }
    
    # Make one field invalid
    if invalid_field == "ollama_endpoint":
        config_dict["ollama_endpoint"] = draw(invalid_url_strategy())
    elif invalid_field == "ollama_model":
        config_dict["ollama_model"] = draw(invalid_model_name_strategy())
    elif invalid_field == "auto_refresh_interval":
        config_dict["auto_refresh_interval"] = draw(st.sampled_from([0, -1, 301, 1000]))
    elif invalid_field == "max_history_items":
        config_dict["max_history_items"] = draw(st.sampled_from([0, -1, 1001, 10000]))
    elif invalid_field == "default_batch_size":
        config_dict["default_batch_size"] = draw(st.sampled_from([0, -1, 1001, 10000]))
    
    return DashboardConfig(**config_dict)


# Property tests

@given(valid_config_strategy())
@settings(max_examples=100, deadline=None)
def test_valid_configuration_passes_validation(config):
    """
    Property 7: Configuration validation
    
    For any valid configuration, validation must pass with no errors.
    
    **Validates: Requirements 7.2, 7.5**
    """
    config_manager = ConfigManager()
    is_valid, errors = config_manager.validate_config(config)
    
    assert is_valid, f"Valid configuration should pass validation. Errors: {errors}"
    assert len(errors) == 0, "Valid configuration should have no errors"


@given(invalid_config_strategy())
@settings(max_examples=100, deadline=None)
def test_invalid_configuration_fails_validation(config):
    """
    Property: Invalid configuration rejection
    
    For any invalid configuration, validation must fail with error messages.
    """
    config_manager = ConfigManager()
    is_valid, errors = config_manager.validate_config(config)
    
    assert not is_valid, "Invalid configuration should fail validation"
    assert len(errors) > 0, "Invalid configuration should have error messages"


@given(
    st.text(min_size=0, max_size=100),
    st.text(min_size=2, max_size=50),
    st.integers(min_value=1, max_value=300),
    st.integers(min_value=1, max_value=1000),
    st.integers(min_value=1, max_value=1000),
    st.booleans()
)
@settings(max_examples=100, deadline=None)
def test_empty_endpoint_fails_validation(
    endpoint, model, refresh, history, batch, debug
):
    """
    Property: Empty endpoint rejection
    
    For any configuration with empty endpoint, validation must fail.
    """
    # Only test when endpoint is empty or whitespace
    assume(not endpoint or endpoint.isspace())
    
    config = DashboardConfig(
        ollama_endpoint=endpoint,
        ollama_model=model,
        auto_refresh_interval=refresh,
        max_history_items=history,
        default_batch_size=batch,
        enable_debug_mode=debug
    )
    
    config_manager = ConfigManager()
    is_valid, errors = config_manager.validate_config(config)
    
    assert not is_valid, "Configuration with empty endpoint should fail"
    assert any("endpoint" in error.lower() for error in errors), \
        "Error message should mention endpoint"


@given(
    valid_url_strategy(),
    st.text(min_size=0, max_size=1),
    st.integers(min_value=1, max_value=300),
    st.integers(min_value=1, max_value=1000),
    st.integers(min_value=1, max_value=1000),
    st.booleans()
)
@settings(max_examples=100, deadline=None)
def test_short_model_name_fails_validation(
    endpoint, model, refresh, history, batch, debug
):
    """
    Property: Short model name rejection
    
    For any configuration with model name less than 2 characters,
    validation must fail.
    """
    config = DashboardConfig(
        ollama_endpoint=endpoint,
        ollama_model=model,
        auto_refresh_interval=refresh,
        max_history_items=history,
        default_batch_size=batch,
        enable_debug_mode=debug
    )
    
    config_manager = ConfigManager()
    is_valid, errors = config_manager.validate_config(config)
    
    assert not is_valid, "Configuration with short model name should fail"
    assert any("model" in error.lower() for error in errors), \
        "Error message should mention model"


@given(
    valid_url_strategy(),
    valid_model_name_strategy(),
    st.integers(min_value=-100, max_value=0) | st.integers(min_value=301, max_value=1000),
    st.integers(min_value=1, max_value=1000),
    st.integers(min_value=1, max_value=1000),
    st.booleans()
)
@settings(max_examples=100, deadline=None)
def test_invalid_refresh_interval_fails_validation(
    endpoint, model, refresh, history, batch, debug
):
    """
    Property: Invalid refresh interval rejection
    
    For any configuration with refresh interval outside valid range (1-300),
    validation must fail.
    """
    config = DashboardConfig(
        ollama_endpoint=endpoint,
        ollama_model=model,
        auto_refresh_interval=refresh,
        max_history_items=history,
        default_batch_size=batch,
        enable_debug_mode=debug
    )
    
    config_manager = ConfigManager()
    is_valid, errors = config_manager.validate_config(config)
    
    assert not is_valid, "Configuration with invalid refresh interval should fail"
    assert any("refresh" in error.lower() or "interval" in error.lower() for error in errors), \
        "Error message should mention refresh interval"


@given(
    valid_url_strategy(),
    valid_model_name_strategy(),
    st.integers(min_value=1, max_value=300),
    st.integers(min_value=-100, max_value=0) | st.integers(min_value=1001, max_value=10000),
    st.integers(min_value=1, max_value=1000),
    st.booleans()
)
@settings(max_examples=100, deadline=None)
def test_invalid_history_items_fails_validation(
    endpoint, model, refresh, history, batch, debug
):
    """
    Property: Invalid history items rejection
    
    For any configuration with history items outside valid range (1-1000),
    validation must fail.
    """
    config = DashboardConfig(
        ollama_endpoint=endpoint,
        ollama_model=model,
        auto_refresh_interval=refresh,
        max_history_items=history,
        default_batch_size=batch,
        enable_debug_mode=debug
    )
    
    config_manager = ConfigManager()
    is_valid, errors = config_manager.validate_config(config)
    
    assert not is_valid, "Configuration with invalid history items should fail"
    assert any("history" in error.lower() for error in errors), \
        "Error message should mention history"


@given(
    valid_url_strategy(),
    valid_model_name_strategy(),
    st.integers(min_value=1, max_value=300),
    st.integers(min_value=1, max_value=1000),
    st.integers(min_value=-100, max_value=0) | st.integers(min_value=1001, max_value=10000),
    st.booleans()
)
@settings(max_examples=100, deadline=None)
def test_invalid_batch_size_fails_validation(
    endpoint, model, refresh, history, batch, debug
):
    """
    Property: Invalid batch size rejection
    
    For any configuration with batch size outside valid range (1-1000),
    validation must fail.
    """
    config = DashboardConfig(
        ollama_endpoint=endpoint,
        ollama_model=model,
        auto_refresh_interval=refresh,
        max_history_items=history,
        default_batch_size=batch,
        enable_debug_mode=debug
    )
    
    config_manager = ConfigManager()
    is_valid, errors = config_manager.validate_config(config)
    
    assert not is_valid, "Configuration with invalid batch size should fail"
    assert any("batch" in error.lower() for error in errors), \
        "Error message should mention batch size"


@given(valid_config_strategy())
@settings(max_examples=100, deadline=None)
def test_config_round_trip_preserves_values(config):
    """
    Property: Configuration round trip
    
    For any valid configuration, converting to dict and back should
    preserve all values.
    """
    config_dict = config.to_dict()
    restored_config = DashboardConfig.from_dict(config_dict)
    
    assert restored_config.ollama_endpoint == config.ollama_endpoint
    assert restored_config.ollama_model == config.ollama_model
    assert restored_config.auto_refresh_interval == config.auto_refresh_interval
    assert restored_config.max_history_items == config.max_history_items
    assert restored_config.default_batch_size == config.default_batch_size
    assert restored_config.enable_debug_mode == config.enable_debug_mode


@given(valid_config_strategy(), st.dictionaries(
    st.sampled_from(["ollama_endpoint", "ollama_model", "auto_refresh_interval"]),
    st.one_of(st.text(min_size=2, max_size=50), st.integers(min_value=1, max_value=300))
))
@settings(max_examples=100, deadline=None)
def test_config_update_validation(config, updates):
    """
    Property: Configuration update validation
    
    For any valid configuration and updates, the update process should
    validate the resulting configuration.
    """
    config_manager = ConfigManager()
    
    # Filter updates to only include valid types for each field
    filtered_updates = {}
    for key, value in updates.items():
        if key in ["ollama_endpoint", "ollama_model"] and isinstance(value, str):
            filtered_updates[key] = value
        elif key == "auto_refresh_interval" and isinstance(value, int):
            filtered_updates[key] = value
    
    if not filtered_updates:
        return  # Skip if no valid updates
    
    updated_config, is_valid, errors = config_manager.update_config(config, filtered_updates)
    
    # If validation passes, updated config should be returned
    if is_valid:
        assert len(errors) == 0
        assert updated_config != config or filtered_updates == {}
    else:
        # If validation fails, original config should be returned
        assert len(errors) > 0
        assert updated_config == config
