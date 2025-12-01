"""
Unit tests for styling utilities.

Tests the styling utility functions to ensure they generate
correct HTML and CSS for enhanced visual appearance.
"""

import pytest
from dashboard.utils.styling import (
    render_status_badge,
    render_progress_card,
    render_metric_card,
    render_info_box,
    render_divider,
    render_alert
)


def test_render_status_badge_executing():
    """Test status badge rendering for executing status."""
    badge_html = render_status_badge("executing")
    
    assert "EXECUTING" in badge_html
    assert "#FFF3CD" in badge_html  # Background color
    assert "#856404" in badge_html  # Text color
    assert "inline-block" in badge_html


def test_render_status_badge_completed():
    """Test status badge rendering for completed status."""
    badge_html = render_status_badge("completed")
    
    assert "COMPLETED" in badge_html
    assert "#D4EDDA" in badge_html  # Background color
    assert "#155724" in badge_html  # Text color


def test_render_status_badge_failed():
    """Test status badge rendering for failed status."""
    badge_html = render_status_badge("failed")
    
    assert "FAILED" in badge_html
    assert "#F8D7DA" in badge_html  # Background color
    assert "#721C24" in badge_html  # Text color


def test_render_status_badge_idle():
    """Test status badge rendering for idle status."""
    badge_html = render_status_badge("idle")
    
    assert "IDLE" in badge_html
    assert "#E9ECEF" in badge_html  # Background color
    assert "#6C757D" in badge_html  # Text color


def test_render_status_badge_custom_text():
    """Test status badge with custom text."""
    badge_html = render_status_badge("executing", text="In Progress")
    
    assert "In Progress" in badge_html
    assert "EXECUTING" not in badge_html


def test_render_status_badge_unknown_status():
    """Test status badge with unknown status defaults to idle."""
    badge_html = render_status_badge("unknown")
    
    assert "UNKNOWN" in badge_html
    assert "#E9ECEF" in badge_html  # Should use idle colors


def test_status_badge_html_structure():
    """Test that status badge has correct HTML structure."""
    badge_html = render_status_badge("completed")
    
    # Check for span tag
    assert "<span" in badge_html
    assert "</span>" in badge_html
    
    # Check for styling attributes
    assert "border-radius" in badge_html
    assert "padding" in badge_html
    assert "font-weight" in badge_html


def test_status_badge_case_insensitive():
    """Test that status badge works with different cases."""
    badge_lower = render_status_badge("executing")
    badge_upper = render_status_badge("EXECUTING")
    badge_mixed = render_status_badge("Executing")
    
    # All should produce similar output (same colors)
    assert "#FFF3CD" in badge_lower
    assert "#FFF3CD" in badge_upper
    assert "#FFF3CD" in badge_mixed


def test_render_divider_without_text():
    """Test divider rendering without text."""
    # This would normally be called in Streamlit context
    # We're just testing it doesn't raise errors
    try:
        render_divider()
    except Exception as e:
        # Should not raise any exceptions
        pytest.fail(f"render_divider() raised {e}")


def test_render_divider_with_text():
    """Test divider rendering with text."""
    try:
        render_divider("Section Break")
    except Exception as e:
        pytest.fail(f"render_divider('Section Break') raised {e}")


def test_render_alert_types():
    """Test alert rendering for different types."""
    alert_types = ["success", "info", "warning", "error"]
    
    for alert_type in alert_types:
        try:
            render_alert(f"Test {alert_type} message", alert_type=alert_type)
        except Exception as e:
            pytest.fail(f"render_alert with type '{alert_type}' raised {e}")


def test_render_info_box():
    """Test info box rendering."""
    try:
        render_info_box(
            title="Test Title",
            content="Test content",
            icon="ℹ️",
            color="#17A2B8"
        )
    except Exception as e:
        pytest.fail(f"render_info_box() raised {e}")


def test_render_progress_card():
    """Test progress card rendering."""
    try:
        render_progress_card(
            title="Test Progress",
            current=50,
            total=100,
            color="#FF4B4B"
        )
    except Exception as e:
        pytest.fail(f"render_progress_card() raised {e}")


def test_render_metric_card():
    """Test metric card rendering."""
    try:
        render_metric_card(
            label="Test Metric",
            value="100",
            delta="+10%",
            delta_color="normal"
        )
    except Exception as e:
        pytest.fail(f"render_metric_card() raised {e}")


def test_render_metric_card_without_delta():
    """Test metric card rendering without delta."""
    try:
        render_metric_card(
            label="Test Metric",
            value="100"
        )
    except Exception as e:
        pytest.fail(f"render_metric_card() without delta raised {e}")


def test_status_badge_all_statuses():
    """Test that all common statuses produce valid HTML."""
    statuses = ["idle", "executing", "completed", "failed"]
    
    for status in statuses:
        badge_html = render_status_badge(status)
        
        # Check basic HTML structure
        assert "<span" in badge_html
        assert "</span>" in badge_html
        assert "style=" in badge_html
        
        # Check status text is present
        assert status.upper() in badge_html


def test_status_badge_styling_consistency():
    """Test that status badges have consistent styling."""
    badge_html = render_status_badge("completed")
    
    # Check for consistent styling properties
    required_styles = [
        "display",
        "padding",
        "border-radius",
        "font-size",
        "font-weight",
        "text-transform",
        "letter-spacing",
        "background",
        "color"
    ]
    
    for style in required_styles:
        assert style in badge_html.lower(), f"Missing style property: {style}"


def test_progress_calculation():
    """Test progress percentage calculation in progress card."""
    # This is implicitly tested by render_progress_card
    # We're testing that it handles edge cases
    
    # Zero total should not crash
    try:
        render_progress_card("Test", 0, 0)
    except ZeroDivisionError:
        pytest.fail("Progress card should handle zero total")
    
    # Current > total should not crash
    try:
        render_progress_card("Test", 150, 100)
    except Exception as e:
        pytest.fail(f"Progress card should handle current > total: {e}")


def test_alert_default_type():
    """Test that alert defaults to info type."""
    try:
        render_alert("Test message")
    except Exception as e:
        pytest.fail(f"render_alert() with default type raised {e}")


def test_metric_card_delta_colors():
    """Test metric card with different delta colors."""
    delta_colors = ["normal", "inverse", "off"]
    
    for color in delta_colors:
        try:
            render_metric_card(
                label="Test",
                value="100",
                delta="+10",
                delta_color=color
            )
        except Exception as e:
            pytest.fail(f"render_metric_card with delta_color '{color}' raised {e}")
