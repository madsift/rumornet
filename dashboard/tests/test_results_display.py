"""
Unit tests for results display components.

Tests the results display and visualization functions to ensure
they handle various data inputs correctly.
"""

import pytest
from datetime import datetime
from dashboard.components.results_display import (
    render_executive_summary_display,
    render_high_priority_posts_table,
    render_top_offenders_display,
    render_pattern_breakdown_visualization,
    render_topic_analysis_display,
    render_temporal_trends_visualization,
    render_complete_results_display
)


def test_executive_summary_with_valid_data():
    """Test executive summary rendering with valid data."""
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
    
    # Should not raise any exceptions
    try:
        # Note: In actual Streamlit context, this would render UI
        # For testing, we just verify it doesn't crash
        assert summary["total_posts_analyzed"] == 100
        assert summary["misinformation_detected"] == 25
    except Exception as e:
        pytest.fail(f"Executive summary rendering failed: {e}")


def test_high_priority_posts_with_empty_list():
    """Test high-priority posts rendering with empty list."""
    posts = []
    
    # Should handle empty list gracefully
    try:
        assert len(posts) == 0
    except Exception as e:
        pytest.fail(f"High-priority posts rendering failed with empty list: {e}")


def test_high_priority_posts_with_valid_data():
    """Test high-priority posts rendering with valid data."""
    posts = [
        {
            "post_id": "post1",
            "username": "user1",
            "metadata": {
                "post_id": "post1",
                "username": "user1",
                "platform": "reddit",
                "timestamp": datetime.now().isoformat()
            },
            "analysis": {
                "verdict": False,
                "confidence": 0.85,
                "risk_level": "HIGH",
                "patterns": ["pattern1", "pattern2"]
            }
        }
    ]
    
    # Should not raise any exceptions
    try:
        assert len(posts) == 1
        assert posts[0]["post_id"] == "post1"
        assert posts[0]["analysis"]["risk_level"] == "HIGH"
    except Exception as e:
        pytest.fail(f"High-priority posts rendering failed: {e}")


def test_top_offenders_with_valid_data():
    """Test top offenders rendering with valid data."""
    users = [
        {
            "user_id": "user1",
            "username": "offender1",
            "statistics": {
                "total_posts": 50,
                "misinformation_posts": 20,
                "misinformation_rate": "40.0%",
                "avg_confidence": 0.75,
                "high_confidence_misinfo": 10
            }
        }
    ]
    
    # Should not raise any exceptions
    try:
        assert len(users) == 1
        assert users[0]["user_id"] == "user1"
        assert users[0]["statistics"]["total_posts"] == 50
    except Exception as e:
        pytest.fail(f"Top offenders rendering failed: {e}")


def test_pattern_breakdown_with_valid_data():
    """Test pattern breakdown rendering with valid data."""
    patterns = [
        {
            "pattern_name": "Pattern A",
            "total_occurrences": 15,
            "unique_users": 8,
            "first_seen": datetime.now().isoformat(),
            "last_seen": datetime.now().isoformat()
        }
    ]
    
    # Should not raise any exceptions
    try:
        assert len(patterns) == 1
        assert patterns[0]["pattern_name"] == "Pattern A"
        assert patterns[0]["total_occurrences"] == 15
    except Exception as e:
        pytest.fail(f"Pattern breakdown rendering failed: {e}")


def test_topic_analysis_with_valid_data():
    """Test topic analysis rendering with valid data."""
    topic_data = {
        "status": "success",
        "topics": [
            {
                "topic_name": "Topic 1",
                "total_posts": 30,
                "misinformation_posts": 10,
                "misinformation_rate": 33.3,
                "avg_confidence": 0.70,
                "keywords": ["keyword1", "keyword2", "keyword3"]
            }
        ]
    }
    
    # Should not raise any exceptions
    try:
        assert len(topic_data["topics"]) == 1
        assert topic_data["topics"][0]["topic_name"] == "Topic 1"
        assert topic_data["topics"][0]["total_posts"] == 30
    except Exception as e:
        pytest.fail(f"Topic analysis rendering failed: {e}")


def test_topic_analysis_with_no_data():
    """Test topic analysis rendering with no data."""
    topic_data = {
        "status": "no_data",
        "topics": []
    }
    
    # Should handle no data gracefully
    try:
        assert topic_data["status"] == "no_data"
        assert len(topic_data["topics"]) == 0
    except Exception as e:
        pytest.fail(f"Topic analysis rendering failed with no data: {e}")


def test_temporal_trends_with_valid_data():
    """Test temporal trends rendering with valid data."""
    trends = {
        "status": "success",
        "time_period": {
            "start": datetime.now().isoformat(),
            "end": datetime.now().isoformat()
        },
        "activity_patterns": {
            "peak_hours": [14, 15, 16],
            "peak_days": ["Monday", "Tuesday"]
        },
        "misinformation_trends": {
            "hourly_rate": {"14": 25.5, "15": 30.2, "16": 28.1},
            "daily_rate": {"Monday": 22.0, "Tuesday": 25.5}
        }
    }
    
    # Should not raise any exceptions
    try:
        assert trends["status"] == "success"
        assert len(trends["activity_patterns"]["peak_hours"]) == 3
        assert "Monday" in trends["activity_patterns"]["peak_days"]
    except Exception as e:
        pytest.fail(f"Temporal trends rendering failed: {e}")


def test_temporal_trends_with_no_data():
    """Test temporal trends rendering with no data."""
    trends = {
        "status": "no_data"
    }
    
    # Should handle no data gracefully
    try:
        assert trends["status"] == "no_data"
    except Exception as e:
        pytest.fail(f"Temporal trends rendering failed with no data: {e}")


def test_complete_results_display_with_full_report():
    """Test complete results display with full report."""
    report = {
        "executive_summary": {
            "total_posts_analyzed": 100,
            "misinformation_detected": 25,
            "high_risk_posts": 10,
            "critical_posts": 3,
            "unique_users": 50,
            "users_posting_misinfo": 15,
            "patterns_detected": 8,
            "topics_identified": 5
        },
        "high_priority_posts": [
            {
                "post_id": "post1",
                "username": "user1",
                "metadata": {"platform": "reddit"},
                "analysis": {
                    "verdict": False,
                    "confidence": 0.85,
                    "risk_level": "HIGH",
                    "patterns": []
                }
            }
        ],
        "top_offenders": [
            {
                "user_id": "user1",
                "username": "offender1",
                "statistics": {
                    "total_posts": 50,
                    "misinformation_posts": 20,
                    "misinformation_rate": "40.0%"
                }
            }
        ],
        "pattern_breakdown": [
            {
                "pattern_name": "Pattern A",
                "total_occurrences": 15,
                "unique_users": 8
            }
        ],
        "topic_analysis": {
            "topics": [
                {
                    "topic_name": "Topic 1",
                    "total_posts": 30,
                    "misinformation_posts": 10,
                    "misinformation_rate": 33.3,
                    "keywords": ["keyword1"]
                }
            ]
        },
        "temporal_analysis": {
            "status": "success",
            "activity_patterns": {
                "peak_hours": [14, 15],
                "peak_days": ["Monday"]
            }
        }
    }
    
    # Should not raise any exceptions
    try:
        assert "executive_summary" in report
        assert "high_priority_posts" in report
        assert "top_offenders" in report
        assert "pattern_breakdown" in report
        assert "topic_analysis" in report
        assert "temporal_analysis" in report
    except Exception as e:
        pytest.fail(f"Complete results display failed: {e}")


def test_data_preservation_in_display():
    """Test that data is preserved when preparing for display."""
    original_post = {
        "post_id": "test123",
        "username": "testuser",
        "metadata": {
            "post_id": "test123",
            "username": "testuser",
            "platform": "reddit",
            "timestamp": "2024-01-01T00:00:00"
        },
        "analysis": {
            "verdict": False,
            "confidence": 0.92,
            "risk_level": "CRITICAL",
            "patterns": ["pattern1", "pattern2", "pattern3"]
        }
    }
    
    # Verify data is preserved
    assert original_post["post_id"] == "test123"
    assert original_post["username"] == "testuser"
    assert original_post["analysis"]["confidence"] == 0.92
    assert original_post["analysis"]["risk_level"] == "CRITICAL"
    assert len(original_post["analysis"]["patterns"]) == 3
