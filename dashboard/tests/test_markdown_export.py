"""
Unit tests for markdown export functionality.

Tests the markdown export components to ensure they generate
correct markdown output with proper formatting and metadata.
"""

import pytest
from datetime import datetime
import tempfile
import os

from dashboard.components.markdown_export import (
    generate_markdown_with_metadata,
    export_markdown_to_string,
    save_markdown_to_file,
    get_markdown_stats
)


def create_mock_report():
    """Create a mock report for testing."""
    return {
        "executive_summary": {
            "total_posts_analyzed": 100,
            "misinformation_detected": 25,
            "high_risk_posts": 10,
            "critical_posts": 3,
            "unique_users": 50,
            "users_posting_misinfo": 15,
            "patterns_detected": 8,
            "topics_identified": 5,
            "report_generated": datetime.now().isoformat()
        },
        "high_priority_posts": [
            {
                "post_id": "post1",
                "username": "user1",
                "metadata": {
                    "platform": "reddit",
                    "timestamp": datetime.now().isoformat()
                },
                "analysis": {
                    "verdict": False,
                    "confidence": 0.85,
                    "risk_level": "HIGH",
                    "patterns": ["pattern1"]
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
        }
    }


def test_generate_markdown_with_metadata():
    """Test markdown generation with metadata."""
    report = create_mock_report()
    markdown = generate_markdown_with_metadata(report, execution_id="test123")
    
    # Check that markdown is generated
    assert markdown is not None
    assert len(markdown) > 0
    
    # Check for title
    assert "# Misinformation Detection Analysis Report" in markdown
    
    # Check for metadata section
    assert "## Report Metadata" in markdown
    assert "test123" in markdown


def test_generate_markdown_without_metadata():
    """Test markdown generation without metadata."""
    report = create_mock_report()
    markdown = generate_markdown_with_metadata(report, include_metadata=False)
    
    # Check that markdown is generated
    assert markdown is not None
    assert len(markdown) > 0
    
    # Check that metadata section is not included
    assert "## Report Metadata" not in markdown


def test_markdown_contains_executive_summary():
    """Test that markdown contains executive summary."""
    report = create_mock_report()
    markdown = export_markdown_to_string(report)
    
    # Check for executive summary section
    assert "## Executive Summary" in markdown
    assert "Total Posts Analyzed" in markdown
    assert "100" in markdown
    assert "Misinformation Detected" in markdown
    assert "25" in markdown


def test_markdown_contains_high_priority_posts():
    """Test that markdown contains high-priority posts."""
    report = create_mock_report()
    markdown = export_markdown_to_string(report)
    
    # Check for high-priority posts section
    assert "## High-Priority Posts" in markdown
    assert "post1" in markdown
    assert "user1" in markdown


def test_markdown_contains_top_offenders():
    """Test that markdown contains top offenders."""
    report = create_mock_report()
    markdown = export_markdown_to_string(report)
    
    # Check for top offenders section
    assert "## Top Offenders" in markdown
    assert "offender1" in markdown
    assert "40.0%" in markdown


def test_markdown_contains_pattern_breakdown():
    """Test that markdown contains pattern breakdown."""
    report = create_mock_report()
    markdown = export_markdown_to_string(report)
    
    # Check for pattern breakdown section
    assert "## Pattern Breakdown" in markdown
    assert "Pattern A" in markdown


def test_markdown_contains_topic_analysis():
    """Test that markdown contains topic analysis."""
    report = create_mock_report()
    markdown = export_markdown_to_string(report)
    
    # Check for topic analysis section
    assert "## Topic Analysis" in markdown
    assert "Topic 1" in markdown


def test_save_markdown_to_file():
    """Test saving markdown to file."""
    report = create_mock_report()
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.md') as f:
        filepath = f.name
    
    try:
        # Save markdown
        success = save_markdown_to_file(report, filepath)
        
        assert success is True
        assert os.path.exists(filepath)
        
        # Read and verify content
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "# Misinformation Detection Analysis Report" in content
        assert "## Executive Summary" in content
    
    finally:
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)


def test_get_markdown_stats():
    """Test getting markdown statistics."""
    report = create_mock_report()
    stats = get_markdown_stats(report)
    
    # Check that stats are returned
    assert stats is not None
    assert "total_characters" in stats
    assert "total_lines" in stats
    assert "total_words" in stats
    assert "sections" in stats
    assert "section_count" in stats
    
    # Check that values are reasonable
    assert stats["total_characters"] > 0
    assert stats["total_lines"] > 0
    assert stats["total_words"] > 0
    assert stats["section_count"] > 0


def test_markdown_stats_sections():
    """Test that markdown stats correctly identify sections."""
    report = create_mock_report()
    stats = get_markdown_stats(report)
    
    # Check that sections are identified
    sections = stats["sections"]
    assert "Executive Summary" in sections
    assert "High-Priority Posts" in sections
    assert "Top Offenders" in sections
    assert "Pattern Breakdown" in sections
    assert "Topic Analysis" in sections


def test_markdown_with_empty_report():
    """Test markdown generation with minimal report."""
    report = {
        "executive_summary": {
            "total_posts_analyzed": 0,
            "misinformation_detected": 0,
            "high_risk_posts": 0,
            "critical_posts": 0,
            "unique_users": 0,
            "users_posting_misinfo": 0,
            "patterns_detected": 0,
            "topics_identified": 0
        }
    }
    
    markdown = export_markdown_to_string(report)
    
    # Should still generate valid markdown
    assert markdown is not None
    assert len(markdown) > 0
    assert "# Misinformation Detection Analysis Report" in markdown
    assert "## Executive Summary" in markdown


def test_markdown_formatting():
    """Test that markdown has proper formatting."""
    report = create_mock_report()
    markdown = export_markdown_to_string(report)
    
    # Check for proper markdown headers
    assert markdown.count("# ") >= 1  # At least one H1
    assert markdown.count("## ") >= 1  # At least one H2
    
    # Check for proper list formatting
    assert "- **" in markdown  # Bullet points with bold
    
    # Check for tables (if present)
    if "high_priority_posts" in report and report["high_priority_posts"]:
        assert "|" in markdown  # Table separator


def test_markdown_metadata_timestamp():
    """Test that metadata includes timestamp."""
    report = create_mock_report()
    markdown = generate_markdown_with_metadata(
        report,
        execution_id="test123",
        include_metadata=True
    )
    
    # Check for timestamp in metadata
    assert "Generated:" in markdown
    assert "Execution ID:" in markdown
    assert "test123" in markdown


def test_export_markdown_to_string_with_metadata():
    """Test exporting markdown to string with metadata."""
    report = create_mock_report()
    
    # With metadata
    markdown_with = export_markdown_to_string(
        report,
        include_metadata=True,
        execution_id="test456"
    )
    
    # Without metadata
    markdown_without = export_markdown_to_string(
        report,
        include_metadata=False
    )
    
    # With metadata should be longer
    assert len(markdown_with) > len(markdown_without)
    
    # Check metadata presence
    assert "## Report Metadata" in markdown_with
    assert "test456" in markdown_with
    assert "## Report Metadata" not in markdown_without


def test_markdown_data_preservation():
    """Test that all data is preserved in markdown."""
    report = create_mock_report()
    markdown = export_markdown_to_string(report)
    
    # Check that key data points are present
    exec_summary = report["executive_summary"]
    assert str(exec_summary["total_posts_analyzed"]) in markdown
    assert str(exec_summary["misinformation_detected"]) in markdown
    assert str(exec_summary["high_risk_posts"]) in markdown
    
    # Check high-priority post data
    post = report["high_priority_posts"][0]
    assert post["post_id"] in markdown
    assert post["username"] in markdown
    
    # Check top offender data
    offender = report["top_offenders"][0]
    assert offender["username"] in markdown
    
    # Check pattern data
    pattern = report["pattern_breakdown"][0]
    assert pattern["pattern_name"] in markdown


def test_markdown_special_characters():
    """Test markdown generation with special characters."""
    report = {
        "executive_summary": {
            "total_posts_analyzed": 10,
            "misinformation_detected": 5,
            "high_risk_posts": 2,
            "critical_posts": 1,
            "unique_users": 8,
            "users_posting_misinfo": 3,
            "patterns_detected": 2,
            "topics_identified": 1
        },
        "high_priority_posts": [
            {
                "post_id": "post_with_special_chars",
                "username": "user@test",
                "metadata": {"platform": "reddit"},
                "analysis": {
                    "verdict": False,
                    "confidence": 0.9,
                    "risk_level": "HIGH",
                    "patterns": ["pattern with spaces"]
                }
            }
        ]
    }
    
    markdown = export_markdown_to_string(report)
    
    # Should handle special characters
    assert "user@test" in markdown
    assert "pattern with spaces" in markdown
