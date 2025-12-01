"""
Property-based tests for markdown export completeness.

**Feature: agent-monitoring-dashboard, Property 5: Markdown export completeness**
**Validates: Requirements 5.2, 5.3**

Tests that all sections present in the dashboard report are included
in the generated markdown output.
"""

import pytest
from hypothesis import given, strategies as st, settings
from datetime import datetime
from typing import Dict, Any, List

from dashboard.utils.markdown_generator import MarkdownGenerator


# Strategies for generating test data

@st.composite
def executive_summary_strategy(draw):
    """Generate random executive summary data."""
    return {
        "total_posts_analyzed": draw(st.integers(min_value=0, max_value=10000)),
        "misinformation_detected": draw(st.integers(min_value=0, max_value=1000)),
        "high_risk_posts": draw(st.integers(min_value=0, max_value=500)),
        "critical_posts": draw(st.integers(min_value=0, max_value=100)),
        "unique_users": draw(st.integers(min_value=0, max_value=5000)),
        "users_posting_misinfo": draw(st.integers(min_value=0, max_value=1000)),
        "patterns_detected": draw(st.integers(min_value=0, max_value=50)),
        "topics_identified": draw(st.integers(min_value=0, max_value=20)),
        "report_generated": datetime.now().isoformat()
    }


@st.composite
def high_priority_post_strategy(draw):
    """Generate random high-priority post data."""
    patterns = draw(st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=5))
    tactics = draw(st.lists(st.text(min_size=1, max_size=30), min_size=0, max_size=5))
    examples = draw(st.lists(st.text(min_size=1, max_size=50), min_size=0, max_size=5))
    
    return {
        "post_id": draw(st.text(min_size=1, max_size=20)),
        "user_id": draw(st.text(min_size=1, max_size=20)),
        "username": draw(st.text(min_size=1, max_size=20)),
        "timestamp": datetime.now().isoformat(),
        "platform": draw(st.sampled_from(["reddit", "twitter", "facebook"])),
        "subreddit": draw(st.one_of(st.none(), st.text(min_size=1, max_size=20))),
        "text_preview": draw(st.text(min_size=0, max_size=300)),
        "full_text_length": draw(st.integers(min_value=0, max_value=5000)),
        "engagement": {
            "upvotes": draw(st.integers(min_value=0, max_value=10000)),
            "comments": draw(st.integers(min_value=0, max_value=1000)),
            "shares": draw(st.integers(min_value=0, max_value=1000))
        },
        "analysis": {
            "verdict": draw(st.sampled_from(["MISINFORMATION", "TRUE", "UNCERTAIN"])),
            "confidence": draw(st.floats(min_value=0.0, max_value=1.0)),
            "risk_level": draw(st.sampled_from(["LOW", "MODERATE", "HIGH", "CRITICAL"])),
            "detected_language": draw(st.sampled_from(["en", "es", "fr", "de"])),
            "patterns": patterns,
            "manipulation_tactics": tactics,
            "specific_examples": examples
        },
        "recommended_action": draw(st.text(min_size=1, max_size=100))
    }


@st.composite
def top_offender_strategy(draw):
    """Generate random top offender data."""
    patterns = draw(st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=5))
    tactics = draw(st.lists(st.text(min_size=1, max_size=30), min_size=0, max_size=5))
    languages = draw(st.lists(st.sampled_from(["en", "es", "fr", "de"]), min_size=0, max_size=3))
    
    total_posts = draw(st.integers(min_value=1, max_value=1000))
    misinfo_posts = draw(st.integers(min_value=0, max_value=total_posts))
    misinfo_rate = (misinfo_posts / total_posts * 100) if total_posts > 0 else 0
    
    return {
        "user_id": draw(st.text(min_size=1, max_size=20)),
        "username": draw(st.text(min_size=1, max_size=20)),
        "statistics": {
            "total_posts": total_posts,
            "misinformation_posts": misinfo_posts,
            "high_confidence_misinfo": draw(st.integers(min_value=0, max_value=misinfo_posts)),
            "misinformation_rate": f"{misinfo_rate:.1f}%",
            "avg_confidence": draw(st.floats(min_value=0.0, max_value=1.0))
        },
        "activity_period": {
            "first_seen": datetime.now().isoformat(),
            "last_seen": datetime.now().isoformat()
        },
        "patterns_used": patterns,
        "manipulation_tactics": tactics,
        "languages": languages,
        "recent_posts": draw(st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=5)),
        "recommended_action": draw(st.text(min_size=1, max_size=100))
    }


@st.composite
def pattern_strategy(draw):
    """Generate random pattern data."""
    examples = []
    num_examples = draw(st.integers(min_value=0, max_value=5))
    for _ in range(num_examples):
        examples.append({
            "post_id": draw(st.text(min_size=1, max_size=20)),
            "user_id": draw(st.text(min_size=1, max_size=20)),
            "timestamp": datetime.now().isoformat(),
            "confidence": draw(st.floats(min_value=0.0, max_value=1.0))
        })
    
    return {
        "pattern_name": draw(st.text(min_size=1, max_size=50)),
        "total_occurrences": draw(st.integers(min_value=0, max_value=1000)),
        "unique_users": draw(st.integers(min_value=0, max_value=500)),
        "first_seen": datetime.now().isoformat(),
        "last_seen": datetime.now().isoformat(),
        "recent_examples": examples
    }


@st.composite
def topic_strategy(draw):
    """Generate random topic data."""
    keywords = draw(st.lists(st.text(min_size=1, max_size=15), min_size=1, max_size=10))
    top_users = draw(st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=5))
    patterns = draw(st.lists(st.text(min_size=1, max_size=30), min_size=0, max_size=5))
    
    total_posts = draw(st.integers(min_value=1, max_value=1000))
    misinfo_posts = draw(st.integers(min_value=0, max_value=total_posts))
    misinfo_rate = (misinfo_posts / total_posts * 100) if total_posts > 0 else 0
    
    return {
        "topic_name": draw(st.text(min_size=1, max_size=50)),
        "total_posts": total_posts,
        "misinformation_posts": misinfo_posts,
        "misinformation_rate": misinfo_rate,
        "avg_confidence": draw(st.floats(min_value=0.0, max_value=1.0)),
        "keywords": keywords,
        "top_users": top_users,
        "patterns": patterns
    }


@st.composite
def temporal_trends_strategy(draw):
    """Generate random temporal trends data."""
    return {
        "status": "success",
        "time_period": {
            "start": datetime.now().isoformat(),
            "end": datetime.now().isoformat()
        },
        "activity_patterns": {
            "peak_hours": draw(st.lists(st.integers(min_value=0, max_value=23), min_size=0, max_size=5)),
            "peak_days": draw(st.lists(st.sampled_from(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]), min_size=0, max_size=3))
        },
        "misinformation_trends": {
            "hourly_rate": {str(i): draw(st.floats(min_value=0.0, max_value=100.0)) for i in range(3)},
            "daily_rate": {day: draw(st.floats(min_value=0.0, max_value=100.0)) for day in ["Monday", "Tuesday", "Wednesday"]}
        },
        "bursts_detected": draw(st.lists(
            st.fixed_dictionaries({
                "time": st.just(datetime.now().isoformat()),
                "post_count": st.integers(min_value=1, max_value=100),
                "user_count": st.integers(min_value=1, max_value=50),
                "misinfo_rate": st.floats(min_value=0.0, max_value=100.0)
            }),
            min_size=0,
            max_size=3
        ))
    }


@st.composite
def complete_report_strategy(draw):
    """Generate a complete report with all sections."""
    # Decide which sections to include
    include_high_priority = draw(st.booleans())
    include_top_offenders = draw(st.booleans())
    include_patterns = draw(st.booleans())
    include_topics = draw(st.booleans())
    include_temporal = draw(st.booleans())
    
    report = {
        "executive_summary": draw(executive_summary_strategy())
    }
    
    if include_high_priority:
        report["high_priority_posts"] = draw(st.lists(high_priority_post_strategy(), min_size=0, max_size=5))
    
    if include_top_offenders:
        report["top_offenders"] = draw(st.lists(top_offender_strategy(), min_size=0, max_size=5))
    
    if include_patterns:
        report["pattern_breakdown"] = draw(st.lists(pattern_strategy(), min_size=0, max_size=5))
    
    if include_topics:
        topics = draw(st.lists(topic_strategy(), min_size=0, max_size=5))
        report["topic_analysis"] = {"status": "success", "topics": topics}
    
    if include_temporal:
        report["temporal_analysis"] = draw(temporal_trends_strategy())
    
    return report


# Property tests

@given(complete_report_strategy())
@settings(max_examples=100, deadline=None)
def test_markdown_export_includes_all_report_sections(report):
    """
    Property 5: Markdown export completeness
    
    For any report with sections, the generated markdown must include
    all sections that are present in the report.
    
    **Validates: Requirements 5.2, 5.3**
    """
    generator = MarkdownGenerator()
    markdown = generator.generate_markdown_report(report)
    
    # Check that markdown is not empty
    assert markdown, "Markdown output should not be empty"
    
    # Check for header
    assert "# Misinformation Detection Analysis Report" in markdown, \
        "Markdown should include report title"
    
    # Check executive summary is always present
    assert "## Executive Summary" in markdown, \
        "Markdown should include Executive Summary section"
    
    # Check that all present sections are included in markdown
    if "high_priority_posts" in report and report["high_priority_posts"]:
        assert "## High-Priority Posts" in markdown, \
            "Markdown should include High-Priority Posts section when present in report"
    
    if "top_offenders" in report and report["top_offenders"]:
        assert "## Top Offenders" in markdown, \
            "Markdown should include Top Offenders section when present in report"
    
    if "pattern_breakdown" in report and report["pattern_breakdown"]:
        assert "## Pattern Breakdown" in markdown, \
            "Markdown should include Pattern Breakdown section when present in report"
    
    if "topic_analysis" in report:
        assert "## Topic Analysis" in markdown, \
            "Markdown should include Topic Analysis section when present in report"
    
    if "temporal_analysis" in report:
        assert "## Temporal Trends" in markdown, \
            "Markdown should include Temporal Trends section when present in report"


@given(executive_summary_strategy())
@settings(max_examples=100, deadline=None)
def test_executive_summary_contains_all_metrics(summary):
    """
    Property: Executive summary completeness
    
    For any executive summary, the markdown must include all key metrics.
    """
    generator = MarkdownGenerator()
    markdown = generator.format_executive_summary(summary)
    
    # Check that all metrics are present
    assert "Total Posts Analyzed" in markdown
    assert "Misinformation Detected" in markdown
    assert "High Risk Posts" in markdown
    assert "Critical Posts" in markdown
    assert "Unique Users" in markdown
    assert "Users Posting Misinformation" in markdown
    assert "Patterns Detected" in markdown
    assert "Topics Identified" in markdown
    
    # Check that values are present
    assert str(summary["total_posts_analyzed"]) in markdown
    assert str(summary["misinformation_detected"]) in markdown
    assert str(summary["high_risk_posts"]) in markdown


@given(st.lists(high_priority_post_strategy(), min_size=1, max_size=3))
@settings(max_examples=100, deadline=None)
def test_high_priority_posts_table_includes_all_posts(posts):
    """
    Property: High-priority posts completeness
    
    For any list of high-priority posts, the markdown must include
    all posts in the table.
    """
    generator = MarkdownGenerator()
    markdown = generator.format_high_priority_posts(posts)
    
    # Check table header is present
    assert "| Post ID | User | Risk Level | Confidence | Patterns | Action |" in markdown
    
    # Check that each post appears in the markdown
    for post in posts:
        post_id = post["post_id"]
        username = post["username"]
        
        # Post should appear in table or detailed section
        assert post_id in markdown, f"Post {post_id} should appear in markdown"
        assert username in markdown, f"Username {username} should appear in markdown"


@given(st.lists(top_offender_strategy(), min_size=1, max_size=3))
@settings(max_examples=100, deadline=None)
def test_top_offenders_table_includes_all_users(users):
    """
    Property: Top offenders completeness
    
    For any list of top offenders, the markdown must include
    all users in the table.
    """
    generator = MarkdownGenerator()
    markdown = generator.format_top_offenders(users)
    
    # Check table header is present
    assert "| User ID | Username | Total Posts | Misinfo Posts | Misinfo Rate | Action |" in markdown
    
    # Check that each user appears in the markdown
    for user in users:
        user_id = user["user_id"]
        username = user["username"]
        
        # User should appear in table or detailed section
        assert user_id in markdown, f"User {user_id} should appear in markdown"
        assert username in markdown, f"Username {username} should appear in markdown"


@given(st.lists(pattern_strategy(), min_size=1, max_size=3))
@settings(max_examples=100, deadline=None)
def test_pattern_breakdown_includes_all_patterns(patterns):
    """
    Property: Pattern breakdown completeness
    
    For any list of patterns, the markdown must include all patterns.
    """
    generator = MarkdownGenerator()
    markdown = generator.format_pattern_breakdown(patterns)
    
    # Check table header is present
    assert "| Pattern | Occurrences | Unique Users | First Seen | Last Seen |" in markdown
    
    # Check that each pattern appears in the markdown
    for pattern in patterns:
        pattern_name = pattern["pattern_name"]
        assert pattern_name in markdown, f"Pattern {pattern_name} should appear in markdown"


@given(st.lists(topic_strategy(), min_size=1, max_size=3))
@settings(max_examples=100, deadline=None)
def test_topic_analysis_includes_all_topics(topics):
    """
    Property: Topic analysis completeness
    
    For any list of topics, the markdown must include all topics.
    """
    generator = MarkdownGenerator()
    topic_data = {"status": "success", "topics": topics}
    markdown = generator.format_topic_analysis(topic_data)
    
    # Check table header is present
    assert "| Topic | Total Posts | Misinfo Posts | Misinfo Rate | Top Keywords |" in markdown
    
    # Check that each topic appears in the markdown
    for topic in topics:
        topic_name = topic["topic_name"]
        assert topic_name in markdown, f"Topic {topic_name} should appear in markdown"


@given(temporal_trends_strategy())
@settings(max_examples=100, deadline=None)
def test_temporal_trends_includes_key_sections(trends):
    """
    Property: Temporal trends completeness
    
    For any temporal trends data, the markdown must include key sections.
    """
    generator = MarkdownGenerator()
    markdown = generator.format_temporal_trends(trends)
    
    # Check main section header
    assert "## Temporal Trends" in markdown
    
    # If we have activity patterns, they should be included
    if "activity_patterns" in trends:
        patterns = trends["activity_patterns"]
        if "peak_hours" in patterns and patterns["peak_hours"]:
            assert "Peak Activity Hours" in markdown
        if "peak_days" in patterns and patterns["peak_days"]:
            assert "Peak Activity Days" in markdown
