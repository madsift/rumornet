"""
Filtering and search functionality for the Agent Monitoring Dashboard.

This module provides comprehensive filtering and search capabilities for
analysis results including risk level filters, confidence thresholds,
search by post/user ID, and pattern filters.

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
"""

import streamlit as st
from typing import Dict, List, Any, Optional, Callable
import pandas as pd


def create_search_ui() -> str:
    """
    Create search UI for post IDs and user IDs.
    
    Provides a text input field for searching posts by post ID or user ID.
    
    Returns:
        Search term entered by user
        
    Requirements: 9.1
    """
    search_term = st.text_input(
        "🔍 Search by Post ID or User ID",
        placeholder="Enter post ID or user ID...",
        help="Search for specific posts or users. Case-insensitive partial matching.",
        key="search_input"
    )
    
    return search_term.strip() if search_term else ""


def create_risk_level_filter() -> List[str]:
    """
    Create risk level filter UI.
    
    Provides multiselect for filtering by risk level.
    
    Returns:
        List of selected risk levels
        
    Requirements: 9.2
    """
    risk_levels = st.multiselect(
        "⚠️ Risk Level",
        options=["LOW", "MODERATE", "HIGH", "CRITICAL"],
        default=["HIGH", "CRITICAL"],
        help="Filter results by risk level. Select one or more levels.",
        key="risk_level_filter"
    )
    
    return risk_levels


def create_confidence_threshold_filter() -> float:
    """
    Create confidence threshold filter UI.
    
    Provides a slider for setting minimum confidence threshold.
    
    Returns:
        Minimum confidence threshold (0.0 to 1.0)
        
    Requirements: 9.3
    """
    confidence_min = st.slider(
        "📊 Minimum Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Filter results by minimum confidence score",
        key="confidence_threshold"
    )
    
    return confidence_min


def create_pattern_filter() -> str:
    """
    Create pattern filter UI.
    
    Provides a text input for filtering by detected patterns.
    
    Returns:
        Pattern filter term
        
    Requirements: 9.4
    """
    pattern_filter = st.text_input(
        "🔍 Pattern Filter",
        placeholder="Enter pattern name...",
        help="Filter by specific misinformation pattern. Case-insensitive partial matching.",
        key="pattern_filter"
    )
    
    return pattern_filter.strip() if pattern_filter else ""


def create_filter_panel() -> Dict[str, Any]:
    """
    Create comprehensive filter panel with all filter controls.
    
    Combines all filter UI elements into a single panel and returns
    the current filter state.
    
    Returns:
        Dictionary containing all filter values
        
    Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
    """
    st.subheader("🔧 Filters")
    
    filters = {}
    
    # Search
    filters["search"] = create_search_ui()
    
    # Risk level
    filters["risk_levels"] = create_risk_level_filter()
    
    # Confidence threshold
    filters["confidence_min"] = create_confidence_threshold_filter()
    
    # Pattern filter
    filters["pattern"] = create_pattern_filter()
    
    # Filter controls
    col1, col2 = st.columns(2)
    
    with col1:
        apply_filters = st.button(
            "✅ Apply Filters",
            type="primary",
            use_container_width=True,
            key="apply_filters_btn"
        )
    
    with col2:
        clear_filters = st.button(
            "🔄 Clear Filters",
            use_container_width=True,
            key="clear_filters_btn"
        )
    
    filters["apply"] = apply_filters
    filters["clear"] = clear_filters
    
    return filters


def apply_search_filter(posts: List[Dict[str, Any]], search_term: str) -> List[Dict[str, Any]]:
    """
    Apply search filter to posts.
    
    Filters posts by searching for the term in post ID or user ID.
    Case-insensitive partial matching.
    
    Args:
        posts: List of post data
        search_term: Search term to match
        
    Returns:
        Filtered list of posts
        
    Requirements: 9.1
    """
    if not search_term:
        return posts
    
    search_lower = search_term.lower()
    filtered = []
    
    for post in posts:
        # Check post ID
        post_id = post.get("post_id", post.get("metadata", {}).get("post_id", ""))
        if search_lower in post_id.lower():
            filtered.append(post)
            continue
        
        # Check user ID
        user_id = post.get("user_id", post.get("metadata", {}).get("user_id", ""))
        if search_lower in user_id.lower():
            filtered.append(post)
            continue
        
        # Check username
        username = post.get("username", post.get("metadata", {}).get("username", ""))
        if search_lower in username.lower():
            filtered.append(post)
            continue
    
    return filtered


def apply_risk_level_filter(posts: List[Dict[str, Any]], risk_levels: List[str]) -> List[Dict[str, Any]]:
    """
    Apply risk level filter to posts.
    
    Filters posts to include only those with specified risk levels.
    
    Args:
        posts: List of post data
        risk_levels: List of risk levels to include
        
    Returns:
        Filtered list of posts
        
    Requirements: 9.2
    """
    if not risk_levels:
        return posts
    
    filtered = []
    
    for post in posts:
        # Get risk level from post or analysis
        risk_level = post.get("risk_level")
        if not risk_level:
            risk_level = post.get("analysis", {}).get("risk_level")
        
        if risk_level in risk_levels:
            filtered.append(post)
    
    return filtered


def apply_confidence_filter(posts: List[Dict[str, Any]], confidence_min: float) -> List[Dict[str, Any]]:
    """
    Apply confidence threshold filter to posts.
    
    Filters posts to include only those with confidence >= threshold.
    
    Args:
        posts: List of post data
        confidence_min: Minimum confidence threshold
        
    Returns:
        Filtered list of posts
        
    Requirements: 9.3
    """
    filtered = []
    
    for post in posts:
        # Get confidence from post or analysis
        confidence = post.get("confidence")
        if confidence is None:
            confidence = post.get("analysis", {}).get("confidence", 0.0)
        
        if confidence >= confidence_min:
            filtered.append(post)
    
    return filtered


def apply_pattern_filter(posts: List[Dict[str, Any]], pattern: str) -> List[Dict[str, Any]]:
    """
    Apply pattern filter to posts.
    
    Filters posts to include only those containing the specified pattern.
    Case-insensitive partial matching.
    
    Args:
        posts: List of post data
        pattern: Pattern name to match
        
    Returns:
        Filtered list of posts
        
    Requirements: 9.4
    """
    if not pattern:
        return posts
    
    pattern_lower = pattern.lower()
    filtered = []
    
    for post in posts:
        # Get patterns from post or analysis
        patterns = post.get("patterns", [])
        if not patterns:
            patterns = post.get("analysis", {}).get("patterns", [])
            if not patterns:
                patterns = post.get("analysis", {}).get("patterns_detected", [])
        
        # Check if any pattern matches
        if any(pattern_lower in p.lower() for p in patterns):
            filtered.append(post)
    
    return filtered


def apply_all_filters(posts: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Apply all filters to posts.
    
    Applies all filter criteria in sequence, ensuring only posts that
    match ALL criteria are included in the result.
    
    Args:
        posts: List of post data
        filters: Dictionary containing all filter criteria
        
    Returns:
        Filtered list of posts
        
    Requirements: 9.5
    """
    filtered = posts
    
    # Apply search filter
    if filters.get("search"):
        filtered = apply_search_filter(filtered, filters["search"])
    
    # Apply risk level filter
    if filters.get("risk_levels"):
        filtered = apply_risk_level_filter(filtered, filters["risk_levels"])
    
    # Apply confidence filter
    if "confidence_min" in filters:
        filtered = apply_confidence_filter(filtered, filters["confidence_min"])
    
    # Apply pattern filter
    if filters.get("pattern"):
        filtered = apply_pattern_filter(filtered, filters["pattern"])
    
    return filtered


def filter_high_priority_posts(
    posts: List[Dict[str, Any]],
    filters: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Filter high-priority posts based on filter criteria.
    
    Convenience function for filtering high-priority posts specifically.
    
    Args:
        posts: List of high-priority post data
        filters: Dictionary containing filter criteria
        
    Returns:
        Filtered list of posts
        
    Requirements: 9.5
    """
    return apply_all_filters(posts, filters)


def filter_top_offenders(
    users: List[Dict[str, Any]],
    search_term: str = ""
) -> List[Dict[str, Any]]:
    """
    Filter top offenders by search term.
    
    Filters users by searching for the term in user ID or username.
    
    Args:
        users: List of user data
        search_term: Search term to match
        
    Returns:
        Filtered list of users
        
    Requirements: 9.1
    """
    if not search_term:
        return users
    
    search_lower = search_term.lower()
    filtered = []
    
    for user in users:
        # Check user ID
        user_id = user.get("user_id", "")
        if search_lower in user_id.lower():
            filtered.append(user)
            continue
        
        # Check username
        username = user.get("username", "")
        if search_lower in username.lower():
            filtered.append(user)
            continue
    
    return filtered


def filter_patterns(
    patterns: List[Dict[str, Any]],
    search_term: str = ""
) -> List[Dict[str, Any]]:
    """
    Filter patterns by search term.
    
    Filters patterns by searching for the term in pattern name.
    
    Args:
        patterns: List of pattern data
        search_term: Search term to match
        
    Returns:
        Filtered list of patterns
        
    Requirements: 9.4
    """
    if not search_term:
        return patterns
    
    search_lower = search_term.lower()
    filtered = []
    
    for pattern in patterns:
        pattern_name = pattern.get("pattern_name", "")
        if search_lower in pattern_name.lower():
            filtered.append(pattern)
    
    return filtered


def filter_topics(
    topics: List[Dict[str, Any]],
    search_term: str = ""
) -> List[Dict[str, Any]]:
    """
    Filter topics by search term.
    
    Filters topics by searching for the term in topic name or keywords.
    
    Args:
        topics: List of topic data
        search_term: Search term to match
        
    Returns:
        Filtered list of topics
        
    Requirements: 9.1
    """
    if not search_term:
        return topics
    
    search_lower = search_term.lower()
    filtered = []
    
    for topic in topics:
        # Check topic name
        topic_name = topic.get("topic_name", "")
        if search_lower in topic_name.lower():
            filtered.append(topic)
            continue
        
        # Check keywords
        keywords = topic.get("keywords", [])
        if any(search_lower in keyword.lower() for keyword in keywords):
            filtered.append(topic)
            continue
    
    return filtered


def display_filter_summary(original_count: int, filtered_count: int):
    """
    Display summary of filter results.
    
    Shows how many items were filtered and provides visual feedback.
    
    Args:
        original_count: Number of items before filtering
        filtered_count: Number of items after filtering
        
    Requirements: 9.5
    """
    if filtered_count < original_count:
        filtered_out = original_count - filtered_count
        st.info(
            f"📊 Showing {filtered_count} of {original_count} items "
            f"({filtered_out} filtered out)"
        )
    elif filtered_count == 0 and original_count > 0:
        st.warning("⚠️ No items match the current filter criteria. Try adjusting your filters.")
    elif filtered_count == original_count and original_count > 0:
        st.success(f"✅ Showing all {original_count} items (no filters applied)")


def update_visualizations_with_filters(
    report: Dict[str, Any],
    filters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update all visualizations with applied filters.
    
    Applies filters to all sections of the report and returns
    the filtered report.
    
    Args:
        report: Complete orchestrator report
        filters: Dictionary containing filter criteria
        
    Returns:
        Filtered report with all sections updated
        
    Requirements: 9.5
    """
    filtered_report = report.copy()
    
    # Filter high-priority posts
    if "high_priority_posts" in report:
        original_posts = report["high_priority_posts"]
        filtered_posts = apply_all_filters(original_posts, filters)
        filtered_report["high_priority_posts"] = filtered_posts
        
        # Update executive summary counts
        if "executive_summary" in filtered_report:
            filtered_report["executive_summary"]["high_risk_posts"] = len(filtered_posts)
    
    # Filter top offenders (by search only)
    if "top_offenders" in report and filters.get("search"):
        original_users = report["top_offenders"]
        filtered_users = filter_top_offenders(original_users, filters["search"])
        filtered_report["top_offenders"] = filtered_users
    
    # Filter patterns (by search only)
    if "pattern_breakdown" in report and filters.get("pattern"):
        original_patterns = report["pattern_breakdown"]
        filtered_patterns = filter_patterns(original_patterns, filters["pattern"])
        filtered_report["pattern_breakdown"] = filtered_patterns
    
    # Filter topics (by search only)
    if "topic_analysis" in report and "topics" in report["topic_analysis"]:
        if filters.get("search"):
            original_topics = report["topic_analysis"]["topics"]
            filtered_topics = filter_topics(original_topics, filters["search"])
            filtered_report["topic_analysis"]["topics"] = filtered_topics
    
    return filtered_report


def render_filter_panel_with_results(
    report: Dict[str, Any],
    on_filter_change: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Render filter panel and return filtered results.
    
    Creates the filter UI, applies filters to the report, and returns
    the filtered report along with filter state.
    
    Args:
        report: Complete orchestrator report
        on_filter_change: Optional callback when filters change
        
    Returns:
        Dictionary containing filtered report and filter state
        
    Requirements: 9.5
    """
    # Create filter panel
    with st.sidebar:
        st.divider()
        filters = create_filter_panel()
    
    # Apply filters if requested
    if filters.get("apply"):
        filtered_report = update_visualizations_with_filters(report, filters)
        
        # Show filter summary
        if "high_priority_posts" in report:
            original_count = len(report["high_priority_posts"])
            filtered_count = len(filtered_report["high_priority_posts"])
            display_filter_summary(original_count, filtered_count)
        
        # Call callback if provided
        if on_filter_change:
            on_filter_change(filters)
        
        return {
            "report": filtered_report,
            "filters": filters,
            "filtered": True
        }
    
    # Clear filters if requested
    if filters.get("clear"):
        st.rerun()
    
    # Return original report if no action taken
    return {
        "report": report,
        "filters": filters,
        "filtered": False
    }
