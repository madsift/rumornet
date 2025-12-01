"""
Property-based tests for filter consistency.

**Feature: agent-monitoring-dashboard, Property 8: Filter consistency**
**Validates: Requirements 9.2, 9.3, 9.4, 9.5**

Tests that filtered datasets contain only items matching all filter criteria.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from datetime import datetime
from typing import Dict, Any, List


# Strategies for generating test data

@st.composite
def post_result_strategy(draw):
    """Generate random post result data."""
    patterns = draw(st.lists(
        st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=32, max_codepoint=126)),
        min_size=0,
        max_size=5
    ))
    
    return {
        "post_id": draw(st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=65, max_codepoint=90))),
        "user_id": draw(st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=65, max_codepoint=90))),
        "username": draw(st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=97, max_codepoint=122))),
        "risk_level": draw(st.sampled_from(["LOW", "MODERATE", "HIGH", "CRITICAL"])),
        "confidence": draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        "patterns": patterns,
        "timestamp": datetime.now().isoformat()
    }


@st.composite
def filter_criteria_strategy(draw):
    """Generate random filter criteria."""
    # Generate risk levels to filter by
    all_risk_levels = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
    risk_levels = draw(st.lists(
        st.sampled_from(all_risk_levels),
        min_size=1,
        max_size=4,
        unique=True
    ))
    
    # Generate confidence threshold
    confidence_min = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    
    # Generate search term (optional)
    search_term = draw(st.one_of(
        st.none(),
        st.text(min_size=1, max_size=10, alphabet=st.characters(min_codepoint=65, max_codepoint=90))
    ))
    
    # Generate pattern filter (optional)
    pattern_filter = draw(st.one_of(
        st.none(),
        st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=32, max_codepoint=126))
    ))
    
    return {
        "risk_levels": risk_levels,
        "confidence_min": confidence_min,
        "search": search_term,
        "pattern": pattern_filter
    }


# Filter functions

def apply_risk_level_filter(posts: List[Dict[str, Any]], risk_levels: List[str]) -> List[Dict[str, Any]]:
    """Filter posts by risk level."""
    return [post for post in posts if post.get("risk_level") in risk_levels]


def apply_confidence_filter(posts: List[Dict[str, Any]], confidence_min: float) -> List[Dict[str, Any]]:
    """Filter posts by minimum confidence threshold."""
    return [post for post in posts if post.get("confidence", 0.0) >= confidence_min]


def apply_search_filter(posts: List[Dict[str, Any]], search_term: str) -> List[Dict[str, Any]]:
    """Filter posts by search term (post ID or user ID)."""
    if not search_term:
        return posts
    
    search_lower = search_term.lower()
    return [
        post for post in posts
        if search_lower in post.get("post_id", "").lower()
        or search_lower in post.get("user_id", "").lower()
    ]


def apply_pattern_filter(posts: List[Dict[str, Any]], pattern: str) -> List[Dict[str, Any]]:
    """Filter posts by pattern."""
    if not pattern:
        return posts
    
    pattern_lower = pattern.lower()
    return [
        post for post in posts
        if any(pattern_lower in p.lower() for p in post.get("patterns", []))
    ]


def apply_all_filters(posts: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Apply all filters to posts."""
    filtered = posts
    
    # Apply risk level filter
    if filters.get("risk_levels"):
        filtered = apply_risk_level_filter(filtered, filters["risk_levels"])
    
    # Apply confidence filter
    if "confidence_min" in filters:
        filtered = apply_confidence_filter(filtered, filters["confidence_min"])
    
    # Apply search filter
    if filters.get("search"):
        filtered = apply_search_filter(filtered, filters["search"])
    
    # Apply pattern filter
    if filters.get("pattern"):
        filtered = apply_pattern_filter(filtered, filters["pattern"])
    
    return filtered


# Property tests

@given(st.lists(post_result_strategy(), min_size=0, max_size=50), filter_criteria_strategy())
@settings(max_examples=100, deadline=None)
def test_risk_level_filter_consistency(posts, filters):
    """
    Property 8: Filter consistency - Risk Level
    
    For any set of posts and risk level filter, all filtered posts must
    have a risk level that matches one of the selected risk levels.
    
    **Validates: Requirements 9.2**
    """
    risk_levels = filters["risk_levels"]
    filtered_posts = apply_risk_level_filter(posts, risk_levels)
    
    # All filtered posts must have matching risk level
    for post in filtered_posts:
        assert post.get("risk_level") in risk_levels, \
            f"Post with risk level {post.get('risk_level')} should not pass filter for {risk_levels}"
    
    # No posts with matching risk level should be excluded
    for post in posts:
        if post.get("risk_level") in risk_levels:
            assert post in filtered_posts, \
                f"Post with risk level {post.get('risk_level')} should pass filter for {risk_levels}"


@given(st.lists(post_result_strategy(), min_size=0, max_size=50), st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=100, deadline=None)
def test_confidence_threshold_filter_consistency(posts, confidence_min):
    """
    Property 8: Filter consistency - Confidence Threshold
    
    For any set of posts and confidence threshold, all filtered posts must
    have confidence greater than or equal to the threshold.
    
    **Validates: Requirements 9.3**
    """
    filtered_posts = apply_confidence_filter(posts, confidence_min)
    
    # All filtered posts must meet confidence threshold
    for post in filtered_posts:
        assert post.get("confidence", 0.0) >= confidence_min, \
            f"Post with confidence {post.get('confidence')} should not pass filter for threshold {confidence_min}"
    
    # No posts meeting threshold should be excluded
    for post in posts:
        if post.get("confidence", 0.0) >= confidence_min:
            assert post in filtered_posts, \
                f"Post with confidence {post.get('confidence')} should pass filter for threshold {confidence_min}"


@given(st.lists(post_result_strategy(), min_size=0, max_size=50), st.text(min_size=1, max_size=10, alphabet=st.characters(min_codepoint=65, max_codepoint=90)))
@settings(max_examples=100, deadline=None)
def test_search_filter_consistency(posts, search_term):
    """
    Property 8: Filter consistency - Search
    
    For any set of posts and search term, all filtered posts must contain
    the search term in either post ID or user ID.
    
    **Validates: Requirements 9.2**
    """
    filtered_posts = apply_search_filter(posts, search_term)
    
    search_lower = search_term.lower()
    
    # All filtered posts must contain search term
    for post in filtered_posts:
        post_id_match = search_lower in post.get("post_id", "").lower()
        user_id_match = search_lower in post.get("user_id", "").lower()
        
        assert post_id_match or user_id_match, \
            f"Post {post.get('post_id')} with user {post.get('user_id')} should not pass search filter for '{search_term}'"
    
    # No posts containing search term should be excluded
    for post in posts:
        post_id_match = search_lower in post.get("post_id", "").lower()
        user_id_match = search_lower in post.get("user_id", "").lower()
        
        if post_id_match or user_id_match:
            assert post in filtered_posts, \
                f"Post {post.get('post_id')} with user {post.get('user_id')} should pass search filter for '{search_term}'"


@given(st.lists(post_result_strategy(), min_size=0, max_size=50), st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=32, max_codepoint=126)))
@settings(max_examples=100, deadline=None)
def test_pattern_filter_consistency(posts, pattern):
    """
    Property 8: Filter consistency - Pattern
    
    For any set of posts and pattern filter, all filtered posts must contain
    the specified pattern in their patterns list.
    
    **Validates: Requirements 9.4**
    """
    filtered_posts = apply_pattern_filter(posts, pattern)
    
    pattern_lower = pattern.lower()
    
    # All filtered posts must contain the pattern
    for post in filtered_posts:
        patterns = post.get("patterns", [])
        has_pattern = any(pattern_lower in p.lower() for p in patterns)
        
        assert has_pattern, \
            f"Post {post.get('post_id')} with patterns {patterns} should not pass pattern filter for '{pattern}'"
    
    # No posts containing pattern should be excluded
    for post in posts:
        patterns = post.get("patterns", [])
        has_pattern = any(pattern_lower in p.lower() for p in patterns)
        
        if has_pattern:
            assert post in filtered_posts, \
                f"Post {post.get('post_id')} with patterns {patterns} should pass pattern filter for '{pattern}'"


@given(st.lists(post_result_strategy(), min_size=0, max_size=50), filter_criteria_strategy())
@settings(max_examples=100, deadline=None)
def test_combined_filters_consistency(posts, filters):
    """
    Property 8: Filter consistency - Combined Filters
    
    For any set of posts and multiple filter criteria, all filtered posts must
    match ALL filter criteria (AND logic).
    
    **Validates: Requirements 9.2, 9.3, 9.4, 9.5**
    """
    filtered_posts = apply_all_filters(posts, filters)
    
    # All filtered posts must match all criteria
    for post in filtered_posts:
        # Check risk level
        if filters.get("risk_levels"):
            assert post.get("risk_level") in filters["risk_levels"], \
                f"Post risk level {post.get('risk_level')} doesn't match filter {filters['risk_levels']}"
        
        # Check confidence
        if "confidence_min" in filters:
            assert post.get("confidence", 0.0) >= filters["confidence_min"], \
                f"Post confidence {post.get('confidence')} doesn't meet threshold {filters['confidence_min']}"
        
        # Check search term
        if filters.get("search"):
            search_lower = filters["search"].lower()
            post_id_match = search_lower in post.get("post_id", "").lower()
            user_id_match = search_lower in post.get("user_id", "").lower()
            assert post_id_match or user_id_match, \
                f"Post doesn't match search term '{filters['search']}'"
        
        # Check pattern
        if filters.get("pattern"):
            pattern_lower = filters["pattern"].lower()
            patterns = post.get("patterns", [])
            has_pattern = any(pattern_lower in p.lower() for p in patterns)
            assert has_pattern, \
                f"Post doesn't contain pattern '{filters['pattern']}'"


@given(st.lists(post_result_strategy(), min_size=1, max_size=50), filter_criteria_strategy())
@settings(max_examples=100, deadline=None)
def test_filter_subset_property(posts, filters):
    """
    Property 8: Filter consistency - Subset Property
    
    For any set of posts and filter criteria, the filtered result must be
    a subset of the original posts (no new posts added).
    
    **Validates: Requirements 9.5**
    """
    filtered_posts = apply_all_filters(posts, filters)
    
    # Filtered posts must be a subset of original posts
    for post in filtered_posts:
        assert post in posts, \
            "Filtered result contains post not in original dataset"
    
    # Filtered count must not exceed original count
    assert len(filtered_posts) <= len(posts), \
        "Filtered result has more posts than original dataset"


@given(st.lists(post_result_strategy(), min_size=0, max_size=50))
@settings(max_examples=100, deadline=None)
def test_empty_filter_returns_all(posts):
    """
    Property 8: Filter consistency - Empty Filter
    
    For any set of posts, applying no filters should return all posts.
    
    **Validates: Requirements 9.5**
    """
    # Empty filter criteria
    filters = {
        "risk_levels": ["LOW", "MODERATE", "HIGH", "CRITICAL"],
        "confidence_min": 0.0,
        "search": None,
        "pattern": None
    }
    
    filtered_posts = apply_all_filters(posts, filters)
    
    # Should return all posts
    assert len(filtered_posts) == len(posts), \
        "Empty filter should return all posts"
    
    for post in posts:
        assert post in filtered_posts, \
            "Empty filter should include all posts"


@given(st.lists(post_result_strategy(), min_size=0, max_size=50), filter_criteria_strategy(), filter_criteria_strategy())
@settings(max_examples=100, deadline=None)
def test_filter_idempotence(posts, filters1, filters2):
    """
    Property 8: Filter consistency - Idempotence
    
    For any set of posts, applying the same filter twice should produce
    the same result as applying it once.
    
    **Validates: Requirements 9.5**
    """
    # Apply filter once
    filtered_once = apply_all_filters(posts, filters1)
    
    # Apply filter twice
    filtered_twice = apply_all_filters(filtered_once, filters1)
    
    # Results should be identical
    assert len(filtered_once) == len(filtered_twice), \
        "Applying filter twice should produce same result as once"
    
    for post in filtered_once:
        assert post in filtered_twice, \
            "Applying filter twice should produce same result as once"


@given(st.lists(post_result_strategy(), min_size=0, max_size=50), st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False), st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=100, deadline=None)
def test_filter_monotonicity(posts, threshold1, threshold2):
    """
    Property 8: Filter consistency - Monotonicity
    
    For any set of posts, a more restrictive filter should return fewer
    or equal results than a less restrictive filter.
    
    **Validates: Requirements 9.3, 9.5**
    """
    # Ensure threshold1 <= threshold2
    if threshold1 > threshold2:
        threshold1, threshold2 = threshold2, threshold1
    
    # Apply less restrictive filter
    filtered_less = apply_confidence_filter(posts, threshold1)
    
    # Apply more restrictive filter
    filtered_more = apply_confidence_filter(posts, threshold2)
    
    # More restrictive filter should return fewer or equal results
    assert len(filtered_more) <= len(filtered_less), \
        "More restrictive filter should return fewer or equal results"
    
    # All posts in more restrictive result should be in less restrictive result
    for post in filtered_more:
        assert post in filtered_less, \
            "More restrictive filter result should be subset of less restrictive"
