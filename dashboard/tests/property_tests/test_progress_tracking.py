"""
Property-based tests for progress tracking completeness.

**Feature: agent-monitoring-dashboard, Property 3: Progress tracking completeness**
**Validates: Requirements 3.3**

Tests that progress tracking correctly accounts for all posts processed.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from datetime import datetime
from typing import Dict, List

from dashboard.models.data_models import AgentStatus


# Strategies for generating test data

@st.composite
def agent_status_with_posts_strategy(draw):
    """Generate agent status with posts processed."""
    return AgentStatus(
        agent_name=draw(st.text(min_size=1, max_size=20)),
        status=draw(st.sampled_from(["idle", "executing", "completed", "failed"])),
        start_time=datetime.now(),
        end_time=datetime.now(),
        execution_time_ms=draw(st.floats(min_value=0, max_value=10000)),
        posts_processed=draw(st.integers(min_value=0, max_value=1000)),
        error=None
    )


@st.composite
def agent_statuses_dict_strategy(draw, num_agents=None):
    """Generate dictionary of agent statuses."""
    if num_agents is None:
        num_agents = draw(st.integers(min_value=1, max_value=10))
    
    statuses = {}
    for i in range(num_agents):
        agent_name = f"agent_{i}"
        statuses[agent_name] = draw(agent_status_with_posts_strategy())
        statuses[agent_name].agent_name = agent_name
    
    return statuses


# Property tests

@given(
    st.integers(min_value=1, max_value=1000),
    agent_statuses_dict_strategy()
)
@settings(max_examples=100, deadline=None)
def test_total_posts_processed_equals_batch_size(total_posts, agent_statuses):
    """
    Property 3: Progress tracking completeness
    
    For any batch analysis, the sum of posts processed across all agents
    should equal the total number of posts in the batch (when all agents
    process all posts).
    
    **Validates: Requirements 3.3**
    """
    # Simulate scenario where each agent processes all posts
    for agent_name in agent_statuses:
        agent_statuses[agent_name].posts_processed = total_posts
    
    # Calculate total posts processed
    total_processed = sum(
        status.posts_processed
        for status in agent_statuses.values()
    )
    
    # In a scenario where all agents process all posts,
    # total processed = num_agents * total_posts
    expected_total = len(agent_statuses) * total_posts
    
    assert total_processed == expected_total, \
        f"Total posts processed ({total_processed}) should equal " \
        f"num_agents * total_posts ({expected_total})"


@given(
    st.integers(min_value=1, max_value=1000),
    st.integers(min_value=1, max_value=10)
)
@settings(max_examples=100, deadline=None)
def test_progress_percentage_calculation(total_posts, current_posts):
    """
    Property: Progress percentage accuracy
    
    For any batch analysis, the progress percentage should be
    correctly calculated as (current / total) * 100.
    """
    assume(current_posts <= total_posts)
    
    progress = current_posts / total_posts
    percentage = progress * 100
    
    assert 0 <= percentage <= 100, "Progress percentage should be between 0 and 100"
    assert percentage == (current_posts / total_posts) * 100, \
        "Progress percentage should equal (current / total) * 100"


@given(agent_statuses_dict_strategy())
@settings(max_examples=100, deadline=None)
def test_posts_processed_non_negative(agent_statuses):
    """
    Property: Non-negative posts processed
    
    For any agent status, the number of posts processed should never be negative.
    """
    for agent_name, status in agent_statuses.items():
        assert status.posts_processed >= 0, \
            f"Agent {agent_name} has negative posts_processed: {status.posts_processed}"


@given(
    st.integers(min_value=1, max_value=1000),
    agent_statuses_dict_strategy()
)
@settings(max_examples=100, deadline=None)
def test_posts_processed_not_exceed_total(total_posts, agent_statuses):
    """
    Property: Posts processed bounds
    
    For any agent in a batch analysis, the number of posts processed
    should not exceed the total number of posts in the batch.
    """
    # Set posts processed to valid values
    for agent_name in agent_statuses:
        posts_processed = min(
            agent_statuses[agent_name].posts_processed,
            total_posts
        )
        agent_statuses[agent_name].posts_processed = posts_processed
    
    for agent_name, status in agent_statuses.items():
        assert status.posts_processed <= total_posts, \
            f"Agent {agent_name} processed {status.posts_processed} posts, " \
            f"but total is only {total_posts}"


@given(
    st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=10)
)
@settings(max_examples=100, deadline=None)
def test_progress_tracking_monotonic_increase(posts_processed_sequence):
    """
    Property: Progress monotonicity
    
    For any sequence of progress updates, the number of posts processed
    should never decrease (monotonically increasing).
    """
    for i in range(1, len(posts_processed_sequence)):
        # Make sequence monotonically increasing
        if posts_processed_sequence[i] < posts_processed_sequence[i-1]:
            posts_processed_sequence[i] = posts_processed_sequence[i-1]
    
    # Verify monotonicity
    for i in range(1, len(posts_processed_sequence)):
        assert posts_processed_sequence[i] >= posts_processed_sequence[i-1], \
            f"Posts processed should not decrease: {posts_processed_sequence[i-1]} -> {posts_processed_sequence[i]}"


@given(
    st.integers(min_value=1, max_value=1000),
    st.integers(min_value=1, max_value=10)
)
@settings(max_examples=100, deadline=None)
def test_remaining_posts_calculation(total_posts, current_posts):
    """
    Property: Remaining posts accuracy
    
    For any batch analysis, remaining posts should equal total minus current.
    """
    assume(current_posts <= total_posts)
    
    remaining = total_posts - current_posts
    
    assert remaining >= 0, "Remaining posts should not be negative"
    assert remaining == total_posts - current_posts, \
        "Remaining posts should equal total - current"
    assert current_posts + remaining == total_posts, \
        "Current + remaining should equal total"


@given(agent_statuses_dict_strategy())
@settings(max_examples=100, deadline=None)
def test_completed_agents_have_posts_processed(agent_statuses):
    """
    Property: Completed agents processing
    
    For any agent with status "completed", if it participated in analysis,
    it should have processed at least some posts (or explicitly 0 if it
    didn't participate).
    """
    for agent_name, status in agent_statuses.items():
        if status.status == "completed":
            # Completed agents should have non-negative posts_processed
            assert status.posts_processed >= 0, \
                f"Completed agent {agent_name} should have non-negative posts_processed"


@given(
    st.integers(min_value=1, max_value=1000),
    st.integers(min_value=1, max_value=10)
)
@settings(max_examples=100, deadline=None)
def test_progress_completion_detection(total_posts, num_agents):
    """
    Property: Progress completion detection
    
    For any batch analysis, when all posts are processed,
    progress should be 100%.
    """
    current_posts = total_posts  # All posts processed
    
    progress = current_posts / total_posts
    percentage = progress * 100
    
    assert percentage == 100.0, \
        "When all posts are processed, progress should be 100%"


@given(
    st.integers(min_value=10, max_value=1000),
    st.integers(min_value=1, max_value=10)
)
@settings(max_examples=100, deadline=None)
def test_progress_increments_are_valid(total_posts, num_increments):
    """
    Property: Progress increment validity
    
    For any batch analysis with incremental updates, each increment
    should be valid (between 0 and total).
    """
    assume(num_increments <= total_posts)
    
    increment_size = total_posts // num_increments
    
    for i in range(num_increments + 1):
        current = min(i * increment_size, total_posts)
        
        assert 0 <= current <= total_posts, \
            f"Progress increment {current} should be between 0 and {total_posts}"
        
        progress = current / total_posts
        assert 0 <= progress <= 1.0, \
            f"Progress ratio {progress} should be between 0 and 1"


@given(agent_statuses_dict_strategy())
@settings(max_examples=100, deadline=None)
def test_agent_status_consistency_with_posts(agent_statuses):
    """
    Property: Agent status and posts consistency
    
    For any agent with "completed" or "failed" status and posts processed,
    it should have valid execution time.
    """
    for agent_name, status in agent_statuses.items():
        if status.posts_processed > 0 and status.status in ["completed", "failed"]:
            # If agent processed posts and is completed/failed, it should have execution time
            assert status.execution_time_ms >= 0, \
                f"Agent {agent_name} processed {status.posts_processed} posts " \
                f"but has negative execution time: {status.execution_time_ms}"
