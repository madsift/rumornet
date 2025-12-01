"""
Reusable UI components for the Agent Monitoring Dashboard.

This module provides Streamlit-based UI components for displaying
agent status, metrics, results, and other dashboard elements.
"""

import streamlit as st
from typing import Dict, List, Any, Optional
from datetime import datetime

from models.data_models import AgentStatus, ExecutionMetrics


def render_agent_status_card(agent_status: AgentStatus):
    """
    Render a status card for a single agent.
    
    Displays the agent's current status, execution time, posts processed,
    and any error information in a visually appealing card format.
    
    Args:
        agent_status: AgentStatus object containing agent information
        
    Requirements: 1.1, 1.5
    """
    # Determine status color and icon
    status_config = {
        "idle": {"color": "🔵", "bg_color": "#E8F4F8"},
        "executing": {"color": "🟡", "bg_color": "#FFF9E6"},
        "completed": {"color": "🟢", "bg_color": "#E8F8E8"},
        "failed": {"color": "🔴", "bg_color": "#FFE8E8"}
    }
    
    config = status_config.get(agent_status.status, status_config["idle"])
    
    # Create card container
    with st.container():
        # Status header
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### {config['color']} {agent_status.agent_name}")
        with col2:
            st.markdown(f"**{agent_status.status.upper()}**")
        
        # Execution details
        if agent_status.status in ["executing", "completed", "failed"]:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if agent_status.start_time:
                    st.metric(
                        "Started",
                        agent_status.start_time.strftime("%H:%M:%S")
                    )
            
            with col2:
                if agent_status.execution_time_ms > 0:
                    st.metric(
                        "Duration",
                        f"{agent_status.execution_time_ms:.0f} ms"
                    )
            
            with col3:
                if agent_status.posts_processed > 0:
                    st.metric(
                        "Posts",
                        agent_status.posts_processed
                    )
        
        # Error display
        if agent_status.error:
            st.error(f"❌ Error: {agent_status.error}")
        
        st.divider()


def render_metrics_dashboard(metrics: ExecutionMetrics):
    """
    Render performance metrics dashboard.
    
    Displays aggregated metrics including execution counts, success rates,
    execution times, and throughput in a comprehensive dashboard layout.
    
    Args:
        metrics: ExecutionMetrics object containing performance data
        
    Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
    """
    st.subheader("📊 Performance Metrics")
    
    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Executions",
            metrics.total_executions,
            help="Total number of analysis runs"
        )
    
    with col2:
        success_rate = (
            (metrics.successful_executions / metrics.total_executions * 100)
            if metrics.total_executions > 0
            else 0
        )
        st.metric(
            "Success Rate",
            f"{success_rate:.1f}%",
            help="Percentage of successful executions"
        )
    
    with col3:
        st.metric(
            "Avg Execution Time",
            f"{metrics.average_execution_time_ms:.0f} ms",
            help="Average time per execution"
        )
    
    with col4:
        st.metric(
            "Throughput",
            f"{metrics.posts_per_second:.2f} posts/s",
            help="Posts processed per second"
        )
    
    # Execution breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "✅ Successful",
            metrics.successful_executions
        )
    
    with col2:
        st.metric(
            "❌ Failed",
            metrics.failed_executions
        )
    
    st.divider()
    
    # Per-agent metrics
    if metrics.agent_metrics:
        st.subheader("Per-Agent Metrics")
        
        # Create table data
        agent_data = []
        for agent_name, agent_metrics in metrics.agent_metrics.items():
            agent_data.append({
                "Agent": agent_name,
                "Status": agent_metrics.get("current_status", "N/A"),
                "Posts Processed": agent_metrics.get("posts_processed", 0),
                "Avg Time (ms)": f"{agent_metrics.get('avg_execution_time_ms', 0):.0f}",
                "Last Error": agent_metrics.get("last_error", "None") or "None"
            })
        
        # Display as dataframe
        if agent_data:
            st.dataframe(
                agent_data,
                use_container_width=True,
                hide_index=True
            )


def render_results_table(
    results: List[Dict[str, Any]],
    title: str = "Analysis Results",
    max_rows: int = 10
):
    """
    Render analysis results in a table format.
    
    Displays post analysis results with key information including
    post ID, user, verdict, confidence, and risk level.
    
    Args:
        results: List of analysis result dictionaries
        title: Title for the results table
        max_rows: Maximum number of rows to display
        
    Requirements: 1.1, 1.5
    """
    st.subheader(title)
    
    if not results:
        st.info("No results to display.")
        return
    
    # Prepare table data
    table_data = []
    for result in results[:max_rows]:
        # Extract data safely
        metadata = result.get("metadata", {})
        analysis = result.get("analysis", {})
        
        verdict = analysis.get("verdict")
        if verdict is False:
            verdict_str = "❌ MISINFO"
        elif verdict is True:
            verdict_str = "✅ TRUE"
        else:
            verdict_str = "❓ UNCERTAIN"
        
        table_data.append({
            "Post ID": metadata.get("post_id", "N/A"),
            "User": metadata.get("username", "N/A"),
            "Verdict": verdict_str,
            "Confidence": f"{analysis.get('confidence', 0):.2f}",
            "Risk Level": analysis.get("risk_level", "N/A"),
            "Patterns": len(analysis.get("patterns_detected", []))
        })
    
    # Display table
    st.dataframe(
        table_data,
        use_container_width=True,
        hide_index=True
    )
    
    # Show count
    if len(results) > max_rows:
        st.caption(f"Showing {max_rows} of {len(results)} results")


def render_execution_timeline(statuses: Dict[str, AgentStatus]):
    """
    Render execution timeline visualization.
    
    Displays a timeline showing the execution order and status of all agents,
    with visual indicators for current execution state.
    
    Args:
        statuses: Dictionary mapping agent names to AgentStatus objects
        
    Requirements: 1.1, 1.5
    """
    st.subheader("🕐 Execution Timeline")
    
    if not statuses:
        st.info("No execution data available.")
        return
    
    # Sort agents by start time (if available)
    sorted_agents = sorted(
        statuses.items(),
        key=lambda x: x[1].start_time if x[1].start_time else datetime.max
    )
    
    # Create timeline visualization
    for agent_name, status in sorted_agents:
        # Determine status indicator
        if status.status == "executing":
            indicator = "🟡 ▶️"
            status_text = "EXECUTING"
        elif status.status == "completed":
            indicator = "🟢 ✓"
            status_text = "COMPLETED"
        elif status.status == "failed":
            indicator = "🔴 ✗"
            status_text = "FAILED"
        else:
            indicator = "⚪"
            status_text = "IDLE"
        
        # Create timeline entry
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        
        with col1:
            st.markdown(f"{indicator} **{agent_name}**")
        
        with col2:
            if status.start_time:
                st.text(f"Start: {status.start_time.strftime('%H:%M:%S')}")
            else:
                st.text("Not started")
        
        with col3:
            if status.execution_time_ms > 0:
                st.text(f"Duration: {status.execution_time_ms:.0f}ms")
            else:
                st.text("-")
        
        with col4:
            st.text(status_text)
    
    st.divider()


def render_progress_bar(
    current: int,
    total: int,
    label: str = "Progress"
):
    """
    Render progress bar for batch processing.
    
    Displays a progress bar with current/total counts and percentage,
    useful for tracking batch analysis progress.
    
    Args:
        current: Current number of items processed
        total: Total number of items to process
        label: Label for the progress bar
        
    Requirements: 1.5
    """
    if total <= 0:
        st.warning("Invalid progress values")
        return
    
    # Calculate percentage
    percentage = min(current / total, 1.0)
    
    # Display progress bar
    st.progress(percentage, text=f"{label}: {current}/{total} ({percentage*100:.1f}%)")
    
    # Additional metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Processed", current)
    
    with col2:
        st.metric("Remaining", total - current)
    
    with col3:
        st.metric("Total", total)


def render_error_panel(
    errors: List[Dict[str, Any]],
    title: str = "Errors & Warnings"
):
    """
    Render error panel for displaying errors and debugging information.
    
    Displays errors with timestamps, component names, error messages,
    and optional stack traces for debugging.
    
    Args:
        errors: List of error dictionaries with timestamp, component, message, and stack_trace
        title: Title for the error panel
        
    Requirements: 10.1, 10.2, 10.3
    """
    st.subheader(f"⚠️ {title}")
    
    if not errors:
        st.success("✅ No errors detected")
        return
    
    # Display error count
    st.error(f"Found {len(errors)} error(s)")
    
    # Display each error
    for i, error in enumerate(errors, 1):
        with st.expander(
            f"Error {i}: {error.get('component', 'Unknown')} - {error.get('timestamp', 'N/A')}",
            expanded=(i == 1)  # Expand first error by default
        ):
            # Error details
            st.markdown(f"**Component:** {error.get('component', 'Unknown')}")
            st.markdown(f"**Timestamp:** {error.get('timestamp', 'N/A')}")
            st.markdown(f"**Level:** {error.get('level', 'ERROR')}")
            
            # Error message
            st.markdown("**Message:**")
            st.code(error.get('message', 'No message available'), language=None)
            
            # Stack trace (if available)
            if error.get('stack_trace'):
                st.markdown("**Stack Trace:**")
                st.code(error.get('stack_trace'), language="python")
            
            # Additional context (if available)
            if error.get('context'):
                st.markdown("**Context:**")
                st.json(error.get('context'))


def render_summary_cards(summary: Dict[str, Any]):
    """
    Render summary cards for executive summary display.
    
    Displays key metrics in a card layout for quick overview of analysis results.
    
    Args:
        summary: Dictionary containing summary metrics
        
    Requirements: 1.1, 1.5
    """
    st.subheader("📋 Executive Summary")
    
    # First row - main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Posts",
            summary.get("total_posts_analyzed", 0),
            help="Total number of posts analyzed"
        )
    
    with col2:
        misinfo_count = summary.get("misinformation_detected", 0)
        total = summary.get("total_posts_analyzed", 1)
        misinfo_rate = (misinfo_count / total * 100) if total > 0 else 0
        st.metric(
            "Misinformation",
            misinfo_count,
            delta=f"{misinfo_rate:.1f}%",
            delta_color="inverse",
            help="Number of posts flagged as misinformation"
        )
    
    with col3:
        st.metric(
            "High Risk",
            summary.get("high_risk_posts", 0),
            help="Posts requiring immediate attention"
        )
    
    with col4:
        st.metric(
            "Critical",
            summary.get("critical_posts", 0),
            help="Critical posts requiring urgent action"
        )
    
    # Second row - user and pattern metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Unique Users",
            summary.get("unique_users", 0)
        )
    
    with col2:
        st.metric(
            "Users w/ Misinfo",
            summary.get("users_posting_misinfo", 0)
        )
    
    with col3:
        st.metric(
            "Patterns",
            summary.get("patterns_detected", 0)
        )
    
    with col4:
        st.metric(
            "Topics",
            summary.get("topics_identified", 0)
        )
    
    st.divider()


def render_filter_panel(
    on_filter_change: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Render filter panel for results filtering.
    
    Provides UI controls for filtering results by various criteria
    including risk level, confidence threshold, and search terms.
    
    Args:
        on_filter_change: Optional callback function when filters change
        
    Returns:
        Dictionary containing current filter values
        
    Requirements: 1.5
    """
    st.subheader("🔍 Filters")
    
    filters = {}
    
    # Search box
    filters["search"] = st.text_input(
        "Search (Post ID or User ID)",
        placeholder="Enter post ID or user ID...",
        help="Search for specific posts or users"
    )
    
    # Risk level filter
    filters["risk_levels"] = st.multiselect(
        "Risk Level",
        options=["LOW", "MODERATE", "HIGH", "CRITICAL"],
        default=["HIGH", "CRITICAL"],
        help="Filter by risk level"
    )
    
    # Confidence threshold
    filters["confidence_min"] = st.slider(
        "Minimum Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence threshold"
    )
    
    # Pattern filter
    filters["patterns"] = st.text_input(
        "Pattern Filter",
        placeholder="Enter pattern name...",
        help="Filter by specific pattern"
    )
    
    # Apply button
    if st.button("Apply Filters", type="primary"):
        if on_filter_change:
            on_filter_change(filters)
    
    # Clear button
    if st.button("Clear Filters"):
        st.rerun()
    
    return filters


def render_agent_grid(statuses: Dict[str, AgentStatus], columns: int = 3):
    """
    Render agents in a grid layout.
    
    Displays multiple agent status cards in a responsive grid layout.
    
    Args:
        statuses: Dictionary mapping agent names to AgentStatus objects
        columns: Number of columns in the grid
        
    Requirements: 1.1, 1.5
    """
    if not statuses:
        st.info("No agents to display.")
        return
    
    # Convert to list for grid layout
    agent_list = list(statuses.items())
    
    # Create grid
    for i in range(0, len(agent_list), columns):
        cols = st.columns(columns)
        
        for j, col in enumerate(cols):
            if i + j < len(agent_list):
                agent_name, status = agent_list[i + j]
                
                with col:
                    # Simplified card for grid view
                    status_emoji = {
                        "idle": "⚪",
                        "executing": "🟡",
                        "completed": "🟢",
                        "failed": "🔴"
                    }.get(status.status, "⚪")
                    
                    st.markdown(f"**{status_emoji} {agent_name}**")
                    st.caption(status.status.upper())
                    
                    if status.execution_time_ms > 0:
                        st.caption(f"{status.execution_time_ms:.0f}ms")
