"""
Execution history viewer components for the Agent Monitoring Dashboard.

This module provides components for viewing and analyzing historical execution data,
including history display, run selection, trends visualization, and comparison features.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
"""

import streamlit as st
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

from models.data_models import ExecutionResult
from core.data_manager import DataManager


def render_history_list(
    history: List[ExecutionResult],
    max_display: int = 20
) -> Optional[str]:
    """
    Render execution history list with selection capability.
    
    Displays a list of historical execution runs with timestamps,
    summary metrics, and selection functionality.
    
    Args:
        history: List of ExecutionResult objects
        max_display: Maximum number of history items to display
        
    Returns:
        Selected execution ID or None
        
    Requirements: 6.1, 6.2
    """
    st.subheader("📜 Execution History")
    
    if not history:
        st.info("No execution history available.")
        return None
    
    st.write(f"Showing {min(len(history), max_display)} of {len(history)} executions")
    
    # Prepare data for table
    history_data = []
    for result in history[:max_display]:
        # Calculate success rate
        success_rate = 0.0
        if result.total_posts > 0:
            success_rate = (result.posts_analyzed / result.total_posts) * 100
        
        history_data.append({
            "Execution ID": result.execution_id[:12] + "...",  # Truncate for display
            "Timestamp": result.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "Total Posts": result.total_posts,
            "Analyzed": result.posts_analyzed,
            "Misinfo": result.misinformation_detected,
            "High Risk": result.high_risk_posts,
            "Duration (s)": f"{result.execution_time_ms / 1000:.2f}",
            "Success Rate": f"{success_rate:.1f}%",
            "Full ID": result.execution_id  # Hidden column for selection
        })
    
    # Display as dataframe
    df = pd.DataFrame(history_data)
    
    # Display table (without Full ID column)
    display_df = df.drop(columns=["Full ID"])
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Selection dropdown
    st.write("---")
    st.write("**Select an execution to view details:**")
    
    # Create selection options
    selection_options = {}
    for i, result in enumerate(history[:max_display]):
        label = f"{result.timestamp.strftime('%Y-%m-%d %H:%M:%S')} - {result.total_posts} posts"
        selection_options[label] = result.execution_id
    
    if selection_options:
        selected_label = st.selectbox(
            "Choose execution",
            options=list(selection_options.keys()),
            key="history_selection"
        )
        
        if selected_label:
            return selection_options[selected_label]
    
    return None


def render_execution_details(result: ExecutionResult):
    """
    Render detailed view of a selected execution.
    
    Displays comprehensive information about a specific execution including
    all metrics, agent statuses, and full analysis results.
    
    Args:
        result: ExecutionResult to display
        
    Requirements: 6.3, 6.4
    """
    st.header(f"📊 Execution Details")
    st.write(f"**Execution ID:** {result.execution_id}")
    st.write(f"**Timestamp:** {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    
    st.divider()
    
    # Summary metrics
    st.subheader("Summary Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Posts", result.total_posts)
    
    with col2:
        st.metric("Posts Analyzed", result.posts_analyzed)
    
    with col3:
        st.metric("Misinformation", result.misinformation_detected)
    
    with col4:
        st.metric("High Risk", result.high_risk_posts)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Execution Time", f"{result.execution_time_ms / 1000:.2f}s")
    
    with col2:
        success_rate = (result.posts_analyzed / result.total_posts * 100) if result.total_posts > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with col3:
        misinfo_rate = (result.misinformation_detected / result.posts_analyzed * 100) if result.posts_analyzed > 0 else 0
        st.metric("Misinfo Rate", f"{misinfo_rate:.1f}%")
    
    st.divider()
    
    # Agent statuses
    st.subheader("Agent Execution Status")
    
    if result.agent_statuses:
        agent_data = []
        for agent_name, status in result.agent_statuses.items():
            agent_data.append({
                "Agent": agent_name,
                "Status": status.status,
                "Duration (ms)": f"{status.execution_time_ms:.0f}",
                "Posts Processed": status.posts_processed,
                "Error": status.error if status.error else "None"
            })
        
        df = pd.DataFrame(agent_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No agent status information available.")
    
    st.divider()
    
    # Full report preview
    with st.expander("📄 View Full Report"):
        if result.full_report:
            st.json(result.full_report)
        else:
            st.info("No full report available.")
    
    # Markdown report preview
    with st.expander("📝 View Markdown Report"):
        if result.markdown_report:
            # Add download button
            col1, col2 = st.columns([3, 1])
            with col2:
                st.download_button(
                    label="⬇️ Download",
                    data=result.markdown_report,
                    file_name=f"report_{result.execution_id[:8]}.md",
                    mime="text/markdown"
                )
            
            st.markdown(result.markdown_report)
        else:
            st.info("No markdown report available.")


def render_execution_trends(history: List[ExecutionResult]):
    """
    Render trends visualization for execution history.
    
    Displays charts showing trends in execution time, success rates,
    misinformation detection, and other metrics over time.
    
    Args:
        history: List of ExecutionResult objects
        
    Requirements: 6.5
    """
    st.header("📈 Execution Trends")
    
    if not history or len(history) < 2:
        st.info("Need at least 2 executions to show trends.")
        return
    
    # Prepare data for visualization
    trend_data = []
    for result in history:
        success_rate = (result.posts_analyzed / result.total_posts * 100) if result.total_posts > 0 else 0
        misinfo_rate = (result.misinformation_detected / result.posts_analyzed * 100) if result.posts_analyzed > 0 else 0
        
        trend_data.append({
            "Timestamp": result.timestamp,
            "Execution Time (s)": result.execution_time_ms / 1000,
            "Total Posts": result.total_posts,
            "Posts Analyzed": result.posts_analyzed,
            "Misinformation Detected": result.misinformation_detected,
            "High Risk Posts": result.high_risk_posts,
            "Success Rate (%)": success_rate,
            "Misinfo Rate (%)": misinfo_rate
        })
    
    df = pd.DataFrame(trend_data)
    df = df.sort_values("Timestamp")
    
    # Execution time trend
    st.subheader("Execution Time Trend")
    fig_time = px.line(
        df,
        x="Timestamp",
        y="Execution Time (s)",
        title="Execution Time Over Time",
        markers=True
    )
    fig_time.update_layout(
        xaxis_title="Timestamp",
        yaxis_title="Execution Time (seconds)",
        hovermode="x unified"
    )
    st.plotly_chart(fig_time, use_container_width=True)
    
    # Success rate trend
    st.subheader("Success Rate Trend")
    fig_success = px.line(
        df,
        x="Timestamp",
        y="Success Rate (%)",
        title="Success Rate Over Time",
        markers=True
    )
    fig_success.update_layout(
        xaxis_title="Timestamp",
        yaxis_title="Success Rate (%)",
        hovermode="x unified"
    )
    st.plotly_chart(fig_success, use_container_width=True)
    
    # Misinformation detection trend
    st.subheader("Misinformation Detection Trend")
    
    fig_misinfo = go.Figure()
    fig_misinfo.add_trace(go.Scatter(
        x=df["Timestamp"],
        y=df["Misinformation Detected"],
        mode="lines+markers",
        name="Misinfo Detected",
        line=dict(color="red")
    ))
    fig_misinfo.add_trace(go.Scatter(
        x=df["Timestamp"],
        y=df["High Risk Posts"],
        mode="lines+markers",
        name="High Risk",
        line=dict(color="orange")
    ))
    fig_misinfo.update_layout(
        title="Misinformation Detection Over Time",
        xaxis_title="Timestamp",
        yaxis_title="Count",
        hovermode="x unified"
    )
    st.plotly_chart(fig_misinfo, use_container_width=True)
    
    # Volume trend
    st.subheader("Processing Volume Trend")
    fig_volume = px.bar(
        df,
        x="Timestamp",
        y=["Total Posts", "Posts Analyzed"],
        title="Processing Volume Over Time",
        barmode="group"
    )
    fig_volume.update_layout(
        xaxis_title="Timestamp",
        yaxis_title="Number of Posts",
        hovermode="x unified"
    )
    st.plotly_chart(fig_volume, use_container_width=True)
    
    # Summary statistics
    st.subheader("Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_time = df["Execution Time (s)"].mean()
        st.metric("Avg Execution Time", f"{avg_time:.2f}s")
    
    with col2:
        avg_success = df["Success Rate (%)"].mean()
        st.metric("Avg Success Rate", f"{avg_success:.1f}%")
    
    with col3:
        avg_misinfo = df["Misinfo Rate (%)"].mean()
        st.metric("Avg Misinfo Rate", f"{avg_misinfo:.1f}%")
    
    with col4:
        total_analyzed = df["Posts Analyzed"].sum()
        st.metric("Total Posts Analyzed", int(total_analyzed))


def render_history_comparison(history: List[ExecutionResult]):
    """
    Render comparison view for multiple executions.
    
    Allows users to select and compare multiple execution runs side-by-side.
    
    Args:
        history: List of ExecutionResult objects
        
    Requirements: 6.5
    """
    st.header("🔄 Compare Executions")
    
    if not history or len(history) < 2:
        st.info("Need at least 2 executions to compare.")
        return
    
    # Create selection options
    selection_options = {}
    for result in history[:20]:  # Limit to 20 most recent
        label = f"{result.timestamp.strftime('%Y-%m-%d %H:%M:%S')} - {result.total_posts} posts"
        selection_options[label] = result.execution_id
    
    # Multi-select for comparison
    selected_labels = st.multiselect(
        "Select executions to compare (2-4 recommended)",
        options=list(selection_options.keys()),
        max_selections=4,
        key="comparison_selection"
    )
    
    if len(selected_labels) < 2:
        st.info("Please select at least 2 executions to compare.")
        return
    
    # Get selected results
    selected_ids = [selection_options[label] for label in selected_labels]
    selected_results = [r for r in history if r.execution_id in selected_ids]
    
    # Comparison table
    st.subheader("Comparison Table")
    
    comparison_data = []
    for result in selected_results:
        success_rate = (result.posts_analyzed / result.total_posts * 100) if result.total_posts > 0 else 0
        misinfo_rate = (result.misinformation_detected / result.posts_analyzed * 100) if result.posts_analyzed > 0 else 0
        
        comparison_data.append({
            "Timestamp": result.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "Total Posts": result.total_posts,
            "Analyzed": result.posts_analyzed,
            "Misinfo": result.misinformation_detected,
            "High Risk": result.high_risk_posts,
            "Duration (s)": f"{result.execution_time_ms / 1000:.2f}",
            "Success Rate": f"{success_rate:.1f}%",
            "Misinfo Rate": f"{misinfo_rate:.1f}%"
        })
    
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Comparison charts
    st.subheader("Visual Comparison")
    
    # Metrics comparison
    col1, col2 = st.columns(2)
    
    with col1:
        fig_posts = px.bar(
            df,
            x="Timestamp",
            y=["Total Posts", "Analyzed", "Misinfo"],
            title="Posts Comparison",
            barmode="group"
        )
        st.plotly_chart(fig_posts, use_container_width=True)
    
    with col2:
        # Extract numeric values for rates
        success_rates = [float(x.rstrip('%')) for x in df["Success Rate"]]
        misinfo_rates = [float(x.rstrip('%')) for x in df["Misinfo Rate"]]
        
        fig_rates = go.Figure()
        fig_rates.add_trace(go.Bar(
            x=df["Timestamp"],
            y=success_rates,
            name="Success Rate",
            marker_color="green"
        ))
        fig_rates.add_trace(go.Bar(
            x=df["Timestamp"],
            y=misinfo_rates,
            name="Misinfo Rate",
            marker_color="red"
        ))
        fig_rates.update_layout(
            title="Rates Comparison",
            yaxis_title="Percentage (%)",
            barmode="group"
        )
        st.plotly_chart(fig_rates, use_container_width=True)


def render_history_sidebar(data_manager: DataManager):
    """
    Render history viewer in sidebar.
    
    Provides a compact history view in the sidebar with quick access
    to recent executions and history management.
    
    Args:
        data_manager: DataManager instance for loading history
        
    Requirements: 6.1, 6.2
    """
    st.sidebar.header("📜 History")
    
    # Load history
    history = data_manager.load_all_execution_results()
    
    if not history:
        st.sidebar.info("No execution history")
        return None
    
    # Show count
    st.sidebar.write(f"**Total Executions:** {len(history)}")
    
    # Show most recent
    if history:
        latest = history[0]
        st.sidebar.write(f"**Latest:** {latest.timestamp.strftime('%Y-%m-%d %H:%M')}")
        st.sidebar.write(f"Posts: {latest.total_posts}, Misinfo: {latest.misinformation_detected}")
    
    # Quick selection
    st.sidebar.write("---")
    st.sidebar.write("**Quick Access:**")
    
    # Show last 5 executions
    for i, result in enumerate(history[:5], 1):
        label = f"{i}. {result.timestamp.strftime('%m/%d %H:%M')} ({result.total_posts} posts)"
        if st.sidebar.button(label, key=f"sidebar_history_{i}"):
            return result.execution_id
    
    # History management
    st.sidebar.write("---")
    if st.sidebar.button("🗑️ Clear History"):
        if st.sidebar.checkbox("Confirm clear all history"):
            data_manager.clear_all_history()
            st.sidebar.success("History cleared!")
            st.rerun()
    
    return None


def render_complete_history_viewer(data_manager: DataManager):
    """
    Render complete history viewer with all features.
    
    Displays comprehensive history view including list, details,
    trends, and comparison features.
    
    Args:
        data_manager: DataManager instance for loading history
        
    Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
    """
    st.title("📜 Execution History Viewer")
    
    # Load history
    history = data_manager.load_all_execution_results()
    
    if not history:
        st.info("No execution history available. Run an analysis to create history.")
        return
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 History List",
        "📊 Execution Details",
        "📈 Trends",
        "🔄 Compare"
    ])
    
    with tab1:
        # History list with selection
        selected_id = render_history_list(history)
        
        if selected_id:
            st.session_state["selected_execution_id"] = selected_id
            st.success(f"Selected execution: {selected_id[:12]}...")
    
    with tab2:
        # Execution details
        if "selected_execution_id" in st.session_state:
            selected_id = st.session_state["selected_execution_id"]
            
            with st.spinner("Loading execution details..."):
                result = data_manager.load_execution_result(selected_id, try_s3=True)
            
            if result:
                render_execution_details(result)
            else:
                st.error(f"Failed to load execution details for ID: {selected_id[:12]}...")
                st.info("This execution may have been deleted or is not accessible.")
        else:
            st.info("Select an execution from the History List tab to view details.")
    
    with tab3:
        # Trends visualization
        render_execution_trends(history)
    
    with tab4:
        # Comparison view
        render_history_comparison(history)
