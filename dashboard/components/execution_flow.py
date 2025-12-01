"""
Execution flow visualization components for the Agent Monitoring Dashboard.

This module provides visualization components for displaying agent execution flow,
pipeline order, data flow between agents, and parallel vs sequential execution patterns.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

import streamlit as st
from typing import Dict, List, Any, Optional
from datetime import datetime

from models.data_models import AgentStatus


# Agent pipeline configuration - defines execution order and dependencies
AGENT_PIPELINE = [
    {
        "name": "reasoning",
        "display_name": "Reasoning Agent",
        "description": "Analyzes post content for misinformation",
        "dependencies": [],
        "parallel_group": 1
    },
    {
        "name": "pattern",
        "display_name": "Pattern Detection Agent",
        "description": "Detects misinformation patterns",
        "dependencies": [],
        "parallel_group": 1
    },
    {
        "name": "evidence",
        "display_name": "Evidence Gathering Agent",
        "description": "Gathers supporting evidence",
        "dependencies": ["reasoning"],
        "parallel_group": 2
    },
    {
        "name": "social_behavior",
        "display_name": "Social Behavior Agent",
        "description": "Analyzes social behavior patterns",
        "dependencies": ["reasoning", "pattern"],
        "parallel_group": 2
    },
    {
        "name": "topic_modeling",
        "display_name": "Topic Modeling Agent",
        "description": "Identifies topics and themes",
        "dependencies": [],
        "parallel_group": 3
    },
    {
        "name": "topic_evidence",
        "display_name": "Topic Evidence Agent",
        "description": "Gathers topic-specific evidence",
        "dependencies": ["topic_modeling"],
        "parallel_group": 4
    },
    {
        "name": "coordination_detector",
        "display_name": "Coordination Detector",
        "description": "Detects coordinated behavior",
        "dependencies": ["social_behavior"],
        "parallel_group": 5
    },
    {
        "name": "echo_chamber_detector",
        "display_name": "Echo Chamber Detector",
        "description": "Detects echo chamber patterns",
        "dependencies": ["social_behavior"],
        "parallel_group": 5
    }
]


def render_agent_pipeline_display(
    statuses: Dict[str, AgentStatus],
    show_dependencies: bool = True
):
    """
    Render agent pipeline display showing execution order.
    
    Displays agents in their execution order with visual indicators
    for current status and dependencies between agents.
    
    Args:
        statuses: Dictionary mapping agent names to AgentStatus objects
        show_dependencies: Whether to show dependency arrows
        
    Requirements: 8.1, 8.2
    """
    st.subheader("🔄 Agent Execution Pipeline")
    
    if not statuses:
        st.info("No agent execution data available.")
        return
    
    # Group agents by parallel execution group
    parallel_groups = {}
    for agent_config in AGENT_PIPELINE:
        group = agent_config["parallel_group"]
        if group not in parallel_groups:
            parallel_groups[group] = []
        parallel_groups[group].append(agent_config)
    
    # Render each parallel group
    for group_num in sorted(parallel_groups.keys()):
        agents_in_group = parallel_groups[group_num]
        
        # Show group header for parallel execution
        if len(agents_in_group) > 1:
            st.markdown(f"**Parallel Group {group_num}** (executes simultaneously)")
        else:
            st.markdown(f"**Stage {group_num}**")
        
        # Create columns for agents in this group
        cols = st.columns(len(agents_in_group))
        
        for i, agent_config in enumerate(agents_in_group):
            agent_name = agent_config["name"]
            agent_status = statuses.get(agent_name)
            
            with cols[i]:
                _render_pipeline_agent_card(
                    agent_config,
                    agent_status,
                    show_dependencies
                )
        
        # Add visual separator between groups
        if group_num < max(parallel_groups.keys()):
            st.markdown("⬇️" * len(agents_in_group), unsafe_allow_html=True)
            st.markdown("---")


def _render_pipeline_agent_card(
    agent_config: Dict[str, Any],
    agent_status: Optional[AgentStatus],
    show_dependencies: bool
):
    """
    Render a single agent card in the pipeline view.
    
    Args:
        agent_config: Agent configuration dictionary
        agent_status: Current status of the agent
        show_dependencies: Whether to show dependencies
    """
    agent_name = agent_config["name"]
    display_name = agent_config["display_name"]
    description = agent_config["description"]
    dependencies = agent_config["dependencies"]
    
    # Determine status and styling
    if agent_status:
        status = agent_status.status
        
        if status == "executing":
            status_emoji = "🟡"
            status_text = "EXECUTING"
            border_color = "#FFC107"
            bg_color = "#FFF9E6"
        elif status == "completed":
            status_emoji = "🟢"
            status_text = "COMPLETED"
            border_color = "#4CAF50"
            bg_color = "#E8F8E8"
        elif status == "failed":
            status_emoji = "🔴"
            status_text = "FAILED"
            border_color = "#F44336"
            bg_color = "#FFE8E8"
        else:  # idle
            status_emoji = "⚪"
            status_text = "IDLE"
            border_color = "#9E9E9E"
            bg_color = "#F5F5F5"
    else:
        status_emoji = "⚪"
        status_text = "NOT TRACKED"
        border_color = "#9E9E9E"
        bg_color = "#F5F5F5"
    
    # Create card with custom styling
    card_html = f"""
    <div style="
        border: 3px solid {border_color};
        border-radius: 10px;
        padding: 15px;
        background-color: {bg_color};
        margin: 10px 0;
        min-height: 150px;
    ">
        <div style="font-size: 24px; text-align: center; margin-bottom: 10px;">
            {status_emoji}
        </div>
        <div style="font-weight: bold; text-align: center; margin-bottom: 5px;">
            {display_name}
        </div>
        <div style="text-align: center; font-size: 12px; color: #666; margin-bottom: 10px;">
            {description}
        </div>
        <div style="text-align: center; font-weight: bold; color: {border_color};">
            {status_text}
        </div>
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)
    
    # Show execution details if available
    if agent_status and agent_status.status != "idle":
        with st.expander("Details", expanded=False):
            if agent_status.execution_time_ms > 0:
                st.metric("Execution Time", f"{agent_status.execution_time_ms:.0f} ms")
            if agent_status.posts_processed > 0:
                st.metric("Posts Processed", agent_status.posts_processed)
            if agent_status.error:
                st.error(f"Error: {agent_status.error}")
    
    # Show dependencies if requested
    if show_dependencies and dependencies:
        st.caption(f"⬆️ Depends on: {', '.join(dependencies)}")


def render_data_flow_visualization(statuses: Dict[str, AgentStatus]):
    """
    Render data flow visualization between agents.
    
    Shows how data flows through the agent pipeline with
    visual indicators for data transfer and processing.
    
    Args:
        statuses: Dictionary mapping agent names to AgentStatus objects
        
    Requirements: 8.3
    """
    st.subheader("📊 Data Flow Visualization")
    
    if not statuses:
        st.info("No data flow information available.")
        return
    
    # Create a flow diagram using text-based visualization
    st.markdown("### Data Flow Through Pipeline")
    
    # Group agents by parallel execution group
    parallel_groups = {}
    for agent_config in AGENT_PIPELINE:
        group = agent_config["parallel_group"]
        if group not in parallel_groups:
            parallel_groups[group] = []
        parallel_groups[group].append(agent_config)
    
    # Build flow visualization
    flow_lines = []
    flow_lines.append("```")
    flow_lines.append("INPUT POSTS")
    flow_lines.append("     |")
    flow_lines.append("     v")
    
    for group_num in sorted(parallel_groups.keys()):
        agents_in_group = parallel_groups[group_num]
        
        if len(agents_in_group) == 1:
            # Single agent in group
            agent = agents_in_group[0]
            agent_status = statuses.get(agent["name"])
            status_indicator = _get_flow_status_indicator(agent_status)
            
            flow_lines.append(f"[{agent['display_name']}] {status_indicator}")
            flow_lines.append("     |")
            flow_lines.append("     v")
        else:
            # Multiple agents in parallel
            flow_lines.append("     |")
            flow_lines.append("  ┌──┴──┐")
            
            # Show parallel branches
            for i, agent in enumerate(agents_in_group):
                agent_status = statuses.get(agent["name"])
                status_indicator = _get_flow_status_indicator(agent_status)
                
                if i == 0:
                    flow_lines.append(f"  v     v")
                
                flow_lines.append(f"[{agent['display_name']}] {status_indicator}")
            
            flow_lines.append("  └──┬──┘")
            flow_lines.append("     |")
            flow_lines.append("     v")
    
    flow_lines.append("FINAL REPORT")
    flow_lines.append("```")
    
    st.markdown("\n".join(flow_lines))
    
    # Show data transfer statistics
    st.markdown("### Data Transfer Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_posts = sum(
            status.posts_processed
            for status in statuses.values()
            if status.posts_processed > 0
        )
        st.metric("Total Posts Processed", total_posts)
    
    with col2:
        active_agents = sum(
            1 for status in statuses.values()
            if status.status in ["executing", "completed"]
        )
        st.metric("Active Agents", active_agents)
    
    with col3:
        completed_agents = sum(
            1 for status in statuses.values()
            if status.status == "completed"
        )
        st.metric("Completed Agents", completed_agents)


def _get_flow_status_indicator(agent_status: Optional[AgentStatus]) -> str:
    """Get status indicator for flow visualization."""
    if not agent_status:
        return "⚪"
    
    if agent_status.status == "executing":
        return "🟡 ▶️"
    elif agent_status.status == "completed":
        return "🟢 ✓"
    elif agent_status.status == "failed":
        return "🔴 ✗"
    else:
        return "⚪"


def render_complete_execution_timeline(statuses: Dict[str, AgentStatus]):
    """
    Render complete execution timeline display.
    
    Shows a detailed timeline of all agent executions with
    start times, end times, and duration bars.
    
    Args:
        statuses: Dictionary mapping agent names to AgentStatus objects
        
    Requirements: 8.4
    """
    st.subheader("⏱️ Complete Execution Timeline")
    
    if not statuses:
        st.info("No execution timeline data available.")
        return
    
    # Filter agents that have execution data
    executed_agents = [
        (name, status) for name, status in statuses.items()
        if status.start_time is not None
    ]
    
    if not executed_agents:
        st.info("No agents have started execution yet.")
        return
    
    # Sort by start time
    executed_agents.sort(key=lambda x: x[1].start_time)
    
    # Find earliest and latest times for timeline scale
    earliest_time = min(status.start_time for _, status in executed_agents)
    latest_time = max(
        status.end_time if status.end_time else status.start_time
        for _, status in executed_agents
    )
    
    total_duration = (latest_time - earliest_time).total_seconds()
    
    # Render timeline
    st.markdown("### Timeline View")
    
    for agent_name, status in executed_agents:
        # Find agent config for display name
        agent_config = next(
            (a for a in AGENT_PIPELINE if a["name"] == agent_name),
            {"display_name": agent_name}
        )
        display_name = agent_config.get("display_name", agent_name)
        
        # Calculate timeline position and width
        start_offset = (status.start_time - earliest_time).total_seconds()
        
        if status.end_time:
            duration = (status.end_time - status.start_time).total_seconds()
        else:
            duration = 0.1  # Small duration for executing agents
        
        # Calculate percentages for visualization
        if total_duration > 0:
            start_percent = (start_offset / total_duration) * 100
            width_percent = (duration / total_duration) * 100
        else:
            start_percent = 0
            width_percent = 100
        
        # Determine color based on status
        if status.status == "completed":
            bar_color = "#4CAF50"
        elif status.status == "failed":
            bar_color = "#F44336"
        elif status.status == "executing":
            bar_color = "#FFC107"
        else:
            bar_color = "#9E9E9E"
        
        # Create timeline bar
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown(f"**{display_name}**")
            st.caption(f"{status.start_time.strftime('%H:%M:%S.%f')[:-3]}")
        
        with col2:
            # Create visual timeline bar
            timeline_html = f"""
            <div style="
                position: relative;
                height: 40px;
                background-color: #f0f0f0;
                border-radius: 5px;
                margin: 5px 0;
            ">
                <div style="
                    position: absolute;
                    left: {start_percent}%;
                    width: {max(width_percent, 2)}%;
                    height: 100%;
                    background-color: {bar_color};
                    border-radius: 5px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-size: 12px;
                    font-weight: bold;
                ">
                    {status.execution_time_ms:.0f}ms
                </div>
            </div>
            """
            st.markdown(timeline_html, unsafe_allow_html=True)
    
    # Show timeline statistics
    st.markdown("### Timeline Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Duration", f"{total_duration:.2f}s")
    
    with col2:
        avg_execution = sum(
            status.execution_time_ms for _, status in executed_agents
        ) / len(executed_agents)
        st.metric("Avg Agent Time", f"{avg_execution:.0f}ms")
    
    with col3:
        max_execution = max(
            status.execution_time_ms for _, status in executed_agents
        )
        st.metric("Longest Agent", f"{max_execution:.0f}ms")
    
    with col4:
        st.metric("Agents Executed", len(executed_agents))


def render_parallel_vs_sequential_visualization(statuses: Dict[str, AgentStatus]):
    """
    Render visualization for parallel vs sequential execution patterns.
    
    Shows which agents executed in parallel vs sequentially,
    with timing analysis and efficiency metrics.
    
    Args:
        statuses: Dictionary mapping agent names to AgentStatus objects
        
    Requirements: 8.5
    """
    st.subheader("⚡ Parallel vs Sequential Execution")
    
    if not statuses:
        st.info("No execution pattern data available.")
        return
    
    # Analyze execution patterns
    parallel_groups = {}
    for agent_config in AGENT_PIPELINE:
        group = agent_config["parallel_group"]
        agent_name = agent_config["name"]
        agent_status = statuses.get(agent_name)
        
        if agent_status and agent_status.start_time:
            if group not in parallel_groups:
                parallel_groups[group] = []
            parallel_groups[group].append({
                "name": agent_name,
                "display_name": agent_config["display_name"],
                "status": agent_status,
                "config": agent_config
            })
    
    if not parallel_groups:
        st.info("No execution data available for pattern analysis.")
        return
    
    # Display execution pattern analysis
    st.markdown("### Execution Pattern Analysis")
    
    for group_num in sorted(parallel_groups.keys()):
        agents_in_group = parallel_groups[group_num]
        
        if len(agents_in_group) > 1:
            pattern_type = "🔀 Parallel Execution"
            pattern_color = "#2196F3"
        else:
            pattern_type = "➡️ Sequential Execution"
            pattern_color = "#9E9E9E"
        
        with st.expander(f"**Group {group_num}: {pattern_type}**", expanded=True):
            st.markdown(f"<div style='color: {pattern_color}; font-weight: bold;'>{pattern_type}</div>", unsafe_allow_html=True)
            
            # Show agents in this group
            for agent_data in agents_in_group:
                agent_status = agent_data["status"]
                display_name = agent_data["display_name"]
                
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                
                with col1:
                    status_emoji = {
                        "executing": "🟡",
                        "completed": "🟢",
                        "failed": "🔴",
                        "idle": "⚪"
                    }.get(agent_status.status, "⚪")
                    st.markdown(f"{status_emoji} {display_name}")
                
                with col2:
                    if agent_status.start_time:
                        st.caption(f"Start: {agent_status.start_time.strftime('%H:%M:%S')}")
                
                with col3:
                    if agent_status.execution_time_ms > 0:
                        st.caption(f"Time: {agent_status.execution_time_ms:.0f}ms")
                
                with col4:
                    st.caption(agent_status.status.upper())
            
            # Calculate group statistics
            if len(agents_in_group) > 1:
                # Parallel execution analysis
                group_start = min(a["status"].start_time for a in agents_in_group if a["status"].start_time)
                group_end = max(
                    a["status"].end_time if a["status"].end_time else a["status"].start_time
                    for a in agents_in_group
                )
                
                parallel_duration = (group_end - group_start).total_seconds() * 1000
                sequential_duration = sum(
                    a["status"].execution_time_ms for a in agents_in_group
                )
                
                time_saved = sequential_duration - parallel_duration
                efficiency = (time_saved / sequential_duration * 100) if sequential_duration > 0 else 0
                
                st.markdown("**Parallel Execution Benefits:**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Parallel Time", f"{parallel_duration:.0f}ms")
                
                with col2:
                    st.metric("Sequential Time", f"{sequential_duration:.0f}ms")
                
                with col3:
                    st.metric("Time Saved", f"{time_saved:.0f}ms", delta=f"{efficiency:.1f}%")
    
    # Overall execution pattern summary
    st.markdown("### Overall Execution Summary")
    
    total_parallel_groups = sum(1 for g in parallel_groups.values() if len(g) > 1)
    total_sequential_groups = sum(1 for g in parallel_groups.values() if len(g) == 1)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Parallel Groups", total_parallel_groups)
    
    with col2:
        st.metric("Sequential Groups", total_sequential_groups)
    
    with col3:
        total_groups = len(parallel_groups)
        parallel_ratio = (total_parallel_groups / total_groups * 100) if total_groups > 0 else 0
        st.metric("Parallelization", f"{parallel_ratio:.1f}%")


def render_execution_flow_dashboard(statuses: Dict[str, AgentStatus]):
    """
    Render complete execution flow dashboard.
    
    Combines all execution flow visualizations into a comprehensive dashboard
    with tabs for different views.
    
    Args:
        statuses: Dictionary mapping agent names to AgentStatus objects
        
    Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
    """
    st.header("🔄 Execution Flow Visualization")
    
    if not statuses:
        st.warning("No execution data available. Run an analysis to see execution flow.")
        return
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "Pipeline View",
        "Data Flow",
        "Timeline",
        "Parallel Analysis"
    ])
    
    with tab1:
        render_agent_pipeline_display(statuses, show_dependencies=True)
    
    with tab2:
        render_data_flow_visualization(statuses)
    
    with tab3:
        render_complete_execution_timeline(statuses)
    
    with tab4:
        render_parallel_vs_sequential_visualization(statuses)
