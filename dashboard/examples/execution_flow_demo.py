"""
Demo script for execution flow visualization components.

This script demonstrates how to use the execution flow visualization
components in a Streamlit dashboard.

To run this demo:
    streamlit run dashboard/examples/execution_flow_demo.py
"""

import streamlit as st
from datetime import datetime, timedelta
from dashboard.models.data_models import AgentStatus
from dashboard.components.execution_flow import (
    render_agent_pipeline_display,
    render_data_flow_visualization,
    render_complete_execution_timeline,
    render_parallel_vs_sequential_visualization,
    render_execution_flow_dashboard
)


def create_sample_statuses():
    """Create sample agent statuses for demonstration."""
    base_time = datetime.now()
    
    return {
        "reasoning": AgentStatus(
            agent_name="reasoning",
            status="completed",
            start_time=base_time,
            end_time=base_time + timedelta(milliseconds=150),
            execution_time_ms=150.0,
            posts_processed=10
        ),
        "pattern": AgentStatus(
            agent_name="pattern",
            status="completed",
            start_time=base_time + timedelta(milliseconds=10),
            end_time=base_time + timedelta(milliseconds=200),
            execution_time_ms=190.0,
            posts_processed=10
        ),
        "evidence": AgentStatus(
            agent_name="evidence",
            status="completed",
            start_time=base_time + timedelta(milliseconds=200),
            end_time=base_time + timedelta(milliseconds=350),
            execution_time_ms=150.0,
            posts_processed=8
        ),
        "social_behavior": AgentStatus(
            agent_name="social_behavior",
            status="completed",
            start_time=base_time + timedelta(milliseconds=200),
            end_time=base_time + timedelta(milliseconds=400),
            execution_time_ms=200.0,
            posts_processed=10
        ),
        "topic_modeling": AgentStatus(
            agent_name="topic_modeling",
            status="completed",
            start_time=base_time + timedelta(milliseconds=400),
            end_time=base_time + timedelta(milliseconds=550),
            execution_time_ms=150.0,
            posts_processed=10
        ),
        "topic_evidence": AgentStatus(
            agent_name="topic_evidence",
            status="completed",
            start_time=base_time + timedelta(milliseconds=550),
            end_time=base_time + timedelta(milliseconds=650),
            execution_time_ms=100.0,
            posts_processed=5
        ),
        "coordination_detector": AgentStatus(
            agent_name="coordination_detector",
            status="completed",
            start_time=base_time + timedelta(milliseconds=650),
            end_time=base_time + timedelta(milliseconds=750),
            execution_time_ms=100.0,
            posts_processed=3
        ),
        "echo_chamber_detector": AgentStatus(
            agent_name="echo_chamber_detector",
            status="completed",
            start_time=base_time + timedelta(milliseconds=650),
            end_time=base_time + timedelta(milliseconds=800),
            execution_time_ms=150.0,
            posts_processed=3
        )
    }


def main():
    """Main demo application."""
    st.set_page_config(
        page_title="Execution Flow Visualization Demo",
        page_icon="🔄",
        layout="wide"
    )
    
    st.title("🔄 Execution Flow Visualization Demo")
    st.markdown("---")
    
    # Create sample data
    sample_statuses = create_sample_statuses()
    
    # Sidebar controls
    st.sidebar.header("Demo Controls")
    
    view_mode = st.sidebar.radio(
        "Select View",
        [
            "Complete Dashboard",
            "Pipeline View",
            "Data Flow",
            "Timeline",
            "Parallel Analysis"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This demo shows the execution flow visualization components "
        "for the Agent Monitoring Dashboard. The sample data represents "
        "a typical analysis run with multiple agents."
    )
    
    # Render selected view
    if view_mode == "Complete Dashboard":
        st.markdown("## Complete Execution Flow Dashboard")
        st.markdown("This view combines all execution flow visualizations in tabs.")
        render_execution_flow_dashboard(sample_statuses)
    
    elif view_mode == "Pipeline View":
        st.markdown("## Agent Pipeline Display")
        st.markdown(
            "Shows agents in execution order with status indicators "
            "and dependency information."
        )
        render_agent_pipeline_display(sample_statuses, show_dependencies=True)
    
    elif view_mode == "Data Flow":
        st.markdown("## Data Flow Visualization")
        st.markdown(
            "Visualizes how data flows through the agent pipeline "
            "with processing statistics."
        )
        render_data_flow_visualization(sample_statuses)
    
    elif view_mode == "Timeline":
        st.markdown("## Complete Execution Timeline")
        st.markdown(
            "Detailed timeline showing when each agent started and ended, "
            "with duration bars."
        )
        render_complete_execution_timeline(sample_statuses)
    
    elif view_mode == "Parallel Analysis":
        st.markdown("## Parallel vs Sequential Execution")
        st.markdown(
            "Analyzes execution patterns showing which agents ran in parallel "
            "and the efficiency gains."
        )
        render_parallel_vs_sequential_visualization(sample_statuses)
    
    # Footer
    st.markdown("---")
    st.caption(
        "Execution Flow Visualization Demo | "
        "Agent Monitoring Dashboard | "
        "RumorNet Misinformation Detection System"
    )


if __name__ == "__main__":
    main()
