"""
Main Dashboard Application for Agent Monitoring.

This is the main entry point for the Agent Monitoring Dashboard.
It provides a comprehensive interface for monitoring agent execution,
viewing analysis results, and managing configuration.

To run the dashboard:
    streamlit run dashboard/dashboard.py

Requirements: 1.5, 7.1
"""

import streamlit as st
import asyncio
import sys
import os
import requests
from datetime import datetime
from typing import Optional, Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import core modules
from core.state_manager import (
    initialize_session_state,
    get_agent_status,
    get_all_agent_statuses,
    update_agent_status,
    add_execution_result,
    get_execution_history,
    get_current_result,
    set_current_result,
    log_error,
    get_error_log,
    clear_error_log,
    start_new_execution
)
from core.data_manager import DataManager
from core.orchestrator_monitor import OrchestratorMonitor
from models.data_models import (
    AgentStatus,
    ExecutionResult,
    DashboardConfig
)

# Import UI components directly (avoid circular imports)
from components.ui_components import render_agent_grid, render_metrics_dashboard, render_summary_cards, render_progress_bar
from components.results_display import render_complete_results_display
from components.history_viewer import render_complete_history_viewer
from components.markdown_export import render_complete_markdown_export
from components.error_logging import render_complete_error_logging_interface, get_error_logger
from components.execution_flow import render_execution_flow_dashboard

# Import batch analysis component (API-enabled version)
from components.batch_analysis_api import render_batch_analysis_workflow_with_api

# Import styling utilities
from utils.styling import (
    initialize_styling,
    render_loading_spinner,
    render_status_badge,
    render_divider,
    render_alert
)


# Page configuration
st.set_page_config(
    page_title="RumorNet",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize styling
initialize_styling()


def initialize_dashboard():
    """Initialize dashboard state and configuration."""
    # Initialize session state
    initialize_session_state(st.session_state)
    
    # Initialize data manager
    if "data_manager" not in st.session_state:
        st.session_state.data_manager = DataManager()
    
    # Initialize error logger
    if "error_logger" not in st.session_state:
        st.session_state.error_logger = get_error_logger()
    
    # Initialize orchestrator monitor (will be set when orchestrator is available)
    if "orchestrator_monitor" not in st.session_state:
        st.session_state.orchestrator_monitor = None
        st.session_state.orchestrator_init_attempted = False
    
    # Auto-refresh state (enabled by default)
    if "auto_refresh_enabled" not in st.session_state:
        st.session_state.auto_refresh_enabled = True
    
    if "last_refresh_time" not in st.session_state:
        st.session_state.last_refresh_time = datetime.now()


def render_sidebar():
    """Render sidebar with configuration and controls."""
    st.sidebar.title("🔍 RumorNet")
    st.sidebar.markdown("---")
    
    # Configuration section - SIMPLIFIED (removed Ollama Settings)
    config = st.session_state.config
    
    st.sidebar.header("⚙️ Settings")
    
    with st.sidebar.expander("Dashboard Settings", expanded=False):
        auto_refresh_interval_config = st.number_input(
            "Auto-Refresh Interval (seconds)",
            min_value=1,
            max_value=300,
            value=config.auto_refresh_interval,
            help="Seconds between auto-refreshes",
            key="config_refresh_interval"
        )
        
        max_history_items = st.number_input(
            "Max History Items",
            min_value=1,
            max_value=1000,
            value=config.max_history_items,
            help="Maximum number of history items to keep",
            key="config_max_history"
        )
        
        default_batch_size = st.number_input(
            "Default Batch Size",
            min_value=1,
            max_value=1000,
            value=config.default_batch_size,
            help="Default batch size for processing",
            key="config_batch_size"
        )
        
        enable_debug_mode = st.checkbox(
            "Enable Debug Mode",
            value=config.enable_debug_mode,
            help="Enable debug logging",
            key="config_debug"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save", type="primary", use_container_width=True, key="config_save"):
                st.session_state.config.auto_refresh_interval = int(auto_refresh_interval_config)
                st.session_state.config.max_history_items = int(max_history_items)
                st.session_state.config.default_batch_size = int(default_batch_size)
                st.session_state.config.enable_debug_mode = enable_debug_mode
                st.success("✅ Saved!")
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset", use_container_width=True, key="config_reset"):
                st.session_state.config = DashboardConfig.default()
                st.success("✅ Reset!")
                st.rerun()
    
    st.sidebar.markdown("---")
    
    # Quick stats
    st.sidebar.subheader("📊 Quick Stats")
    
    agent_statuses = get_all_agent_statuses(st.session_state)
    
    if agent_statuses:
        executing_count = sum(
            1 for status in agent_statuses.values()
            if status.status == "executing"
        )
        completed_count = sum(
            1 for status in agent_statuses.values()
            if status.status == "completed"
        )
        failed_count = sum(
            1 for status in agent_statuses.values()
            if status.status == "failed"
        )
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Executing", executing_count)
            st.metric("Completed", completed_count)
        with col2:
            st.metric("Failed", failed_count)
            st.metric("Total", len(agent_statuses))
    else:
        st.sidebar.info("No agent data available")
    
    st.sidebar.markdown("---")
    
    # Auto-refresh controls
    st.sidebar.subheader("🔄 Auto-Refresh")
    
    # Check if analysis is running
    is_analyzing = st.session_state.get('analysis_in_progress', False)
    
    auto_refresh = st.sidebar.checkbox(
        "Enable Auto-Refresh",
        value=st.session_state.auto_refresh_enabled,
        help="Automatically refresh the dashboard",
        disabled=is_analyzing
    )
    
    if auto_refresh != st.session_state.auto_refresh_enabled:
        st.session_state.auto_refresh_enabled = auto_refresh
    
    if auto_refresh and not is_analyzing:
        refresh_interval = st.sidebar.slider(
            "Refresh Interval (seconds)",
            min_value=1,
            max_value=30,
            value=st.session_state.config.auto_refresh_interval,
            help="How often to refresh the dashboard"
        )
        
        # Update config if changed
        if refresh_interval != st.session_state.config.auto_refresh_interval:
            st.session_state.config.auto_refresh_interval = refresh_interval
        
        # Check if it's time to refresh
        time_since_refresh = (datetime.now() - st.session_state.last_refresh_time).total_seconds()
        
        if time_since_refresh >= refresh_interval:
            st.session_state.last_refresh_time = datetime.now()
            st.rerun()
        
        # Show countdown
        remaining = refresh_interval - time_since_refresh
        st.sidebar.caption(f"Next refresh in {remaining:.1f}s")
    elif is_analyzing:
        st.sidebar.caption("⏸️ Paused during analysis")
    
    st.sidebar.markdown("---")
    
    # Error log quick view
    error_log = get_error_log(st.session_state)
    if error_log:
        st.sidebar.error(f"⚠️ {len(error_log)} error(s) logged")
        if st.sidebar.button("View Errors"):
            st.session_state.active_tab = "Errors"
            st.rerun()
    
    # History quick access
    history = get_execution_history(st.session_state)
    if history:
        st.sidebar.info(f"📚 {len(history)} execution(s) in history")


def render_overview_tab():
    """Render the overview tab with key metrics and status."""
    st.header("📊 Dashboard Overview")
    
    # Get current data
    agent_statuses = get_all_agent_statuses(st.session_state)
    current_result = get_current_result(st.session_state)
    
    # Summary cards
    if current_result and current_result.full_report:
        report = current_result.full_report
        summary = report.get("executive_summary", {})
        
        render_summary_cards(summary)
        
        st.markdown("---")
        
        # Markdown Report Display
        st.subheader("📄 Markdown Report")
        
        # Generate markdown report
        from utils.markdown_generator import MarkdownGenerator
        markdown_gen = MarkdownGenerator()
        markdown_report = markdown_gen.generate_markdown_report(report)
        
        # Display markdown in expandable section (expanded by default)
        with st.expander("📖 View Full Markdown Report", expanded=True):
            st.markdown(markdown_report)
        
        # Download button
        st.download_button(
            label="⬇️ Download Markdown Report",
            data=markdown_report,
            file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )
    else:
        st.info("No analysis results available. Run an analysis to see metrics.")
    
    st.markdown("---")
    
    # Agent status grid
    st.subheader("🤖 Agent Status")
    
    if agent_statuses:
        render_agent_grid(agent_statuses, columns=4)
    else:
        st.info("No agent status data available.")
    
    st.markdown("---")
    
    # Metrics dashboard
    if st.session_state.metrics:
        render_metrics_dashboard(st.session_state.metrics)
    else:
        st.info("No performance metrics available yet.")


def render_analysis_tab():
    """Render the analysis tab for running batch analysis."""
    st.header("🔬 Batch Analysis")
    
    st.markdown("""
    Upload or paste post data to analyze for misinformation.
    The system will process posts through the agent pipeline and provide detailed results.
    """)
    
    # Render batch analysis interface (API-enabled)
    render_batch_analysis_workflow_with_api(
        orchestrator_monitor=st.session_state.orchestrator_monitor,
        data_manager=st.session_state.data_manager
    )


def render_results_tab():
    """Render the results tab with detailed analysis results."""
    st.header("📈 Analysis Results")
    
    current_result = get_current_result(st.session_state)
    
    if current_result and current_result.full_report:
        # Render complete results display
        render_complete_results_display(current_result.full_report)
    else:
        st.info("No analysis results available. Run an analysis to see results.")
        
        # Show option to load from history
        history = get_execution_history(st.session_state)
        if history:
            st.markdown("### Load from History")
            
            selected_id = st.selectbox(
                "Select a previous execution",
                options=[r.execution_id for r in history],
                format_func=lambda x: f"{x[:8]}... - {next((r.timestamp.strftime('%Y-%m-%d %H:%M:%S') for r in history if r.execution_id == x), 'Unknown')}"
            )
            
            if st.button("Load Results"):
                for result in history:
                    if result.execution_id == selected_id:
                        set_current_result(st.session_state, result)
                        st.success("Results loaded!")
                        st.rerun()


def render_execution_flow_tab():
    """Render the execution flow visualization tab."""
    st.header("🔄 Execution Flow")
    
    # Try to get agent statuses from current result first
    current_result = get_current_result(st.session_state)
    agent_statuses = None
    
    if current_result and current_result.agent_statuses:
        agent_statuses = current_result.agent_statuses
    else:
        # Fall back to session state agent statuses
        agent_statuses = get_all_agent_statuses(st.session_state)
    
    if agent_statuses:
        # Check if any agents have execution data
        has_execution_data = any(
            status.start_time is not None or status.execution_time_ms > 0
            for status in agent_statuses.values()
        )
        
        if has_execution_data:
            render_execution_flow_dashboard(agent_statuses)
        else:
            st.info("No execution flow data available. Agents are idle.")
            st.markdown("Run an analysis to see the execution flow visualization.")
    else:
        st.info("No agent data available. Run an analysis to see execution flow.")
        
        # Show option to load from history
        history = get_execution_history(st.session_state)
        if history:
            st.markdown("### Load from History")
            
            selected_id = st.selectbox(
                "Select a previous execution",
                options=[r.execution_id for r in history],
                format_func=lambda x: f"{x[:8]}... - {next((r.timestamp.strftime('%Y-%m-%d %H:%M:%S') for r in history if r.execution_id == x), 'Unknown')}",
                key="execution_flow_history_select"
            )
            
            if st.button("Load Execution Flow", key="load_execution_flow"):
                for result in history:
                    if result.execution_id == selected_id:
                        set_current_result(st.session_state, result)
                        st.success("Execution flow loaded!")
                        st.rerun()


def render_history_tab():
    """Render the history tab with execution history."""
    st.header("📚 Execution History")
    
    # Pass data_manager instead of history list
    render_complete_history_viewer(st.session_state.data_manager)


def render_export_tab():
    """Render the export tab for markdown generation."""
    st.header("📄 Export Results")
    
    current_result = get_current_result(st.session_state)
    
    if current_result and current_result.full_report:
        render_complete_markdown_export(current_result.full_report)
    else:
        st.info("No results available to export. Run an analysis first.")


def render_errors_tab():
    """Render the errors tab with error logging."""
    st.header("⚠️ Errors & Debugging")
    
    error_log = get_error_log(st.session_state)
    
    if error_log:
        render_complete_error_logging_interface(error_log)
        
        # Clear errors button
        if st.button("Clear Error Log", type="secondary"):
            clear_error_log(st.session_state)
            st.success("Error log cleared!")
            st.rerun()
    else:
        st.success("✅ No errors logged")
        st.info("Errors will appear here when they occur during analysis.")


def render_main_content():
    """Render main content area with tabs."""
    # Initialize active tab if not set
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Overview"
    
    # Create tabs
    tabs = st.tabs([
        "📊 Overview",
        "🔬 Analysis",
        "📈 Results",
        "🔄 Execution Flow",
        "📚 History",
        "📄 Export",
        "⚠️ Errors"
    ])
    
    # Render content based on active tab
    with tabs[0]:
        render_overview_tab()
    
    with tabs[1]:
        render_analysis_tab()
    
    with tabs[2]:
        render_results_tab()
    
    with tabs[3]:
        render_execution_flow_tab()
    
    with tabs[4]:
        render_history_tab()
    
    with tabs[5]:
        render_export_tab()
    
    with tabs[6]:
        render_errors_tab()


def render_header():
    """Render dashboard header."""
    col1, col2, col3 = st.columns([2, 3, 2])
    
    with col1:
        st.title("🔍 RumorNet")
    
    with col2:
        st.markdown("### Misinformation Detection")
    
    with col3:
        # Check Lambda health
        LAMBDA_API_URL = "https://mgbsx1x8l1.execute-api.us-east-1.amazonaws.com/Prod"
        
        # Check if analysis is in progress
        if st.session_state.get('analysis_in_progress', False):
            st.markdown("🟡 **Status:** Analyzing")
        else:
            # Try to get Lambda health (with caching to avoid too many requests)
            if 'lambda_health_check' not in st.session_state or \
               (datetime.now() - st.session_state.get('lambda_health_check_time', datetime.min)).seconds > 30:
                try:
                    response = requests.get(f"{LAMBDA_API_URL}/health", timeout=2)
                    if response.status_code == 200:
                        st.session_state.lambda_health_check = "healthy"
                    else:
                        st.session_state.lambda_health_check = "error"
                    st.session_state.lambda_health_check_time = datetime.now()
                except:
                    st.session_state.lambda_health_check = "error"
                    st.session_state.lambda_health_check_time = datetime.now()
            
            if st.session_state.get('lambda_health_check') == "healthy":
                st.markdown("🟢 **Lambda:** Healthy")
            else:
                st.markdown("🔴 **Lambda:** Offline")
    
    st.markdown("---")


def render_footer():
    """Render dashboard footer."""
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption("RumorNet Dashboard v1.0")
    
    with col2:
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    with col3:
        if st.session_state.auto_refresh_enabled:
            st.caption("🔄 Auto-refresh: ON")
        else:
            st.caption("⏸️ Auto-refresh: OFF")


def main():
    """Main dashboard application."""
    try:
        # Initialize dashboard
        initialize_dashboard()
        
        # Render header
        render_header()
        
        # Render sidebar
        render_sidebar()
        
        # Render main content
        render_main_content()
        
        # Render footer
        render_footer()
        
    except Exception as e:
        st.error(f"Dashboard Error: {e}")
        
        # Log error
        log_error(
            st.session_state,
            component="main_dashboard",
            error_message=str(e),
            stack_trace=None
        )
        
        # Show debug info
        with st.expander("Debug Information"):
            st.exception(e)
            
            if st.button("Clear Session State"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()


if __name__ == "__main__":
    main()
