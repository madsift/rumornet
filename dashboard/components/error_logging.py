"""
Error logging and debugging features for the Agent Monitoring Dashboard.

This module provides comprehensive error logging, display, and debugging
capabilities including error tracking, log viewing, and filtering.

Requirements: 10.1, 10.2, 10.3, 10.4, 10.5
"""

import streamlit as st
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import traceback


class ErrorLogger:
    """
    Centralized error logger for the dashboard.
    
    Provides methods for logging errors with timestamps, component names,
    and additional context for debugging.
    """
    
    def __init__(self):
        """Initialize the error logger."""
        self.errors: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
    
    def log_error(
        self,
        component: str,
        message: str,
        level: str = "ERROR",
        stack_trace: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log an error with complete information.
        
        Args:
            component: Name of the component where error occurred
            message: Error message
            level: Error level (ERROR, WARNING, CRITICAL, INFO)
            stack_trace: Optional stack trace
            context: Optional additional context
            
        Returns:
            Logged error dictionary
            
        Requirements: 10.1, 10.3
        """
        error_log = {
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "message": message,
            "level": level
        }
        
        if stack_trace:
            error_log["stack_trace"] = stack_trace
        
        if context:
            error_log["context"] = context
        
        # Add to errors list
        self.errors.append(error_log)
        
        # Also log to Python logger
        log_method = getattr(self.logger, level.lower(), self.logger.error)
        log_method(f"[{component}] {message}")
        
        return error_log
    
    def log_exception(
        self,
        component: str,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log an exception with stack trace.
        
        Args:
            component: Name of the component where exception occurred
            exception: The exception object
            context: Optional additional context
            
        Returns:
            Logged error dictionary
            
        Requirements: 10.2, 10.3
        """
        stack_trace = traceback.format_exc()
        message = f"{type(exception).__name__}: {str(exception)}"
        
        return self.log_error(
            component=component,
            message=message,
            level="ERROR",
            stack_trace=stack_trace,
            context=context
        )
    
    def get_errors(
        self,
        level: Optional[str] = None,
        component: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get logged errors with optional filtering.
        
        Args:
            level: Filter by error level
            component: Filter by component name
            limit: Maximum number of errors to return
            
        Returns:
            List of error dictionaries
            
        Requirements: 10.4, 10.5
        """
        filtered_errors = self.errors
        
        if level:
            filtered_errors = [e for e in filtered_errors if e.get("level") == level]
        
        if component:
            filtered_errors = [e for e in filtered_errors if e.get("component") == component]
        
        if limit:
            filtered_errors = filtered_errors[-limit:]
        
        return filtered_errors
    
    def clear_errors(self):
        """Clear all logged errors."""
        self.errors = []
    
    def get_error_count(self, level: Optional[str] = None) -> int:
        """Get count of errors, optionally filtered by level."""
        if level:
            return len([e for e in self.errors if e.get("level") == level])
        return len(self.errors)


def render_error_display(
    errors: List[Dict[str, Any]],
    title: str = "⚠️ Errors & Warnings",
    show_stack_trace: bool = True
):
    """
    Render error display section.
    
    Displays errors with timestamps, component names, messages,
    and optional stack traces for debugging.
    
    Args:
        errors: List of error dictionaries
        title: Title for the error section
        show_stack_trace: Whether to show stack traces
        
    Requirements: 10.1, 10.2, 10.3
    """
    st.subheader(title)
    
    if not errors:
        st.success("✅ No errors detected")
        return
    
    # Show error count by level
    error_counts = {}
    for error in errors:
        level = error.get("level", "ERROR")
        error_counts[level] = error_counts.get(level, 0) + 1
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Errors", len(errors))
    
    with col2:
        st.metric("Critical", error_counts.get("CRITICAL", 0))
    
    with col3:
        st.metric("Errors", error_counts.get("ERROR", 0))
    
    with col4:
        st.metric("Warnings", error_counts.get("WARNING", 0))
    
    st.divider()
    
    # Display each error
    for i, error in enumerate(errors, 1):
        level = error.get("level", "ERROR")
        
        # Choose icon and color based on level
        if level == "CRITICAL":
            icon = "🔴"
            color = "red"
        elif level == "ERROR":
            icon = "🟠"
            color = "orange"
        elif level == "WARNING":
            icon = "🟡"
            color = "yellow"
        else:
            icon = "🔵"
            color = "blue"
        
        with st.expander(
            f"{icon} Error {i}: {error.get('component', 'Unknown')} - {error.get('timestamp', 'N/A')}",
            expanded=(i == 1)  # Expand first error by default
        ):
            # Error details
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Component:** {error.get('component', 'Unknown')}")
                st.markdown(f"**Level:** {level}")
            
            with col2:
                st.markdown(f"**Timestamp:** {error.get('timestamp', 'N/A')}")
            
            # Error message
            st.markdown("**Message:**")
            st.error(error.get('message', 'No message available'))
            
            # Stack trace (if available and enabled)
            if show_stack_trace and error.get('stack_trace'):
                st.markdown("**Stack Trace:**")
                st.code(error.get('stack_trace'), language="python")
            
            # Additional context (if available)
            if error.get('context'):
                st.markdown("**Context:**")
                st.json(error.get('context'))


def render_log_viewer(
    error_logger: ErrorLogger,
    max_display: int = 50
):
    """
    Render log viewer with filtering capabilities.
    
    Provides an interface for viewing and filtering error logs
    with search and level filtering.
    
    Args:
        error_logger: ErrorLogger instance
        max_display: Maximum number of logs to display
        
    Requirements: 10.4, 10.5
    """
    st.subheader("📋 Log Viewer")
    
    # Filtering controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        level_filter = st.selectbox(
            "Filter by Level",
            options=["All", "CRITICAL", "ERROR", "WARNING", "INFO"],
            key="log_level_filter"
        )
    
    with col2:
        # Get unique components
        all_errors = error_logger.get_errors()
        components = sorted(set(e.get("component", "Unknown") for e in all_errors))
        component_filter = st.selectbox(
            "Filter by Component",
            options=["All"] + components,
            key="log_component_filter"
        )
    
    with col3:
        limit = st.number_input(
            "Max Logs",
            min_value=10,
            max_value=500,
            value=max_display,
            step=10,
            key="log_limit"
        )
    
    # Apply filters
    level = None if level_filter == "All" else level_filter
    component = None if component_filter == "All" else component_filter
    
    filtered_errors = error_logger.get_errors(
        level=level,
        component=component,
        limit=int(limit)
    )
    
    # Display filtered errors
    st.write(f"Showing {len(filtered_errors)} log(s)")
    
    if filtered_errors:
        render_error_display(filtered_errors, title="Filtered Logs")
    else:
        st.info("No logs match the current filters.")
    
    # Clear logs button
    st.divider()
    if st.button("🗑️ Clear All Logs"):
        if st.checkbox("Confirm clear all logs"):
            error_logger.clear_errors()
            st.success("All logs cleared!")
            st.rerun()


def render_error_summary(error_logger: ErrorLogger):
    """
    Render error summary dashboard.
    
    Displays summary statistics and trends for logged errors.
    
    Args:
        error_logger: ErrorLogger instance
        
    Requirements: 10.1, 10.5
    """
    st.subheader("📊 Error Summary")
    
    errors = error_logger.get_errors()
    
    if not errors:
        st.info("No errors logged yet.")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Errors", len(errors))
    
    with col2:
        critical_count = error_logger.get_error_count("CRITICAL")
        st.metric("Critical", critical_count)
    
    with col3:
        error_count = error_logger.get_error_count("ERROR")
        st.metric("Errors", error_count)
    
    with col4:
        warning_count = error_logger.get_error_count("WARNING")
        st.metric("Warnings", warning_count)
    
    # Component breakdown
    st.divider()
    st.write("**Errors by Component:**")
    
    component_counts = {}
    for error in errors:
        component = error.get("component", "Unknown")
        component_counts[component] = component_counts.get(component, 0) + 1
    
    if component_counts:
        import pandas as pd
        df = pd.DataFrame([
            {"Component": comp, "Error Count": count}
            for comp, count in sorted(component_counts.items(), key=lambda x: x[1], reverse=True)
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Recent errors
    st.divider()
    st.write("**Recent Errors (Last 5):**")
    recent_errors = errors[-5:]
    
    for error in reversed(recent_errors):
        level_icon = {
            "CRITICAL": "🔴",
            "ERROR": "🟠",
            "WARNING": "🟡",
            "INFO": "🔵"
        }.get(error.get("level", "ERROR"), "⚪")
        
        st.write(f"{level_icon} **{error.get('component')}**: {error.get('message')[:100]}...")


def render_complete_error_logging_interface(error_logger: ErrorLogger):
    """
    Render complete error logging interface.
    
    Provides comprehensive error logging and debugging interface
    with all features including display, filtering, and summary.
    
    Args:
        error_logger: ErrorLogger instance
        
    Requirements: 10.1, 10.2, 10.3, 10.4, 10.5
    """
    st.title("🐛 Error Logging & Debugging")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs([
        "📊 Summary",
        "📋 Log Viewer",
        "⚙️ Settings"
    ])
    
    with tab1:
        render_error_summary(error_logger)
    
    with tab2:
        render_log_viewer(error_logger)
    
    with tab3:
        st.subheader("Logging Settings")
        
        st.write("**Log Levels:**")
        st.write("- 🔴 **CRITICAL**: System-critical errors requiring immediate attention")
        st.write("- 🟠 **ERROR**: Errors that prevent normal operation")
        st.write("- 🟡 **WARNING**: Warnings about potential issues")
        st.write("- 🔵 **INFO**: Informational messages")
        
        st.divider()
        
        st.write("**Log Management:**")
        total_logs = len(error_logger.get_errors())
        st.write(f"Total logs stored: {total_logs}")
        
        if st.button("Export Logs to File"):
            st.info("Export functionality coming soon!")
        
        st.divider()
        
        st.write("**Debug Mode:**")
        debug_mode = st.checkbox("Enable debug mode", value=False)
        
        if debug_mode:
            st.info("Debug mode enabled - showing additional diagnostic information")


# Convenience function for session state integration
def get_error_logger() -> ErrorLogger:
    """
    Get or create error logger from session state.
    
    Returns:
        ErrorLogger instance
    """
    if "error_logger" not in st.session_state:
        st.session_state.error_logger = ErrorLogger()
    
    return st.session_state.error_logger
