"""
UI Components for the Agent Monitoring Dashboard.

This package provides reusable Streamlit components for building
the dashboard interface.
"""

from components.ui_components import (
    render_agent_status_card,
    render_metrics_dashboard,
    render_results_table,
    render_execution_timeline,
    render_progress_bar,
    render_error_panel,
    render_summary_cards,
    render_filter_panel,
    render_agent_grid
)

from components.results_display import (
    render_executive_summary_display,
    render_high_priority_posts_table,
    render_top_offenders_display,
    render_pattern_breakdown_visualization,
    render_topic_analysis_display,
    render_temporal_trends_visualization,
    render_complete_results_display
)

from components.history_viewer import (
    render_history_list,
    render_execution_details,
    render_execution_trends,
    render_history_comparison,
    render_history_sidebar,
    render_complete_history_viewer
)

from components.markdown_export import (
    generate_markdown_with_metadata,
    render_markdown_preview,
    render_markdown_download_button,
    render_markdown_export_options,
    render_complete_markdown_export,
    export_markdown_to_string,
    save_markdown_to_file,
    get_markdown_stats
)

from components.error_logging import (
    ErrorLogger,
    render_error_display,
    render_log_viewer,
    render_error_summary,
    render_complete_error_logging_interface,
    get_error_logger
)

from components.execution_flow import (
    render_agent_pipeline_display,
    render_data_flow_visualization,
    render_complete_execution_timeline,
    render_parallel_vs_sequential_visualization,
    render_execution_flow_dashboard
)

__all__ = [
    "render_agent_status_card",
    "render_metrics_dashboard",
    "render_results_table",
    "render_execution_timeline",
    "render_progress_bar",
    "render_error_panel",
    "render_summary_cards",
    "render_filter_panel",
    "render_agent_grid",
    "render_executive_summary_display",
    "render_high_priority_posts_table",
    "render_top_offenders_display",
    "render_pattern_breakdown_visualization",
    "render_topic_analysis_display",
    "render_temporal_trends_visualization",
    "render_complete_results_display",
    "render_history_list",
    "render_execution_details",
    "render_execution_trends",
    "render_history_comparison",
    "render_history_sidebar",
    "render_complete_history_viewer",
    "generate_markdown_with_metadata",
    "render_markdown_preview",
    "render_markdown_download_button",
    "render_markdown_export_options",
    "render_complete_markdown_export",
    "export_markdown_to_string",
    "save_markdown_to_file",
    "get_markdown_stats",
    "ErrorLogger",
    "render_error_display",
    "render_log_viewer",
    "render_error_summary",
    "render_complete_error_logging_interface",
    "get_error_logger",
    "render_agent_pipeline_display",
    "render_data_flow_visualization",
    "render_complete_execution_timeline",
    "render_parallel_vs_sequential_visualization",
    "render_execution_flow_dashboard"
]
