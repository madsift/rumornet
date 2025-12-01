"""
Core Module

Core functionality for orchestrator monitoring, data management, and report generation.
"""

from core.state_manager import (
    initialize_session_state,
    update_agent_status,
    get_agent_status,
    get_all_agent_statuses,
    reset_agent_statuses,
    add_execution_result,
    get_execution_history,
    get_execution_by_id,
    clear_execution_history,
    update_config,
    get_config,
    update_filters,
    get_filters,
    log_error,
    get_error_log,
    clear_error_log,
    generate_execution_id,
    start_new_execution,
    set_current_result,
    get_current_result
)

from core.orchestrator_monitor import OrchestratorMonitor
from core.data_manager import DataManager

__all__ = [
    "initialize_session_state",
    "update_agent_status",
    "get_agent_status",
    "get_all_agent_statuses",
    "reset_agent_statuses",
    "add_execution_result",
    "get_execution_history",
    "get_execution_by_id",
    "clear_execution_history",
    "update_config",
    "get_config",
    "update_filters",
    "get_filters",
    "log_error",
    "get_error_log",
    "clear_error_log",
    "generate_execution_id",
    "start_new_execution",
    "set_current_result",
    "get_current_result",
    "OrchestratorMonitor",
    "DataManager"
]
