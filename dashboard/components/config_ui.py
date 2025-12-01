"""
Configuration UI components for the Agent Monitoring Dashboard.

This module provides Streamlit UI components for displaying and
editing dashboard configuration settings.
"""

import streamlit as st
from typing import Optional, Callable

from models.data_models import DashboardConfig
from utils.config_manager import ConfigManager


def render_config_sidebar(
    config: DashboardConfig,
    on_save: Optional[Callable[[DashboardConfig], None]] = None,
    on_reset: Optional[Callable[[], None]] = None
) -> Optional[DashboardConfig]:
    """
    Render configuration UI in sidebar.
    
    Displays configuration settings with input controls and validation.
    Allows users to modify, save, and reset configuration.
    
    Args:
        config: Current DashboardConfig object
        on_save: Optional callback function when configuration is saved
        on_reset: Optional callback function when configuration is reset
        
    Returns:
        Updated DashboardConfig if changes were made, None otherwise
        
    Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
    """
    st.sidebar.header("⚙️ Configuration")
    
    # Create config manager for validation
    config_manager = ConfigManager()
    
    # Configuration form
    with st.sidebar.form("config_form"):
        st.subheader("Ollama Settings")
        
        ollama_endpoint = st.text_input(
            "Ollama Endpoint",
            value=config.ollama_endpoint,
            help="URL for Ollama API endpoint"
        )
        
        ollama_model = st.text_input(
            "Ollama Model",
            value=config.ollama_model,
            help="Model name to use for analysis"
        )
        
        st.divider()
        st.subheader("Dashboard Settings")
        
        auto_refresh_interval = st.number_input(
            "Auto-Refresh Interval (seconds)",
            min_value=1,
            max_value=300,
            value=config.auto_refresh_interval,
            help="Seconds between auto-refreshes"
        )
        
        max_history_items = st.number_input(
            "Max History Items",
            min_value=1,
            max_value=1000,
            value=config.max_history_items,
            help="Maximum number of history items to keep"
        )
        
        default_batch_size = st.number_input(
            "Default Batch Size",
            min_value=1,
            max_value=1000,
            value=config.default_batch_size,
            help="Default batch size for processing"
        )
        
        enable_debug_mode = st.checkbox(
            "Enable Debug Mode",
            value=config.enable_debug_mode,
            help="Enable debug logging"
        )
        
        # Form buttons
        col1, col2 = st.columns(2)
        
        with col1:
            save_button = st.form_submit_button(
                "💾 Save",
                type="primary",
                use_container_width=True
            )
        
        with col2:
            reset_button = st.form_submit_button(
                "🔄 Reset",
                use_container_width=True
            )
    
    # Handle reset
    if reset_button:
        default_config = DashboardConfig.default()
        
        if on_reset:
            on_reset()
        
        st.sidebar.success("✅ Configuration reset to defaults")
        st.rerun()
        return default_config
    
    # Handle save
    if save_button:
        # Create updated config
        updates = {
            "ollama_endpoint": ollama_endpoint,
            "ollama_model": ollama_model,
            "auto_refresh_interval": int(auto_refresh_interval),
            "max_history_items": int(max_history_items),
            "default_batch_size": int(default_batch_size),
            "enable_debug_mode": enable_debug_mode
        }
        
        # Update and validate
        updated_config, is_valid, errors = config_manager.update_config(config, updates)
        
        if is_valid:
            # Save configuration
            if config_manager.save_config(updated_config):
                if on_save:
                    on_save(updated_config)
                
                st.sidebar.success("✅ Configuration saved successfully!")
                return updated_config
            else:
                st.sidebar.error("❌ Failed to save configuration")
        else:
            # Display validation errors
            st.sidebar.error("❌ Invalid configuration:")
            for error in errors:
                st.sidebar.error(f"• {error}")
    
    return None


def render_config_panel(
    config: DashboardConfig,
    editable: bool = True
) -> Optional[DashboardConfig]:
    """
    Render configuration panel in main area.
    
    Displays configuration settings in a more detailed panel format
    with validation guidance and help text.
    
    Args:
        config: Current DashboardConfig object
        editable: Whether configuration can be edited
        
    Returns:
        Updated DashboardConfig if changes were made, None otherwise
        
    Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
    """
    st.header("⚙️ Dashboard Configuration")
    
    # Create config manager for validation
    config_manager = ConfigManager()
    
    if not editable:
        # Display read-only configuration
        st.info("Configuration is read-only in this view")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Ollama Settings")
            st.text_input("Ollama Endpoint", value=config.ollama_endpoint, disabled=True)
            st.text_input("Ollama Model", value=config.ollama_model, disabled=True)
        
        with col2:
            st.subheader("Dashboard Settings")
            st.number_input("Auto-Refresh Interval", value=config.auto_refresh_interval, disabled=True)
            st.number_input("Max History Items", value=config.max_history_items, disabled=True)
            st.number_input("Default Batch Size", value=config.default_batch_size, disabled=True)
            st.checkbox("Enable Debug Mode", value=config.enable_debug_mode, disabled=True)
        
        return None
    
    # Editable configuration
    with st.form("config_panel_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Ollama Settings")
            
            ollama_endpoint = st.text_input(
                "Ollama Endpoint",
                value=config.ollama_endpoint,
                help=config_manager.get_validation_guidance("ollama_endpoint")
            )
            
            ollama_model = st.text_input(
                "Ollama Model",
                value=config.ollama_model,
                help=config_manager.get_validation_guidance("ollama_model")
            )
        
        with col2:
            st.subheader("Dashboard Settings")
            
            auto_refresh_interval = st.number_input(
                "Auto-Refresh Interval (seconds)",
                min_value=1,
                max_value=300,
                value=config.auto_refresh_interval,
                help=config_manager.get_validation_guidance("auto_refresh_interval")
            )
            
            max_history_items = st.number_input(
                "Max History Items",
                min_value=1,
                max_value=1000,
                value=config.max_history_items,
                help=config_manager.get_validation_guidance("max_history_items")
            )
            
            default_batch_size = st.number_input(
                "Default Batch Size",
                min_value=1,
                max_value=1000,
                value=config.default_batch_size,
                help=config_manager.get_validation_guidance("default_batch_size")
            )
            
            enable_debug_mode = st.checkbox(
                "Enable Debug Mode",
                value=config.enable_debug_mode,
                help=config_manager.get_validation_guidance("enable_debug_mode")
            )
        
        # Form buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            save_button = st.form_submit_button(
                "💾 Save Configuration",
                type="primary"
            )
        
        with col2:
            reset_button = st.form_submit_button("🔄 Reset to Defaults")
        
        with col3:
            validate_button = st.form_submit_button("✓ Validate Only")
    
    # Handle validation only
    if validate_button:
        updates = {
            "ollama_endpoint": ollama_endpoint,
            "ollama_model": ollama_model,
            "auto_refresh_interval": int(auto_refresh_interval),
            "max_history_items": int(max_history_items),
            "default_batch_size": int(default_batch_size),
            "enable_debug_mode": enable_debug_mode
        }
        
        _, is_valid, errors = config_manager.update_config(config, updates)
        
        if is_valid:
            st.success("✅ Configuration is valid!")
        else:
            st.error("❌ Configuration validation failed:")
            for error in errors:
                st.error(f"• {error}")
        
        return None
    
    # Handle reset
    if reset_button:
        default_config = DashboardConfig.default()
        
        if config_manager.save_config(default_config):
            st.success("✅ Configuration reset to defaults!")
            st.rerun()
            return default_config
        else:
            st.error("❌ Failed to reset configuration")
        
        return None
    
    # Handle save
    if save_button:
        updates = {
            "ollama_endpoint": ollama_endpoint,
            "ollama_model": ollama_model,
            "auto_refresh_interval": int(auto_refresh_interval),
            "max_history_items": int(max_history_items),
            "default_batch_size": int(default_batch_size),
            "enable_debug_mode": enable_debug_mode
        }
        
        updated_config, is_valid, errors = config_manager.update_config(config, updates)
        
        if is_valid:
            if config_manager.save_config(updated_config):
                st.success("✅ Configuration saved successfully!")
                return updated_config
            else:
                st.error("❌ Failed to save configuration")
        else:
            st.error("❌ Configuration validation failed:")
            for error in errors:
                st.error(f"• {error}")
    
    return None


def render_config_status(config: DashboardConfig):
    """
    Render configuration status summary.
    
    Displays a compact summary of current configuration settings.
    
    Args:
        config: Current DashboardConfig object
    """
    st.subheader("Current Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Ollama Model", config.ollama_model)
        st.metric("Batch Size", config.default_batch_size)
    
    with col2:
        st.metric("Refresh Interval", f"{config.auto_refresh_interval}s")
        st.metric("Max History", config.max_history_items)
    
    with col3:
        debug_status = "Enabled" if config.enable_debug_mode else "Disabled"
        st.metric("Debug Mode", debug_status)
        
        # Validate current config
        config_manager = ConfigManager()
        is_valid, _ = config_manager.validate_config(config)
        status = "✅ Valid" if is_valid else "❌ Invalid"
        st.metric("Config Status", status)
