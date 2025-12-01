"""
Configuration Loader Utility

Loads and validates dashboard configuration from config.yaml
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


def get_config_path() -> Path:
    """Get the path to the config.yaml file."""
    dashboard_dir = Path(__file__).parent.parent
    return dashboard_dir / "config.yaml"


def load_config() -> Dict[str, Any]:
    """
    Load configuration from config.yaml.
    
    Returns:
        Dict containing configuration settings
        
    Raises:
        FileNotFoundError: If config.yaml is not found
        yaml.YAMLError: If config.yaml is invalid
    """
    config_path = get_config_path()
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_orchestrator_config() -> Dict[str, Any]:
    """Get orchestrator configuration section."""
    config = load_config()
    return config.get('orchestrator', {})


def get_dashboard_config() -> Dict[str, Any]:
    """Get dashboard configuration section."""
    config = load_config()
    return config.get('dashboard', {})


def get_data_config() -> Dict[str, Any]:
    """Get data persistence configuration section."""
    config = load_config()
    return config.get('data', {})


def get_visualization_config() -> Dict[str, Any]:
    """Get visualization configuration section."""
    config = load_config()
    return config.get('visualization', {})


def get_logging_config() -> Dict[str, Any]:
    """Get logging configuration section."""
    config = load_config()
    return config.get('logging', {})


def ensure_directories() -> None:
    """
    Ensure all required directories exist.
    Creates directories specified in the configuration if they don't exist.
    """
    config = load_config()
    dashboard_dir = Path(__file__).parent.parent
    
    # Create data directories
    data_config = config.get('data', {})
    history_dir = dashboard_dir / data_config.get('history_dir', 'data/history')
    export_dir = dashboard_dir / data_config.get('export_dir', 'data/exports')
    
    history_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logs directory
    logging_config = config.get('logging', {})
    log_file = logging_config.get('log_file', 'logs/dashboard.log')
    log_dir = dashboard_dir / Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
