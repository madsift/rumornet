"""
Configuration management for the Agent Monitoring Dashboard.

This module handles loading, saving, validating, and managing
dashboard configuration settings.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from urllib.parse import urlparse

from dashboard.models.data_models import DashboardConfig


class ConfigValidationError(Exception):
    """Exception raised when configuration validation fails."""
    pass


class ConfigManager:
    """
    Manages dashboard configuration with validation and persistence.
    
    Handles loading configuration from file, validating settings,
    saving changes, and providing default values.
    """
    
    def __init__(self, config_path: str = "dashboard/config.yaml"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.logger = logging.getLogger(f"{__name__}.config_manager")
        
        # Ensure config directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
    
    def load_config(self) -> DashboardConfig:
        """
        Load configuration from file.
        
        Returns default configuration if file doesn't exist or is invalid.
        
        Returns:
            DashboardConfig object
        """
        try:
            if not self.config_path.exists():
                self.logger.info("Config file not found, using defaults")
                return DashboardConfig.default()
            
            # Read config file
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if self.config_path.suffix == '.json':
                    config_dict = json.load(f)
                else:
                    # For YAML files, try to parse as JSON first (simple format)
                    content = f.read()
                    try:
                        config_dict = json.loads(content)
                    except json.JSONDecodeError:
                        # If not JSON, parse as simple key-value YAML
                        config_dict = self._parse_simple_yaml(content)
            
            # Create config from dictionary
            config = DashboardConfig.from_dict(config_dict)
            
            self.logger.info("Configuration loaded successfully")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            self.logger.info("Using default configuration")
            return DashboardConfig.default()
    
    def save_config(self, config: DashboardConfig) -> bool:
        """
        Save configuration to file.
        
        Args:
            config: DashboardConfig object to save
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            # Convert to dictionary
            config_dict = config.to_dict()
            
            # Write to file
            with open(self.config_path, 'w', encoding='utf-8') as f:
                if self.config_path.suffix == '.json':
                    json.dump(config_dict, f, indent=2)
                else:
                    # Write as simple YAML format
                    f.write(self._format_simple_yaml(config_dict))
            
            self.logger.info("Configuration saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False
    
    def validate_config(self, config: DashboardConfig) -> Tuple[bool, List[str]]:
        """
        Validate configuration settings.
        
        Checks all configuration values for validity and returns
        validation status with error messages.
        
        Args:
            config: DashboardConfig object to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
            
        Requirements: 7.2, 7.5
        """
        errors = []
        
        # Validate ollama_endpoint
        if not config.ollama_endpoint:
            errors.append("Ollama endpoint cannot be empty")
        else:
            try:
                parsed = urlparse(config.ollama_endpoint)
                if not parsed.scheme or not parsed.netloc:
                    errors.append("Ollama endpoint must be a valid URL (e.g., http://localhost:11434)")
                elif parsed.scheme not in ["http", "https"]:
                    errors.append("Ollama endpoint must use http or https protocol")
            except Exception:
                errors.append("Ollama endpoint must be a valid URL")
        
        # Validate ollama_model
        if not config.ollama_model:
            errors.append("Ollama model cannot be empty")
        elif len(config.ollama_model) < 2:
            errors.append("Ollama model name must be at least 2 characters")
        
        # Validate auto_refresh_interval
        if config.auto_refresh_interval < 1:
            errors.append("Auto-refresh interval must be at least 1 second")
        elif config.auto_refresh_interval > 300:
            errors.append("Auto-refresh interval cannot exceed 300 seconds (5 minutes)")
        
        # Validate max_history_items
        if config.max_history_items < 1:
            errors.append("Max history items must be at least 1")
        elif config.max_history_items > 1000:
            errors.append("Max history items cannot exceed 1000")
        
        # Validate default_batch_size
        if config.default_batch_size < 1:
            errors.append("Default batch size must be at least 1")
        elif config.default_batch_size > 1000:
            errors.append("Default batch size cannot exceed 1000")
        
        # enable_debug_mode is boolean, no validation needed
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def validate_config_dict(self, config_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate configuration dictionary before creating DashboardConfig.
        
        Args:
            config_dict: Dictionary containing configuration values
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required fields
        required_fields = [
            "ollama_endpoint",
            "ollama_model",
            "auto_refresh_interval",
            "max_history_items",
            "default_batch_size",
            "enable_debug_mode"
        ]
        
        for field in required_fields:
            if field not in config_dict:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return False, errors
        
        # Create temporary config and validate
        try:
            config = DashboardConfig.from_dict(config_dict)
            return self.validate_config(config)
        except Exception as e:
            errors.append(f"Invalid configuration format: {str(e)}")
            return False, errors
    
    def update_config(
        self,
        current_config: DashboardConfig,
        updates: Dict[str, Any]
    ) -> Tuple[DashboardConfig, bool, List[str]]:
        """
        Update configuration with new values.
        
        Validates the updated configuration before applying changes.
        
        Args:
            current_config: Current DashboardConfig object
            updates: Dictionary of fields to update
            
        Returns:
            Tuple of (updated_config, is_valid, error_messages)
        """
        # Create updated config dictionary
        config_dict = current_config.to_dict()
        config_dict.update(updates)
        
        # Validate updated config
        is_valid, errors = self.validate_config_dict(config_dict)
        
        if is_valid:
            updated_config = DashboardConfig.from_dict(config_dict)
            return updated_config, True, []
        else:
            return current_config, False, errors
    
    def reset_to_defaults(self) -> DashboardConfig:
        """
        Reset configuration to default values.
        
        Returns:
            Default DashboardConfig object
        """
        self.logger.info("Resetting configuration to defaults")
        return DashboardConfig.default()
    
    def export_config(self, config: DashboardConfig, output_path: str) -> bool:
        """
        Export configuration to a file.
        
        Args:
            config: DashboardConfig object to export
            output_path: Path for the output file
            
        Returns:
            True if export was successful, False otherwise
        """
        try:
            config_dict = config.to_dict()
            
            output_path_obj = Path(output_path)
            
            with open(output_path_obj, 'w', encoding='utf-8') as f:
                if output_path_obj.suffix == '.json':
                    json.dump(config_dict, f, indent=2)
                else:
                    f.write(self._format_simple_yaml(config_dict))
            
            self.logger.info(f"Configuration exported to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export configuration: {e}")
            return False
    
    def import_config(self, input_path: str) -> Optional[DashboardConfig]:
        """
        Import configuration from a file.
        
        Args:
            input_path: Path to the configuration file
            
        Returns:
            DashboardConfig object if successful, None otherwise
        """
        try:
            input_path_obj = Path(input_path)
            
            if not input_path_obj.exists():
                self.logger.error(f"Config file not found: {input_path}")
                return None
            
            with open(input_path_obj, 'r', encoding='utf-8') as f:
                if input_path_obj.suffix == '.json':
                    config_dict = json.load(f)
                else:
                    content = f.read()
                    try:
                        config_dict = json.loads(content)
                    except json.JSONDecodeError:
                        config_dict = self._parse_simple_yaml(content)
            
            # Validate before importing
            is_valid, errors = self.validate_config_dict(config_dict)
            
            if not is_valid:
                self.logger.error(f"Invalid configuration: {', '.join(errors)}")
                return None
            
            config = DashboardConfig.from_dict(config_dict)
            self.logger.info(f"Configuration imported from {input_path}")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to import configuration: {e}")
            return None
    
    def _parse_simple_yaml(self, content: str) -> Dict[str, Any]:
        """
        Parse simple YAML format (key: value pairs).
        
        Args:
            content: YAML content string
            
        Returns:
            Dictionary of parsed values
        """
        config_dict = {}
        
        for line in content.split('\n'):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Parse key: value
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # Convert value to appropriate type
                if value.lower() == 'true':
                    config_dict[key] = True
                elif value.lower() == 'false':
                    config_dict[key] = False
                elif value.isdigit():
                    config_dict[key] = int(value)
                else:
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    config_dict[key] = value
        
        return config_dict
    
    def _format_simple_yaml(self, config_dict: Dict[str, Any]) -> str:
        """
        Format configuration as simple YAML.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            YAML formatted string
        """
        lines = ["# Dashboard Configuration", ""]
        
        for key, value in config_dict.items():
            if isinstance(value, bool):
                lines.append(f"{key}: {str(value).lower()}")
            elif isinstance(value, str):
                lines.append(f'{key}: "{value}"')
            else:
                lines.append(f"{key}: {value}")
        
        return '\n'.join(lines)
    
    def get_validation_guidance(self, field_name: str) -> str:
        """
        Get validation guidance for a specific configuration field.
        
        Args:
            field_name: Name of the configuration field
            
        Returns:
            Guidance string for the field
        """
        guidance = {
            "ollama_endpoint": "Must be a valid URL (e.g., http://localhost:11434)",
            "ollama_model": "Model name must be at least 2 characters (e.g., llama2, mistral)",
            "auto_refresh_interval": "Must be between 1 and 300 seconds",
            "max_history_items": "Must be between 1 and 1000",
            "default_batch_size": "Must be between 1 and 1000",
            "enable_debug_mode": "Boolean value (true or false)"
        }
        
        return guidance.get(field_name, "No guidance available")
