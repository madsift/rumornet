"""
API configuration for dashboard.

This module provides configuration for switching between:
- Direct orchestrator mode (local)
- API mode (FastAPI server mimicking Lambda)
"""

import os
from typing import Optional


class APIConfig:
    """Configuration for API mode."""
    
    # API Mode settings
    USE_API_MODE: bool = os.environ.get('USE_API_MODE', 'true').lower() == 'true'
    API_BASE_URL: str = os.environ.get('API_BASE_URL', 'http://localhost:3000')
    API_TIMEOUT: int = int(os.environ.get('API_TIMEOUT', '1200'))  # 20 minutes for large batches
    
    # Orchestrator settings (for direct mode)
    OLLAMA_ENDPOINT: str = os.environ.get('OLLAMA_ENDPOINT', 'http://192.168.10.68:11434')
    OLLAMA_MODEL: str = os.environ.get('OLLAMA_MODEL', 'gemma3:4b')
    
    @classmethod
    def get_mode_description(cls) -> str:
        """Get description of current mode."""
        if cls.USE_API_MODE:
            return f"API Mode (Server: {cls.API_BASE_URL})"
        else:
            return "Direct Mode (Local Orchestrator)"
    
    @classmethod
    def is_api_mode(cls) -> bool:
        """Check if API mode is enabled."""
        return cls.USE_API_MODE
    
    @classmethod
    def get_api_url(cls) -> str:
        """Get API base URL."""
        return cls.API_BASE_URL
    
    @classmethod
    def set_api_mode(cls, enabled: bool, base_url: Optional[str] = None):
        """
        Set API mode configuration.
        
        Args:
            enabled: Whether to enable API mode
            base_url: Optional API base URL
        """
        cls.USE_API_MODE = enabled
        if base_url:
            cls.API_BASE_URL = base_url


# Default configuration
DEFAULT_CONFIG = APIConfig()
