"""
Orchestrator Factory

Provides a factory function to select between sequential and concurrent orchestrators
based on configuration or environment variables.

Usage:
    from orchestrator_factory import get_orchestrator_class
    
    # Get appropriate orchestrator class
    OrchestratorClass = get_orchestrator_class()
    
    # Create instance
    orchestrator = OrchestratorClass(config=config)
    await orchestrator.initialize_agents()
"""

import os
import logging

logger = logging.getLogger(__name__)


def get_orchestrator_class(use_concurrent: bool = None):
    """
    Get the appropriate orchestrator class based on configuration.
    
    Args:
        use_concurrent: If True, use concurrent version. If False, use sequential.
                       If None, determine from environment or provider.
    
    Returns:
        GranularMisinformationOrchestrator class (sequential or concurrent)
    
    Decision Logic:
    1. If use_concurrent is explicitly set, use that
    2. If USE_CONCURRENT env var is set, use that
    3. If LLM_PROVIDER is 'bedrock' or 'openai', default to concurrent
    4. If LLM_PROVIDER is 'ollama', default to sequential
    5. Otherwise, default to sequential (safe choice)
    """
    # Check explicit parameter
    if use_concurrent is not None:
        if use_concurrent:
            logger.info("🚀 Using CONCURRENT orchestrator (explicit)")
            from granular_misinformation_orchestrator_concurrent import GranularMisinformationOrchestrator
            return GranularMisinformationOrchestrator
        else:
            logger.info("🔄 Using SEQUENTIAL orchestrator (explicit)")
            from granular_misinformation_orchestrator import GranularMisinformationOrchestrator
            return GranularMisinformationOrchestrator
    
    # Check environment variable
    use_concurrent_env = os.environ.get('USE_CONCURRENT', '').lower()
    if use_concurrent_env in ('true', '1', 'yes'):
        logger.info("🚀 Using CONCURRENT orchestrator (USE_CONCURRENT=true)")
        from granular_misinformation_orchestrator_concurrent import GranularMisinformationOrchestrator
        return GranularMisinformationOrchestrator
    elif use_concurrent_env in ('false', '0', 'no'):
        logger.info("🔄 Using SEQUENTIAL orchestrator (USE_CONCURRENT=false)")
        from granular_misinformation_orchestrator import GranularMisinformationOrchestrator
        return GranularMisinformationOrchestrator
    
    # Auto-detect based on provider
    llm_provider = os.environ.get('LLM_PROVIDER', 'ollama').lower()
    
    if llm_provider in ('bedrock', 'openai'):
        # Bedrock and OpenAI support concurrent requests
        logger.info(f"🚀 Using CONCURRENT orchestrator (provider={llm_provider} supports concurrency)")
        from granular_misinformation_orchestrator_concurrent import GranularMisinformationOrchestrator
        return GranularMisinformationOrchestrator
    else:
        # Ollama or unknown provider - use sequential (safe default)
        logger.info(f"🔄 Using SEQUENTIAL orchestrator (provider={llm_provider}, single GPU assumed)")
        from granular_misinformation_orchestrator import GranularMisinformationOrchestrator
        return GranularMisinformationOrchestrator


def create_orchestrator(config: dict = None, use_concurrent: bool = None):
    """
    Create an orchestrator instance with the appropriate version.
    
    Args:
        config: Configuration dictionary
        use_concurrent: If True, use concurrent version. If False, use sequential.
                       If None, auto-detect based on provider.
    
    Returns:
        Initialized orchestrator instance (not yet initialized agents)
    
    Example:
        # Auto-detect based on provider
        orchestrator = create_orchestrator(config={'llm_provider': 'bedrock', ...})
        
        # Explicit concurrent
        orchestrator = create_orchestrator(config=config, use_concurrent=True)
        
        # Explicit sequential
        orchestrator = create_orchestrator(config=config, use_concurrent=False)
    """
    OrchestratorClass = get_orchestrator_class(use_concurrent=use_concurrent)
    return OrchestratorClass(config=config)


# Convenience functions
def create_sequential_orchestrator(config: dict = None):
    """Create a sequential orchestrator (for Ollama/single GPU)."""
    logger.info("🔄 Creating SEQUENTIAL orchestrator")
    from granular_misinformation_orchestrator import GranularMisinformationOrchestrator
    return GranularMisinformationOrchestrator(config=config)


def create_concurrent_orchestrator(config: dict = None):
    """Create a concurrent orchestrator (for Bedrock/OpenAI)."""
    logger.info("🚀 Creating CONCURRENT orchestrator")
    from granular_misinformation_orchestrator_concurrent import GranularMisinformationOrchestrator
    return GranularMisinformationOrchestrator(config=config)
