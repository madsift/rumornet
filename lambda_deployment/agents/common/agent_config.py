"""
Centralized Agent Configuration

Provides factory functions for LLM and embedding model initialization.
Supports multiple providers (Ollama, Bedrock, etc.) through simple configuration.

Usage:
    from agents.common import get_llm_agent, get_embedding_model

    # Get LLM agent
    llm_agent = get_llm_agent(config)

    # Get embedding model
    embedding_model = get_embedding_model(config)
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for agent initialization."""

    # LLM Configuration
    llm_provider: str = "ollama"  # Options: "ollama", "bedrock", "openai"
    ollama_endpoint: str = "http://192.168.10.68:11434"
    ollama_model: str = "gemma3:4b"

    # Embedding Configuration
    embedding_provider: str = "ollama"  # Options: "ollama", "bedrock", "openai"
    embedding_endpoint: str = "http://192.168.10.68:11434"
    embedding_model: str = "all-minilm:22m"

    # Bedrock Configuration (if using AWS Bedrock)
    bedrock_region: str = "us-east-1"
    bedrock_model_id: str = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
    bedrock_embedding_model_id: str = "cohere.embed-v4:0"

    # OpenAI Configuration (if using OpenAI)
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4"
    openai_embedding_model: str = "text-embedding-ada-002"

    # General settings
    timeout: int = 60
    max_retries: int = 3

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AgentConfig':
        """Create AgentConfig from dictionary."""
        return cls(
            llm_provider=config_dict.get('llm_provider', 'ollama'),
            ollama_endpoint=config_dict.get('ollama_endpoint', 'http://192.168.10.68:11434'),
            ollama_model=config_dict.get('ollama_model', 'gemma3:4b'),
            embedding_provider=config_dict.get('embedding_provider', 'ollama'),
            embedding_endpoint=config_dict.get('embedding_endpoint', 'http://192.168.10.68:11434'),
            embedding_model=config_dict.get('embedding_model', 'all-minilm:22m'),
            bedrock_region=config_dict.get('bedrock_region', 'us-east-1'),
            bedrock_model_id=config_dict.get('bedrock_model_id', 'us.anthropic.claude-3-5-haiku-20241022-v1:0'),
            bedrock_embedding_model_id=config_dict.get('bedrock_embedding_model_id', 'cohere.embed-v4:0'),
            openai_api_key=config_dict.get('openai_api_key'),
            openai_model=config_dict.get('openai_model', 'gpt-4'),
            openai_embedding_model=config_dict.get('openai_embedding_model', 'text-embedding-ada-002'),
            timeout=config_dict.get('timeout', 60),
            max_retries=config_dict.get('max_retries', 3)
        )


def get_llm_agent(config: AgentConfig):
    """
    Get LLM agent based on configuration.

    Args:
        config: AgentConfig with provider settings

    Returns:
        Initialized LLM agent (Strands Agent or equivalent)
    """
    try:
        if config.llm_provider == "ollama":
            raise ValueError("Ollama provider not supported in this deployment. Use 'bedrock' instead.")

        elif config.llm_provider == "bedrock":
            from strands import Agent
            from strands.models import BedrockModel

            bedrock_model = BedrockModel(
                model_id=config.bedrock_model_id,
                temperature=0.3,
                streaming=False,
                region_name=config.bedrock_region
            )

            agent = Agent(model=bedrock_model)
            logger.info(f"Initialized Bedrock LLM: {config.bedrock_model_id} in {config.bedrock_region}")
            return agent

        elif config.llm_provider == "openai":
            # Placeholder for OpenAI implementation
            logger.warning("OpenAI LLM not yet implemented")
            return None

        else:
            logger.error(f"Unknown LLM provider: {config.llm_provider}")
            return None

    except Exception as e:
        logger.error(f"Failed to initialize LLM agent: {e}")
        return None


def get_embedding_model(config: AgentConfig):
    """
    Get embedding model based on configuration.

    Args:
        config: AgentConfig with provider settings

    Returns:
        Initialized embedding model
    """
    try:
        if config.embedding_provider == "ollama":
            # Import OllamaEmbeddings from true_batch_topic_modeling
            import sys
            import os
            # Add parent directory to path
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            from true_batch_topic_modeling import OllamaEmbeddings

            embedding_model = OllamaEmbeddings(
                endpoint=config.embedding_endpoint,
                model=config.embedding_model
            )

            logger.info(f"Initialized Ollama embeddings: {config.embedding_model} at {config.embedding_endpoint}")
            return embedding_model

        elif config.embedding_provider == "bedrock":
            # Import Bedrock embeddings wrapper
            from agents.common.bedrock_embeddings import BedrockEmbeddings
            
            embedding_model = BedrockEmbeddings(
                model_id=config.bedrock_embedding_model_id,
                region=config.bedrock_region
            )
            
            logger.info(f"Initialized Bedrock embeddings: {config.bedrock_embedding_model_id} in {config.bedrock_region}")
            return embedding_model

        elif config.embedding_provider == "openai":
            # Placeholder for OpenAI implementation
            logger.warning("OpenAI embeddings not yet implemented")
            return None

        else:
            logger.error(f"Unknown embedding provider: {config.embedding_provider}")
            return None

    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {e}")
        return None


# Convenience function for simple config dict
def create_default_config(
    ollama_endpoint: str = "http://192.168.10.68:11434",
    ollama_model: str = "gemma3:4b",
    embedding_model: str = "all-minilm:22m"
) -> AgentConfig:
    """
    Create default Ollama configuration.

    Args:
        ollama_endpoint: Ollama server endpoint
        ollama_model: LLM model name
        embedding_model: Embedding model name

    Returns:
        AgentConfig with Ollama settings
    """
    return AgentConfig(
        llm_provider="ollama",
        ollama_endpoint=ollama_endpoint,
        ollama_model=ollama_model,
        embedding_provider="ollama",
        embedding_endpoint=ollama_endpoint,
        embedding_model=embedding_model
    )

def create_bedrock_config(
    bedrock_model_id: str = "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    bedrock_embedding_model_id: str = "cohere.embed-v4:0",
    bedrock_region: str = "us-east-1"
) -> AgentConfig:
    """
    Create default bedrock configuration.

    Args:
        bedrock_model_id: LLM model name
        bedrock_embedding_model_id: Embedding model name
        bedrock_region: AWS region for Bedrock

    Returns:
        AgentConfig with Bedrock settings
    """
    return AgentConfig(
        llm_provider="bedrock",
        bedrock_model_id=bedrock_model_id,
        bedrock_embedding_model_id=bedrock_embedding_model_id,
        bedrock_region=bedrock_region,
        embedding_provider="bedrock"
    )
