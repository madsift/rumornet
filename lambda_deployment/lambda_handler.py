"""
AWS Lambda handler for misinformation detection API.

This module provides a Lambda-compatible interface for the orchestrator,
making it easy to deploy to AWS Lambda + API Gateway.

API Endpoints:
- POST /analyze - Analyze posts for misinformation
- GET /health - Health check endpoint
"""

import json
import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime

# Import orchestrator factory (will select sequential or concurrent version)
from orchestrator_factory import create_orchestrator
from utils.data_transformer import transform_posts_batch, enrich_post_metadata

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Global orchestrator instance (reused across Lambda invocations)
_orchestrator = None


def get_orchestrator():
    """
    Get or create orchestrator instance.
    
    Lambda containers are reused, so we cache the orchestrator
    to avoid reinitializing on every request.
    
    Supports multiple LLM providers via environment variables:
    - Ollama (local): LLM_PROVIDER=ollama, OLLAMA_ENDPOINT=..., OLLAMA_MODEL=...
    - Bedrock (AWS): LLM_PROVIDER=bedrock, BEDROCK_REGION=..., BEDROCK_MODEL_ID=...
    """
    global _orchestrator
    
    if _orchestrator is None:
        logger.info("Initializing orchestrator...")
        
        # Get configuration from environment variables
        import os
        
        # Determine provider (default to ollama for backward compatibility)
        llm_provider = os.environ.get('LLM_PROVIDER', 'ollama').lower()
        
        # Build config based on provider
        config = {
            'llm_provider': llm_provider
        }
        
        if llm_provider == 'ollama':
            # Ollama configuration (local development)
            config.update({
                'ollama_endpoint': os.environ.get('OLLAMA_ENDPOINT', 'http://192.168.10.68:11434'),
                'ollama_model': os.environ.get('OLLAMA_MODEL', 'gemma3:4b'),
                'embedding_provider': 'ollama',
                'embedding_endpoint': os.environ.get('OLLAMA_ENDPOINT', 'http://192.168.10.68:11434'),
                'embedding_model': os.environ.get('EMBEDDING_MODEL', 'all-minilm:22m')
            })
            logger.info(f"Using Ollama: {config['ollama_model']} at {config['ollama_endpoint']}")
            
        elif llm_provider == 'bedrock':
            # AWS Bedrock configuration (production)
            config.update({
                'bedrock_region': os.environ.get('BEDROCK_REGION', 'us-east-1'),
                'bedrock_model_id': os.environ.get('BEDROCK_MODEL_ID', 'anthropic.claude-v2'),
                'embedding_provider': 'bedrock',
                'bedrock_embedding_model_id': os.environ.get('BEDROCK_EMBEDDING_MODEL_ID', 'amazon.titan-embed-text-v1')
            })
            logger.info(f"Using Bedrock: {config['bedrock_model_id']} in {config['bedrock_region']}")
            
        elif llm_provider == 'openai':
            # OpenAI configuration (alternative)
            config.update({
                'openai_api_key': os.environ.get('OPENAI_API_KEY'),
                'openai_model': os.environ.get('OPENAI_MODEL', 'gpt-4'),
                'embedding_provider': 'openai',
                'openai_embedding_model': os.environ.get('OPENAI_EMBEDDING_MODEL', 'text-embedding-ada-002')
            })
            logger.info(f"Using OpenAI: {config['openai_model']}")
            
        else:
            logger.warning(f"Unknown LLM provider: {llm_provider}, defaulting to Ollama")
            config.update({
                'llm_provider': 'ollama',
                'ollama_endpoint': 'http://192.168.10.68:11434',
                'ollama_model': 'gemma3:4b'
            })
        
        # Use factory to select appropriate orchestrator version
        from orchestrator_factory import create_orchestrator
        _orchestrator = create_orchestrator(config=config)
        
        # Initialize agents
        try:
            asyncio.run(_orchestrator.initialize_agents())
            logger.info("Orchestrator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize agents: {e}")
            # Continue anyway - some agents may work
    
    return _orchestrator


def create_response(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create API Gateway response.
    
    Args:
        status_code: HTTP status code
        body: Response body dictionary
        
    Returns:
        API Gateway response format
    """
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',  # Enable CORS
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Methods': 'GET,POST,OPTIONS'
        },
        'body': json.dumps(body, default=str)  # default=str handles datetime
    }


def handle_analyze(event_body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle POST /analyze request.
    
    Request body:
    {
        "posts": [...],  // Array of posts (optional if demo_file provided)
        "demo_file": "demo1.json",  // Load demo from filesystem (optional)
        "format": "reddit" | "standard" | "auto",  // Optional
        "use_batch": true | false  // Optional, default true
    }
    
    Response:
    {
        "status": "success" | "error",
        "execution_time_ms": 1234.56,
        "results": [...],
        "report": {...},
        "error": "error message"  // Only if status is "error"
    }
    """
    try:
        # Extract parameters
        posts = event_body.get('posts', [])
        demo_file = event_body.get('demo_file')
        format_type = event_body.get('format', 'auto')
        use_batch = event_body.get('use_batch', True)
        
        # Load posts from demo file OR use provided posts
        if demo_file:
            logger.info(f"Loading demo file: {demo_file}")
            
            import os
            
            # Try multiple possible paths
            possible_paths = [
                demo_file,
                os.path.join("examples", demo_file)
            ]
            
            demo_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    demo_path = path
                    break
            
            if not demo_path:
                return create_response(404, {
                    'status': 'error',
                    'error': f'Demo file not found: {demo_file}. Tried: {possible_paths}'
                })
            
            # Load demo file
            with open(demo_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract posts from nested structure
            from utils.data_transformer import extract_posts_from_nested_structure
            posts = extract_posts_from_nested_structure(data)
            
            if not posts:
                return create_response(400, {
                    'status': 'error',
                    'error': f'Could not extract posts from demo file: {demo_file}'
                })
            
            logger.info(f"Loaded {len(posts)} posts from {demo_path}")
        
        elif not posts:
            return create_response(400, {
                'status': 'error',
                'error': 'Either "posts" or "demo_file" must be provided'
            })
        
        logger.info(f"Analyzing {len(posts)} posts (format={format_type}, batch={use_batch})")
        
        # Transform posts to standard format
        transformed_posts = transform_posts_batch(posts, format_type=format_type)
        transformed_posts = [enrich_post_metadata(post) for post in transformed_posts]
        
        # Get orchestrator
        orchestrator = get_orchestrator()
        
        # Run analysis
        start_time = datetime.now()
        
        if use_batch:
            results = asyncio.run(orchestrator.analyze_batch_true_batch(transformed_posts))
        else:
            results = []
            for post in transformed_posts:
                result = asyncio.run(orchestrator.analyze_post_with_metadata(post))
                results.append(result)
        
        # Generate report
        report = orchestrator.generate_actionable_report()
        
        execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Get agent statuses
        agent_statuses = {}
        if hasattr(orchestrator, 'agents') and orchestrator.agents:
            for agent_name, agent in orchestrator.agents.items():
                agent_statuses[agent_name] = {
                    'status': 'completed',
                    'execution_time_ms': 0  # Would need to track per-agent timing
                }
        
        logger.info(f"Analysis completed in {execution_time_ms:.2f}ms")
        
        return create_response(200, {
            'status': 'success',
            'execution_time_ms': execution_time_ms,
            'total_posts': len(posts),
            'results': results,
            'report': report,
            'agent_statuses': agent_statuses
        })
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return create_response(500, {
            'status': 'error',
            'error': str(e)
        })


def handle_health(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle GET /health request.
    
    Response:
    {
        "status": "healthy",
        "timestamp": "2024-01-01T12:00:00",
        "orchestrator_initialized": true | false
    }
    """
    return create_response(200, {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'orchestrator_initialized': _orchestrator is not None
    })


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler function.
    
    This is the main entry point for Lambda invocations.
    
    Args:
        event: API Gateway event
        context: Lambda context
        
    Returns:
        API Gateway response
    """
    try:
        # Log request
        logger.info(f"Received request: {event.get('httpMethod')} {event.get('path')}")
        
        # Handle OPTIONS for CORS preflight
        if event.get('httpMethod') == 'OPTIONS':
            return create_response(200, {'message': 'OK'})
        
        # Route based on path and method
        path = event.get('path', '/')
        method = event.get('httpMethod', 'GET')
        
        if path == '/health' and method == 'GET':
            return handle_health(event)
        
        elif path == '/analyze' and method == 'POST':
            # Parse body
            body = event.get('body', '{}')
            if isinstance(body, str):
                body = json.loads(body)
            
            return handle_analyze(body)
        
        else:
            return create_response(404, {
                'status': 'error',
                'error': f'Not found: {method} {path}'
            })
    
    except Exception as e:
        logger.error(f"Handler error: {e}", exc_info=True)
        return create_response(500, {
            'status': 'error',
            'error': 'Internal server error'
        })


# For local testing
if __name__ == "__main__":
    # Test event
    test_event = {
        'httpMethod': 'POST',
        'path': '/analyze',
        'body': json.dumps({
            'posts': [
                {
                    "submission_id": "15uzos2",
                    "author_name": "user_42486",
                    "posts": "Brazil arrests police officials over January 8 attacks",
                    "score": 25,
                    "num_comments": 1,
                    "upvote_ratio": 0.91,
                    "created_utc": "2023-08-19 07:20:07",
                    "subreddit": "GlobalTalk"
                }
            ],
            'format': 'reddit',
            'use_batch': True
        })
    }
    
    # Test locally
    response = lambda_handler(test_event, None)
    print(json.dumps(response, indent=2))
