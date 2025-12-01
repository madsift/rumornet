"""
AWS Lambda handler with ASYNC processing for misinformation detection API.

This version returns immediately to avoid API Gateway 29-second timeout,
then continues processing in the background.

API Endpoints:
- POST /analyze - Start async analysis (returns immediately with job_id)
- GET /status/{job_id} - Check analysis status
- GET /results/{job_id} - Get analysis results
- GET /health - Health check endpoint
"""

import json
import asyncio
import logging
import uuid
import os
from typing import Dict, Any
from datetime import datetime
import boto3

# Import orchestrator factory
from orchestrator_factory import create_orchestrator
from utils.data_transformer import transform_posts_batch, enrich_post_metadata

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Global orchestrator instance (reused across Lambda invocations)
_orchestrator = None

# S3 client for storing results
s3_client = boto3.client('s3')
RESULTS_BUCKET = os.environ.get('RESULTS_BUCKET', 'rumornet')


async def get_orchestrator():
    """
    Get or create orchestrator instance.
    
    Lambda containers are reused, so we cache the orchestrator
    to avoid reinitializing on every request (warm start optimization).
    """
    global _orchestrator
    
    if _orchestrator is None:
        logger.info("🔧 Initializing orchestrator (cold start)...")
        
        import os
        
        # Get provider from environment
        llm_provider = os.environ.get('LLM_PROVIDER', 'bedrock').lower()
        
        # Build config
        config = {'llm_provider': llm_provider}
        
        if llm_provider == 'bedrock':
            config.update({
                'bedrock_region': os.environ.get('BEDROCK_REGION', 'us-east-1'),
                'bedrock_model_id': os.environ.get('BEDROCK_MODEL_ID', 'us.anthropic.claude-3-5-haiku-20241022-v1:0'),
                'embedding_provider': 'bedrock',
                'bedrock_embedding_model_id': os.environ.get('BEDROCK_EMBEDDING_MODEL_ID', 'cohere.embed-v4:0')
            })
            logger.info(f"☁️ Using Bedrock: {config['bedrock_model_id']}")
        
        # Create orchestrator
        _orchestrator = create_orchestrator(config=config)
        
        # Initialize agents asynchronously
        try:
            await _orchestrator.initialize_agents()
            logger.info("✅ Orchestrator initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize agents: {e}")
    
    return _orchestrator


def create_response(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    """Create API Gateway response with CORS headers."""
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Methods': 'GET,POST,OPTIONS'
        },
        'body': json.dumps(body, default=str)
    }


def store_job_status(job_id: str, status: str, data: Dict[str, Any] = None):
    """Store job status in S3 with timestamp."""
    try:
        timestamp = datetime.now().isoformat()
        status_data = {
            'job_id': job_id,
            'status': status,
            'timestamp': timestamp,
            'data': data or {}
        }
        
        # Store in rumornet bucket under misinformation-detection/jobs/
        s3_client.put_object(
            Bucket=RESULTS_BUCKET,
            Key=f'misinformation-detection/jobs/{job_id}/status.json',
            Body=json.dumps(status_data),
            ContentType='application/json',
            Metadata={
                'job-id': job_id,
                'status': status,
                'timestamp': timestamp
            }
        )
    except Exception as e:
        logger.error(f"Failed to store job status: {e}")


def store_job_results(job_id: str, results: Dict[str, Any]):
    """Store job results in S3 with timestamp."""
    try:
        timestamp = datetime.now().isoformat()
        
        # Add timestamp to results
        results['stored_at'] = timestamp
        results['job_id'] = job_id
        
        # Store in rumornet bucket under misinformation-detection/jobs/
        s3_client.put_object(
            Bucket=RESULTS_BUCKET,
            Key=f'misinformation-detection/jobs/{job_id}/results.json',
            Body=json.dumps(results, default=str),
            ContentType='application/json',
            Metadata={
                'job-id': job_id,
                'timestamp': timestamp,
                'total-posts': str(results.get('total_posts', 0))
            }
        )
        
        # Also store a timestamped copy in reports folder for archival
        timestamp_safe = timestamp.replace(':', '-').replace('.', '-')
        s3_client.put_object(
            Bucket=RESULTS_BUCKET,
            Key=f'misinformation-detection/reports/{timestamp_safe}_{job_id}.json',
            Body=json.dumps(results, default=str),
            ContentType='application/json',
            Metadata={
                'job-id': job_id,
                'timestamp': timestamp,
                'total-posts': str(results.get('total_posts', 0))
            }
        )
        
        logger.info(f"Stored results for job {job_id} at {timestamp}")
    except Exception as e:
        logger.error(f"Failed to store job results: {e}")


def get_job_status(job_id: str) -> Dict[str, Any]:
    """Get job status from S3."""
    try:
        response = s3_client.get_object(
            Bucket=RESULTS_BUCKET,
            Key=f'misinformation-detection/jobs/{job_id}/status.json'
        )
        return json.loads(response['Body'].read())
    except s3_client.exceptions.NoSuchKey:
        return None
    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        return None


def get_job_results(job_id: str) -> Dict[str, Any]:
    """Get job results from S3."""
    try:
        response = s3_client.get_object(
            Bucket=RESULTS_BUCKET,
            Key=f'misinformation-detection/jobs/{job_id}/results.json'
        )
        return json.loads(response['Body'].read())
    except s3_client.exceptions.NoSuchKey:
        return None
    except Exception as e:
        logger.error(f"Failed to get job results: {e}")
        return None


async def process_analysis_async(job_id: str, posts: list, use_batch: bool = True):
    """
    Process analysis asynchronously in background.
    
    This runs after the initial response is sent.
    """
    try:
        logger.info(f"🚀 Starting async analysis for job {job_id}")
        
        # Update status to processing
        store_job_status(job_id, 'processing', {
            'total_posts': len(posts),
            'started_at': datetime.now().isoformat()
        })
        
        # Get orchestrator
        orchestrator = await get_orchestrator()
        
        # Run analysis
        start_time = datetime.now()
        
        if use_batch:
            results = await orchestrator.analyze_batch_true_batch(posts)
        else:
            results = []
            for post in posts:
                result = await orchestrator.analyze_post_with_metadata(post)
                results.append(result)
        
        # Generate report
        report = orchestrator.generate_actionable_report()
        
        execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Store results
        final_results = {
            'status': 'completed',
            'execution_time_ms': execution_time_ms,
            'total_posts': len(posts),
            'results': results,
            'report': report,
            'completed_at': datetime.now().isoformat()
        }
        
        store_job_results(job_id, final_results)
        store_job_status(job_id, 'completed', {
            'execution_time_ms': execution_time_ms,
            'total_posts': len(posts)
        })
        
        logger.info(f"✅ Analysis completed for job {job_id} in {execution_time_ms:.2f}ms")
        
    except Exception as e:
        logger.error(f"❌ Analysis failed for job {job_id}: {e}", exc_info=True)
        store_job_status(job_id, 'failed', {'error': str(e)})


def handle_analyze(event_body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle POST /analyze request - ASYNC VERSION.
    
    Returns immediately with job_id, processing continues in background.
    
    Request body:
    {
        "posts": [...],
        "demo_file": "demo1.json",  // Optional
        "format": "reddit" | "standard" | "auto",
        "use_batch": true | false
    }
    
    Response (immediate):
    {
        "status": "accepted",
        "job_id": "uuid",
        "message": "Analysis started",
        "status_url": "/status/{job_id}",
        "results_url": "/results/{job_id}"
    }
    """
    try:
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Extract parameters
        posts = event_body.get('posts', [])
        demo_file = event_body.get('demo_file')
        format_type = event_body.get('format', 'auto')
        use_batch = event_body.get('use_batch', True)
        
        # Load posts from demo file OR use provided posts
        if demo_file:
            logger.info(f"Loading demo file: {demo_file}")
            
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
                    'error': f'Demo file not found: {demo_file}'
                })
            
            with open(demo_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            from utils.data_transformer import extract_posts_from_nested_structure
            posts = extract_posts_from_nested_structure(data)
            
            if not posts:
                return create_response(400, {
                    'status': 'error',
                    'error': f'Could not extract posts from demo file'
                })
        
        elif not posts:
            return create_response(400, {
                'status': 'error',
                'error': 'Either "posts" or "demo_file" must be provided'
            })
        
        logger.info(f"📝 Job {job_id}: Analyzing {len(posts)} posts")
        
        # Transform posts
        transformed_posts = transform_posts_batch(posts, format_type=format_type)
        transformed_posts = [enrich_post_metadata(post) for post in transformed_posts]
        
        # Store initial status
        store_job_status(job_id, 'processing', {
            'total_posts': len(posts),
            'created_at': datetime.now().isoformat()
        })
        
        # Run processing synchronously and return results
        asyncio.run(process_analysis_async(job_id, transformed_posts, use_batch))
        
        # Get and return results
        results = get_job_results(job_id)
        if results:
            return create_response(200, results)
        else:
            return create_response(500, {
                'status': 'error',
                'error': 'Analysis completed but results not found'
            })
        
    except Exception as e:
        logger.error(f"Failed to start analysis: {e}", exc_info=True)
        return create_response(500, {
            'status': 'error',
            'error': str(e)
        })


def handle_status(job_id: str) -> Dict[str, Any]:
    """
    Handle GET /status/{job_id} request.
    
    Response:
    {
        "job_id": "uuid",
        "status": "queued" | "processing" | "completed" | "failed",
        "timestamp": "2024-01-01T12:00:00",
        "data": {...}
    }
    """
    status = get_job_status(job_id)
    
    if not status:
        return create_response(404, {
            'status': 'error',
            'error': f'Job not found: {job_id}'
        })
    
    return create_response(200, status)


def handle_results(job_id: str) -> Dict[str, Any]:
    """
    Handle GET /results/{job_id} request.
    
    Response:
    {
        "status": "completed",
        "execution_time_ms": 1234.56,
        "results": [...],
        "report": {...}
    }
    """
    # Check status first
    status = get_job_status(job_id)
    
    if not status:
        return create_response(404, {
            'status': 'error',
            'error': f'Job not found: {job_id}'
        })
    
    if status['status'] != 'completed':
        return create_response(202, {
            'status': status['status'],
            'message': f'Job is {status["status"]}, results not ready yet',
            'job_id': job_id
        })
    
    # Get results
    results = get_job_results(job_id)
    
    if not results:
        return create_response(404, {
            'status': 'error',
            'error': f'Results not found for job: {job_id}'
        })
    
    return create_response(200, results)


def handle_health(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle GET /health request.
    
    Response:
    {
        "status": "healthy",
        "timestamp": "2024-01-01T12:00:00",
        "orchestrator_ready": true,
        "agents_count": 10
    }
    """
    orchestrator_ready = _orchestrator is not None
    agents_count = len(_orchestrator.agents) if orchestrator_ready else 0
    
    return create_response(200, {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'orchestrator_ready': orchestrator_ready,
        'agents_count': agents_count,
        'provider': os.environ.get('LLM_PROVIDER', 'bedrock')
    })


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler function.
    
    Routes:
    - POST /analyze -> Analyze posts (synchronous, returns results)
    - GET /status/{job_id} -> Check status
    - GET /results/{job_id} -> Get results
    - GET /health -> Health check
    """
    try:
        # Log request
        method = event.get('httpMethod', 'GET')
        path = event.get('path', '/')
        logger.info(f"📨 {method} {path}")
        
        # Handle OPTIONS for CORS
        if method == 'OPTIONS':
            return create_response(200, {'message': 'OK'})
        
        # Route based on path
        if path == '/health' and method == 'GET':
            return handle_health(event)
        
        elif path == '/analyze' and method == 'POST':
            body = event.get('body', '{}')
            if isinstance(body, str):
                body = json.loads(body)
            return handle_analyze(body)
        
        elif path.startswith('/status/') and method == 'GET':
            job_id = path.split('/')[-1]
            return handle_status(job_id)
        
        elif path.startswith('/results/') and method == 'GET':
            job_id = path.split('/')[-1]
            return handle_results(job_id)
        
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
