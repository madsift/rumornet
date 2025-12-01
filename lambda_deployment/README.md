# Lambda Deployment Guide

This directory contains everything needed to deploy the misinformation detection system to AWS Lambda + API Gateway.

## Architecture

```
Client → API Gateway → Lambda Function → Ollama (self-hosted or API)
```

## Prerequisites

1. **AWS Account** with appropriate permissions
2. **Serverless Framework** installed: `npm install -g serverless`
3. **AWS CLI** configured with credentials
4. **Ollama** accessible via HTTP (can be self-hosted or API)

## Deployment Steps

### Option 1: Using Serverless Framework (Recommended)

```bash
# Install Serverless Framework
npm install -g serverless

# Install plugins
serverless plugin install -n serverless-python-requirements

# Set environment variables
export OLLAMA_ENDPOINT="http://your-ollama-endpoint:11434"
export OLLAMA_MODEL="llama3.2"

# Deploy
cd lambda_deployment
serverless deploy

# Test
serverless invoke -f analyze --data '{"posts": [...]}'
```

### Option 2: Manual Deployment

```bash
# 1. Create deployment package
cd lambda_deployment
pip install -r requirements.txt -t package/
cp ../lambda_handler.py package/
cp -r ../agents package/
cp ../granular_misinformation_orchestrator.py package/
cd package
zip -r ../deployment.zip .

# 2. Upload to Lambda via AWS Console or CLI
aws lambda create-function \
  --function-name misinformation-detection \
  --runtime python3.9 \
  --role arn:aws:iam::YOUR_ACCOUNT:role/lambda-execution-role \
  --handler lambda_handler.lambda_handler \
  --zip-file fileb://../deployment.zip \
  --timeout 300 \
  --memory-size 3008 \
  --environment Variables="{OLLAMA_ENDPOINT=http://your-endpoint:11434,OLLAMA_MODEL=llama3.2}"

# 3. Create API Gateway
# Use AWS Console to create REST API and link to Lambda function
```

## API Endpoints

### POST /analyze

Analyze posts for misinformation.

**Request:**
```json
{
  "posts": [
    {
      "submission_id": "15uzos2",
      "author_name": "user_42486",
      "posts": "Brazil arrests police officials...",
      "score": 25,
      "num_comments": 1,
      "upvote_ratio": 0.91,
      "created_utc": "2023-08-19 07:20:07",
      "subreddit": "GlobalTalk"
    }
  ],
  "format": "reddit",
  "use_batch": true
}
```

**Response:**
```json
{
  "status": "success",
  "execution_time_ms": 1234.56,
  "total_posts": 1,
  "results": [...],
  "report": {...}
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00",
  "orchestrator_initialized": true
}
```

## Testing Locally

```python
# Test the Lambda handler locally
python lambda_handler.py
```

## Environment Variables

- `OLLAMA_ENDPOINT`: Ollama API endpoint (default: http://localhost:11434)
- `OLLAMA_MODEL`: Model to use (default: llama3.2)

## Cost Optimization

- Lambda is billed per request and execution time
- Use batch processing (`use_batch: true`) for better efficiency
- Consider using Lambda provisioned concurrency for consistent performance
- Monitor CloudWatch logs for optimization opportunities

## Monitoring

- CloudWatch Logs: Automatic logging of all requests
- CloudWatch Metrics: Track invocations, errors, duration
- X-Ray: Enable for distributed tracing (optional)

## Scaling

- Lambda automatically scales based on demand
- API Gateway handles rate limiting
- Consider using SQS for async processing of large batches

## Security

- Use API Gateway API keys for authentication
- Enable AWS WAF for DDoS protection
- Use VPC for private Ollama endpoints
- Encrypt environment variables with KMS

## Troubleshooting

### Timeout Issues
- Increase Lambda timeout (max 15 minutes)
- Use batch processing
- Consider async processing with SQS

### Memory Issues
- Increase Lambda memory (max 10GB)
- Optimize model loading
- Use Lambda layers for dependencies

### Cold Start Issues
- Use provisioned concurrency
- Keep Lambda warm with scheduled pings
- Optimize initialization code
