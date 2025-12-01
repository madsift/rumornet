# Lambda Container Image for RumorNet Misinformation Detection
# Optimized for AWS Bedrock with async processing

FROM public.ecr.aws/lambda/python:3.11

# Set working directory
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY lambda_handler_async.py .
COPY orchestrator_factory.py .
COPY granular_misinformation_orchestrator_concurrent.py .
COPY true_batch_pattern_detection.py .
COPY true_batch_reasoning.py .
COPY true_batch_topic_modeling.py .

# Copy agents folder
COPY agents/ ./agents/

# Copy utils folder
COPY utils/ ./utils/

# Copy demo data
COPY demo1.json .

# Set Lambda handler
CMD ["lambda_handler_async.lambda_handler"]
