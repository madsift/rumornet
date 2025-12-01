FROM python:3.10-slim

# Install system packages (optional)
RUN apt-get update && apt-get install -y build-essential

# Create app directory
WORKDIR /app

# Install python deps first (cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app
COPY . .

# Expose port Render will use
EXPOSE 10000

# Streamlit needs these to run in Docker properly
ENV STREAMLIT_SERVER_PORT=10000
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false

# AWS credentials will be passed as environment variables at runtime
# Set these when running: -e AWS_ACCESS_KEY_ID=xxx -e AWS_SECRET_ACCESS_KEY=xxx -e AWS_DEFAULT_REGION=us-east-1

# Command to start Streamlit
CMD ["streamlit", "run", "dashboard.py"]
