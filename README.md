# RumorNet: AI-Powered Misinformation Detection System

A production-ready misinformation detection system using AWS Lambda, Bedrock AI, and Streamlit dashboard.

## 🎯 Features

- **Multi-Agent AI Pipeline**: 4 specialized agents for comprehensive analysis
  - Multilingual reasoning agent
  - Pattern detection agent  
  - Evidence gathering agent
  - Social behavior analysis agent

- **Concurrent Processing**: Analyzes 132 posts in ~3-4 minutes using TRUE BATCH pattern
- **AWS Lambda Deployment**: Serverless, scalable architecture
- **S3 History**: Automatic storage and retrieval of all analysis results
- **Interactive Dashboard**: Real-time monitoring and visualization
- **Markdown Reports**: Exportable analysis reports

## 🏗️ Architecture

```
User → Streamlit Dashboard → API Gateway → Lambda Function
                                              ↓
                                         Bedrock AI
                                              ↓
                                         S3 Storage
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- AWS Account with Bedrock access
- Docker (optional, for containerized deployment)

### Local Development

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd rumors
```

2. **Set up AWS credentials**
```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1
```

3. **Configure authentication**
```bash
cd dashboard/.streamlit
cp secrets.toml.example secrets.toml
# Edit secrets.toml with your username/password
```

4. **Run the dashboard**
```bash
cd dashboard
pip install -r requirements.txt
streamlit run dashboard.py
```

### Docker Deployment

#### Option 1: Build and Run Locally

1. **Build the image**
```bash
cd dashboard
docker build -t rumornet-dashboard .
```

2. **Run with environment variables** (recommended)
```bash
docker run -p 8501:10000 \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  -e AWS_DEFAULT_REGION=us-east-1 \
  -e AUTH_USERNAME=admin \
  -e AUTH_PASSWORD=your_secure_password \
  --rm rumornet-dashboard
```

#### Option 2: Pull from Docker Hub

1. **Pull the image**
```bash
docker pull yourusername/rumornet-dashboard:latest
```

2. **Run with credentials**
```bash
docker run -p 8501:10000 \
  -e AWS_ACCESS_KEY_ID=your_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret \
  -e AWS_DEFAULT_REGION=us-east-1 \
  -e AUTH_USERNAME=admin \
  -e AUTH_PASSWORD=your_secure_password \
  --rm yourusername/rumornet-dashboard:latest
```

3. **Access the dashboard**
```
http://localhost:8501
```

#### Option 3: Docker Compose (Easiest)

1. **Create .env file**
```bash
cd dashboard
cp .env.example .env
# Edit .env with your credentials
```

2. **Run with docker-compose**
```bash
docker-compose up -d
```

3. **View logs**
```bash
docker-compose logs -f
```

4. **Stop**
```bash
docker-compose down
```

#### Publishing to Docker Hub

```bash
# Tag the image
docker tag rumornet-dashboard yourusername/rumornet-dashboard:latest

# Login to Docker Hub
docker login

# Push to Docker Hub
docker push yourusername/rumornet-dashboard:latest
```

## 📊 Dashboard Features

### Overview Tab
- Executive summary with key metrics
- Full markdown report (auto-generated)
- Download button for reports

### Analysis Tab
- One-click analysis of demo posts
- Real-time progress tracking
- Automatic S3 polling for results

### Results Tab
- Detailed analysis breakdown
- High-priority posts
- Pattern detection results
- Topic analysis

### History Tab
- All past executions (local + S3)
- Load and view any previous analysis
- Markdown generation for S3 results

## 🔧 Configuration

### Environment Variables

The dashboard supports the following environment variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `AWS_ACCESS_KEY_ID` | Yes | AWS access key for S3 and Bedrock |
| `AWS_SECRET_ACCESS_KEY` | Yes | AWS secret key |
| `AWS_DEFAULT_REGION` | Yes | AWS region (default: us-east-1) |
| `AUTH_USERNAME` | Yes* | Dashboard login username |
| `AUTH_PASSWORD` | Yes* | Dashboard login password |

*Can also be configured in `.streamlit/secrets.toml` for local development

### Lambda Function

Lambda Deployment is done

**Timeout Handling**: API Gateway has a 29-second timeout. For longer analyses, the dashboard automatically polls S3 for results.

### S3 Storage

Results are stored in:
```
s3://rumornet/misinformation-detection/reports/
```

### Bedrock Models

- **Reasoning**: Claude 3.5 Sonnet
- **Pattern Detection**: Claude 3 Haiku
- **Evidence**: Claude 3 Haiku
- **Social Analysis**: Claude 3 Haiku

## 🛠️ Development

### Project Structure

```
rumors/
├── dashboard/                 # Streamlit dashboard
│   ├── components/           # UI components
│   ├── core/                 # Core functionality
│   ├── models/               # Data models
│   ├── utils/                # Utilities
│   └── dashboard.py          # Main app
├── lambda_deployment/        # Lambda function
│   ├── agents/              # AI agents
│   ├── utils/               # Utilities
│   └── lambda_handler_async.py
└── README.md
```

### Key Files

- `dashboard/components/batch_analysis_api.py` - Lambda API integration
- `dashboard/core/data_manager.py` - S3 and local storage
- `lambda_deployment/lambda_handler_async.py` - Lambda entry point
- `lambda_deployment/granular_misinformation_orchestrator_concurrent.py` - Concurrent processing

## 🔐 Security

- AWS credentials via environment variables
- IAM roles for Lambda execution
- S3 bucket policies for data access
- No credentials in code or version control

## 📈 Performance

- **Processing Time**: ~220 seconds for 132 posts
- **Concurrency**: 5 parallel chunks
- **Memory**: 481 MB max in Lambda
- **Patterns Detected**: 70+ per analysis


## 📧 Contact

[madsift@gmail.com]

---
