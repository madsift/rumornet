# RumorNet: Multi-Agent Misinformation Detection System

## Technical Architecture & Implementation

**Author**: Saket Kunwar 
**Date**: November 2025  
**Status**: Production Deployment on AWS Lambda + Streamlit Dashboard

---

## Executive Summary

RumorNet is a serverless, multi-agent AI system designed to detect and analyze misinformation at scale with a groundbreaking focus on **multilingual, culturally-aware analysis**. Built on AWS Lambda with Bedrock AI, it processes 132 social media posts in ~3-4 minutes using concurrent agent orchestration. The system features a real-time Streamlit dashboard with S3-backed history, authentication, and comprehensive reporting capabilities.

### The Multilingual Challenge: Why This Matters

Misinformation doesn't respect language barriers. A false claim about a political event in Brazil spreads in Portuguese, gets translated to Spanish, morphs in cultural context, and reaches English-speaking audiences with different connotations. Traditional misinformation detection systems fail here because they:

1. **Lack Cultural Context**: A statement that's clearly false in one culture might be ambiguous in another
2. **Miss Nuance**: Sarcasm, idioms, and cultural references don't translate literally
3. **Ignore Regional Knowledge**: Local political context, historical events, and social norms vary dramatically
4. **Treat Languages Independently**: Cross-lingual misinformation campaigns are invisible to monolingual systems

### Recent LLM Breakthroughs Enable This

**The Capability Gap**: Until recently, multilingual misinformation detection required:
- Separate models per language
- Manual cultural expertise
- Translation layers (losing nuance)
- Language-specific training data

**The LLM Revolution**: Modern large language models like Gemma 3, Claude 3.5, and GPT-4 have changed this:
- **Native Multilingual Understanding**: Trained on diverse language corpora
- **Cultural Context Awareness**: Implicit knowledge of cultural norms and references
- **Cross-Lingual Reasoning**: Can compare claims across languages
- **Nuance Preservation**: Understand sarcasm, idioms, and context-dependent meaning

**Our Implementation**: We chose **Claude 3.5 Haiku** for RumorNet because:
- ✅ Strong multilingual capabilities across 100+ languages
- ✅ Cultural context understanding
- ✅ Fast inference (critical for Lambda)
- ✅ Cost-effective for batch processing
- ⚠️ **Note**: Gemma 3 was our first choice but unavailable in AWS Bedrock us-east-1 region

### RumorNet's Multilingual Innovation

Our **Multilingual Knowledge Graph Reasoning Agent** goes beyond simple translation:

**1. Automatic Language Detection**
```python
# Detects language without explicit input
post = "Este político nunca dijo eso, es fake news"
→ Detected: Portuguese (Brazil)
→ Cultural Context: Brazilian political discourse
→ Analysis: Considers local political climate
```

**2. Cultural Context Integration**
- Understands regional political dynamics
- Recognizes culturally-specific misinformation patterns
- Accounts for local media landscape
- Considers historical context

**3. Cross-Lingual Pattern Recognition**
- Identifies misinformation campaigns across languages
- Detects translated false narratives
- Tracks claim evolution across linguistic boundaries
- Recognizes coordinated multilingual campaigns

**4. Nuanced Reasoning**
- Distinguishes between literal and figurative language
- Understands context-dependent truth values
- Recognizes cultural idioms and references
- Accounts for regional variations in language

### Why This Is Groundbreaking

**Traditional Approach**:
```
English Post → English Model → Analysis
Spanish Post → Translation → English Model → Analysis (loses nuance)
```

**RumorNet Approach**:
```
Any Language Post → Multilingual KG Reasoning → Culturally-Aware Analysis
                  ↓
            Native understanding + Cultural context + Cross-lingual patterns
```

**Real-World Impact**:
- **Global Misinformation Campaigns**: Detect coordinated campaigns across languages
- **Diaspora Communities**: Understand misinformation in multilingual communities
- **Cross-Border Narratives**: Track how false claims evolve across cultures
- **Regional Conflicts**: Analyze misinformation in conflict zones with multiple languages

### Key Metrics

- **Processing Speed**: 220 seconds for 132 posts
- **Languages Supported**: 100+ (via Claude 3.5 Haiku)
- **Concurrency**: 5 parallel chunks (TRUE BATCH pattern)
- **Memory Usage**: 481 MB peak in Lambda
- **Detection Rate**: 70+ patterns per analysis
- **Cultural Context**: Integrated into every analysis
- **Uptime**: Serverless (scales to zero)

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Agent Framework](#agent-framework)
3. [Orchestration Strategy](#orchestration-strategy)
4. [Deployment Architecture](#deployment-architecture)
5. [Dashboard & Monitoring](#dashboard--monitoring)
6. [Technical Decisions](#technical-decisions)
7. [Future Roadmap](#future-roadmap)

---

## System Architecture

### High-Level Overview

```
┌─────────────────┐
│  User Request   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│     Streamlit Dashboard (Docker)        │
│  - Authentication (username/password)   │
│  - Real-time progress tracking          │
│  - S3 history integration               │
└────────┬────────────────────────────────┘
         │ HTTPS POST
         ▼
┌─────────────────────────────────────────┐
│      AWS API Gateway (29s timeout)      │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│    AWS Lambda (15 min max timeout)      │
│  - 3008 MB memory                       │
│  - Python 3.11 runtime                  │
│  - Async orchestration                  │
└────────┬────────────────────────────────┘
         │
         ├──────────────────┬──────────────────┐
         ▼                  ▼                  ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   Bedrock    │   │   Bedrock    │   │   Bedrock    │
│  Claude      │   │ Claude Haiku │   │ Claude Haiku │
│   Embedding      │   │              │   │          │
└──────────────┘   └──────────────┘   └──────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         S3 Storage (rumornet)           │
│  - Analysis reports (JSON)              │
│  - Execution history                    │
│  - Timestamped results                  │
└─────────────────────────────────────────┘
```

### Component Breakdown

#### 1. **Frontend Layer** (Streamlit Dashboard)
- **Technology**: Streamlit 1.28+, Plotly, Pandas
- **Deployment**: Docker container on Render.com
- **Features**:
  - Form-based authentication (env vars or secrets.toml)
  - Real-time spinner during analysis (3-4 min)
  - S3 polling for results (handles 504 Gateway timeout)
  - Markdown report generation
  - Historical analysis viewer
  - Auto-refresh (pauses during analysis)

#### 2. **API Layer** (AWS API Gateway)
- **Type**: REST API
- **Endpoint**: `/Prod/analyze`
- **Timeout**: 29 seconds (AWS hard limit)
- **Handling**: Dashboard polls S3 after 504 timeout

#### 3. **Compute Layer** (AWS Lambda)
- **Runtime**: Python 3.11
- **Memory**: 3008 MB
- **Timeout**: 900 seconds (15 minutes)
- **Concurrency**: Async/await with asyncio
- **Cold Start**: ~2-3 seconds

#### 4. **AI Layer** (AWS Bedrock)
- **Model**: Claude 3.5 Haiku (all agents)
- **Model ID**: `us.anthropic.claude-3-5-haiku-20241022-v1:0`
- **API**: Converse API (streaming disabled for Lambda)
- **Region**: us-east-1
- **Rationale**: Haiku chosen for speed and cost-effectiveness in Lambda environment

#### 5. **Storage Layer** (AWS S3)
- **Bucket**: `rumornet`
- **Path**: `misinformation-detection/reports/`
- **Format**: JSON with timestamp + UUID
- **Retention**: Indefinite (manual cleanup)

---

## Agent Framework

### Core Agents (Production)

#### 1. **Multilingual KG Reasoning Agent**
**File**: `agents/multilingual_kg_reasoning_agent.py`

**Purpose**: Primary reasoning engine for misinformation detection

**Model**: Claude 3.5 Haiku

**Capabilities**:
- Multilingual claim analysis (auto-detects language)
- Knowledge graph reasoning
- Chain-of-thought verification
- Confidence scoring (0.0-1.0)
- Contextual understanding

**Input**: Social media post text  
**Output**: 
```python
{
    "verdict": bool,           # True = misinformation
    "confidence": float,       # 0.0-1.0
    "detected_language": str,  # ISO code
    "reasoning_chain": [...]   # Step-by-step logic
}
```

**Performance**: ~940ms per post (concurrent)

---

#### 2. **Pattern Detector Agent**
**File**: `agents/pattern_detector_agent.py`

**Purpose**: Identify misinformation patterns and tactics

**Model**: Claude 3.5 Haiku

**Patterns Detected**:
- Emotional manipulation
- False authority claims
- Cherry-picking data
- Conspiracy theories
- Deepfakes/manipulated media
- Coordinated campaigns
- Bot-like behavior
- Astroturfing

**Input**: Post text + metadata  
**Output**:
```python
{
    "patterns": [
        {
            "type": "emotional_manipulation",
            "confidence": 0.85,
            "indicators": ["fear-mongering", "urgency"]
        }
    ]
}
```

**Performance**: ~800ms per post

---

#### 3. **Evidence Gatherer Agent**
**File**: `agents/evidence_gatherer_agent.py`

**Purpose**: Collect supporting/refuting evidence

**Model**: Claude 3.5 Haiku

**Capabilities**:
- Fact extraction
- Source credibility assessment
- Cross-reference detection
- Evidence quality scoring

**Input**: Claim + context  
**Output**:
```python
{
    "supporting_evidence": [...],
    "refuting_evidence": [...],
    "credibility_score": float
}
```

**Status**: Implemented but not in main pipeline (future integration)

---

#### 4. **Social Behavior Analysis Agent**
**File**: `agents/social_behavior_analysis_agent.py`

**Purpose**: Analyze social dynamics and echo chambers

**Model**: Claude 3.5 Haiku

**Capabilities**:
- ✅ Echo chamber detection (active)
- ✅ Coordinated behavior detection (active)
- ⚠️ Bot network identification (implemented, not yet integrated)
- Network density analysis
- Content homogeneity scoring

**Input**: User interactions + post metadata  
**Output**:
```python
{
    "echo_chamber_detected": True,
    "echo_chamber_score": 0.66,
    "network_density": 0.72,
    "content_homogeneity": 0.60,
    "coordination_indicators": [...]
}
```

**Performance**: Batch analysis across all posts

**Status**: ✅ **ACTIVE** - Echo chamber and coordination detection running in production. Bot detection implemented but not yet integrated into main pipeline.

---

**Note**: Echo chamber detection is **actively used** as part of the Social Behavior Analysis Agent (Agent #4 above). The standalone `echo_chamber_detector_agent.py` exists but the functionality is integrated into the main social behavior agent for efficiency.

### Specialized Agents (Implemented, Not Yet Integrated)

---

#### 6. **Topic Intelligence Engine Agent**
**File**: `agents/misinformation_intelligence/topic_intelligence_engine_agent.py`

**Purpose**: Topic-based misinformation tracking

**Future Use**: Will power the intelligence dashboard showing trending misinformation topics, emerging narratives, and cross-topic patterns.

---

#### 7. **Topic Social Analysis Agent**
**File**: `agents/misinformation_intelligence/topic_social_analysis_agent.py`

**Purpose**: Social dynamics within specific topics

**Future Use**: Will analyze how misinformation spreads differently across topics (e.g., health vs. politics).

---

## Orchestration Strategy

### TRUE BATCH Concurrent Processing

**Pattern**: Divide posts into chunks, process chunks in parallel

```python
# Orchestration Flow
1. Load 132 posts
2. Split into 5 chunks of ~26 posts each
3. Process chunks concurrently:
   ├─ Chunk 1 → [Reasoning, Pattern, Evidence] (parallel)
   ├─ Chunk 2 → [Reasoning, Pattern, Evidence] (parallel)
   ├─ Chunk 3 → [Reasoning, Pattern, Evidence] (parallel)
   ├─ Chunk 4 → [Reasoning, Pattern, Evidence] (parallel)
   └─ Chunk 5 → [Reasoning, Pattern, Evidence] (parallel)
4. Aggregate results
5. Social behavior analysis (batch)
6. Generate report
7. Store to S3
```

### Concurrency Implementation

**File**: `granular_misinformation_orchestrator_concurrent.py`

**Key Techniques**:
- `asyncio.gather()` for parallel agent execution
- `asyncio.Semaphore()` for rate limiting
- Chunk-based batching to avoid memory issues
- Error isolation (one chunk failure doesn't kill the job)

**Code Snippet**:
```python
async def analyze_batch_true_batch(self, posts):
    # Split into chunks
    chunks = self._create_chunks(posts, chunk_size=32)
    
    # Process chunks concurrently
    chunk_tasks = [
        self._process_chunk_concurrent(chunk, chunk_id)
        for chunk_id, chunk in enumerate(chunks)
    ]
    
    chunk_results = await asyncio.gather(*chunk_tasks)
    
    # Aggregate
    return self._aggregate_results(chunk_results)
```

### Performance Optimization

**Strategies**:
1. **Model Selection**: Claude 3.5 Haiku for optimal speed/cost balance
2. **Concurrent Execution**: 5x speedup vs sequential
3. **Streaming Disabled**: Reduces Lambda complexity
4. **Batch Aggregation**: Single social analysis pass
5. **S3 Async Writes**: Non-blocking storage

**Bottlenecks**:
- API Gateway 29s timeout (solved with S3 polling)
- Bedrock rate limits (handled with semaphores)
- Lambda memory (3008 MB sufficient for 132 posts)

---

## Deployment Architecture

### Lambda Deployment

**Infrastructure as Code**: AWS SAM (Serverless Application Model)

**File**: `template.yaml`

**Key Configuration**:
```yaml
Globals:
  Function:
    Timeout: 900          # 15 minutes
    MemorySize: 3008      # ~3 GB
    Runtime: python3.11
    Environment:
      Variables:
        LLM_PROVIDER: bedrock
        BEDROCK_REGION: us-east-1
        BEDROCK_MODEL_ID: us.anthropic.claude-3-5-haiku-20241022-v1:0
        RESULTS_BUCKET: rumornet
```

**IAM Permissions**:
```yaml
- bedrock:InvokeModel
- bedrock:InvokeAgent
- s3:PutObject
- s3:GetObject
- s3:ListBucket
```

**Deployment Command**:
```bash
sam build
sam deploy --guided
```

**API Endpoint**:
```
https://mgbsx1x8l1.execute-api.us-east-1.amazonaws.com/Prod/analyze
```

---

### Dashboard Deployment

**Technology**: Docker + Streamlit

**Dockerfile**:
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV STREAMLIT_SERVER_PORT=10000
CMD ["streamlit", "run", "dashboard.py"]
```

**Environment Variables**:
```bash
AWS_ACCESS_KEY_ID=xxx
AWS_SECRET_ACCESS_KEY=xxx
AWS_DEFAULT_REGION=us-east-1
AUTH_USERNAME=admin
AUTH_PASSWORD=xxx
```

**Deployment Options**:
1. **Docker Hub**: Public image, pull and run
2. **Render.com**: Automatic deployment from GitHub
3. **Local**: `docker-compose up`

**Docker Compose**:
```yaml
services:
  dashboard:
    build: .
    ports:
      - "8501:10000"
    environment:
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AUTH_USERNAME=${AUTH_USERNAME}
      - AUTH_PASSWORD=${AUTH_PASSWORD}
```

---

## Dashboard & Monitoring

### Features

#### 1. **Overview Tab**
- Executive summary (posts analyzed, misinformation detected, high-risk posts)
- Auto-generated markdown report
- Download button for reports
- Agent status grid

#### 2. **Analysis Tab**
- One-click demo analysis (132 posts)
- Authentication required
- Real-time progress spinner
- S3 polling for results (handles 504 timeout)

#### 3. **Results Tab**
- Detailed breakdown by post
- High-priority posts table
- Pattern frequency chart
- Topic analysis (when available)
- Temporal trends

#### 4. **History Tab**
- Combined local + S3 history
- Sorted by timestamp (newest first)
- Load any previous analysis
- Markdown generation for S3 results
- Timezone-aware comparison

### S3 Integration

**Challenge**: API Gateway 504 timeout after 29 seconds

**Solution**: Dashboard polls S3 for results

**Flow**:
```
1. User clicks "Analyze"
2. Dashboard sends request to Lambda
3. API Gateway times out at 29s (504)
4. Dashboard starts polling S3 every 5 seconds
5. Lambda continues processing (3-4 minutes)
6. Lambda stores result to S3
7. Dashboard detects new S3 file
8. Dashboard loads and displays result
```

**Implementation**:
```python
# Poll S3 for up to 10 minutes
for i in range(120):
    time.sleep(5)
    results = data_manager.load_all_execution_results(load_from_s3=True)
    
    # Find results created after analysis started
    new_results = [r for r in results if r.timestamp > start_time]
    
    if new_results:
        display_results(new_results[0])
        break
```

### Authentication

**Method**: Username/password (simple, effective)

**Storage**:
- Local: `.streamlit/secrets.toml` (gitignored)
- Docker: Environment variables

**Implementation**:
```python
correct_user = os.getenv("AUTH_USERNAME") or st.secrets["auth"]["username"]
correct_pass = os.getenv("AUTH_PASSWORD") or st.secrets["auth"]["password"]

if username == correct_user and password == correct_pass:
    st.session_state.authenticated = True
```

**Security**:
- Secrets never committed to Git
- Environment variables in production
- Session-based authentication

---

## Technical Decisions

### Why Not MCP (Model Context Protocol)?

**Decision**: Custom agent framework over MCP

**Reasoning**:

1. **Lambda Constraints**: Spawning ephemeral MCP servers in Lambda is resource-intensive
   - Cold start overhead
   - Memory footprint
   - Process management complexity

2. **Orchestration Control**: Direct agent invocation provides:
   - Fine-grained concurrency control
   - Custom error handling
   - Performance optimization
   - Simpler debugging

3. **Bedrock Integration**: Native Bedrock API is:
   - More reliable in Lambda
   - Lower latency
   - Better cost control
   - Easier to monitor

4. **Future Flexibility**: Custom framework allows:
   - Easy agent swapping
   - Model upgrades
   - Provider changes (Bedrock → OpenAI → Anthropic)

**Trade-off**: Less standardization, but better performance and control

---

### Why Async/Await Over Threading?

**Decision**: asyncio over threading

**Reasoning**:
- I/O-bound workload (API calls)
- Better resource utilization
- Simpler error handling
- Native Python 3.11 support
- Easier to reason about

---

### Why S3 Polling Over WebSockets?

**Decision**: S3 polling over real-time updates

**Reasoning**:
- Lambda is stateless (no persistent connections)
- API Gateway timeout is unavoidable
- S3 is reliable and cheap
- Polling is simple and works
- No need for complex infrastructure

---

## Future Roadmap

### Phase 1: Observability (Q1 2026)

**Goal**: Production-grade monitoring and debugging

**Components**:

1. **Distributed Tracing**
   - AWS X-Ray integration
   - Agent execution timelines
   - Bottleneck identification
   - Error propagation tracking

2. **Metrics Dashboard**
   - CloudWatch custom metrics
   - Agent success rates
   - Latency percentiles (p50, p95, p99)
   - Cost per analysis

3. **Logging Enhancement**
   - Structured logging (JSON)
   - Log aggregation (CloudWatch Insights)
   - Error alerting (SNS)
   - Audit trails

**Implementation**:
```python
# X-Ray tracing
from aws_xray_sdk.core import xray_recorder

@xray_recorder.capture('reasoning_agent')
async def analyze_post(post):
    # Agent logic
    pass
```

---

### Phase 2: Orchestrator Memory (Q2 2026)

**Goal**: Cross-execution learning and context retention

**Architecture**: Graph Database (Neo4j or Amazon Neptune)

**Use Cases**:

1. **Pattern Learning**
   - Store detected patterns
   - Build pattern taxonomy
   - Identify emerging tactics
   - Cross-reference historical data

2. **Entity Tracking**
   - User behavior over time
   - Source credibility history
   - Topic evolution
   - Network relationships

3. **Context Retention**
   - Previous analyses
   - Related claims
   - Fact-check results
   - Expert annotations

**Schema**:
```cypher
// Nodes
(Post)-[:CONTAINS]->(Claim)
(Claim)-[:DETECTED_BY]->(Pattern)
(User)-[:POSTED]->(Post)
(User)-[:CONNECTED_TO]->(User)
(Topic)-[:MENTIONED_IN]->(Post)

// Queries
MATCH (u:User)-[:POSTED]->(p:Post)-[:CONTAINS]->(c:Claim)
WHERE c.misinformation = true
RETURN u, count(c) as misinfo_count
ORDER BY misinfo_count DESC
```

**Benefits**:
- Faster analysis (leverage past results)
- Better accuracy (learn from corrections)
- Trend detection (spot emerging narratives)
- Network analysis (identify coordinated campaigns)

---

### Phase 3: Burst-Driven Analytics (Q3 2026)

**Goal**: Automated analysis triggered by events

**Triggers**:

1. **Volume Spikes**
   - Sudden increase in posts about a topic
   - Viral content detection
   - Breaking news events

2. **Pattern Anomalies**
   - New misinformation tactics
   - Coordinated campaigns
   - Bot swarms

3. **Scheduled Scans**
   - Daily trending topics
   - Weekly summaries
   - Monthly reports

**Architecture**:
```
EventBridge → Lambda → Orchestrator → S3 → Dashboard
     ↑
     └─ CloudWatch Alarms
     └─ SQS Queue
     └─ Cron Schedule
```

**Memory Integration**:
- Query graph DB for context
- Compare to historical patterns
- Identify novel tactics
- Update knowledge base

**Example**:
```python
# Detect burst
if post_volume > threshold:
    # Query memory
    similar_events = graph_db.query(
        "MATCH (e:Event) WHERE e.topic = $topic RETURN e",
        topic=current_topic
    )
    
    # Analyze with context
    result = await orchestrator.analyze_with_memory(
        posts=new_posts,
        context=similar_events
    )
```

---

### Phase 4: Advanced Agent Integration (Q4 2026)

**Goal**: Activate dormant agents and add new capabilities

#### 1. **Evidence Gatherer Integration**
- Real-time fact-checking API integration
- Source verification
- Cross-reference validation

#### 2. **Topic Intelligence Dashboard**
- Trending misinformation topics
- Topic evolution over time
- Cross-topic pattern analysis
- Narrative tracking

#### 3. **Bot Network Detection Integration**
- **Status**: Implemented but not integrated
- **Reason**: Requires behavioral data not currently collected:
  - Posting frequency tracking
  - Interval variance calculation
  - Content uniqueness scoring
- **Effort**: Low - just needs data collection layer
- **Impact**: High - bot campaigns are major misinformation vectors
- **Priority**: Q4 2026

#### 4. **Echo Chamber Deep Dive**
- Network graph visualization
- Influence propagation modeling
- Community detection
- Bridge identification

#### 5. **Multimodal Analysis**
- Image/video deepfake detection
- Audio manipulation detection
- Cross-modal consistency checking

---

### Phase 5: Scale & Performance (2027)

**Goals**:
- 10,000 posts per analysis
- Sub-60-second processing
- Multi-region deployment
- 99.9% uptime

**Techniques**:
- Lambda provisioned concurrency
- Bedrock batch inference
- DynamoDB for hot data
- CloudFront for dashboard
- Multi-region S3 replication

---

## Conclusion

RumorNet represents a production-grade, serverless multi-agent system for misinformation detection. By leveraging AWS Lambda, Bedrock AI, and concurrent orchestration, it achieves:

- **Scalability**: Serverless architecture scales to zero
- **Performance**: 132 posts in 3-4 minutes
- **Reliability**: S3-backed persistence, error isolation
- **Usability**: Streamlit dashboard with authentication
- **Extensibility**: Modular agent framework

The decision to build a custom agent framework over MCP was driven by Lambda's constraints and the need for fine-grained control. Future phases will add observability, memory (graph DB), and burst-driven analytics, transforming RumorNet into an intelligent, self-learning misinformation detection platform.

**Current Status**: ✅ Production-ready  
**Next Milestone**: Observability & monitoring (Q1 2026)

---

## Appendix: Key Files

### Lambda
- `lambda_handler_async.py` - Entry point
- `granular_misinformation_orchestrator_concurrent.py` - Orchestrator
- `agents/multilingual_kg_reasoning_agent.py` - Primary reasoning
- `agents/pattern_detector_agent.py` - Pattern detection
- `agents/social_behavior_analysis_agent.py` - Social analysis
- `template.yaml` - Infrastructure as Code

### Dashboard
- `dashboard.py` - Main application
- `components/batch_analysis_api.py` - Lambda integration
- `core/data_manager.py` - S3 + local storage
- `utils/markdown_generator.py` - Report generation
- `Dockerfile` - Container definition

### Configuration
- `.streamlit/secrets.toml` - Local auth (gitignored)
- `docker-compose.yml` - Easy deployment
- `.env.example` - Environment template

---

**Built with ❤️ using AWS Bedrock, Lambda, and Streamlit**
