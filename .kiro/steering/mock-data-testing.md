# Mock Data and Testing Guidelines

## Critical Testing Requirements

### Real Implementation Priority
- **ALWAYS use real Ollama calls** for LLM operations - never mock LLM responses
- **ALWAYS implement actual functions** - never create empty stubs or placeholder implementations
- **ALWAYS use real FastMCP protocol** - never mock MCP communication
- **ALWAYS use real memory backends** (Oxigraph, LanceDB) when possible

### When Mocking is Acceptable

#### 1. External Service Dependencies (Only When Unavailable)
```python
# Acceptable: Mock external APIs that aren't available locally
class MockExternalAPI:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    async def fetch_data(self, query: str) -> dict:
        self.logger.warning("Using mock external API - replace with real implementation")
        # Return realistic mock data structure
        return {"status": "success", "data": [...]}
```

#### 2. Test Data Generation
```python
# Acceptable: Generate realistic test data
def generate_test_claims() -> List[dict]:
    """Generate realistic test claims for misinformation detection."""
    return [
        {
            "text": "The Earth is flat and NASA is hiding the truth",
            "type": "conspiracy_theory",
            "expected_difficulty": 0.8,
            "ground_truth": False
        },
        {
            "text": "It's raining today in Seattle",
            "type": "weather_claim", 
            "expected_difficulty": 0.2,
            "ground_truth": None  # Depends on actual weather
        }
    ]
```

#### 3. Expensive Operations (With Clear Warnings)
```python
# Acceptable: Mock expensive training operations for quick testing
class MockTrainer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def train_policy(self, episodes: List[dict]) -> dict:
        self.logger.warning("MOCK TRAINER: Replace with real RL training for production")
        # Simulate training metrics
        return {"loss": 0.1, "accuracy": 0.85, "epochs": 10}
```

### Mandatory Real Implementations

#### 1. Core Orchestration Logic
```python
# MUST BE REAL: Never mock orchestrator core logic
class AdaptiveOrchestrator:
    async def process_task(self, task: dict, session_id: str) -> dict:
        # Real implementation required - no mocking allowed
        context = await self.memory.get_context(session_id, task)
        difficulty = self.difficulty_estimator.estimate(task["text"], context)
        workflow = self.policy.plan(task, difficulty, context)
        result = await self._execute_workflow(workflow, session_id)
        return result
```

#### 2. FastMCP Agent Communication
```python
# MUST BE REAL: Never mock FastMCP protocol
class KGReasoningMCPAgent:
    @mcp.tool()
    async def query_kg(self, claim: str) -> dict:
        # Real implementation using actual Ollama calls
        response = await self.ollama_model.generate(
            prompt=f"Analyze this claim: {claim}",
            model="llama3.1:8b"
        )
        return {"evidence": response.content, "confidence": 0.85}
```

#### 3. Memory Operations
```python
# MUST BE REAL: Never mock memory storage/retrieval
class MemoryManager:
    async def store_episode(self, session_id: str, task: dict, workflow: List[dict], result: dict):
        # Real implementation - actually store in Oxigraph and LanceDB
        await self.oxigraph.store_graph(session_id, task, workflow, result)
        await self.lancedb.store_embedding(task["embedding"], {"task": task, "result": result})
```

### Mock Data Guidelines

#### 1. Realistic Data Structures
```python
# Good: Realistic mock data that matches expected schemas
MOCK_EPISODES = [
    {
        "session_id": "session_001",
        "task": {
            "type": "claim_verification",
            "text": "Vaccines cause autism",
            "embedding": np.random.rand(768).tolist()  # Realistic embedding dimension
        },
        "workflow": [
            {"agent": "kg_agent", "tool": "query_kg", "args": {"claim": "Vaccines cause autism"}},
            {"agent": "evidence_agent", "tool": "gather_evidence", "args": {"claim": "Vaccines cause autism"}}
        ],
        "result": {
            "verdict": False,
            "confidence": 0.95,
            "evidence": ["Multiple peer-reviewed studies show no link between vaccines and autism"]
        },
        "metrics": {
            "tokens": 1250,
            "cost": 0.025,
            "latency_ms": 3400
        }
    }
]
```

#### 2. Inform User About Mocking
```python
def create_mock_data():
    """
    Create mock test data for orchestration framework.
    
    WARNING: This generates synthetic data for testing purposes.
    In production, use real historical episodes and claims.
    """
    logger.warning("Using mock data - replace with real data for production")
    return MOCK_EPISODES
```

### Testing Strategy

#### 1. Integration Tests with Real Components
```python
async def test_orchestrator_integration():
    """Test orchestrator with real Ollama and memory backends."""
    # Use real Ollama endpoint
    ollama_model = OllamaModel(
        host="http://192.168.10.68:11434",
        model_id="gemma3:4b"
    )
    
    # Use real memory backends
    memory = MemoryManager(
        oxigraph_endpoint="http://localhost:7878",
        lancedb_path="./test_lancedb"
    )
    
    # Real orchestrator with real components
    orchestrator = AdaptiveOrchestrator(
        agent_configs={"test_agent": "stdio://python test_agent.py"},
        memory=memory
    )
    
    # Test with realistic task
    result = await orchestrator.process_task(
        task={"type": "claim_verification", "text": "Test claim"},
        session_id="test_session"
    )
    
    assert result is not None
    assert "verdict" in result
```

#### 2. Unit Tests with Minimal Mocking
```python
def test_difficulty_estimator():
    """Test difficulty estimator with real VAE model."""
    estimator = DifficultyEstimator()
    
    # Use real text encoding
    z, difficulty = estimator.estimate("Complex conspiracy theory claim", {})
    
    assert isinstance(z, np.ndarray)
    assert 0.0 <= difficulty <= 1.0
    assert z.shape == (16,)  # Expected latent dimension
```

### Prohibited Practices

#### ❌ Never Do This:
```python
# WRONG: Empty stub functions
def process_task(self, task):
    pass  # TODO: Implement later

# WRONG: Fake LLM responses
def mock_ollama_call(prompt):
    return "This is a fake response"

# WRONG: Non-functional FastMCP agents
@mcp.tool()
def analyze_claim(claim: str):
    return {"status": "not implemented"}
```

#### ✅ Always Do This:
```python
# RIGHT: Real implementation with proper error handling
async def process_task(self, task: dict, session_id: str) -> dict:
    try:
        # Real implementation
        context = await self.memory.get_context(session_id, task)
        difficulty = self.difficulty_estimator.estimate(task["text"], context)
        workflow = self.policy.plan(task, difficulty, context)
        result = await self._execute_workflow(workflow, session_id)
        return result
    except Exception as e:
        self.logger.error(f"Task processing failed: {e}")
        raise

# RIGHT: Real Ollama integration
@mcp.tool()
async def analyze_claim(claim: str) -> dict:
    try:
        response = await self.ollama_model.generate(
            prompt=f"Analyze this claim for misinformation: {claim}",
            model="llama3.1:8b"
        )
        return {
            "verdict": self._parse_verdict(response.content),
            "confidence": self._calculate_confidence(response.content),
            "evidence": self._extract_evidence(response.content)
        }
    except Exception as e:
        self.logger.error(f"Claim analysis failed: {e}")
        raise
```

### Environment Configuration

#### Required Real Services
- **Ollama**: Must be running at http://192.168.10.68:11434
- **Oxigraph**: Should be available for graph storage
- **LanceDB**: Should be available for vector storage
- **Python Environment**: Use conda agent environment

#### Mock Service Alternatives (Only When Necessary)
```python
# Only if real services unavailable - with clear warnings
class MockOxigraph:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.warning("Using mock Oxigraph - replace with real service")
        self.data = {}
    
    async def store_graph(self, session_id: str, data: dict):
        self.logger.warning("Mock storage - data will not persist")
        self.data[session_id] = data
```

### Validation Requirements

Before any implementation:
1. **Verify Ollama is accessible** at http://192.168.10.68:11434
2. **Confirm llama3.1:8b model is available**
3. **Test FastMCP protocol communication**
4. **Validate memory backend connections**
5. **Inform user of any necessary mocking with clear justification**

### Summary

- **Real implementations are mandatory** for core functionality
- **Mock only when absolutely necessary** and with clear warnings
- **Always inform the user** when mocking is used
- **Use realistic mock data** that matches expected schemas
- **Never create empty stubs** or non-functional placeholders
- **Prioritize integration with real Ollama** over any mocking