---
inclusion: always
---

# Ollama Configuration for KG Reasoning System

## CRITICAL REQUIREMENT: NO BEDROCK CALLS DURING TESTING

**Under no circumstances should Amazon Bedrock be called during testing. Always use the local Ollama setup.**

## Correct IP Address

**IMPORTANT: The correct Ollama endpoint is `http://192.168.10.68:11434` (NOT .69)**

## Correct Strands Agent Pattern

When using AWS Strands with Ollama, always follow this exact pattern:

```python
from strands.models.ollama import OllamaModel
from strands import Agent

# Initialize Ollama model with specific endpoint
ollama_model = OllamaModel(
    host="http://192.168.10.68:11434",
    model_id="gemma3:4b"
)

# Create agent with Ollama model
agent = Agent(model=ollama_model)

# Use the agent
response = agent("what is capital of china?")
```

## Configuration Requirements

### Environment Variables
```bash
export KG_LLM_BACKEND=ollama
export KG_OLLAMA_ENDPOINT=http://192.168.10.68:11434
export KG_OLLAMA_MODEL=gemma3:4b
export KG_OLLAMA_TIMEOUT=60
```

### YAML Configuration
```yaml
llm_backend: "ollama"
ollama_endpoint: "http://192.168.10.68:11434"
ollama_model: "gemma3:4b"
ollama_timeout: 60
```

### Programmatic Configuration
```python
from kg_reasoning.config.agent_config import AgentConfig, LLMBackend

config = AgentConfig(
    llm_backend=LLMBackend.OLLAMA,
    ollama_endpoint="http://192.168.10.68:11434",
    ollama_model="gemma3:4b",
    ollama_timeout=60
)
```

## Implementation in KG Reasoning Agent

The `KGReasoningAgent._get_llm_model()` method has been updated to:

1. **Always prefer Ollama** over Bedrock
2. **Override Bedrock configuration** to use Ollama during testing
3. **Use hardcoded Ollama settings** as fallback
4. **Fail explicitly** if Ollama is not available (no Bedrock fallback)

```python
def _get_llm_model(self):
    """Get configured LLM model - ALWAYS use Ollama for testing."""
    try:
        # CRITICAL: Always use Ollama - NO Bedrock calls during testing
        if self.config.llm_backend == LLMBackend.OLLAMA:
            return OllamaModel(
                host=self.config.ollama_endpoint,
                model_id=self.config.ollama_model,
                timeout=self.config.ollama_timeout
            )
        elif self.config.llm_backend == LLMBackend.BEDROCK:
            # OVERRIDE: Force Ollama instead of Bedrock for testing
            self.logger.warning("Bedrock configured but overriding to use Ollama for testing")
            return OllamaModel(
                host="http://192.168.10.68:11434",
                model_id="gemm3:4b",
                timeout=60
            )
        else:
            # Default to Ollama configuration
            return OllamaModel(
                host="http://192.168.10.68:11434",
                model_id="gemma3:4b",
                timeout=60
            )
    except Exception as e:
        raise Exception(f"Ollama model initialization failed: {e}. No fallback to Bedrock allowed.")
```

## Testing Verification

Before running any tests, verify Ollama is working:

```python
# Test Ollama connection
import asyncio
import aiohttp

async def test_ollama():
    async with aiohttp.ClientSession() as session:
        async with session.get("http://192.168.10.68:11434/api/tags") as response:
            if response.status == 200:
                data = await response.json()
                models = [model['name'] for model in data.get('models', [])]
                print(f"Available models: {models}")
                assert "gemma3:4b" in models, "gemma3:4b model not found"
                print("✅ Ollama configuration verified")
            else:
                raise Exception(f"Ollama not accessible: {response.status}")

# Run verification
asyncio.run(test_ollama())
```

## Cost Implications

Using Ollama instead of Bedrock:
- **Zero API costs** - runs locally
- **No token charges** - unlimited usage
- **Faster iteration** - no network latency to AWS
- **Privacy** - all data stays local

## Task 12 Specific Configuration

For Task 12 social media analysis scenarios:
- Use **Chain-of-Thought (CoT)** initially for lower computational cost
- Limit **max_steps** to 5-10 for faster execution
- Enable **caching** to avoid repeated computations
- Use **batch processing** for multiple test scenarios

## Troubleshooting

If Ollama connection fails:
1. Verify Ollama is running: `curl http://192.168.10.68:11434/api/tags`
2. Check model availability: `ollama list`
3. Pull model if missing: `ollama pull gemma3:4b`
4. Check network connectivity to the endpoint
5. Verify firewall settings allow port 11434

## Enforcement

The system is configured to:
- **Reject Bedrock calls** during testing
- **Log warnings** when Bedrock is configured but overridden
- **Fail explicitly** if Ollama is not available
- **Use hardcoded Ollama settings** as ultimate fallback

This ensures no accidental Bedrock usage and associated costs during development and testing.