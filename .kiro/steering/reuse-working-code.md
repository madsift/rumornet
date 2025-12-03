---
inclusion: always
---

# Code Reuse and Pattern Recognition

## CRITICAL RULE: DON'T REINVENT THE WHEEL

Before writing ANY new code, ALWAYS:

1. **Search for existing working implementations**
2. **Read and understand the working code**
3. **Copy the working patterns**
4. **Only modify what's necessary**

## ❌ NEVER Do This:
- Write new code from scratch when similar code exists
- Assume you know better than working code
- Ignore existing patterns and create new ones
- Simplify or "improve" working code without testing
- Create new abstractions when simple code works

## ✅ ALWAYS Do This:
- Search the codebase for similar functionality
- Read the working implementation completely
- Copy the exact patterns that work
- Keep the same structure and flow
- Only add/modify the specific new feature needed
- Test that the copied pattern still works

## Pattern Recognition Process

### Step 1: Search for Working Code
```bash
# Search for similar functionality
grepSearch: "process.*batch"
grepSearch: "initialize.*agents"
grepSearch: "fetch.*posts"
```

### Step 2: Read the Working Implementation
- Read the ENTIRE file, not just snippets
- Understand the data flow
- Note all dependencies and imports
- Identify the key patterns

### Step 3: Copy, Don't Recreate
- Copy the working code structure
- Keep variable names consistent
- Maintain the same error handling
- Use the same logging patterns
- Keep the same async/await patterns

### Step 4: Minimal Modifications
- Only change what's absolutely necessary
- Add new features incrementally
- Test after each small change
- Don't "clean up" or "improve" working code

## Example: Adding Memory to Working Orchestrator

### ❌ WRONG Approach:
```python
# Creating entirely new orchestrator from scratch
class NewMemoryOrchestrator:
    def __init__(self):
        # New initialization logic
        pass
    
    async def process_posts(self):
        # Completely new processing logic
        pass
```

### ✅ CORRECT Approach:
```python
# Copy from jaeger_working_orchestrator.py
class JaegerWorkingOrchestrator:
    def __init__(self):
        # KEEP ALL EXISTING CODE
        self.agents = {}
        self.results = []
        # ... existing code ...
        
        # ADD ONLY NEW MEMORY FEATURE
        self.memory = None
        self.memory_connected = False
    
    async def process_posts_batch(self, posts):
        # KEEP EXACT SAME STRUCTURE
        start_time = time.time()
        
        if self.tracer:
            batch_span = self.tracer.start_span("real_posts_batch_processing")
            # ... existing tracing code ...
        
        # KEEP EXACT SAME PROCESSING LOGIC
        claims = []
        post_metadata = []
        for i, post in enumerate(posts):
            text = post.get("posts", "").strip()
            # ... existing extraction code ...
        
        # ADD MEMORY FEATURE HERE (minimal addition)
        if self.memory_connected:
            warm_start = await self.memory.get_warm_start_workflow(task)
        
        # KEEP EXACT SAME AGENT CALLS
        if "multilingual_kg_reasoning" in self.agents:
            agent = self.agents["multilingual_kg_reasoning"]
            reasoning_result = await self._call_multilingual_reason(...)
            # ... existing code ...
```

## Common Patterns to Reuse

### 1. Agent Initialization
**Source**: `jaeger_working_orchestrator.py`
```python
async def initialize_agents(self):
    """Initialize existing agents with extensive logging."""
    import sys
    sys.path.append('.')

    if self.tracer:
        with self.tracer.start_as_current_span("agent_initialization") as span:
            # Multilingual KG Reasoning Agent
            try:
                from agents.multilingual_kg_reasoning_agent import MultilingualKGReasoningMCPAgent
                config = {
                    "ollama_endpoint": GLOBAL_CONFIG["ollama_endpoint"], 
                    "ollama_model": GLOBAL_CONFIG["ollama_model"],
                    "default_response_language": "auto"
                }
                kg_agent = MultilingualKGReasoningMCPAgent(config=config)
                self.agents["multilingual_kg_reasoning"] = kg_agent
```

### 2. Post Processing
**Source**: `jaeger_working_orchestrator.py`
```python
# Extract claims from posts
claims = []
post_metadata = []

for i, post in enumerate(posts):
    text = post.get("posts", "").strip()
    if text:
        claims.append(text)
        post_metadata.append({
            "post_id": post.get("submission_id", f"unknown_{i}"),
            "subreddit": post.get("subreddit", "unknown"),
            "original_text": text
        })
```

### 3. Agent Calling Pattern
**Source**: `jaeger_working_orchestrator.py`
```python
async def _call_multilingual_reason(self, agent, claim: str, strategy: str, 
                                   response_language: str, context: Optional[str]) -> Dict[str, Any]:
    """Call individual multilingual reasoning using internal methods."""
    try:
        from agents.kg_reasoning_agent import ReasoningStrategy
        
        try:
            reasoning_strategy = ReasoningStrategy(strategy)
        except ValueError:
            reasoning_strategy = ReasoningStrategy.COT
        
        result = await agent._execute_multilingual_reasoning(
            claim=claim,
            strategy=reasoning_strategy,
            response_language=response_language,
            context=context
        )
        
        return result
```

### 4. GPA Tracking Pattern
**Source**: `jaeger_working_orchestrator.py`
```python
if self.gpa_tracker and ProcessingMetrics:
    detected_lang = reasoning_result.get("detected_language", "unknown")
    
    if self.gpa_tracer:
        with self.gpa_tracer.start_as_current_span(f"gpa_claim_analysis_{detected_lang}") as gpa_span:
            gpa_metrics = ProcessingMetrics(
                processing_time_ms=claim_processing_time,
                success_rate=1.0 if not reasoning_result.get("error") else 0.0,
                language_detected=detected_lang,
                content_length=len(claim),
                timestamp=datetime.now()
            )
            
            self.gpa_tracker.track_language_processing(
                language=detected_lang,
                content_type="social_media_claim",
                metrics=gpa_metrics
            )
```

### 5. Fetch Posts Pattern
**Source**: `jaeger_working_orchestrator.py`
```python
async def fetch_posts_from_mcp(timestamp: str) -> List[Dict[str, Any]]:
    """Fetch posts from MCP server."""
    server_url = f"{GLOBAL_CONFIG['mcp_server_url']}/tools/get_posts_before"
    query_payload = {
        "subreddit": GLOBAL_CONFIG["subreddit"],
        "timestamp": timestamp,
        "limit": GLOBAL_CONFIG["post_limit"]
    }

    try:
        response = requests.post(server_url, json=query_payload, timeout=30)
        response.raise_for_status()
        posts = response.json()
        # ... handle response ...
```

## When to Create New Code

Only create new code when:
1. **No similar functionality exists** in the codebase
2. **The existing code is broken** and needs replacement
3. **The requirement is completely different** from anything existing
4. **You've thoroughly searched** and found nothing reusable

Even then:
- Copy the code style and patterns from similar files
- Use the same naming conventions
- Follow the same error handling patterns
- Match the logging style

## Verification Checklist

Before submitting new code, verify:
- [ ] Searched for similar existing code
- [ ] Read the working implementation completely
- [ ] Copied the working patterns
- [ ] Kept the same structure and flow
- [ ] Only modified what's necessary
- [ ] Tested that it still works
- [ ] Didn't "improve" or "simplify" working code

## Remember

**Working code is sacred. Copy it, don't recreate it.**
