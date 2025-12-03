---
inclusion: always
---

# NO FAKE TESTS - PRODUCTION READINESS ONLY

## CRITICAL RULE: NEVER DUMB DOWN TESTS TO MAKE THEM PASS

### What is FORBIDDEN:

1. **Testing only if imports work** - This proves NOTHING
2. **Testing only if objects can be created** - This proves NOTHING
3. **Skipping actual functionality tests** - This proves NOTHING
4. **Marking tests as PASS when they don't test real behavior** - This is LYING
5. **Creating "simplified" tests that avoid the hard parts** - This is CHEATING
6. **Testing only initialization without testing actual tool calls** - This proves NOTHING
7. **Accepting "connection successful" as a pass without testing tools** - This proves NOTHING

### What is REQUIRED for a test to be valid:

1. **Discover ALL tools** via the actual protocol (MCP, HTTP, etc.)
2. **Call EVERY tool** with real test data
3. **Validate EVERY response** for correctness
4. **Check actual output values** (verdicts, confidence scores, etc.)
5. **Verify the tool actually performed its function** (not just returned something)
6. **Test with realistic data** that exercises the full code path
7. **Fail the test if ANY tool fails** - no partial credit

### Example of FAKE test (FORBIDDEN):

```python
# WRONG - This is a fake test
def test_agent():
    agent = MyAgent()
    assert agent is not None  # This proves NOTHING
    print("✓ PASSED")
```

### Example of REAL test (REQUIRED):

```python
# CORRECT - This is a real test
def test_agent():
    agent = MyAgent()
    
    # Discover tools
    tools = discover_tools_via_protocol(agent)
    assert len(tools) > 0, "No tools discovered"
    
    # Call each tool
    for tool in tools:
        result = call_tool(tool, real_test_data)
        
        # Validate response
        assert result is not None, f"Tool {tool} returned None"
        assert "error" not in result, f"Tool {tool} returned error: {result['error']}"
        
        # Validate actual output
        if tool == "analyze":
            assert result.get("verdict") is not None, "Verdict is None"
            assert result.get("confidence") > 0, "Confidence is 0"
            assert len(result.get("reasoning_chain", [])) > 0, "No reasoning steps"
    
    print("✓ PASSED - All tools work correctly")
```

### Production Readiness Criteria:

A system is NOT production ready unless:

1. **ALL agents can be started** and stay running
2. **ALL tools can be discovered** via the intended protocol
3. **ALL tools can be called** with real data
4. **ALL tools return valid responses** without errors
5. **ALL responses contain expected data** (not empty, not None, not errors)
6. **The system works end-to-end** as it will be used in production

### If tests fail:

1. **DO NOT simplify the test** to make it pass
2. **DO NOT skip failing parts** and test only what works
3. **DO NOT create alternative "easier" tests**
4. **FIX THE ACTUAL PROBLEM** in the code
5. **Report honestly** what is broken and what needs fixing

### Consequences of fake tests:

- Wastes user's time and money
- Deploys broken systems to production
- Destroys trust
- Creates technical debt
- Leads to production failures

### Remember:

**A test that doesn't test real functionality is worse than no test at all - it gives false confidence.**

**If you can't make the real test pass, say so. Don't fake it.**
