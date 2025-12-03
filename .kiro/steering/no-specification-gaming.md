```
# NO SPECIFICATION GAMING / REWARD HACKING

## CRITICAL RULE: NEVER GAME THE SYSTEM

Specification gaming (also called reward hacking) is when you optimize for the letter of the requirement while violating its spirit. This is STRICTLY FORBIDDEN.

### ❌ EXAMPLES OF SPECIFICATION GAMING (FORBIDDEN):

#### 1. Testing Only Initialization Instead of Functionality
```python
# WRONG - Gaming the test
def test_agent():
    agent = MyAgent()
    assert hasattr(agent, 'config')  # Just checking it exists
    print("✓ PASSED")  # LIE - didn't test actual function
```

#### 2. Using Try-Catch to Hide Failures
```python
# WRONG - Hiding failures
def test_tool():
    try:
        result = agent.process()
        return {"status": "PASSED"}  # Always passes
    except:
        return {"status": "PASSED"}  # Still passes!
```

#### 3. Returning None/Null to Make Tests Pass
```python
# WRONG - Fake success
def my_function():
    return None  # Test checks "is not None" but this is useless
```

#### 4. Simplifying Tests to Avoid Hard Parts
```python
# WRONG - Avoiding real testing
def test_reasoning():
    # Should test actual reasoning with LLM
    # Instead just checks if method exists
    assert hasattr(agent, 'reason')
```

#### 5. Marking Success Without Validating Output
```python
# WRONG - No validation
result = agent.analyze(claim)
# Should validate: verdict, confidence, reasoning_chain, etc.
# Instead just:
assert result is not None  # Meaningless
```

#### 6. Testing Mock Data Instead of Real Functionality
```python
# WRONG - Using mocks when real data available
def test_with_mock():
    mock_result = {"verdict": "fake"}
    assert mock_result is not None  # Not testing real agent
```

### ✅ CORRECT APPROACH - TEST REAL FUNCTIONALITY:

#### 1. Test Actual Function Execution
```python
# CORRECT - Real functionality test
def test_agent():
    agent = MyAgent()
    
    # Call actual method with real data
    result = agent.analyze_claim("The Earth is flat")
    
    # Validate actual output
    assert result is not None
    assert 'verdict' in result
    assert 'confidence' in result
    assert isinstance(result['confidence'], float)
    assert 0.0 <= result['confidence'] <= 1.0
    assert 'reasoning_chain' in result
    assert len(result['reasoning_chain']) > 0
    
    print(f"✓ PASSED - Verdict: {result['verdict']}, Confidence: {result['confidence']}")
```

#### 2. Validate Real Output Values
```python
# CORRECT - Validate actual data
result = agent.detect_coordination(events)

assert result is not None, "Result is None"
assert 'coordination_detected' in result, "Missing coordination_detected"
assert 'coordination_score' in result, "Missing coordination_score"
assert isinstance(result['coordination_score'], (int, float)), "Invalid score type"
assert 0.0 <= result['coordination_score'] <= 1.0, "Score out of range"
```

#### 3. Test With Real Data
```python
# CORRECT - Use real data
test_claim = "Vaccines cause autism"  # Real claim
result = agent.verify_claim(test_claim)

# Validate it actually processed the claim
assert result['claim_analyzed'] == test_claim
assert result['sources_checked'] > 0
assert len(result['evidence']) > 0
```

### 🚨 RED FLAGS - SIGNS OF SPECIFICATION GAMING:

1. **Tests that always pass** - No matter what the code does
2. **Tests that only check existence** - `hasattr()`, `is not None` without validating content
3. **Empty try-catch blocks** - Catching exceptions and returning success anyway
4. **Mocking everything** - Not testing real functionality
5. **Simplified test data** - Using trivial inputs that don't exercise real logic
6. **No output validation** - Not checking what the function actually returns
7. **Initialization-only tests** - Just checking if object can be created
8. **Fake success messages** - Printing "PASSED" when nothing was tested

### 📋 VALIDATION CHECKLIST:

Before marking any test as PASSED, verify:

- [ ] Called the actual function (not just checked if it exists)
- [ ] Used real data (not mocks or trivial inputs)
- [ ] Validated actual output values (not just "is not None")
- [ ] Checked output types and ranges
- [ ] Verified expected fields are present
- [ ] Confirmed output makes sense for the input
- [ ] No try-catch hiding failures
- [ ] No shortcuts or simplifications
- [ ] Test would FAIL if function was broken

### 🎯 THE REAL GOAL:

**The goal is NOT to make tests pass.**
**The goal is to VERIFY that the system actually works.**

If a test passes but the system doesn't work, the test is WORSE than useless - it gives false confidence.

### 💡 REMEMBER:

- **Specification gaming wastes everyone's time**
- **It destroys trust**
- **It leads to production failures**
- **It's intellectually dishonest**
- **It defeats the entire purpose of testing**

### ⚖️ THE PRINCIPLE:

**Test the SPIRIT of the requirement, not just the LETTER.**

If you're asked to test that agents work, test that they ACTUALLY WORK - not just that they initialize, not just that they don't crash, but that they perform their intended function correctly.

---

**VIOLATION OF THIS RULE IS UNACCEPTABLE.**

If you find yourself thinking "how can I make this test pass easily?" - STOP.
Ask instead: "how can I verify this actually works?"
```
