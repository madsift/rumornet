---
inclusion: always
---

# No Premature Documentation

## CRITICAL RULE: TEST FIRST, DOCUMENT LATER

### ❌ NEVER Do This:
- Write extensive documentation before testing the code
- Create README files, guides, or examples without running them
- Document features that haven't been verified to work
- Write "how to use" guides before confirming functionality
- Create integration examples without testing integration

### ✅ ALWAYS Do This:
1. **Write the code**
2. **Test the code** - actually run it and verify it works
3. **Fix any issues** found during testing
4. **Only then** write minimal documentation if explicitly requested

### Documentation Rules:
- Documentation is ONLY created when:
  - User explicitly requests it
  - Code has been tested and verified working
  - There's a specific need (API reference, deployment guide, etc.)
- Keep documentation minimal and focused
- Prefer inline code comments over separate docs
- Don't create examples that haven't been tested

### What to Do Instead:
- Focus on making code work first
- Test thoroughly
- Fix bugs
- Add inline comments for complex logic
- Only document when user asks or when deploying to production

### Remember:
**Untested documentation is worse than no documentation - it wastes time and creates false confidence.**
