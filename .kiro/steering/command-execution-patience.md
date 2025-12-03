---
inclusion: always
---

# Command Execution Patience

## Critical Rule: ALWAYS Wait for Command Completion

**NEVER interrupt or create new commands while a previous command is still executing.**

### Execution Patience Guidelines:

1. **Wait Time**: Always wait at least 2-3 minutes for Python commands to complete
2. **Long Operations**: For complex operations (tests, installations, builds), wait up to 5-10 minutes
3. **No Interruption**: Never create alternative/simpler commands while waiting
4. **User Notification**: If a command takes longer than expected, the user will inform you
5. **Trust the Process**: Commands may appear slow but are often still running

### Indicators to Wait For:
- Python test suites (can take 1-5 minutes)
- Package installations 
- Database operations
- File processing operations
- Network operations

### What NOT to Do:
- ❌ Create "simpler" tests while waiting
- ❌ Assume a command failed if no immediate output
- ❌ Run multiple commands simultaneously
- ❌ Interrupt with alternative approaches

### What TO Do:
- ✅ Wait patiently for completion
- ✅ Trust that the command is running
- ✅ Let the user inform you of any issues
- ✅ Only proceed after seeing complete output

**Remember: Patience with command execution is critical for successful task completion.**