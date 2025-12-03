# Python Environment Configuration

## Conda Environment Usage

Always use the user's conda `agent` environment for Python operations:

- The user's conda environment is located at `$env:USERPROFILE\miniconda3\envs\agent\python.exe`
- If modules are unavailable, ask the user to install them in the conda agent environment
- The user has langdetect and other dependencies installed in this environment
- Do not assume system Python - always use the conda agent environment

## Command Pattern

When running Python commands, use this EXACT pattern:
```powershell
& "$env:USERPROFILE\miniconda3\envs\agent\python.exe" [command]
```

For Python scripts with path setup:
```powershell
& "$env:USERPROFILE\miniconda3\envs\agent\python.exe" -c "import sys; sys.path.append('docker_app/backend'); [your_code]"
```

## NEVER use these patterns (they don't work):
- `conda activate agent && python [command]` 
- `conda activate agent ; python [command]`

## Dependency Management

If a Python module is missing:
1. Ask the user to install it in their conda agent environment
2. Provide the installation command using the correct pattern above
3. Do not attempt to install packages automatically