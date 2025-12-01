# RumorNet Agent Monitoring Dashboard

A comprehensive Streamlit-based monitoring dashboard for the RumorNet misinformation detection system.

## Features

- Real-time agent execution monitoring
- Performance metrics and analytics
- Batch analysis processing
- Interactive results visualization
- Markdown report generation and export
- Execution history tracking
- Advanced filtering and search
- Error logging and debugging tools

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure the dashboard:
Edit `config.yaml` to set your orchestrator endpoint and preferences.

## Usage

Run the dashboard:
```bash
streamlit run dashboard.py
```

The dashboard will be available at `http://localhost:8501`

## Configuration

Configuration is managed through `config.yaml`. Key settings include:

- **Orchestrator Settings**: Ollama endpoint and model configuration
- **Dashboard Settings**: Auto-refresh interval, history limits, debug mode
- **Data Persistence**: Directories for history and exports
- **Visualization**: Color schemes and chart settings
- **Logging**: Log level and file location

## Directory Structure

```
dashboard/
├── __init__.py              # Package initialization
├── dashboard.py             # Main application entry point
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── components/             # UI components
│   └── __init__.py
├── core/                   # Core functionality
│   ├── __init__.py
│   ├── orchestrator_monitor.py
│   ├── data_manager.py
│   └── markdown_generator.py
├── models/                 # Data models
│   └── __init__.py
├── utils/                  # Utility functions
│   └── __init__.py
├── data/                   # Data storage
│   ├── history/           # Execution history
│   └── exports/           # Exported reports
├── logs/                   # Log files
└── tests/                  # Test files
    ├── __init__.py
    ├── test_orchestrator_monitor.py
    ├── test_data_manager.py
    ├── test_markdown_generator.py
    └── property_tests/     # Property-based tests
        └── __init__.py
```

## Testing

Run unit tests:
```bash
pytest tests/
```

Run property-based tests:
```bash
pytest tests/property_tests/
```

## Development

The dashboard is built with:
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **Hypothesis**: Property-based testing
- **pytest**: Unit testing framework

## License

[Add license information]


## Quick Start Guide

### Running the Dashboard

**Option 1: Using the launcher script (Windows)**
```bash
# From project root
run_dashboard.bat
```

**Option 2: Using Streamlit directly**
```bash
cd dashboard
streamlit run dashboard.py
```

**Option 3: From project root**
```bash
streamlit run dashboard/dashboard.py
```

The dashboard will automatically open in your default web browser at `http://localhost:8501`

## Dashboard Sections

### 📊 Overview Tab
- **Real-time Agent Status**: Monitor all agents with live status updates
- **Executive Summary**: Key metrics and statistics at a glance
- **Performance Metrics**: Execution times, success rates, throughput
- **Agent Grid**: Visual grid showing all agents and their current state

### 🔬 Analysis Tab
- **Batch Analysis Interface**: Upload or paste post data for analysis
- **File Upload**: Support for JSON and CSV files
- **Text Input**: Direct paste of post data
- **Progress Tracking**: Real-time progress during analysis
- **Status Updates**: Live agent execution status

### 📈 Results Tab
- **Detailed Results Display**: Comprehensive analysis results
- **High-Priority Posts**: Posts requiring immediate attention
- **Top Offenders**: Users with highest misinformation rates
- **Pattern Breakdown**: Detected patterns and their frequencies
- **Topic Analysis**: Topic-based misinformation analysis
- **Temporal Trends**: Time-based analysis patterns
- **Load from History**: Access previous analysis results

### 🔄 Execution Flow Tab
- **Pipeline Visualization**: See agents in execution order
- **Data Flow Diagram**: Visualize data movement through pipeline
- **Execution Timeline**: Detailed timeline with duration bars
- **Parallel Analysis**: View parallel vs sequential execution
- **Efficiency Metrics**: Time savings from parallelization

### 📚 History Tab
- **Execution History**: Browse all past analyses
- **Historical Comparison**: Compare multiple runs
- **Trends Visualization**: Performance trends over time
- **Quick Load**: Load any previous result instantly

### 📄 Export Tab
- **Markdown Generation**: Create comprehensive reports
- **Preview**: View report before downloading
- **Download**: Export as .md file
- **Customization**: Configure export options
- **Metadata**: Automatic timestamps and execution info

### ⚠️ Errors Tab
- **Error Display**: All errors with timestamps
- **Stack Traces**: Full debugging information
- **Component Tracking**: See which component had errors
- **Log Filtering**: Filter by severity and component
- **Clear Log**: Reset error log when needed

## Configuration

### Sidebar Settings

The sidebar provides quick access to:

1. **Ollama Configuration**
   - Endpoint URL
   - Model selection
   - Connection settings

2. **Dashboard Settings**
   - Auto-refresh interval
   - Maximum history items
   - Default batch size
   - Debug mode

3. **Quick Stats**
   - Executing agents count
   - Completed agents count
   - Failed agents count
   - Total agents

4. **Auto-Refresh Controls**
   - Enable/disable toggle
   - Interval slider (1-30 seconds)
   - Countdown timer

### Auto-Refresh Feature

Enable auto-refresh to keep the dashboard updated automatically:

1. Check "Enable Auto-Refresh" in sidebar
2. Set refresh interval (default: 5 seconds)
3. Dashboard will update automatically
4. Countdown shows time until next refresh

## Navigation Tips

### Tab Navigation
- Click tab names to switch views
- Current tab is highlighted
- All tabs maintain their state

### Sidebar
- Always visible for quick access
- Collapsible on mobile devices
- Persistent configuration

### Keyboard Shortcuts
- `R`: Refresh dashboard
- `Ctrl+K`: Focus search
- `Esc`: Close modals

## Session State Management

The dashboard maintains state across interactions:

- **Agent Statuses**: Current state of all agents
- **Execution History**: Past analysis runs
- **Current Results**: Active analysis results
- **Configuration**: User preferences
- **Filters**: Search and filter settings
- **Error Log**: All logged errors

State persists during the session but resets on browser refresh.

## Common Workflows

### Running an Analysis

1. Go to **Analysis** tab
2. Upload file or paste post data
3. Click "Start Analysis"
4. Monitor progress in real-time
5. View results when complete

### Viewing Results

1. Go to **Results** tab
2. Browse detailed analysis
3. Use filters to focus on specific data
4. Export results if needed

### Monitoring Execution

1. Go to **Execution Flow** tab
2. View pipeline visualization
3. Check timeline for bottlenecks
4. Analyze parallel execution efficiency

### Exporting Reports

1. Go to **Export** tab
2. Preview markdown report
3. Customize export options
4. Download report file

## Troubleshooting

### Dashboard Won't Start

**Problem**: Error when running `streamlit run dashboard.py`

**Solutions**:
- Ensure Streamlit is installed: `pip install streamlit`
- Check Python version (3.7+ required)
- Install all dependencies: `pip install -r requirements.txt`
- Try running from dashboard directory

### No Data Displayed

**Problem**: Dashboard shows "No data available"

**Solutions**:
- Run an analysis first (Analysis tab)
- Check orchestrator initialization
- Verify agent configuration
- Check error log for issues

### Auto-Refresh Not Working

**Problem**: Dashboard doesn't update automatically

**Solutions**:
- Enable auto-refresh in sidebar
- Check refresh interval setting
- Ensure browser allows auto-refresh
- Check for JavaScript errors in console

### Errors Not Showing

**Problem**: Errors occur but don't appear

**Solutions**:
- Check the Errors tab
- Enable debug mode in configuration
- View browser console (F12)
- Check terminal output

### Performance Issues

**Problem**: Dashboard is slow or unresponsive

**Solutions**:
- Reduce auto-refresh interval
- Clear execution history
- Disable debug mode
- Close unused browser tabs
- Restart dashboard

## Advanced Features

### Custom Styling

Add custom CSS in `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### Data Persistence

Enable data persistence by configuring the data manager:

```python
# In dashboard configuration
data_manager = DataManager(
    history_dir="./data/history",
    export_dir="./data/exports"
)
```

### Integration with Orchestrator

The dashboard integrates with `GranularMisinformationOrchestrator`:

```python
from dashboard.core import OrchestratorMonitor

# Wrap orchestrator with monitoring
monitor = OrchestratorMonitor(orchestrator)

# Run analysis with monitoring
result = await monitor.analyze_with_monitoring(posts)

# Get agent statuses
statuses = monitor.get_all_statuses()
```

## API Reference

### Main Functions

- `initialize_dashboard()`: Initialize dashboard state
- `render_sidebar()`: Render sidebar UI
- `render_main_content()`: Render main content area
- `render_header()`: Render dashboard header
- `render_footer()`: Render dashboard footer

### State Management

- `initialize_session_state()`: Initialize session state
- `get_agent_status()`: Get specific agent status
- `get_all_agent_statuses()`: Get all agent statuses
- `add_execution_result()`: Add result to history
- `get_execution_history()`: Get execution history

## Development

### Project Structure

```
dashboard/
├── dashboard.py              # Main application entry point
├── components/               # UI components
│   ├── ui_components.py     # Basic UI elements
│   ├── results_display.py   # Results visualization
│   ├── history_viewer.py    # History management
│   ├── markdown_export.py   # Report generation
│   ├── error_logging.py     # Error handling
│   ├── execution_flow.py    # Flow visualization
│   ├── batch_analysis.py    # Analysis interface
│   ├── config_ui.py         # Configuration UI
│   └── filtering.py         # Filtering logic
├── core/                     # Core functionality
│   ├── orchestrator_monitor.py  # Orchestrator wrapper
│   ├── data_manager.py          # Data persistence
│   └── state_manager.py         # State management
├── models/                   # Data models
│   └── data_models.py       # Core data structures
├── utils/                    # Utilities
│   ├── config_loader.py     # Configuration loading
│   ├── config_manager.py    # Configuration management
│   └── markdown_generator.py # Markdown generation
├── tests/                    # Test suite
│   ├── test_*.py            # Unit tests
│   └── property_tests/      # Property-based tests
├── examples/                 # Example scripts
│   └── execution_flow_demo.py
└── docs/                     # Documentation
    ├── execution_flow_implementation.md
    └── execution_flow_integration_guide.md
```

### Running Tests

**All tests**:
```bash
cd dashboard
pytest tests/ -v
```

**Specific test file**:
```bash
pytest tests/test_execution_flow.py -v
```

**Property-based tests**:
```bash
pytest tests/property_tests/ -v
```

**With coverage**:
```bash
pytest tests/ --cov=dashboard --cov-report=html
```

### Adding New Features

1. Create component in `components/`
2. Add to `components/__init__.py`
3. Import in `dashboard.py`
4. Add to appropriate tab
5. Write tests in `tests/`
6. Update documentation

## Performance Optimization

### Caching

Use Streamlit caching for expensive operations:

```python
@st.cache_data
def expensive_computation(data):
    # Your computation here
    return result
```

### Lazy Loading

Load data only when needed:

```python
if st.button("Load Details"):
    # Load detailed data only when requested
    details = load_detailed_data()
    st.write(details)
```

### Pagination

Paginate large datasets:

```python
page_size = 10
page = st.number_input("Page", min_value=1, max_value=total_pages)
start = (page - 1) * page_size
end = start + page_size
st.dataframe(data[start:end])
```

## Security Considerations

- Input validation on all user inputs
- File upload size limits
- Configuration value validation
- Error message sanitization
- No sensitive data in logs

## Support

For issues, questions, or contributions:

1. Check this README
2. Review documentation in `docs/`
3. Check example scripts in `examples/`
4. Review test files for usage examples

## Version History

- **v1.0**: Initial release with all core features
  - Real-time monitoring
  - Batch analysis
  - Results visualization
  - Execution flow
  - History tracking
  - Markdown export
  - Error logging

## License

Part of the RumorNet Misinformation Detection System
