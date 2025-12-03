# Design Document

## Overview

The Agent Monitoring Dashboard is a Streamlit-based web application that provides comprehensive real-time monitoring and analysis capabilities for the RumorNet misinformation detection system. The dashboard integrates with the GranularMisinformationOrchestrator to display agent execution status, performance metrics, analysis results, and generate exportable markdown reports.

The system follows a reactive architecture where the Streamlit frontend communicates with the orchestrator backend through a state management layer. Real-time updates are achieved through Streamlit's session state and automatic rerun mechanisms.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Sidebar    │  │  Main View   │  │  Export      │     │
│  │  - Config    │  │  - Metrics   │  │  - Markdown  │     │
│  │  - Controls  │  │  - Results   │  │  - Download  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              State Management Layer                          │
│  - Session State (st.session_state)                         │
│  - Execution History                                         │
│  - Agent Status Tracking                                     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│         GranularMisinformationOrchestrator                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │Reasoning │  │ Pattern  │  │ Evidence │  │  Social  │   │
│  │  Agent   │  │  Agent   │  │  Agent   │  │  Agent   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│  ┌──────────┐  ┌──────────┐                                │
│  │  Topic   │  │  Other   │                                │
│  │  Agent   │  │  Agents  │                                │
│  └──────────┘  └──────────┘                                │
└─────────────────────────────────────────────────────────────┘
```

### Component Architecture

1. **Dashboard UI Layer**: Streamlit components for visualization and interaction
2. **State Management Layer**: Session state management for tracking execution and results
3. **Orchestrator Integration Layer**: Wrapper for orchestrator with monitoring hooks
4. **Agent Monitoring Layer**: Real-time status tracking for individual agents
5. **Export Layer**: Markdown generation and file download functionality

## Components and Interfaces

### 1. Dashboard Main Application (`dashboard.py`)

**Responsibilities:**
- Initialize Streamlit application
- Manage page layout and navigation
- Coordinate between UI components
- Handle user interactions

**Key Functions:**
```python
def main():
    """Main dashboard entry point"""
    
def initialize_session_state():
    """Initialize session state variables"""
    
def render_sidebar():
    """Render configuration and control sidebar"""
    
def render_main_view():
    """Render main dashboard view"""
```

### 2. Orchestrator Monitor (`orchestrator_monitor.py`)

**Responsibilities:**
- Wrap orchestrator with monitoring capabilities
- Track agent execution status
- Collect performance metrics
- Provide real-time status updates

**Key Classes:**
```python
class AgentStatus:
    """Track individual agent status"""
    agent_name: str
    status: str  # idle, executing, completed, failed
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    execution_time_ms: float
    error: Optional[str]

class OrchestatorMonitor:
    """Monitor orchestrator execution"""
    
    def __init__(self, orchestrator: GranularMisinformationOrchestrator)
    
    async def analyze_with_monitoring(self, posts: List[Dict]) -> Dict:
        """Execute analysis with real-time monitoring"""
    
    def get_agent_status(self, agent_name: str) -> AgentStatus:
        """Get current status of an agent"""
    
    def get_all_statuses(self) -> Dict[str, AgentStatus]:
        """Get status of all agents"""
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregated performance metrics"""
```

### 3. UI Components (`ui_components.py`)

**Responsibilities:**
- Reusable Streamlit UI components
- Consistent styling and formatting
- Data visualization widgets

**Key Functions:**
```python
def render_agent_status_card(agent_status: AgentStatus):
    """Render status card for single agent"""
    
def render_metrics_dashboard(metrics: Dict):
    """Render performance metrics dashboard"""
    
def render_results_table(results: List[Dict]):
    """Render analysis results in table format"""
    
def render_execution_timeline(statuses: Dict[str, AgentStatus]):
    """Render execution timeline visualization"""
    
def render_progress_bar(current: int, total: int):
    """Render progress bar for batch processing"""
```

### 4. Markdown Generator (`markdown_generator.py`)

**Responsibilities:**
- Generate comprehensive markdown reports
- Format tables and sections
- Include all analysis results

**Key Functions:**
```python
def generate_markdown_report(report: Dict) -> str:
    """Generate complete markdown report"""
    
def format_executive_summary(summary: Dict) -> str:
    """Format executive summary section"""
    
def format_high_priority_posts(posts: List[Dict]) -> str:
    """Format high-priority posts table"""
    
def format_top_offenders(users: List[Dict]) -> str:
    """Format top offenders table"""
    
def format_topic_analysis(topics: Dict) -> str:
    """Format topic analysis section"""
    
def format_temporal_trends(trends: Dict) -> str:
    """Format temporal trends section"""
```

### 5. Data Manager (`data_manager.py`)

**Responsibilities:**
- Manage execution history
- Handle data persistence
- Provide data access methods

**Key Functions:**
```python
def save_execution_result(result: Dict):
    """Save execution result to history"""
    
def load_execution_history() -> List[Dict]:
    """Load execution history"""
    
def get_execution_by_id(execution_id: str) -> Dict:
    """Get specific execution result"""
    
def clear_history():
    """Clear execution history"""
```

## Data Models

### AgentStatus
```python
@dataclass
class AgentStatus:
    agent_name: str
    status: str  # "idle", "executing", "completed", "failed"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time_ms: float = 0.0
    posts_processed: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### ExecutionMetrics
```python
@dataclass
class ExecutionMetrics:
    total_executions: int
    successful_executions: int
    failed_executions: int
    total_execution_time_ms: float
    average_execution_time_ms: float
    posts_per_second: float
    agent_metrics: Dict[str, Dict[str, Any]]
```

### ExecutionResult
```python
@dataclass
class ExecutionResult:
    execution_id: str
    timestamp: datetime
    total_posts: int
    posts_analyzed: int
    misinformation_detected: int
    high_risk_posts: int
    execution_time_ms: float
    agent_statuses: Dict[str, AgentStatus]
    full_report: Dict[str, Any]
    markdown_report: str
```

### DashboardConfig
```python
@dataclass
class DashboardConfig:
    ollama_endpoint: str
    ollama_model: str
    auto_refresh_interval: int  # seconds
    max_history_items: int
    default_batch_size: int
    enable_debug_mode: bool
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Agent status consistency
*For any* agent execution, when an agent transitions from "executing" to "completed", the execution time must be greater than zero and the end time must be after the start time.
**Validates: Requirements 1.2, 1.3**

### Property 2: Metrics accuracy
*For any* set of agent executions, the average execution time must equal the sum of all execution times divided by the count of executions.
**Validates: Requirements 2.4**

### Property 3: Progress tracking completeness
*For any* batch analysis, the sum of posts processed across all agent status updates must equal the total number of posts in the batch.
**Validates: Requirements 3.3**

### Property 4: Result data preservation
*For any* analysis execution, all data from the orchestrator report must be present in the dashboard display without loss or corruption.
**Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**

### Property 5: Markdown export completeness
*For any* generated markdown report, all sections present in the dashboard view must have corresponding sections in the markdown output.
**Validates: Requirements 5.2, 5.3**

### Property 6: Historical data integrity
*For any* execution result saved to history, retrieving that result by ID must return data identical to what was originally saved.
**Validates: Requirements 6.4**

### Property 7: Configuration validation
*For any* configuration update, if the configuration contains invalid values, the dashboard must reject the update and display an error message.
**Validates: Requirements 7.2, 7.5**

### Property 8: Filter consistency
*For any* applied filter on results, the filtered dataset must contain only items that match all filter criteria.
**Validates: Requirements 9.2, 9.3, 9.4, 9.5**

### Property 9: Error logging completeness
*For any* error that occurs during execution, the error must be logged with a timestamp, component name, and error message.
**Validates: Requirements 10.1, 10.3**

### Property 10: Status update ordering
*For any* agent execution sequence, status updates must occur in the correct order: idle → executing → (completed | failed).
**Validates: Requirements 1.1, 1.2, 1.3, 1.4**

## Error Handling

### Error Categories

1. **Orchestrator Errors**: Failures in agent initialization or execution
   - Display error in dedicated error panel
   - Log full stack trace
   - Allow retry with same configuration

2. **Data Input Errors**: Invalid post data or file format
   - Validate input before processing
   - Display clear validation messages
   - Provide example format

3. **Configuration Errors**: Invalid settings
   - Validate before applying
   - Show specific validation errors
   - Revert to previous valid configuration

4. **Export Errors**: Failures in markdown generation or download
   - Catch and display error
   - Provide fallback text export
   - Log error details

### Error Recovery Strategies

- **Graceful Degradation**: If one agent fails, continue with others
- **Retry Logic**: Allow manual retry for failed operations
- **State Preservation**: Maintain valid state even after errors
- **User Feedback**: Always inform user of errors with actionable messages

## Testing Strategy

### Unit Testing

Unit tests will verify individual components and functions:

- **UI Component Tests**: Test rendering functions with mock data
- **Markdown Generator Tests**: Verify correct formatting for various inputs
- **Data Manager Tests**: Test save/load operations with sample data
- **Configuration Validation Tests**: Test validation logic with valid and invalid configs

### Property-Based Testing

Property-based tests will verify universal properties using the Hypothesis library for Python:

- **Property 1 Test**: Generate random agent executions and verify time consistency
- **Property 2 Test**: Generate random execution sets and verify metrics calculations
- **Property 3 Test**: Generate random batch sizes and verify progress tracking
- **Property 4 Test**: Generate random orchestrator reports and verify data preservation
- **Property 5 Test**: Generate random reports and verify markdown completeness
- **Property 6 Test**: Generate random execution results and verify storage/retrieval
- **Property 7 Test**: Generate random configurations (valid and invalid) and verify validation
- **Property 8 Test**: Generate random result sets and filter criteria, verify filtering
- **Property 9 Test**: Generate random errors and verify logging completeness
- **Property 10 Test**: Generate random agent execution sequences and verify status ordering

Each property-based test will run a minimum of 100 iterations to ensure comprehensive coverage across the input space.

### Integration Testing

Integration tests will verify end-to-end workflows:

- Dashboard initialization with orchestrator
- Complete analysis workflow from input to results
- Export workflow from results to markdown file
- Configuration update workflow
- Historical data access workflow

### Testing Configuration

- **Framework**: pytest for unit tests, Hypothesis for property-based tests
- **Minimum Iterations**: 100 per property test
- **Coverage Target**: 80% code coverage
- **Mock Strategy**: Mock orchestrator for UI tests, use real orchestrator for integration tests

## Implementation Notes

### Streamlit-Specific Considerations

1. **Session State Management**: Use `st.session_state` for persistent data across reruns
2. **Async Execution**: Use `asyncio.run()` to execute async orchestrator methods
3. **Progress Updates**: Use `st.progress()` and `st.status()` for real-time feedback
4. **Auto-refresh**: Use `st.rerun()` with time delays for periodic updates
5. **File Upload**: Use `st.file_uploader()` for post data input
6. **Download**: Use `st.download_button()` for markdown export

### Performance Considerations

1. **Caching**: Use `@st.cache_data` for expensive computations
2. **Lazy Loading**: Load historical data only when requested
3. **Pagination**: Paginate large result sets
4. **Async Processing**: Run orchestrator in background to keep UI responsive

### Security Considerations

1. **Input Validation**: Sanitize all user inputs
2. **File Upload Limits**: Restrict file size and type
3. **Configuration Validation**: Validate all configuration values
4. **Error Message Sanitization**: Don't expose sensitive information in errors

## Technology Stack

- **Frontend**: Streamlit 1.28+
- **Backend**: Python 3.8+
- **Orchestrator**: GranularMisinformationOrchestrator (existing)
- **Data Persistence**: JSON files for execution history
- **Testing**: pytest, Hypothesis
- **Visualization**: Plotly (via Streamlit)
- **Markdown**: Python markdown library

## Deployment Considerations

1. **Local Development**: Run with `streamlit run dashboard.py`
2. **Production**: Deploy to Streamlit Cloud or containerize with Docker
3. **Configuration**: Use environment variables for sensitive settings
4. **Logging**: Configure logging to file for production debugging
5. **Monitoring**: Add health check endpoint for production monitoring
