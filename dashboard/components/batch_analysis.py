"""
Batch analysis workflow components for the Agent Monitoring Dashboard.

This module provides UI components for batch analysis including
input handling, progress tracking, and results display.
"""

import streamlit as st
import json
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from models.data_models import ExecutionResult, AgentStatus
from core.orchestrator_monitor import OrchestratorMonitor
from core.data_manager import DataManager
from utils.markdown_generator import MarkdownGenerator
from utils.data_transformer import (
    transform_posts_batch, 
    enrich_post_metadata,
    extract_posts_from_nested_structure
)
import os


def render_batch_input_ui() -> Optional[List[Dict[str, Any]]]:
    """
    Render input UI for batch analysis.
    
    Provides file upload and text area options for inputting post data.
    Validates and parses input data.
    
    Returns:
        List of post dictionaries if valid input provided, None otherwise
        
    Requirements: 3.1
    """
    st.subheader("📥 Input Post Data")
    
    posts = None
    
    # Demo data button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("📂 Load Demo Data", help="Load example data from examples/demo1.json"):
            # Try multiple possible paths
            possible_paths = [
                os.path.join("examples", "demo1.json"),
                os.path.join("dashboard", "examples", "demo1.json"),
                os.path.join("..", "dashboard", "examples", "demo1.json")
            ]
            
            demo_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    demo_path = path
                    break
            
            if demo_path:
                try:
                    with open(demo_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Extract posts from nested structure
                    raw_posts = extract_posts_from_nested_structure(data)
                    
                    if raw_posts:
                        # Transform posts to standard format
                        posts = transform_posts_batch(raw_posts, format_type="auto")
                        
                        # Enrich with metadata
                        posts = [enrich_post_metadata(post) for post in posts]
                        
                        # Store in session state
                        st.session_state['demo_posts_loaded'] = posts
                        st.success(f"✅ Loaded {len(posts)} demo posts from {demo_path}!")
                        st.rerun()
                    else:
                        st.error("❌ Could not extract posts from demo file")
                except Exception as e:
                    st.error(f"❌ Error loading demo file: {str(e)}")
            else:
                st.error(f"❌ Demo file not found. Tried: {', '.join(possible_paths)}")
    
    # Check if demo posts were loaded
    if 'demo_posts_loaded' in st.session_state and st.session_state['demo_posts_loaded']:
        demo_posts = st.session_state['demo_posts_loaded']
        st.info(f"📊 Demo data loaded: {len(demo_posts)} posts ready for analysis")
        return demo_posts
    
    # Input method selection
    input_method = st.radio(
        "Select input method:",
        options=["File Upload", "Text Input"],
        horizontal=True
    )
    
    posts = None
    
    if input_method == "File Upload":
        uploaded_file = st.file_uploader(
            "Upload JSON file with posts",
            type=["json"],
            help="Upload a JSON file containing an array of post objects"
        )
        
        if uploaded_file is not None:
            try:
                # Read and parse JSON file
                content = uploaded_file.read().decode('utf-8')
                
                # Try to parse as standard JSON first
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    # If that fails, try to fix comma-separated objects
                    # Your format: },{ without array brackets
                    content = content.strip()
                    if not content.startswith('['):
                        content = '[' + content
                    if not content.endswith(']'):
                        content = content + ']'
                    data = json.loads(content)
                
                # Extract posts from various structures
                raw_posts = extract_posts_from_nested_structure(data)
                
                if not raw_posts:
                    st.error("❌ Invalid file format. Could not find posts in the data structure")
                    raw_posts = None
                
                if raw_posts:
                    # Transform posts to standard format
                    posts = transform_posts_batch(raw_posts, format_type="auto")
                    
                    # Enrich with metadata
                    posts = [enrich_post_metadata(post) for post in posts]
                    
                    st.success(f"✅ Loaded and transformed {len(posts)} posts from file")
                
            except json.JSONDecodeError as e:
                st.error(f"❌ Invalid JSON format: {str(e)}")
                st.info("💡 Tip: Make sure your JSON is properly formatted as an array: [{...}, {...}]")
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    else:  # Text Input
        text_input = st.text_area(
            "Paste JSON data",
            height=200,
            placeholder='[{"post_id": "1", "text": "Sample post", ...}, ...]',
            help="Paste JSON array of post objects"
        )
        
        if text_input.strip():
            try:
                # Try to parse as standard JSON first
                try:
                    data = json.loads(text_input)
                except json.JSONDecodeError:
                    # If that fails, try to fix comma-separated objects
                    content = text_input.strip()
                    if not content.startswith('['):
                        content = '[' + content
                    if not content.endswith(']'):
                        content = content + ']'
                    data = json.loads(content)
                
                # Extract posts from various structures
                raw_posts = extract_posts_from_nested_structure(data)
                
                if not raw_posts:
                    st.error("❌ Invalid format. Could not find posts in the data structure")
                    raw_posts = None
                
                if raw_posts:
                    # Transform posts to standard format
                    posts = transform_posts_batch(raw_posts, format_type="auto")
                    
                    # Enrich with metadata
                    posts = [enrich_post_metadata(post) for post in posts]
                    
                    st.success(f"✅ Parsed and transformed {len(posts)} posts from input")
                
            except json.JSONDecodeError as e:
                st.error(f"❌ Invalid JSON format: {str(e)}")
                st.info("💡 Tip: Make sure your JSON is properly formatted as an array: [{...}, {...}]")
            except Exception as e:
                st.error(f"❌ Error parsing input: {str(e)}")
    
    # Display sample format
    with st.expander("📖 View Sample Formats"):
        st.write("**Reddit Format (Auto-detected):**")
        st.code('''[
  {
    "submission_id": "15uzos2",
    "author_name": "user_42486",
    "posts": "Brazil arrests police officials over January 8 attacks",
    "score": 25,
    "num_comments": 1,
    "upvote_ratio": 0.91,
    "created_utc": "2023-08-19 07:20:07",
    "subreddit": "GlobalTalk"
  }
]''', language='json')
        
        st.write("**Standard Format:**")
        st.code('''[
  {
    "post_id": "post_001",
    "user_id": "user_123",
    "username": "example_user",
    "text": "This is the post content...",
    "timestamp": "2024-01-01T12:00:00",
    "platform": "reddit",
    "subreddit": "news",
    "upvotes": 100,
    "comments": 25,
    "shares": 10
  }
]''', language='json')
    
    return posts


def render_analysis_controls(
    posts: List[Dict[str, Any]],
    orchestrator_monitor: Optional[OrchestratorMonitor] = None
) -> bool:
    """
    Render analysis control buttons.
    
    Provides buttons to start analysis with different modes.
    
    Args:
        posts: List of posts to analyze
        orchestrator_monitor: Optional orchestrator monitor instance
        
    Returns:
        True if analysis should start, False otherwise
        
    Requirements: 3.2
    """
    st.subheader("🚀 Analysis Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Posts to Analyze", len(posts))
        if len(posts) > 50:
            st.warning(f"⚠️ Large batch ({len(posts)} posts) may take 10-20 minutes")
    
    with col2:
        # Analysis mode selection
        use_batch = st.checkbox(
            "Use Batch Mode",
            value=True,
            help="Process posts in batch (faster) vs sequential"
        )
    
    with col3:
        # Start button - always enabled to show what's needed
        start_analysis = st.button(
            "▶️ Start Analysis",
            type="primary"
        )
    
    # Debug info
    with st.expander("🔍 Debug Info"):
        st.write(f"**Orchestrator Status:**")
        st.write(f"- orchestrator_monitor is None: {orchestrator_monitor is None}")
        if orchestrator_monitor:
            st.write(f"- orchestrator_monitor type: {type(orchestrator_monitor)}")
            st.write(f"- Has orchestrator: {hasattr(orchestrator_monitor, 'orchestrator')}")
            if hasattr(orchestrator_monitor, 'orchestrator'):
                st.write(f"- Orchestrator agents: {list(orchestrator_monitor.orchestrator.agents.keys()) if orchestrator_monitor.orchestrator.agents else 'None'}")
        st.write(f"**Session State:**")
        st.write(f"- Has orchestrator_monitor in session: {'orchestrator_monitor' in st.session_state}")
        if 'orchestrator_monitor' in st.session_state:
            st.write(f"- Session orchestrator is None: {st.session_state.orchestrator_monitor is None}")
    
    if orchestrator_monitor is None:
        st.error("❌ **Orchestrator not initialized!**")
        st.write("**To run analysis, you need to:**")
        st.write("1. Initialize the GranularMisinformationOrchestrator")
        st.write("2. Configure Ollama endpoint and model")
        st.write("3. Ensure all agents are available")
        st.write("")
        st.info("💡 **Quick Setup:** Use `start_dashboard_with_orchestrator.py` to start the dashboard with orchestrator pre-initialized.")
        
        # Show example code
        with st.expander("📝 Show Example Setup Code"):
            st.code('''
# Example: Initialize orchestrator before starting dashboard
from granular_misinformation_orchestrator import GranularMisinformationOrchestrator
from core.orchestrator_monitor import OrchestratorMonitor

# Create orchestrator
config = {
    "ollama_endpoint": "http://192.168.10.68:11434",
    "ollama_model": "gemma3:4b"
}
orchestrator = GranularMisinformationOrchestrator(config=config)

# Initialize agents
await orchestrator.initialize_agents()

# Create monitor
monitor = OrchestratorMonitor(orchestrator)

# Store in session state
st.session_state.orchestrator_monitor = monitor
''', language='python')
    
    # Store batch mode in session state
    if start_analysis:
        st.write("🔴 **DEBUG: Button was clicked!**")
        if orchestrator_monitor is None:
            st.warning("⚠️ Cannot start analysis - orchestrator not initialized. See instructions above.")
            return False
        st.write("🟢 **DEBUG: Orchestrator is available, returning True**")
        st.session_state['use_batch_mode'] = use_batch
        return True
    
    st.write("⚪ **DEBUG: Button was NOT clicked this cycle**")
    return False


async def run_batch_analysis(
    posts: List[Dict[str, Any]],
    orchestrator_monitor: OrchestratorMonitor,
    use_batch: bool = True
) -> Dict[str, Any]:
    """
    Run batch analysis with progress tracking.
    
    Executes analysis through orchestrator monitor and tracks progress.
    
    Args:
        posts: List of posts to analyze
        orchestrator_monitor: Orchestrator monitor instance
        use_batch: Whether to use batch mode
        
    Returns:
        Analysis results dictionary
        
    Requirements: 3.2, 3.3
    """
    # Run analysis
    results = await orchestrator_monitor.analyze_with_monitoring(
        posts=posts,
        use_batch=use_batch
    )
    
    return results


def render_analysis_progress(
    current: int,
    total: int,
    agent_statuses: Dict[str, AgentStatus]
):
    """
    Render analysis progress indicators.
    
    Displays progress bar and agent status during analysis.
    
    Args:
        current: Current number of posts processed
        total: Total number of posts
        agent_statuses: Dictionary of agent statuses
        
    Requirements: 3.3
    """
    st.subheader("⏳ Analysis in Progress")
    
    # Progress bar
    if total > 0:
        progress = current / total
        st.progress(progress, text=f"Processing: {current}/{total} posts ({progress*100:.1f}%)")
    
    # Agent status indicators
    st.write("**Agent Status:**")
    
    cols = st.columns(4)
    for i, (agent_name, status) in enumerate(agent_statuses.items()):
        col_idx = i % 4
        with cols[col_idx]:
            # Status emoji
            if status.status == "executing":
                emoji = "🟡"
            elif status.status == "completed":
                emoji = "🟢"
            elif status.status == "failed":
                emoji = "🔴"
            else:
                emoji = "⚪"
            
            st.write(f"{emoji} {agent_name}")
            if status.execution_time_ms > 0:
                st.caption(f"{status.execution_time_ms:.0f}ms")


def render_analysis_summary(results: Dict[str, Any]):
    """
    Render analysis summary statistics.
    
    Displays summary metrics after analysis completion.
    
    Args:
        results: Analysis results dictionary
        
    Requirements: 3.4
    """
    st.subheader("📊 Analysis Summary")
    
    # Extract summary data
    report = results.get("report", {})
    exec_summary = report.get("executive_summary", {})
    execution_time_ms = results.get("execution_time_ms", 0)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Posts",
            exec_summary.get("total_posts_analyzed", 0)
        )
    
    with col2:
        misinfo_count = exec_summary.get("misinformation_detected", 0)
        total = exec_summary.get("total_posts_analyzed", 1)
        misinfo_rate = (misinfo_count / total * 100) if total > 0 else 0
        st.metric(
            "Misinformation",
            misinfo_count,
            delta=f"{misinfo_rate:.1f}%",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "High Risk",
            exec_summary.get("high_risk_posts", 0)
        )
    
    with col4:
        st.metric(
            "Execution Time",
            f"{execution_time_ms/1000:.2f}s"
        )
    
    # Additional metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Critical Posts",
            exec_summary.get("critical_posts", 0)
        )
    
    with col2:
        st.metric(
            "Unique Users",
            exec_summary.get("unique_users", 0)
        )
    
    with col3:
        st.metric(
            "Patterns Detected",
            exec_summary.get("patterns_detected", 0)
        )
    
    with col4:
        st.metric(
            "Topics Identified",
            exec_summary.get("topics_identified", 0)
        )


def render_analysis_error(error: str, details: Optional[Dict[str, Any]] = None):
    """
    Render analysis error display.
    
    Shows error messages with actionable information.
    
    Args:
        error: Error message
        details: Optional error details dictionary
        
    Requirements: 3.5
    """
    st.error("❌ Analysis Failed")
    
    st.write("**Error Message:**")
    st.code(error, language=None)
    
    if details:
        with st.expander("🔍 Error Details"):
            st.json(details)
    
    # Actionable suggestions
    st.write("**Troubleshooting Steps:**")
    st.write("1. Check that the orchestrator is properly initialized")
    st.write("2. Verify that all required agents are available")
    st.write("3. Ensure post data is in the correct format")
    st.write("4. Check system logs for more details")
    st.write("5. Try analyzing a smaller batch of posts")


def save_analysis_results(
    results: Dict[str, Any],
    posts: List[Dict[str, Any]],
    data_manager: DataManager
) -> Optional[str]:
    """
    Save analysis results to history.
    
    Creates ExecutionResult and saves to data manager.
    
    Args:
        results: Analysis results dictionary
        posts: Original posts list
        data_manager: DataManager instance
        
    Returns:
        Execution ID if successful, None otherwise
    """
    try:
        # Extract data from results
        report = results.get("report", {})
        exec_summary = report.get("executive_summary", {})
        agent_statuses = results.get("agent_statuses", {})
        execution_time_ms = results.get("execution_time_ms", 0)
        
        # Generate markdown report
        markdown_generator = MarkdownGenerator()
        markdown_report = markdown_generator.generate_markdown_report(report)
        
        # Create execution result
        execution_id = f"exec_{int(datetime.now().timestamp())}"
        
        execution_result = ExecutionResult(
            execution_id=execution_id,
            timestamp=datetime.now(),
            total_posts=len(posts),
            posts_analyzed=exec_summary.get("total_posts_analyzed", 0),
            misinformation_detected=exec_summary.get("misinformation_detected", 0),
            high_risk_posts=exec_summary.get("high_risk_posts", 0),
            execution_time_ms=execution_time_ms,
            agent_statuses=agent_statuses,
            full_report=report,
            markdown_report=markdown_report
        )
        
        # Save to data manager
        if data_manager.save_execution_result(execution_result):
            return execution_id
        else:
            return None
            
    except Exception as e:
        logging.error(f"Failed to save analysis results: {e}")
        return None


def render_orchestrator_setup():
    """
    Render orchestrator setup section.
    
    Provides UI for initializing the orchestrator if not already set up.
    """
    from components.orchestrator_init import render_orchestrator_init_ui
    
    st.info("🔧 **Orchestrator Setup Required**")
    
    st.write("""
    The orchestrator needs to be initialized before you can run analysis.
    Initialize it below or ensure Ollama is running.
    """)
    
    # Add initialization UI
    render_orchestrator_init_ui()
    
    with st.expander("📖 Setup Instructions"):
        st.markdown("""
        ### Option 1: Initialize from Code
        
        Before starting the dashboard, run:
        
        ```python
        from granular_misinformation_orchestrator import GranularMisinformationOrchestrator
        from core.orchestrator_monitor import OrchestratorMonitor
        import asyncio
        
        # Create orchestrator
        config = {
            "ollama_endpoint": "http://192.168.10.68:11434",
            "ollama_model": "gemma3:4b"
        }
        orchestrator = GranularMisinformationOrchestrator(config=config)
        
        # Initialize agents
        asyncio.run(orchestrator.initialize_agents())
        
        # Create monitor and store in a way the dashboard can access
        monitor = OrchestratorMonitor(orchestrator)
        ```
        
        ### Option 2: Use Existing Orchestrator Script
        
        If you have `granular_misinformation_orchestrator.py` in your project:
        
        1. Make sure Ollama is running
        2. The orchestrator should be initialized in your main script
        3. Pass it to the dashboard when starting
        
        ### Check Ollama Status
        
        Make sure Ollama is running:
        ```bash
        # Check if Ollama is running
        curl http://192.168.10.68:11434/api/tags
        ```
        """)
    
    st.warning("⚠️ Once the orchestrator is initialized, restart the dashboard or refresh the page.")


def render_batch_analysis_workflow(
    orchestrator_monitor: Optional[OrchestratorMonitor] = None,
    data_manager: Optional[DataManager] = None
):
    """
    Render complete batch analysis workflow.
    
    Combines all batch analysis components into a complete workflow.
    
    Args:
        orchestrator_monitor: Optional orchestrator monitor instance
        data_manager: Optional data manager instance
        
    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
    """
    # Ensure asyncio is available (already imported at module level but being explicit)
    import asyncio as async_lib
    
    st.header("🔬 Batch Analysis")
    
    # Show analysis status at the top
    if st.session_state.get('analysis_running', False):
        st.warning("⚡ ANALYSIS IS RUNNING - Please wait...")
    
    # Auto-initialize orchestrator if not available
    if orchestrator_monitor is None and not st.session_state.get('orchestrator_init_attempted', False):
        with st.spinner("🔧 Auto-initializing orchestrator with default configuration..."):
            st.session_state.orchestrator_init_attempted = True
            
            try:
                # Create config with correct endpoint and model
                config = {
                    "ollama_endpoint": "http://192.168.10.68:11434",
                    "ollama_model": "gemma3:4b"
                }
                
                st.write("📦 Creating orchestrator...")
                
                # Create orchestrator
                from granular_misinformation_orchestrator import GranularMisinformationOrchestrator
                from core.orchestrator_monitor import OrchestratorMonitor
                
                orchestrator = GranularMisinformationOrchestrator(config=config)
                
                st.write("🤖 Initializing agents (this may take 30-60 seconds)...")
                
                # Initialize agents (asyncio already imported at top)
                try:
                    loop = async_lib.get_event_loop()
                except RuntimeError:
                    loop = async_lib.new_event_loop()
                    async_lib.set_event_loop(loop)
                
                loop.run_until_complete(orchestrator.initialize_agents())
                
                st.write("📊 Creating monitor...")
                
                # Create monitor
                monitor = OrchestratorMonitor(orchestrator)
                
                # Store in session state
                st.session_state.orchestrator_monitor = monitor
                orchestrator_monitor = monitor
                
                st.success(f"✅ Orchestrator initialized with {len(orchestrator.agents)} agents!")
                st.info("🔄 Refreshing page...")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Auto-initialization failed: {str(e)}")
                with st.expander("Error Details"):
                    import traceback
                    st.code(traceback.format_exc())
    
    # Show setup instructions if orchestrator not available
    if orchestrator_monitor is None:
        render_orchestrator_setup()
        st.divider()
    
    # Initialize session state
    if 'analysis_running' not in st.session_state:
        st.session_state['analysis_running'] = False
    if 'analysis_results' not in st.session_state:
        st.session_state['analysis_results'] = None
    if 'analysis_posts' not in st.session_state:
        st.session_state['analysis_posts'] = None
    
    # Step 1: Input
    posts = None
    if not st.session_state['analysis_running']:
        posts = render_batch_input_ui()
        
        if posts and len(posts) > 0:
            st.session_state['analysis_posts'] = posts
            
            # Step 2: Controls
            should_start = render_analysis_controls(posts, orchestrator_monitor)
            
            if should_start:
                st.balloons()
                st.success("🚀 **Button clicked! Starting analysis...**")
                st.session_state['analysis_running'] = True
                st.session_state['analysis_started'] = True
                # Force immediate rerun to trigger analysis
                st.rerun()
    
    # Step 3: Run analysis
    if st.session_state['analysis_running'] and st.session_state.get('analysis_posts'):
        posts = st.session_state['analysis_posts']
        use_batch = st.session_state.get('use_batch_mode', True)
        
        st.write("=" * 60)
        st.write("### 🔄 ANALYSIS IN PROGRESS")
        st.write("=" * 60)
        st.write(f"📊 Processing {len(posts)} posts...")
        st.write(f"⚙️ Mode: {'Batch' if use_batch else 'Sequential'}")
        st.write("")
        
        try:
            # Run analysis
            if orchestrator_monitor:
                # Create progress container
                progress_container = st.container()
                
                with progress_container:
                    st.write("### 🔄 Analysis Progress")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.write("✓ Orchestrator found")
                    progress_bar.progress(10)
                    
                    status_text.write("✓ Creating event loop...")
                    
                    # Create new event loop for this execution
                    try:
                        loop = async_lib.get_event_loop()
                    except RuntimeError:
                        loop = async_lib.new_event_loop()
                        async_lib.set_event_loop(loop)
                    
                    progress_bar.progress(20)
                    status_text.write("✓ Event loop ready - Starting analysis...")
                    
                    # Run analysis with proper async handling
                    try:
                        progress_bar.progress(30)
                        status_text.write(f"🤖 Analyzing {len(posts)} posts... (this may take several minutes)")
                        
                        results = loop.run_until_complete(run_batch_analysis(
                            posts=posts,
                            orchestrator_monitor=orchestrator_monitor,
                            use_batch=use_batch
                        ))
                        
                        progress_bar.progress(100)
                        status_text.write("✓ Analysis complete!")
                        st.success("✅ Analysis finished successfully!")
                        
                    except Exception as async_error:
                        st.error(f"❌ Async execution error: {str(async_error)}")
                        raise
                
                # Check if successful
                if results.get("status") == "success":
                    st.session_state['analysis_results'] = results
                    st.session_state['analysis_running'] = False
                    
                    # Store in current result for Results tab
                    from core.state_manager import set_current_result, add_execution_result
                    
                    # Create execution result
                    report = results.get("report", {})
                    exec_summary = report.get("executive_summary", {})
                    
                    execution_result = ExecutionResult(
                        execution_id=f"exec_{int(datetime.now().timestamp())}",
                        timestamp=datetime.now(),
                        total_posts=len(posts),
                        posts_analyzed=exec_summary.get("total_posts_analyzed", 0),
                        misinformation_detected=exec_summary.get("misinformation_detected", 0),
                        high_risk_posts=exec_summary.get("high_risk_posts", 0),
                        execution_time_ms=results.get("execution_time_ms", 0),
                        agent_statuses=results.get("agent_statuses", {}),
                        full_report=report,
                        markdown_report=""
                    )
                    
                    # Set as current result
                    set_current_result(st.session_state, execution_result)
                    add_execution_result(st.session_state, execution_result)
                    
                    # Save results
                    if data_manager:
                        execution_id = save_analysis_results(results, posts, data_manager)
                        if execution_id:
                            st.success(f"✅ Results saved with ID: {execution_id}")
                    
                    st.info("📊 Results are now available in the 'Results' tab!")
                    st.rerun()
                else:
                    # Analysis failed
                    error = results.get("error", "Unknown error")
                    st.session_state['analysis_running'] = False
                    render_analysis_error(error, results)
            else:
                st.session_state['analysis_running'] = False
                st.error("❌ Orchestrator monitor not available")
                st.write("**Debug Info:**")
                st.write(f"- orchestrator_monitor is None: {orchestrator_monitor is None}")
                st.write(f"- Session state keys: {list(st.session_state.keys())}")
                
        except Exception as e:
            st.session_state['analysis_running'] = False
            st.error(f"❌ Exception during analysis: {str(e)}")
            st.write("**Full error details:**")
            import traceback
            st.code(traceback.format_exc())
            render_analysis_error(str(e))
    
    # Step 4: Display results
    if st.session_state['analysis_results'] and not st.session_state['analysis_running']:
        results = st.session_state['analysis_results']
        
        # Show summary
        render_analysis_summary(results)
        
        # Option to start new analysis
        if st.button("🔄 Start New Analysis"):
            st.session_state['analysis_running'] = False
            st.session_state['analysis_results'] = None
            st.session_state['analysis_posts'] = None
            st.rerun()
