"""
Simple working Lambda API integration.
"""

import streamlit as st
import requests
from typing import Optional
from datetime import datetime

from models.data_models import ExecutionResult
from core.data_manager import DataManager
from core.state_manager import set_current_result, add_execution_result, get_current_result


def render_batch_analysis_workflow_with_api(
    orchestrator_monitor=None,
    data_manager: Optional[DataManager] = None
):
    """Call Lambda API and display results."""
    
    st.subheader("🔬 Misinformation Analysis")
    
    LAMBDA_API_URL = "https://mgbsx1x8l1.execute-api.us-east-1.amazonaws.com/Prod"
    
    # Initialize state
    if 'analysis_in_progress' not in st.session_state:
        st.session_state.analysis_in_progress = False
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    
    # Show results if complete
    current_result = get_current_result(st.session_state)
    if current_result and st.session_state.analysis_complete:
        st.success("✅ Analysis complete! View results in Overview and Results tabs.")
        
        report = current_result.full_report
        exec_summary = report.get("executive_summary", {})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Posts Analyzed", exec_summary.get("total_posts_analyzed", 0))
        with col2:
            st.metric("Misinformation", exec_summary.get("misinformation_detected", 0))
        with col3:
            st.metric("High Risk", exec_summary.get("high_risk_posts", 0))
        with col4:
            st.metric("Time (s)", f"{current_result.execution_time_ms / 1000:.1f}")
        
        if st.button("🔄 Start New Analysis", type="secondary"):
            st.session_state.analysis_complete = False
            st.session_state.analysis_in_progress = False
            st.rerun()
        return
    
    # Show button
    if not st.session_state.analysis_in_progress:
        # Check if authenticated
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        
        if not st.session_state.authenticated:
            # Show auth form
            st.info("🔒 Authentication required to run analysis")
            
            import os
            username = st.text_input("Username", key="auth_user")
            password = st.text_input("Password", type="password", key="auth_pass")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("Login", type="primary", key="login_btn"):
                    try:
                        # Try environment variables first (for Docker)
                        correct_user = os.getenv("AUTH_USERNAME") or st.secrets.get("auth", {}).get("username")
                        correct_pass = os.getenv("AUTH_PASSWORD") or st.secrets.get("auth", {}).get("password")
                        
                        if username == correct_user and password == correct_pass:
                            st.session_state.authenticated = True
                            st.rerun()
                        else:
                            st.error("❌ Invalid credentials")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
            return
        
        # If authenticated, show analysis button
        st.success("✅ Authenticated")
        
        def start_analysis():
            st.session_state.analysis_in_progress = True
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.button(
                "▶️ Analyze Demo Posts (132 posts)", 
                type="primary", 
                key="start_analysis_btn",
                on_click=start_analysis
            )
        
        with col2:
            if st.button("🔓 Logout", key="logout_btn"):
                st.session_state.authenticated = False
                st.rerun()
        
        return
    
    # Run analysis
    from datetime import timezone
    start_time = datetime.now(timezone.utc)
    
    with st.spinner("🔍 Analyzing 132 posts... This may take 3-4 minutes"):
        try:
            response = requests.post(
                f"{LAMBDA_API_URL}/analyze",
                json={"demo_file": "demo1.json"},
                timeout=60
            )
            
            if response.status_code == 200:
                results = response.json()
                report = results.get("report", {})
                exec_summary = report.get("executive_summary", {})
                
                execution_result = ExecutionResult(
                    execution_id=f"exec_{int(datetime.now().timestamp())}",
                    timestamp=datetime.now(),
                    total_posts=results.get("total_posts", 132),
                    posts_analyzed=exec_summary.get("total_posts_analyzed", 0),
                    misinformation_detected=exec_summary.get("misinformation_detected", 0),
                    high_risk_posts=exec_summary.get("high_risk_posts", 0),
                    execution_time_ms=results.get("execution_time_ms", 0),
                    agent_statuses={},
                    full_report=report,
                    markdown_report=""
                )
                
                set_current_result(st.session_state, execution_result)
                add_execution_result(st.session_state, execution_result)
                
                st.session_state.analysis_in_progress = False
                st.session_state.analysis_complete = True
                
            elif response.status_code == 504:
                # Gateway timeout - Lambda is still running, poll S3
                if data_manager:
                    import time
                    from datetime import timedelta
                    
                    max_polls = 120
                    poll_interval = 5
                    buffer_time = start_time - timedelta(minutes=1)
                    
                    for i in range(max_polls):
                        time.sleep(poll_interval)
                        
                        all_results = data_manager.load_all_execution_results(load_from_s3=True)
                        
                        # Find new results (timezone-aware comparison)
                        new_results = []
                        for r in all_results:
                            r_time = r.timestamp
                            if r_time.tzinfo is None:
                                r_time = r_time.replace(tzinfo=timezone.utc)
                            if r_time > buffer_time:
                                new_results.append(r)
                        
                        if new_results:
                            latest = new_results[0]
                            set_current_result(st.session_state, latest)
                            add_execution_result(st.session_state, latest)
                            
                            st.session_state.analysis_in_progress = False
                            st.session_state.analysis_complete = True
                            break
                    else:
                        st.session_state.analysis_in_progress = False
                else:
                    st.session_state.analysis_in_progress = False
            else:
                st.session_state.analysis_in_progress = False
                
        except:
            st.session_state.analysis_in_progress = False
    
    # Rerun after analysis
    if st.session_state.analysis_complete:
        st.rerun()
