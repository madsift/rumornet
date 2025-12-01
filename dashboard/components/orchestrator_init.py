"""
Orchestrator initialization component for the dashboard.

Provides UI for initializing the orchestrator directly from the dashboard.
"""

import streamlit as st
import asyncio
import sys
import os
from typing import Optional

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from granular_misinformation_orchestrator import GranularMisinformationOrchestrator
from core.orchestrator_monitor import OrchestratorMonitor


def render_orchestrator_init_ui():
    """
    Render UI for initializing the orchestrator.
    
    Allows users to initialize the orchestrator directly from the dashboard
    with custom configuration.
    """
    st.subheader("🔧 Initialize Orchestrator")
    
    # Check if already initialized
    if 'orchestrator_monitor' in st.session_state and st.session_state.orchestrator_monitor is not None:
        st.success("✅ Orchestrator is already initialized!")
        
        # Show status
        monitor = st.session_state.orchestrator_monitor
        if hasattr(monitor, 'orchestrator') and monitor.orchestrator.agents:
            st.write(f"**Active Agents:** {len(monitor.orchestrator.agents)}")
            with st.expander("View Agent List"):
                for agent_name in monitor.orchestrator.agents.keys():
                    st.write(f"- {agent_name}")
        
        # Option to reinitialize
        if st.button("🔄 Reinitialize Orchestrator"):
            st.session_state.orchestrator_monitor = None
            st.rerun()
        
        return
    
    # Configuration inputs
    st.write("**Configuration:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ollama_endpoint = st.text_input(
            "Ollama Endpoint",
            value="http://192.168.10.68:11434",
            help="URL of the Ollama server"
        )
    
    with col2:
        ollama_model = st.text_input(
            "Ollama Model",
            value="gemma3:4b",
            help="Name of the Ollama model to use"
        )
    
    # Initialize button
    if st.button("🚀 Initialize Orchestrator", type="primary"):
        with st.spinner("Initializing orchestrator and agents..."):
            try:
                # Create config
                config = {
                    "ollama_endpoint": ollama_endpoint,
                    "ollama_model": ollama_model
                }
                
                # Create orchestrator
                st.write("📦 Creating orchestrator...")
                orchestrator = GranularMisinformationOrchestrator(config=config)
                
                # Initialize agents
                st.write("🤖 Initializing agents...")
                
                # Create or get event loop
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Run initialization
                loop.run_until_complete(orchestrator.initialize_agents())
                
                # Create monitor
                st.write("📊 Creating monitor...")
                monitor = OrchestratorMonitor(orchestrator)
                
                # Store in session state
                st.session_state.orchestrator_monitor = monitor
                
                st.success("✅ Orchestrator initialized successfully!")
                st.write(f"**Agents initialized:** {len(orchestrator.agents)}")
                
                # Show agent list
                if orchestrator.agents:
                    with st.expander("View Initialized Agents"):
                        for agent_name in orchestrator.agents.keys():
                            st.write(f"✓ {agent_name}")
                
                st.info("🎉 You can now run analysis! Go to the Analysis tab.")
                
                # Rerun to update UI
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Failed to initialize orchestrator: {str(e)}")
                
                # Show detailed error
                with st.expander("Error Details"):
                    import traceback
                    st.code(traceback.format_exc())
                
                # Troubleshooting tips
                st.write("**Troubleshooting:**")
                st.write("1. Make sure Ollama is running at the specified endpoint")
                st.write("2. Verify the model name is correct")
                st.write("3. Check that all required Python packages are installed")
                st.write("4. Ensure agents are properly configured")


def check_orchestrator_status() -> bool:
    """
    Check if orchestrator is initialized and ready.
    
    Returns:
        True if orchestrator is ready, False otherwise
    """
    if 'orchestrator_monitor' not in st.session_state:
        return False
    
    monitor = st.session_state.orchestrator_monitor
    
    if monitor is None:
        return False
    
    if not hasattr(monitor, 'orchestrator'):
        return False
    
    if not monitor.orchestrator.agents:
        return False
    
    return True


def get_orchestrator_monitor() -> Optional[OrchestratorMonitor]:
    """
    Get the orchestrator monitor from session state.
    
    Returns:
        OrchestratorMonitor instance or None if not initialized
    """
    return st.session_state.get('orchestrator_monitor', None)
