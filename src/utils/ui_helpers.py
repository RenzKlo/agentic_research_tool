"""
UI helper functions for the Streamlit interface.
"""

import streamlit as st
from typing import Dict, Any, List, Optional, Union
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


def render_agent_message(message: str, is_user: bool = False) -> None:
    """Render a message in the chat interface."""
    if is_user:
        with st.chat_message("user"):
            st.write(message)
    else:
        with st.chat_message("assistant"):
            st.write(message)


def render_tool_usage(tool_name: str, tool_input: Dict[str, Any], tool_output: Any) -> None:
    """Render tool usage information in an expandable section."""
    with st.expander(f"🔧 Tool Used: {tool_name}"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input")
            st.json(tool_input)
        
        with col2:
            st.subheader("Output")
            if isinstance(tool_output, dict):
                st.json(tool_output)
            elif isinstance(tool_output, pd.DataFrame):
                st.dataframe(tool_output)
            elif isinstance(tool_output, (go.Figure, px.Figure)):
                st.plotly_chart(tool_output)
            else:
                st.text(str(tool_output))


def render_thinking_process(thoughts: List[str]) -> None:
    """Render the agent's thinking process."""
    with st.expander("🤔 Agent Thinking Process"):
        for i, thought in enumerate(thoughts, 1):
            st.write(f"**Step {i}:** {thought}")


def render_configuration_sidebar(config: Dict[str, Any]) -> Dict[str, Any]:
    """Render configuration options in the sidebar."""
    st.sidebar.header("⚙️ Configuration")
    
    updated_config = {}
    
    # LLM Settings
    st.sidebar.subheader("LLM Settings")
    updated_config["model_name"] = st.sidebar.selectbox(
        "Model",
        ["gpt-4-turbo-preview", "gpt-4", "gpt-3.5-turbo"],
        index=0 if config.get("model_name") == "gpt-4-turbo-preview" else 1
    )
    
    updated_config["temperature"] = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=config.get("temperature", 0.7),
        step=0.1
    )
    
    updated_config["max_tokens"] = st.sidebar.slider(
        "Max Tokens",
        min_value=100,
        max_value=8000,
        value=config.get("max_tokens", 4000),
        step=100
    )
    
    # Agent Settings
    st.sidebar.subheader("Agent Settings")
    updated_config["max_iterations"] = st.sidebar.slider(
        "Max Iterations",
        min_value=1,
        max_value=20,
        value=config.get("max_iterations", 10),
        step=1
    )
    
    updated_config["max_execution_time"] = st.sidebar.slider(
        "Max Execution Time (seconds)",
        min_value=30,
        max_value=600,
        value=config.get("max_execution_time", 300),
        step=30
    )
    
    updated_config["verbose"] = st.sidebar.checkbox(
        "Verbose Mode",
        value=config.get("verbose", False)
    )
    
    # Tool Settings
    st.sidebar.subheader("Tool Settings")
    updated_config["enable_code_execution"] = st.sidebar.checkbox(
        "Enable Code Execution",
        value=config.get("enable_code_execution", False)
    )
    
    return updated_config


def render_example_prompts() -> Optional[str]:
    """Render example prompts for the user."""
    st.sidebar.header("💡 Example Prompts")
    
    examples = [
        "Analyze the sales data in sales.csv and create a visualization",
        "Search for recent news about AI agents and summarize the findings",
        "Create a Python script to process customer data and save the results",
        "Compare the performance of different machine learning models",
        "Generate a report on market trends using web research",
        "Analyze log files to identify error patterns",
        "Create a dashboard for monitoring system metrics",
        "Process and clean a dataset for machine learning",
    ]
    
    selected_example = st.sidebar.selectbox(
        "Select an example:",
        [""] + examples,
        index=0
    )
    
    if selected_example:
        return selected_example
    
    return None


def render_metrics_dashboard(metrics: Dict[str, Any]) -> None:
    """Render a metrics dashboard."""
    st.subheader("📊 Agent Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Queries",
            metrics.get("total_queries", 0)
        )
    
    with col2:
        st.metric(
            "Successful Executions",
            metrics.get("successful_executions", 0)
        )
    
    with col3:
        st.metric(
            "Tools Used",
            metrics.get("tools_used", 0)
        )
    
    with col4:
        st.metric(
            "Avg Response Time",
            f"{metrics.get('avg_response_time', 0):.2f}s"
        )
    
    # Response time chart
    if "response_times" in metrics and metrics["response_times"]:
        fig = px.line(
            x=range(len(metrics["response_times"])),
            y=metrics["response_times"],
            title="Response Time Over Time",
            labels={"x": "Query Number", "y": "Response Time (seconds)"}
        )
        st.plotly_chart(fig, use_container_width=True)


def render_file_uploader(accepted_types: List[str]) -> Optional[Any]:
    """Render a file uploader widget."""
    uploaded_file = st.file_uploader(
        "Upload a file for analysis",
        type=accepted_types,
        help="Upload a file that the agent can analyze or process"
    )
    
    if uploaded_file is not None:
        # Save the uploaded file
        file_path = f"data/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"File uploaded: {uploaded_file.name}")
        return file_path
    
    return None


def render_data_preview(data: Union[pd.DataFrame, Dict, List], title: str = "Data Preview") -> None:
    """Render a data preview."""
    st.subheader(title)
    
    if isinstance(data, pd.DataFrame):
        st.dataframe(data.head())
        
        # Basic statistics
        if st.checkbox("Show Statistics"):
            st.subheader("Data Statistics")
            st.write(data.describe())
    
    elif isinstance(data, dict):
        st.json(data)
    
    elif isinstance(data, list):
        if len(data) > 0:
            if isinstance(data[0], dict):
                df = pd.DataFrame(data)
                st.dataframe(df.head())
            else:
                st.write(data[:10])  # Show first 10 items
    
    else:
        st.text(str(data))


def render_error_message(error: Exception, context: str = "") -> None:
    """Render an error message."""
    st.error(f"❌ Error{' in ' + context if context else ''}: {str(error)}")
    
    if st.checkbox("Show Error Details"):
        st.exception(error)


def render_success_message(message: str) -> None:
    """Render a success message."""
    st.success(f"✅ {message}")


def render_info_message(message: str) -> None:
    """Render an info message."""
    st.info(f"ℹ️ {message}")


def render_warning_message(message: str) -> None:
    """Render a warning message."""
    st.warning(f"⚠️ {message}")


def render_agent_status(status: str, details: Optional[str] = None) -> None:
    """Render the agent's current status."""
    status_colors = {
        "idle": "🟢",
        "thinking": "🟡",
        "executing": "🔵",
        "error": "🔴",
        "completed": "✅"
    }
    
    color = status_colors.get(status, "⚪")
    
    with st.container():
        st.write(f"**Agent Status:** {color} {status.title()}")
        if details:
            st.caption(details)


def render_conversation_history(messages: List[Dict[str, Any]]) -> None:
    """Render the conversation history."""
    st.subheader("💬 Conversation History")
    
    for i, message in enumerate(messages):
        timestamp = message.get("timestamp", datetime.now().isoformat())
        role = message.get("role", "assistant")
        content = message.get("content", "")
        
        with st.expander(f"{role.title()} - {timestamp}"):
            st.write(content)
            
            # Show tool usage if available
            if "tool_usage" in message:
                for tool_use in message["tool_usage"]:
                    render_tool_usage(
                        tool_use.get("tool_name", "Unknown"),
                        tool_use.get("input", {}),
                        tool_use.get("output", "")
                    )


def render_system_info(config: Dict[str, Any]) -> None:
    """Render system information."""
    st.subheader("🖥️ System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Configuration:**")
        for key, value in config.items():
            if not key.endswith("_key"):  # Hide API keys
                st.write(f"- {key}: {value}")
    
    with col2:
        st.write("**System Status:**")
        st.write(f"- Timestamp: {datetime.now().isoformat()}")
        st.write(f"- Memory Usage: Available")
        st.write(f"- Tools Available: {len(config.get('available_tools', []))}")


def format_response_time(seconds: float) -> str:
    """Format response time for display."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"


def create_download_link(data: Union[str, bytes, pd.DataFrame], filename: str, mime_type: str = "text/plain") -> None:
    """Create a download link for data."""
    if isinstance(data, pd.DataFrame):
        data = data.to_csv(index=False)
        mime_type = "text/csv"
    elif isinstance(data, dict):
        data = json.dumps(data, indent=2)
        mime_type = "application/json"
    
    st.download_button(
        label=f"Download {filename}",
        data=data,
        file_name=filename,
        mime=mime_type
    )
