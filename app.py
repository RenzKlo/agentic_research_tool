"""
LangChain Agentic AI - Main Application

This is the main Streamlit application that provides an interface for interacting
with the autonomous LangChain agent that can use tools to accomplish tasks.
"""

import streamlit as st
import os
from typing import List, Dict, Any
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
from src.agent.agent import AutonomousAgent
from src.utils.config import config
from src.utils.ui_helpers import (
    render_agent_message,
    render_configuration_sidebar,
    render_example_prompts,
    render_error_message,
    render_success_message,
    render_agent_status,
    render_file_uploader,
    render_tool_usage
)

# Page configuration
st.set_page_config(
    page_title="LangChain Agentic AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    .agent-response {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2E86AB;
        margin: 1rem 0;
    }
    .tool-usage {
        background-color: #e8f4f8;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .error-message {
        background-color: #ffe6e6;
        color: #d32f2f;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #d32f2f;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "config" not in st.session_state:
        st.session_state.config = config

def create_agent():
    """Create and initialize the autonomous agent"""
    try:
        agent = AutonomousAgent(
            api_key=st.session_state.config.openai_api_key,
            model=st.session_state.config.model_name,
            temperature=st.session_state.config.temperature,
            max_tool_calls=st.session_state.config.max_iterations,
            memory_type=st.session_state.config.memory_type,
            tavily_api_key=st.session_state.config.tavily_api_key,
            enable_code_execution=st.session_state.config.enable_code_execution
        )
        return agent
    except Exception as e:
        st.error(f"Failed to create agent: {str(e)}")
        return None

def process_user_input(user_input: str, agent: AutonomousAgent):
    """Process user input through the agent"""
    try:
        # Show thinking indicator
        with st.spinner("Agent is thinking and selecting tools..."):
            # Use the synchronous run method instead of arun to avoid async issues
            response = agent.run(user_input)
        
        return response
    except Exception as e:
        logger.error(f"Error processing user input: {str(e)}")
        return {"error": str(e)}

def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 LangChain Agentic AI</h1>', unsafe_allow_html=True)
    st.markdown("An autonomous AI agent that can use tools to accomplish complex tasks")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Agent Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key", 
            type="password",
            value=getattr(st.session_state.config, 'openai_api_key', ''),
            help="Enter your OpenAI API key"
        )
        
        if api_key:
            st.session_state.config.openai_api_key = api_key
        
        # Model selection
        model = st.selectbox(
            "Model",
            ["gpt-4-turbo-preview", "gpt-4", "gpt-3.5-turbo"],
            index=0
        )
        st.session_state.config.model_name = model
        
        # Agent settings
        st.subheader("Agent Settings")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
        max_tool_calls = st.slider("Max Tool Calls", 1, 50, 20)
        
        st.session_state.config.temperature = temperature
        st.session_state.config.max_iterations = max_tool_calls
        
        # Available tools info
        st.subheader("Available Tools")
        st.markdown("""
        - 📁 **File Operations**: Read, write, search files
        - 📊 **Data Analysis**: Pandas, statistics, visualization
        - 🌐 **Web Research**: Search engines, web scraping
        - 🔌 **API Integration**: REST APIs, external services
        - 💻 **Code Execution**: Python code runner
        - 🧠 **Memory**: Conversation history and context
        """)
        
        # Clear conversation
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.session_state.agent = None
            st.rerun()
    
    # Initialize agent if not exists
    if st.session_state.agent is None and api_key:
        st.session_state.agent = create_agent()
    
    # Main chat interface
    st.header("💬 Chat with Agent")
    
    # Display conversation history
    for message in st.session_state.messages:
        render_agent_message(message.get("content", ""), message.get("role") == "user")
    
    # User input
    if st.session_state.agent is not None:
        user_input = st.chat_input("Ask the agent to accomplish a task...")
        
        if user_input:
            # Add user message
            user_message = {
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now()
            }
            st.session_state.messages.append(user_message)
            
            # Process through agent
            with st.chat_message("user"):
                st.write(user_input)
            
            with st.chat_message("assistant"):
                # Create placeholder for agent response
                response_placeholder = st.empty()
                tool_placeholder = st.empty()
                
                # Process user input
                try:
                    # Use synchronous call since we changed process_user_input
                    response = process_user_input(user_input, st.session_state.agent)
                    
                    # Debug: Print response structure
                    logger.info(f"Response type: {type(response)}")
                    logger.info(f"Response keys: {response.keys() if isinstance(response, dict) else 'Not a dict'}")
                    
                    if "error" in response:
                        st.error(f"Error: {response['error']}")
                    else:
                        # Display agent response
                        if "content" in response:
                            response_placeholder.markdown(response["content"])
                        
                        # Display tool usage if any
                        if "tool_calls" in response:
                            with tool_placeholder.container():
                                for tool_call in response["tool_calls"]:
                                    render_tool_usage(
                                        tool_call.get("tool_name", "Unknown"),
                                        tool_call.get("input", {}),
                                        tool_call.get("output", "")
                                    )
                        
                        # Add agent message to history
                        agent_message = {
                            "role": "assistant",
                            "content": response.get("content", ""),
                            "tool_calls": response.get("tool_calls", []),
                            "timestamp": datetime.now()
                        }
                        st.session_state.messages.append(agent_message)
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    else:
        if not api_key:
            st.warning("Please enter your OpenAI API key in the sidebar to start.")
        else:
            st.error("Failed to initialize agent. Please check your API key and try again.")
    
    # Example prompts
    st.header("💡 Example Prompts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Data Analysis Tasks:**
        - "Create a sample dataset and analyze it with visualizations"
        - "Search for a CSV file online and perform statistical analysis"
        - "Generate a report on sales data trends"
        """)
        
    with col2:
        st.markdown("""
        **Research Tasks:**
        - "Research the latest developments in AI and summarize them"
        - "Find and analyze recent scientific papers on climate change"
        - "Create a market analysis report for electric vehicles"
        """)

if __name__ == "__main__":
    main()
