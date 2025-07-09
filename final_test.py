#!/usr/bin/env python3
"""Test the agent with a tool-using query."""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Testing agent with tool usage...")

try:
    from src.agent.agent import AutonomousAgent
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No OpenAI API key found in environment")
        sys.exit(1)
    
    # Initialize agent
    agent = AutonomousAgent(
        api_key=api_key,
        model="gpt-3.5-turbo",
        temperature=0.1,
        enable_code_execution=False
    )
    print("✓ Agent initialized")
    
    # Test with a query that should use the file operations tool
    print("Testing query that requires file operations...")
    response = agent.run("List the files in the current directory and tell me what you find.")
    
    print(f"Response type: {type(response)}")
    if isinstance(response, dict):
        if 'content' in response:
            print(f"Content: {response['content']}")
        if 'tool_calls' in response and response['tool_calls']:
            print(f"Tool calls made: {len(response['tool_calls'])}")
            for i, tool_call in enumerate(response['tool_calls']):
                print(f"  Tool {i+1}: {tool_call.get('tool', 'unknown')}")
        if 'error' in response:
            print(f"Error: {response['error']}")
    
    print("✓ Tool usage test completed")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
            api_key=config.openai_api_key,
            model=config.model_name,
            temperature=config.temperature,
            max_tool_calls=config.max_iterations,
            memory_type=config.memory_type,
            tavily_api_key=config.tavily_api_key,
            enable_code_execution=config.enable_code_execution
        )
        print("✅ Agent created successfully")
        
        # Test tool availability
        tools = agent.tools  # Direct access to tools list
        print(f"✅ {len(tools)} tools available:")
        for tool in tools:
            print(f"   - {tool.name}")
        
        # Test tool descriptions
        tool_descriptions = agent.get_tool_descriptions()
        print(f"✅ Tool descriptions available: {len(tool_descriptions)} tools")
        
        # Test Streamlit components
        print("🔍 Testing Streamlit components...")
        import streamlit as st
        from src.utils.ui_helpers import render_configuration_sidebar, render_example_prompts
        print("✅ UI helpers available")
        
        print("\n" + "=" * 50)
        print("🎉 SUCCESS: All core components are working!")
        print("\nNext steps:")
        print("1. Run: .venv/bin/python -m streamlit run app.py")
        print("2. Open http://localhost:8501 in your browser")
        print("3. Configure your OpenAI API key in the sidebar")
        print("4. Start interacting with your agentic AI!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_core_functionality()
    sys.exit(0 if success else 1)
