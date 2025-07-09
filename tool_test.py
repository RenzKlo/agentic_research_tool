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
