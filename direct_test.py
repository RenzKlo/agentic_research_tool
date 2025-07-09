#!/usr/bin/env python3
"""Direct test of the agent with minimal dependencies."""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Testing agent directly...")

try:
    # Import required modules
    from src.agent.agent import AutonomousAgent
    print("✓ Agent imported")
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No OpenAI API key found in environment")
        sys.exit(1)
    
    print("✓ API key found")
    
    # Initialize agent with minimal config
    print("Initializing agent...")
    agent = AutonomousAgent(
        api_key=api_key,
        model="gpt-3.5-turbo",
        temperature=0.1,
        enable_code_execution=False
    )
    print("✓ Agent initialized")
    
    # Test simple query
    print("Testing simple query...")
    response = agent.run("What tools do you have available?")
    
    print(f"Response type: {type(response)}")
    if isinstance(response, dict):
        print(f"Response keys: {list(response.keys())}")
        if 'content' in response:
            print(f"Content preview: {response['content'][:200]}...")
        if 'error' in response:
            print(f"Error: {response['error']}")
    else:
        print(f"Response: {response}")
    
    print("✓ Test completed successfully")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
