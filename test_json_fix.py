#!/usr/bin/env python3
"""
Test script to verify JSON parsing fixes in the agent.
"""

import sys
import os
sys.path.append('/home/renzk/projects/agentic_data_analysis')

from src.agent.agent import AutonomousAgent
from src.utils.config import Config
import asyncio

async def test_web_research():
    """Test web research tool with different input formats."""
    
    # Initialize config and agent
    config = Config()
    agent = AutonomousAgent(
        api_key=config.openai_api_key,
        model=config.model_name,
        temperature=config.temperature,
        max_tool_calls=config.max_iterations,
        memory_type=config.memory_type,
        tavily_api_key=config.tavily_api_key,
        enable_code_execution=config.enable_code_execution
    )
    
    # Test 1: Simple search query that might cause JSON parsing issues
    print("Testing simple web search...")
    try:
        result = await agent.arun("Search for recent news about artificial intelligence")
        print(f"✅ Success: {result[:200]}...")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Another potentially problematic query
    print("\nTesting data analysis request...")
    try:
        result = await agent.arun("Analyze the sales data and create a visualization")
        print(f"✅ Success: {result[:200]}...")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_web_research())
