#!/usr/bin/env python3
"""
Direct test of web research tool for JSON parsing issues.
"""

import sys
import os
sys.path.append('/home/renzk/projects/agentic_data_analysis')

from src.tools.web_research import WebResearchTool

def test_web_research_tool():
    """Test web research tool directly."""
    
    # Initialize the tool
    tool = WebResearchTool()
    
    # Test 1: String input that might cause JSON parsing issues
    print("Testing string input 'search'...")
    try:
        result = tool._run("search")
        print(f"✅ Success: {result}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: String input with search query
    print("\nTesting string input 'search for AI news'...")
    try:
        result = tool._run("search for AI news")
        print(f"✅ Success: {result}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Dict input
    print("\nTesting dict input...")
    try:
        result = tool._run({
            "operation": "search",
            "query": "artificial intelligence",
            "num_results": 3
        })
        print(f"✅ Success: {result}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_web_research_tool()
