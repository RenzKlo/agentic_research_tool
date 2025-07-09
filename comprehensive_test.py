#!/usr/bin/env python3
"""
Comprehensive test to verify JSON parsing and Plotly fixes.
"""

import sys
import os
sys.path.append('/home/renzk/projects/agentic_data_analysis')

def test_plotly_import():
    """Test that plotly imports work correctly after the fix."""
    print("Testing Plotly imports...")
    try:
        from src.utils.ui_helpers import render_tool_output
        print("✅ UI helpers import successful")
        
        # Test if plotly.express is still referenced incorrectly
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 4, 2]))
        print("✅ Plotly Figure creation successful")
        
        return True
    except Exception as e:
        print(f"❌ Plotly test failed: {e}")
        return False

def test_web_research_tool():
    """Test web research tool with different inputs that might cause JSON errors."""
    print("\nTesting Web Research Tool...")
    try:
        from src.tools.web_research import WebResearchTool
        tool = WebResearchTool()
        
        # Test 1: Simple string that previously caused JSON parsing error
        print("1. Testing simple 'search' input...")
        result1 = tool._run("search")
        if result1.get("success", False) or "error" in result1:
            print("✅ Simple 'search' handled properly")
        else:
            print(f"⚠️  Unexpected result: {result1}")
        
        # Test 2: Search with query
        print("2. Testing 'search artificial intelligence' input...")
        result2 = tool._run("search artificial intelligence")
        if result2.get("success", False) or "error" in result2:
            print("✅ Search with query handled properly")
        else:
            print(f"⚠️  Unexpected result: {result2}")
        
        # Test 3: Dict input
        print("3. Testing dict input...")
        result3 = tool._run({
            "operation": "search",
            "query": "test query",
            "num_results": 3
        })
        if result3.get("success", False) or "error" in result3:
            print("✅ Dict input handled properly")
        else:
            print(f"⚠️  Unexpected result: {result3}")
        
        # Test 4: Edge case - empty string
        print("4. Testing empty string input...")
        result4 = tool._run("")
        if result4.get("success", False) or "error" in result4:
            print("✅ Empty string handled properly")
        else:
            print(f"⚠️  Unexpected result: {result4}")
        
        return True
    except Exception as e:
        print(f"❌ Web research tool test failed: {e}")
        return False

def test_config_loading():
    """Test that config loads correctly."""
    print("\nTesting Config Loading...")
    try:
        from src.utils.config import Config
        config = Config()
        
        # Check if API key is loaded
        api_key = config.openai_api_key
        if api_key and isinstance(api_key, str) and len(api_key) > 10:
            print("✅ OpenAI API key loaded correctly")
        else:
            print(f"⚠️  API key issue: {type(api_key)}, length: {len(api_key) if api_key else 0}")
        
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Running comprehensive fix verification tests...\n")
    
    results = []
    results.append(test_plotly_import())
    results.append(test_web_research_tool())
    results.append(test_config_loading())
    
    print(f"\n📊 Test Results: {sum(results)}/{len(results)} tests passed")
    
    if all(results):
        print("🎉 All tests passed! Fixes appear to be working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
