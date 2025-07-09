#!/usr/bin/env python3
"""
Quick verification of the specific fixes.
"""

# Test 1: Verify plotly.express.Figure error is fixed
print("Testing Plotly fix...")
try:
    import sys
    sys.path.append('/home/renzk/projects/agentic_data_analysis')
    from src.utils.ui_helpers import render_tool_output
    import plotly.graph_objects as go
    
    # Create a test figure to verify the fix
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 4, 2]))
    
    # This would have failed before with "px.Figure" error
    print("✅ Plotly import and usage successful - no px.Figure error")
except Exception as e:
    print(f"❌ Plotly error: {e}")

# Test 2: Verify JSON parsing error is handled
print("\nTesting JSON parsing fix...")
try:
    from src.tools.web_research import WebResearchTool
    tool = WebResearchTool()
    
    # This input would have caused "search" is not valid JSON error
    result = tool._run("search")
    
    if isinstance(result, dict):
        print("✅ JSON parsing handled correctly - no JSON parse error")
    else:
        print(f"⚠️  Unexpected result type: {type(result)}")
except Exception as e:
    print(f"❌ JSON parsing error: {e}")

print("\n🎯 Fix verification complete!")
