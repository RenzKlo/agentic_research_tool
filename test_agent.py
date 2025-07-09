#!/usr/bin/env python3
"""
Test script for the Agentic AI system.
This script validates all major components without requiring API keys.
"""

import sys
import os
import traceback
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules can be imported correctly."""
    print("🔍 Testing imports...")
    
    try:
        # Core modules
        from src.utils.config import Config
        from src.utils.ui_helpers import render_configuration_sidebar
        from src.agent.memory import EnhancedMemory
        
        # Tools
        from src.tools.file_operations import FileOperationsTool
        from src.tools.data_analysis import DataAnalysisTool
        from src.tools.web_research import WebResearchTool
        from src.tools.code_execution import CodeExecutionTool
        from src.tools.api_integration import APIIntegrationTool
        
        print("✅ All imports successful")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_configuration():
    """Test the configuration system."""
    print("\n🔍 Testing configuration...")
    
    try:
        from src.utils.config import Config
        
        config = Config()
        
        # Test basic config attributes
        assert hasattr(config, 'openai_api_key')
        assert hasattr(config, 'tavily_api_key')
        assert hasattr(config, 'model_name')
        assert hasattr(config, 'temperature')
        
        print("✅ Configuration system working")
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        traceback.print_exc()
        return False

def test_memory():
    """Test the memory system."""
    print("\n🔍 Testing memory system...")
    
    try:
        from src.agent.memory import EnhancedMemory
        
        memory = EnhancedMemory()
        
        # Test memory operations
        memory.add_message("user", "Hello, how are you?")
        memory.add_message("assistant", "I'm doing great, thanks!")
        
        messages = memory.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        
        # Test memory clearing
        memory.clear()
        assert len(memory.get_messages()) == 0
        
        print("✅ Memory system working")
        return True
        
    except Exception as e:
        print(f"❌ Memory error: {e}")
        traceback.print_exc()
        return False

def test_tools():
    """Test individual tools."""
    print("\n🔍 Testing tools...")
    
    try:
        from src.tools.file_operations import FileOperationsTool
        from src.tools.data_analysis import DataAnalysisTool
        from src.tools.code_execution import CodeExecutionTool
        
        # Test tool instantiation
        file_tool = FileOperationsTool()
        data_tool = DataAnalysisTool()
        code_tool = CodeExecutionTool()
        
        # Test tool properties
        assert hasattr(file_tool, 'name')
        assert hasattr(file_tool, 'description')
        assert hasattr(data_tool, 'name')
        assert hasattr(data_tool, 'description')
        assert hasattr(code_tool, 'name')
        assert hasattr(code_tool, 'description')
        
        print("✅ Tools instantiated successfully")
        return True
        
    except Exception as e:
        print(f"❌ Tools error: {e}")
        traceback.print_exc()
        return False

def test_data_analysis():
    """Test data analysis capabilities."""
    print("\n🔍 Testing data analysis...")
    
    try:
        from src.tools.data_analysis import DataAnalysisTool
        import pandas as pd
        
        # Create sample data
        data = {
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'salary': [50000, 60000, 70000]
        }
        df = pd.DataFrame(data)
        
        # Save sample data
        data_file = project_root / "data" / "test_data.csv"
        df.to_csv(data_file, index=False)
        
        # Test data analysis tool
        data_tool = DataAnalysisTool()
        
        # Test reading data
        result = data_tool._run(f"analyze_csv:{data_file}")
        assert "summary" in result
        
        print("✅ Data analysis working")
        return True
        
    except Exception as e:
        print(f"❌ Data analysis error: {e}")
        traceback.print_exc()
        return False

def test_file_operations():
    """Test file operations."""
    print("\n🔍 Testing file operations...")
    
    try:
        from src.tools.file_operations import FileOperationsTool
        
        file_tool = FileOperationsTool()
        
        # Test file creation
        test_file = project_root / "data" / "test_file.txt"
        result = file_tool._run(f"write_file:{test_file}:This is a test file")
        assert "successfully" in result.lower()
        
        # Test file reading
        result = file_tool._run(f"read_file:{test_file}")
        assert "This is a test file" in result
        
        print("✅ File operations working")
        return True
        
    except Exception as e:
        print(f"❌ File operations error: {e}")
        traceback.print_exc()
        return False

def test_code_execution():
    """Test code execution capabilities."""
    print("\n🔍 Testing code execution...")
    
    try:
        from src.tools.code_execution import CodeExecutionTool
        
        code_tool = CodeExecutionTool()
        
        # Test simple Python code
        result = code_tool._run("print('Hello, World!')")
        assert "Hello, World!" in result
        
        # Test data processing code
        code = """
import pandas as pd
data = {'x': [1, 2, 3], 'y': [4, 5, 6]}
df = pd.DataFrame(data)
print(df.sum())
"""
        result = code_tool._run(code)
        assert "6" in result  # Sum of y column
        
        print("✅ Code execution working")
        return True
        
    except Exception as e:
        print(f"❌ Code execution error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Agentic AI System Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_configuration,
        test_memory,
        test_tools,
        test_data_analysis,
        test_file_operations,
        test_code_execution,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The Agentic AI system is ready.")
        print("\nNext steps:")
        print("1. Set your OpenAI API key in the .env file")
        print("2. Optionally set your Tavily API key for web search")
        print("3. Run: streamlit run app.py")
        print("4. Open http://localhost:8501 in your browser")
    else:
        print(f"❌ {total - passed} tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
