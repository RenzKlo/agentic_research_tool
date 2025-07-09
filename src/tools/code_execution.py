"""
Code Execution Tool for the Agentic AI system.
"""

import subprocess
import sys
import os
import tempfile
import ast
import importlib.util
import traceback
from typing import Dict, Any, List, Optional, Union
from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
import logging
from datetime import datetime
import json
import signal
import threading
import time

logger = logging.getLogger(__name__)


class CodeExecutionInput(BaseModel):
    """Input for code execution operations."""
    code: str = Field(description="The code to execute")
    language: str = Field(default="python", description="Programming language (python, bash, javascript)")
    timeout: int = Field(default=30, description="Timeout in seconds")
    safe_mode: bool = Field(default=True, description="Execute in safe mode with restrictions")


class CodeAnalysisInput(BaseModel):
    """Input for code analysis operations."""
    code: str = Field(description="The code to analyze")
    analysis_type: str = Field(default="syntax", description="Type of analysis: syntax, security, complexity")


# Global configuration to avoid Pydantic field issues
_TOOL_CONFIG = {}

class CodeExecutionTool(BaseTool):
    """Tool for executing and analyzing code safely."""
    
    name = "code_execution"
    description = """
    Execute and analyze code safely:
    - execute: Run Python, Bash, or JavaScript code
    - analyze: Analyze code for syntax, security, or complexity
    - validate: Validate code syntax
    - format: Format and clean code
    - test: Run unit tests
    """
    
    def __init__(self, enable_execution: bool = True, allowed_imports: Optional[List[str]] = None):
        super().__init__()
        # Store config globally to avoid Pydantic field issues
        _TOOL_CONFIG[id(self)] = {
            'enable_execution': enable_execution,
            'allowed_imports': allowed_imports or [
                'os', 'sys', 'json', 'csv', 'pandas', 'numpy', 'matplotlib', 'plotly',
                'requests', 'datetime', 'time', 'math', 'random', 'statistics',
                'collections', 'itertools', 're', 'pathlib', 'typing'
            ],
            'restricted_functions': [
                'exec', 'eval', 'compile', '__import__', 'open', 'file',
                'input', 'raw_input', 'reload', 'vars', 'globals', 'locals',
                'dir', 'getattr', 'setattr', 'delattr', 'hasattr'
            ],
            'temp_dir': tempfile.mkdtemp()
        }
    
    @property
    def config(self):
        return _TOOL_CONFIG.get(id(self), {})
    
    def _run(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute code operation synchronously."""
        try:
            if operation == "execute":
                return self._execute_code(**kwargs)
            elif operation == "analyze":
                return self._analyze_code(**kwargs)
            elif operation == "validate":
                return self._validate_code(**kwargs)
            elif operation == "format":
                return self._format_code(**kwargs)
            elif operation == "test":
                return self._run_tests(**kwargs)
            else:
                raise ValueError(f"Unknown operation: {operation}")
        except Exception as e:
            logger.error(f"Error in code execution operation {operation}: {str(e)}")
            return {"error": str(e), "success": False}
    
    async def _arun(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute code operation asynchronously."""
        # For now, just run synchronously
        return self._run(operation, **kwargs)
    
    def _execute_code(self, code: str, language: str = "python", timeout: int = 30, safe_mode: bool = True) -> Dict[str, Any]:
        """Execute code in a safe environment."""
        if not self.config["enable_execution"]:
            return {
                "error": "Code execution is disabled",
                "success": False
            }
        
        try:
            if language.lower() == "python":
                return self._execute_python(code, timeout, safe_mode)
            elif language.lower() == "bash":
                return self._execute_bash(code, timeout, safe_mode)
            elif language.lower() == "javascript":
                return self._execute_javascript(code, timeout, safe_mode)
            else:
                return {
                    "error": f"Unsupported language: {language}",
                    "success": False
                }
        except Exception as e:
            logger.error(f"Error executing {language} code: {str(e)}")
            return {"error": str(e), "success": False}
    
    def _execute_python(self, code: str, timeout: int = 30, safe_mode: bool = True) -> Dict[str, Any]:
        """Execute Python code safely."""
        try:
            # Validate syntax first
            validation_result = self._validate_python_code(code)
            if not validation_result["success"]:
                return validation_result
            
            # Security check in safe mode
            if safe_mode:
                security_result = self._check_python_security(code)
                if not security_result["success"]:
                    return security_result
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir=self.config["temp_dir"]) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # Execute with timeout
                result = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=self.config["temp_dir"]
                )
                
                return {
                    "success": True,
                    "output": result.stdout,
                    "error": result.stderr,
                    "return_code": result.returncode,
                    "execution_time": "N/A",  # Could be improved with timing
                    "language": "python",
                    "timestamp": datetime.now().isoformat()
                }
            
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file)
                except:
                    pass
        
        except subprocess.TimeoutExpired:
            return {
                "error": f"Code execution timed out after {timeout} seconds",
                "success": False
            }
        except Exception as e:
            logger.error(f"Error executing Python code: {str(e)}")
            return {"error": str(e), "success": False}
    
    def _execute_bash(self, code: str, timeout: int = 30, safe_mode: bool = True) -> Dict[str, Any]:
        """Execute Bash code safely."""
        try:
            # Security check in safe mode
            if safe_mode:
                security_result = self._check_bash_security(code)
                if not security_result["success"]:
                    return security_result
            
            # Execute bash command
            result = subprocess.run(
                code,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.config["temp_dir"]
            )
            
            return {
                "success": True,
                "output": result.stdout,
                "error": result.stderr,
                "return_code": result.returncode,
                "language": "bash",
                "timestamp": datetime.now().isoformat()
            }
        
        except subprocess.TimeoutExpired:
            return {
                "error": f"Code execution timed out after {timeout} seconds",
                "success": False
            }
        except Exception as e:
            logger.error(f"Error executing Bash code: {str(e)}")
            return {"error": str(e), "success": False}
    
    def _execute_javascript(self, code: str, timeout: int = 30, safe_mode: bool = True) -> Dict[str, Any]:
        """Execute JavaScript code using Node.js."""
        try:
            # Check if Node.js is available
            try:
                subprocess.run(['node', '--version'], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                return {
                    "error": "Node.js is not installed or not available",
                    "success": False
                }
            
            # Security check in safe mode
            if safe_mode:
                security_result = self._check_javascript_security(code)
                if not security_result["success"]:
                    return security_result
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False, dir=self.config["temp_dir"]) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # Execute with Node.js
                result = subprocess.run(
                    ['node', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=self.config["temp_dir"]
                )
                
                return {
                    "success": True,
                    "output": result.stdout,
                    "error": result.stderr,
                    "return_code": result.returncode,
                    "language": "javascript",
                    "timestamp": datetime.now().isoformat()
                }
            
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file)
                except:
                    pass
        
        except subprocess.TimeoutExpired:
            return {
                "error": f"Code execution timed out after {timeout} seconds",
                "success": False
            }
        except Exception as e:
            logger.error(f"Error executing JavaScript code: {str(e)}")
            return {"error": str(e), "success": False}
    
    def _validate_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Validate code syntax."""
        try:
            if language.lower() == "python":
                return self._validate_python_code(code)
            elif language.lower() == "javascript":
                return self._validate_javascript_code(code)
            else:
                return {
                    "error": f"Syntax validation not supported for {language}",
                    "success": False
                }
        except Exception as e:
            logger.error(f"Error validating {language} code: {str(e)}")
            return {"error": str(e), "success": False}
    
    def _validate_python_code(self, code: str) -> Dict[str, Any]:
        """Validate Python code syntax."""
        try:
            ast.parse(code)
            return {
                "success": True,
                "valid": True,
                "language": "python",
                "message": "Code syntax is valid"
            }
        except SyntaxError as e:
            return {
                "success": True,
                "valid": False,
                "language": "python",
                "error": str(e),
                "line": e.lineno,
                "offset": e.offset
            }
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def _validate_javascript_code(self, code: str) -> Dict[str, Any]:
        """Validate JavaScript code syntax using Node.js."""
        try:
            # Use Node.js to check syntax
            result = subprocess.run(
                ['node', '--check', '-'],
                input=code,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "valid": True,
                    "language": "javascript",
                    "message": "Code syntax is valid"
                }
            else:
                return {
                    "success": True,
                    "valid": False,
                    "language": "javascript",
                    "error": result.stderr
                }
        
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def _analyze_code(self, code: str, analysis_type: str = "syntax") -> Dict[str, Any]:
        """Analyze code for various aspects."""
        try:
            if analysis_type == "syntax":
                return self._validate_code(code)
            elif analysis_type == "security":
                return self._analyze_security(code)
            elif analysis_type == "complexity":
                return self._analyze_complexity(code)
            else:
                return {
                    "error": f"Unknown analysis type: {analysis_type}",
                    "success": False
                }
        except Exception as e:
            logger.error(f"Error analyzing code: {str(e)}")
            return {"error": str(e), "success": False}
    
    def _analyze_security(self, code: str) -> Dict[str, Any]:
        """Analyze code for security issues."""
        try:
            issues = []
            
            # Check for dangerous functions
            for func in self.config["restricted_functions"]:
                if func in code:
                    issues.append({
                        "type": "dangerous_function",
                        "function": func,
                        "severity": "high",
                        "message": f"Use of potentially dangerous function: {func}"
                    })
            
            # Check for dangerous imports
            if "import os" in code and "os.system" in code:
                issues.append({
                    "type": "dangerous_import",
                    "severity": "high",
                    "message": "Use of os.system() can be dangerous"
                })
            
            # Check for subprocess usage
            if "subprocess" in code:
                issues.append({
                    "type": "subprocess_usage",
                    "severity": "medium",
                    "message": "Use of subprocess module detected"
                })
            
            return {
                "success": True,
                "issues": issues,
                "risk_level": "high" if any(issue["severity"] == "high" for issue in issues) else "medium" if issues else "low"
            }
        
        except Exception as e:
            logger.error(f"Error analyzing security: {str(e)}")
            return {"error": str(e), "success": False}
    
    def _analyze_complexity(self, code: str) -> Dict[str, Any]:
        """Analyze code complexity (basic implementation)."""
        try:
            lines = code.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]
            
            # Count various constructs
            function_count = len([line for line in non_empty_lines if 'def ' in line])
            class_count = len([line for line in non_empty_lines if 'class ' in line])
            loop_count = len([line for line in non_empty_lines if any(keyword in line for keyword in ['for ', 'while '])])
            conditional_count = len([line for line in non_empty_lines if any(keyword in line for keyword in ['if ', 'elif ', 'else:'])])
            
            # Calculate basic complexity score
            complexity_score = len(non_empty_lines) + loop_count * 2 + conditional_count * 1.5
            
            return {
                "success": True,
                "total_lines": len(lines),
                "code_lines": len(non_empty_lines),
                "function_count": function_count,
                "class_count": class_count,
                "loop_count": loop_count,
                "conditional_count": conditional_count,
                "complexity_score": complexity_score,
                "complexity_level": "high" if complexity_score > 100 else "medium" if complexity_score > 50 else "low"
            }
        
        except Exception as e:
            logger.error(f"Error analyzing complexity: {str(e)}")
            return {"error": str(e), "success": False}
    
    def _check_python_security(self, code: str) -> Dict[str, Any]:
        """Check Python code for security issues."""
        try:
            # Parse AST to check for dangerous constructs
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                # Check for dangerous function calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in self.config["restricted_functions"]:
                            return {
                                "success": False,
                                "error": f"Dangerous function call detected: {node.func.id}"
                            }
                
                # Check for dangerous imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name not in self.config["allowed_imports"]:
                            return {
                                "success": False,
                                "error": f"Import not allowed: {alias.name}"
                            }
                
                if isinstance(node, ast.ImportFrom):
                    if node.module not in self.config["allowed_imports"]:
                        return {
                            "success": False,
                            "error": f"Import not allowed: {node.module}"
                        }
            
            return {"success": True}
        
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def _check_bash_security(self, code: str) -> Dict[str, Any]:
        """Check Bash code for security issues."""
        dangerous_commands = [
            'rm -rf', 'dd', 'mkfs', 'fdisk', 'format', 'del /f',
            'shutdown', 'reboot', 'halt', 'poweroff', 'init',
            'kill -9', 'killall', 'pkill', 'fuser -k'
        ]
        
        for cmd in dangerous_commands:
            if cmd in code:
                return {
                    "success": False,
                    "error": f"Dangerous command detected: {cmd}"
                }
        
        return {"success": True}
    
    def _check_javascript_security(self, code: str) -> Dict[str, Any]:
        """Check JavaScript code for security issues."""
        dangerous_patterns = [
            'eval(', 'Function(', 'setTimeout(', 'setInterval(',
            'document.write(', 'innerHTML', 'outerHTML'
        ]
        
        for pattern in dangerous_patterns:
            if pattern in code:
                return {
                    "success": False,
                    "error": f"Potentially dangerous pattern detected: {pattern}"
                }
        
        return {"success": True}
    
    def _format_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Format code (basic implementation)."""
        try:
            if language.lower() == "python":
                # Basic Python formatting
                lines = code.split('\n')
                formatted_lines = []
                indent_level = 0
                
                for line in lines:
                    stripped = line.strip()
                    if not stripped:
                        formatted_lines.append('')
                        continue
                    
                    if stripped.startswith(('def ', 'class ', 'if ', 'elif ', 'else:', 'for ', 'while ', 'try:', 'except:', 'finally:', 'with ')):
                        formatted_lines.append('    ' * indent_level + stripped)
                        if stripped.endswith(':'):
                            indent_level += 1
                    elif stripped in ('else:', 'except:', 'finally:'):
                        indent_level -= 1
                        formatted_lines.append('    ' * indent_level + stripped)
                        indent_level += 1
                    else:
                        formatted_lines.append('    ' * indent_level + stripped)
                
                return {
                    "success": True,
                    "formatted_code": '\n'.join(formatted_lines),
                    "language": language
                }
            else:
                return {
                    "success": True,
                    "formatted_code": code,
                    "language": language,
                    "message": f"Formatting not implemented for {language}"
                }
        
        except Exception as e:
            logger.error(f"Error formatting code: {str(e)}")
            return {"error": str(e), "success": False}
    
    def _run_tests(self, code: str, test_code: str = None) -> Dict[str, Any]:
        """Run unit tests on code."""
        try:
            if test_code:
                # Combine code and tests
                full_code = f"{code}\n\n{test_code}"
            else:
                # Look for test functions in the code
                full_code = code
            
            # Execute the test code
            result = self._execute_python(full_code)
            
            return {
                "success": True,
                "test_result": result,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error running tests: {str(e)}")
            return {"error": str(e), "success": False}
    
    def get_available_operations(self) -> List[str]:
        """Get list of available operations."""
        return ["execute", "analyze", "validate", "format", "test"]
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported programming languages."""
        return ["python", "bash", "javascript"]
    
    def __del__(self):
        """Clean up temporary directory."""
        try:
            import shutil
            shutil.rmtree(self.config["temp_dir"], ignore_errors=True)
        except:
            pass
