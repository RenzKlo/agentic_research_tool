"""
File Operations Tool

This tool allows the agent to perform file system operations including
reading, writing, searching, and analyzing files.
"""

import os
import json
import csv
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import pandas as pd

logger = logging.getLogger(__name__)

class FileOperationInput(BaseModel):
    """Input schema for file operations"""
    operation: str = Field(description="Operation to perform: read, write, search, list, analyze")
    file_path: str = Field(description="Path to the file or directory")
    content: Optional[str] = Field(None, description="Content to write (for write operations)")
    search_term: Optional[str] = Field(None, description="Term to search for (for search operations)")
    encoding: str = Field("utf-8", description="File encoding")

class FileOperationsTool(BaseTool):
    """Tool for file system operations"""
    
    name: str = "file_operations"
    description: str = """
    Perform file system operations. Always specify the operation type and file/directory path.
    
    Available operations:
    - list: List files in a directory (use file_path as directory path)
    - read: Read content from a file (use file_path as file path)
    - write: Write content to a file (use file_path and content)
    - search: Search for text within files (use file_path and search_term)
    - analyze: Analyze file structure and metadata (use file_path)
    
    Example usage for listing directory: {"operation": "list", "file_path": "."}
    Example usage for reading file: {"operation": "read", "file_path": "data.txt"}
    """
    args_schema = FileOperationInput
    
    def _run(self, 
             operation: str, 
             file_path: str, 
             content: Optional[str] = None,
             search_term: Optional[str] = None,
             encoding: str = "utf-8") -> str:
        """Execute file operation"""
        try:
            path = Path(file_path)
            
            if operation == "read":
                return self._read_file(path, encoding)
            elif operation == "write":
                return self._write_file(path, content, encoding)
            elif operation == "search":
                return self._search_files(path, search_term)
            elif operation == "list":
                return self._list_directory(path)
            elif operation == "analyze":
                return self._analyze_file(path)
            else:
                return f"Unknown operation: {operation}"
                
        except Exception as e:
            error_msg = f"File operation error: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _read_file(self, path: Path, encoding: str) -> str:
        """Read content from a file"""
        if not path.exists():
            return f"File not found: {path}"
        
        if not path.is_file():
            return f"Path is not a file: {path}"
        
        try:
            # Handle different file types
            file_extension = path.suffix.lower()
            
            if file_extension == '.csv':
                df = pd.read_csv(path, encoding=encoding)
                return f"CSV file content (first 10 rows):\n{df.head(10).to_string()}\n\nShape: {df.shape}"
            
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(path)
                return f"Excel file content (first 10 rows):\n{df.head(10).to_string()}\n\nShape: {df.shape}"
            
            elif file_extension == '.json':
                with open(path, 'r', encoding=encoding) as f:
                    data = json.load(f)
                return f"JSON content:\n{json.dumps(data, indent=2)[:1000]}..."
            
            else:
                # Read as text file
                with open(path, 'r', encoding=encoding) as f:
                    content = f.read()
                
                if len(content) > 2000:
                    return f"File content (first 2000 characters):\n{content[:2000]}...\n\nTotal length: {len(content)} characters"
                else:
                    return f"File content:\n{content}"
                    
        except UnicodeDecodeError:
            return f"Cannot read file with encoding {encoding}. File may be binary or use different encoding."
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def _write_file(self, path: Path, content: str, encoding: str) -> str:
        """Write content to a file"""
        if content is None:
            return "No content provided for write operation"
        
        try:
            # Create directory if it doesn't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            with open(path, 'w', encoding=encoding) as f:
                f.write(content)
            
            return f"Successfully wrote {len(content)} characters to {path}"
            
        except Exception as e:
            return f"Error writing file: {str(e)}"
    
    def _search_files(self, path: Path, search_term: str) -> str:
        """Search for text within files"""
        if search_term is None:
            return "No search term provided"
        
        results = []
        
        try:
            if path.is_file():
                # Search in single file
                result = self._search_in_file(path, search_term)
                if result:
                    results.append(result)
            elif path.is_dir():
                # Search in directory
                for file_path in path.rglob("*"):
                    if file_path.is_file() and self._is_text_file(file_path):
                        result = self._search_in_file(file_path, search_term)
                        if result:
                            results.append(result)
            
            if results:
                return f"Search results for '{search_term}':\n" + "\n".join(results)
            else:
                return f"No matches found for '{search_term}'"
                
        except Exception as e:
            return f"Error during search: {str(e)}"
    
    def _search_in_file(self, file_path: Path, search_term: str) -> Optional[str]:
        """Search for term in a specific file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            matches = []
            for i, line in enumerate(lines, 1):
                if search_term.lower() in line.lower():
                    matches.append(f"  Line {i}: {line.strip()}")
            
            if matches:
                return f"{file_path}:\n" + "\n".join(matches[:5])  # Limit to 5 matches per file
            
        except Exception:
            pass  # Skip files that can't be read
        
        return None
    
    def _is_text_file(self, path: Path) -> bool:
        """Check if file is likely a text file"""
        text_extensions = {'.txt', '.py', '.js', '.html', '.css', '.json', '.csv', '.md', '.yml', '.yaml', '.xml'}
        return path.suffix.lower() in text_extensions
    
    def _list_directory(self, path: Path) -> str:
        """List contents of a directory"""
        if not path.exists():
            return f"Directory not found: {path}"
        
        if not path.is_dir():
            return f"Path is not a directory: {path}"
        
        try:
            items = []
            for item in sorted(path.iterdir()):
                if item.is_dir():
                    items.append(f"📁 {item.name}/")
                else:
                    size = item.stat().st_size
                    if size < 1024:
                        size_str = f"{size}B"
                    elif size < 1024 * 1024:
                        size_str = f"{size // 1024}KB"
                    else:
                        size_str = f"{size // (1024 * 1024)}MB"
                    items.append(f"📄 {item.name} ({size_str})")
            
            return f"Directory contents of {path}:\n" + "\n".join(items)
            
        except Exception as e:
            return f"Error listing directory: {str(e)}"
    
    def _analyze_file(self, path: Path) -> str:
        """Analyze file structure and metadata"""
        if not path.exists():
            return f"File not found: {path}"
        
        try:
            stat = path.stat()
            analysis = [
                f"File Analysis for: {path}",
                f"Size: {stat.st_size} bytes",
                f"Type: {'Directory' if path.is_dir() else 'File'}",
                f"Extension: {path.suffix}",
            ]
            
            if path.is_file():
                # Additional file analysis
                if path.suffix.lower() == '.csv':
                    try:
                        df = pd.read_csv(path)
                        analysis.extend([
                            f"CSV Rows: {len(df)}",
                            f"CSV Columns: {len(df.columns)}",
                            f"Column Names: {list(df.columns)[:10]}",  # First 10 columns
                            f"Data Types: {df.dtypes.to_dict()}"
                        ])
                    except Exception:
                        analysis.append("Could not analyze CSV structure")
                
                elif path.suffix.lower() in ['.xlsx', '.xls']:
                    try:
                        df = pd.read_excel(path)
                        analysis.extend([
                            f"Excel Rows: {len(df)}",
                            f"Excel Columns: {len(df.columns)}",
                            f"Column Names: {list(df.columns)[:10]}"
                        ])
                    except Exception:
                        analysis.append("Could not analyze Excel structure")
                
                elif path.suffix.lower() == '.json':
                    try:
                        with open(path, 'r') as f:
                            data = json.load(f)
                        if isinstance(data, dict):
                            analysis.append(f"JSON Keys: {list(data.keys())[:10]}")
                        elif isinstance(data, list):
                            analysis.append(f"JSON Array Length: {len(data)}")
                    except Exception:
                        analysis.append("Could not analyze JSON structure")
            
            return "\n".join(analysis)
            
        except Exception as e:
            return f"Error analyzing file: {str(e)}"
