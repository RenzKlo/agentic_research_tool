"""
Configuration management for the Agentic AI system.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging

# Try to load dotenv, but don't fail if it's not available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not available, using only environment variables
except Exception as e:
    pass  # Could not load .env file

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for the agentic AI system."""
    
    def __init__(self):
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from environment variables."""
        # LLM Configuration
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.model_name = os.getenv("MODEL_NAME", "gpt-4-turbo-preview")
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "4000"))
        
        # Tool Configuration
        self.tavily_api_key = os.getenv("TAVILY_API_KEY", "")
        self.enable_code_execution = os.getenv("ENABLE_CODE_EXECUTION", "false").lower() == "true"
        self.max_file_size = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB
        self.allowed_file_extensions = os.getenv("ALLOWED_FILE_EXTENSIONS", ".py,.txt,.csv,.json,.md,.html,.xml").split(",")
        
        # Agent Configuration
        self.max_iterations = int(os.getenv("MAX_ITERATIONS", "10"))
        self.max_execution_time = int(os.getenv("MAX_EXECUTION_TIME", "300"))  # 5 minutes
        self.verbose = os.getenv("VERBOSE", "false").lower() == "true"
        
        # Memory Configuration
        self.memory_type = os.getenv("MEMORY_TYPE", "conversation_buffer")
        self.memory_key = os.getenv("MEMORY_KEY", "chat_history")
        self.max_memory_length = int(os.getenv("MAX_MEMORY_LENGTH", "2000"))
        
        # Streamlit Configuration
        self.streamlit_theme = os.getenv("STREAMLIT_THEME", "light")
        self.show_debug_info = os.getenv("SHOW_DEBUG_INFO", "false").lower() == "true"
        
        # Data paths
        self.data_directory = Path(os.getenv("DATA_DIRECTORY", "./data"))
        self.logs_directory = Path(os.getenv("LOGS_DIRECTORY", "./logs"))
        self.cache_directory = Path(os.getenv("CACHE_DIRECTORY", "./cache"))
        
        # Create directories if they don't exist
        self.data_directory.mkdir(exist_ok=True)
        self.logs_directory.mkdir(exist_ok=True)
        self.cache_directory.mkdir(exist_ok=True)
        
        # Validate required configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate required configuration values."""
        if not self.openai_api_key:
            logger.warning("OpenAI API key not found. Agent will not function properly.")
        
        if not self.tavily_api_key:
            logger.warning("Tavily API key not found. Web research capabilities will be limited.")
        
        if self.max_iterations < 1:
            raise ValueError("MAX_ITERATIONS must be at least 1")
        
        if self.max_execution_time < 30:
            raise ValueError("MAX_EXECUTION_TIME must be at least 30 seconds")
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration as a dictionary."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "openai_api_key": self.openai_api_key,
        }
    
    def get_agent_config(self) -> Dict[str, Any]:
        """Get agent configuration as a dictionary."""
        return {
            "max_iterations": self.max_iterations,
            "max_execution_time": self.max_execution_time,
            "verbose": self.verbose,
        }
    
    def get_memory_config(self) -> Dict[str, Any]:
        """Get memory configuration as a dictionary."""
        return {
            "memory_type": self.memory_type,
            "memory_key": self.memory_key,
            "max_memory_length": self.max_memory_length,
        }
    
    def get_tool_config(self) -> Dict[str, Any]:
        """Get tool configuration as a dictionary."""
        return {
            "tavily_api_key": self.tavily_api_key,
            "enable_code_execution": self.enable_code_execution,
            "max_file_size": self.max_file_size,
            "allowed_file_extensions": self.allowed_file_extensions,
        }
    
    def update_config(self, **kwargs) -> None:
        """Update configuration values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Updated config: {key} = {value}")
            else:
                logger.warning(f"Unknown config key: {key}")
    
    def is_valid(self) -> bool:
        """Check if the configuration is valid for running the agent."""
        return bool(self.openai_api_key)
    
    def get_safe_config(self) -> Dict[str, Any]:
        """Get configuration without sensitive information for display."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_iterations": self.max_iterations,
            "max_execution_time": self.max_execution_time,
            "verbose": self.verbose,
            "memory_type": self.memory_type,
            "max_memory_length": self.max_memory_length,
            "enable_code_execution": self.enable_code_execution,
            "max_file_size": self.max_file_size,
            "allowed_file_extensions": self.allowed_file_extensions,
            "data_directory": str(self.data_directory),
            "logs_directory": str(self.logs_directory),
            "cache_directory": str(self.cache_directory),
        }


# Global configuration instance
config = Config()
