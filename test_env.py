#!/usr/bin/env python3
"""
Test environment variable loading.
"""

import sys
import os
sys.path.append('/home/renzk/projects/agentic_data_analysis')

from src.utils.config import Config
from dotenv import load_dotenv

# Test environment loading
load_dotenv()
print(f"OPENAI_API_KEY from env: {os.getenv('OPENAI_API_KEY')}")

# Test config
config = Config()
print(f"Config openai_api_key: {config.openai_api_key}")
print(f"Config type: {type(config.openai_api_key)}")
