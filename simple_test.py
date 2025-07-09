#!/usr/bin/env python3
"""Simple test to isolate the issue."""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Starting minimal test...")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Project root: {project_root}")

try:
    print("Testing basic imports...")
    import json
    print("✓ json imported")
    
    import logging
    print("✓ logging imported")
    
    import os
    print("✓ os imported")
    
    print("Testing project structure...")
    config_path = project_root / "src" / "utils" / "config.py"
    print(f"Config path exists: {config_path.exists()}")
    
    if config_path.exists():
        print("Trying to import config...")
        from src.utils.config import Config
        print("✓ Config imported successfully")
        
        config = Config()
        print("✓ Config created successfully")
        print(f"Model: {config.model_name}")
        
    else:
        print("❌ Config file not found")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("Test complete.")
