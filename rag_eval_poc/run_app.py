#!/usr/bin/env python
"""
Main entry point for running the Streamlit web UI
Handles path setup for the restructured project
"""
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

if __name__ == "__main__":
    project_root = Path(__file__).parent
    
    # Load environment variables from .env BEFORE running subprocess
    # This ensures all API keys, configs, etc. are available in the subprocess
    env_path = project_root / "config" / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
    else:
        # Try default .env location in project root
        default_env = project_root / ".env"
        if default_env.exists():
            load_dotenv(default_env, override=True)
    
    src_path = project_root / "src"
    app_path = src_path / "app.py"
    
    # Run streamlit directly
    result = subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(app_path)
    ])
    
    # Exit with appropriate code if Streamlit fails
    if result.returncode != 0:
        sys.exit(result.returncode)
