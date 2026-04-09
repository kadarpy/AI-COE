#!/usr/bin/env python
"""
Main entry point for running the Streamlit web UI
Handles path setup for the restructured project
"""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    project_root = Path(__file__).parent
    src_path = project_root / "src"
    app_path = src_path / "app.py"
    
    # Run streamlit directly
    result = subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(app_path)
    ])
    
    # Exit with appropriate code if Streamlit fails
    if result.returncode != 0:
        sys.exit(result.returncode)
