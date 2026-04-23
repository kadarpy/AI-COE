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
    import subprocess
    import sys

    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "src/app.py"
        ])
    except KeyboardInterrupt:
        print("\nStreamlit app stopped gracefully.")
