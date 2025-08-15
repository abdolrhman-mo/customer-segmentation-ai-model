#!/usr/bin/env python3
import subprocess
import sys
import os

def main():
    """Launch the Breast Cancer Diagnosis App"""
    # Check if we're in the right directory
    if not os.path.exists("ui/main_app.py"):
        print("Error: ui/main_app.py not found!")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    
    # Launch the app
    print("🎯 Launching Breast Cancer Diagnosis App...")
    print("📁 Using specialized app from ui/ folder")
    print("🚀 This app demonstrates threshold optimization for maximum recall!")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "ui/main_app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
