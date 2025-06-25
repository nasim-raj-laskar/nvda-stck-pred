#!/usr/bin/env python3
"""
Quick launcher script for the NVIDIA Stock Prediction Dashboard
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'pandas', 
        'numpy',
        'yfinance',
        'plotly',
        'scikit-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages. Please install manually:")
        print("pip install -r requirements.txt")
        return False

def run_streamlit_app():
    """Launch the Streamlit application"""
    print("🚀 Launching NVIDIA Stock Prediction Dashboard...")
    print("📊 The app will open in your default browser")
    print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
    print("\n" + "="*60)
    print("🎯 NVIDIA STOCK PREDICTION DASHBOARD")
    print("="*60)
    print("📈 Features:")
    print("  • Real-time stock data and predictions")
    print("  • Interactive charts and technical analysis") 
    print("  • Machine learning model performance metrics")
    print("  • Backtesting and strategy analysis")
    print("="*60)
    print("\n⚠️  DISCLAIMER: For educational purposes only!")
    print("   Not financial advice. Trade at your own risk.\n")
    
    try:
        # Run streamlit app
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Thanks for using the NVIDIA Stock Prediction Dashboard!")
    except Exception as e:
        print(f"❌ Error running the app: {e}")
        print("💡 Try running manually: streamlit run app.py")

def main():
    """Main function to run the application"""
    print("🔍 Checking system requirements...")
    
    # Check if we're in the right directory
    if not Path("app.py").exists():
        print("❌ app.py not found in current directory")
        print("💡 Please navigate to the project directory first")
        return
    
    # Check for missing packages
    missing = check_requirements()
    
    if missing:
        print(f"📋 Missing packages: {', '.join(missing)}")
        
        # Ask user if they want to install
        response = input("🤔 Would you like to install them now? (y/n): ").lower().strip()
        
        if response in ['y', 'yes']:
            if not install_requirements():
                return
        else:
            print("💡 Please install requirements manually:")
            print("   pip install -r requirements.txt")
            return
    else:
        print("✅ All required packages are installed!")
    
    # Run the app
    run_streamlit_app()

if __name__ == "__main__":
    main()