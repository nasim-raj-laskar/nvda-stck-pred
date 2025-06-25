#!/bin/bash

echo "🚀 Starting NVIDIA Stock Prediction Dashboard..."
echo "📊 The app will open in your default browser"
echo "🔗 If it doesn't open automatically, go to: http://localhost:8501"
echo ""
echo "="*60
echo "🎯 NVIDIA STOCK PREDICTION DASHBOARD"
echo "="*60
echo "📈 Features:"
echo "  • Real-time stock data and predictions"
echo "  • Interactive charts and technical analysis" 
echo "  • Machine learning model performance metrics"
echo "="*60
echo ""
echo "⚠️  DISCLAIMER: For educational purposes only!"
echo "   Not financial advice. Trade at your own risk."
echo ""

# Check if required packages are installed
echo "🔍 Checking dependencies..."

python3 -c "import streamlit, pandas, numpy, yfinance, plotly, sklearn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing required packages..."
    pip3 install streamlit pandas numpy yfinance plotly scikit-learn
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install packages. Please install manually:"
        echo "pip3 install streamlit pandas numpy yfinance plotly scikit-learn"
        exit 1
    fi
fi

echo "✅ All dependencies are ready!"
echo ""

# Try different app versions based on what works
echo "🔍 Testing internet connection..."
python3 -c "import yfinance as yf; data = yf.Ticker('AAPL').history(period='1d'); exit(0 if not data.empty else 1)" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ Internet connection OK - using live data"
    if [ -f "app_simple.py" ]; then
        echo "🚀 Starting simplified dashboard..."
        python3 -m streamlit run app_simple.py
    else
        echo "🚀 Starting main dashboard..."
        python3 -m streamlit run app.py
    fi
else
    echo "⚠️  Internet connection issue - using demo with sample data"
    if [ -f "app_demo.py" ]; then
        echo "🚀 Starting demo dashboard..."
        python3 -m streamlit run app_demo.py
    else
        echo "❌ No demo app available"
        exit 1
    fi
fi