import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score
import warnings
import time
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="NVIDIA Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data():
    """Load NVIDIA stock data with robust error handling"""
    
    # Try different approaches to get data
    approaches = [
        # Approach 1: Recent periods
        {'period': '1y'},
        {'period': '6mo'},
        {'period': '3mo'},
        # Approach 2: Specific date ranges
        {'start': '2023-01-01', 'end': '2024-12-31'},
        {'start': '2023-06-01', 'end': '2024-12-31'},
        {'start': '2024-01-01', 'end': '2024-12-31'},
    ]
    
    for i, params in enumerate(approaches):
        try:
            st.write(f"Trying approach {i+1}...")
            
            # Add delay to avoid rate limiting
            if i > 0:
                time.sleep(1)
            
            ticker = yf.Ticker("NVDA")
            
            if 'period' in params:
                data = ticker.history(period=params['period'])
            else:
                data = ticker.history(start=params['start'], end=params['end'])
            
            if not data.empty and len(data) > 50:
                st.success(f"✅ Successfully loaded {len(data)} days of data using approach {i+1}")
                
                # Create target variable
                data["Tomorrow"] = data["Close"].shift(-1)
                data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
                
                # Add technical indicators
                horizons = [2, 5, 20, 60]
                predictors = []
                
                for horizon in horizons:
                    if len(data) > horizon:
                        rolling_avg = data["Close"].rolling(horizon).mean()
                        ratio_col = f"Close_Ratio_{horizon}"
                        data[ratio_col] = data["Close"] / rolling_avg
                        trend_col = f"Trend_{horizon}"
                        data[trend_col] = data.shift(1).rolling(horizon).sum()["Target"]
                        predictors += [ratio_col, trend_col]
                
                data = data.dropna()
                if not data.empty:
                    return data, predictors
            else:
                st.warning(f"Approach {i+1} returned empty data")
                
        except Exception as e:
            st.warning(f"Approach {i+1} failed: {str(e)}")
            continue
    
    # If all approaches fail, create sample data
    st.error("All data loading approaches failed. Using sample data for demonstration.")
    return create_sample_data()

def create_sample_data():
    """Create sample data for demonstration when API fails"""
    dates = pd.date_range(start='2024-01-01', end='2024-12-01', freq='D')
    dates = dates[dates.weekday < 5]  # Only weekdays
    
    # Generate realistic NVDA-like price data
    np.random.seed(42)
    base_price = 400
    prices = []
    current_price = base_price
    
    for i in range(len(dates)):
        # Add some trend and volatility
        trend = 0.001 * (1 + 0.5 * np.sin(i / 50))  # Slight upward trend with cycles
        volatility = np.random.normal(0, 0.03)  # 3% daily volatility
        current_price *= (1 + trend + volatility)
        prices.append(current_price)
    
    # Create OHLCV data
    data = pd.DataFrame(index=dates)
    data['Close'] = prices
    data['Open'] = data['Close'].shift(1) * (1 + np.random.normal(0, 0.01, len(data)))
    data['High'] = np.maximum(data['Open'], data['Close']) * (1 + np.abs(np.random.normal(0, 0.01, len(data))))
    data['Low'] = np.minimum(data['Open'], data['Close']) * (1 - np.abs(np.random.normal(0, 0.01, len(data))))
    data['Volume'] = np.random.randint(20000000, 100000000, len(data))
    
    # Create target and predictors
    data["Tomorrow"] = data["Close"].shift(-1)
    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
    
    horizons = [2, 5, 20, 60]
    predictors = []
    
    for horizon in horizons:
        rolling_avg = data["Close"].rolling(horizon).mean()
        ratio_col = f"Close_Ratio_{horizon}"
        data[ratio_col] = data["Close"] / rolling_avg
        trend_col = f"Trend_{horizon}"
        data[trend_col] = data.shift(1).rolling(horizon).sum()["Target"]
        predictors += [ratio_col, trend_col]
    
    data = data.dropna()
    
    st.info("📊 Using sample data for demonstration. Features and predictions are simulated.")
    return data, predictors

def main():
    st.title("🚀 NVIDIA Stock Prediction Dashboard")
    
    # Load data
    with st.spinner("Loading NVIDIA stock data..."):
        data, predictors = load_data()
    
    if data is None or data.empty:
        st.error("Failed to load any data. Please try again later.")
        return
    
    # Sidebar
    st.sidebar.header("Model Settings")
    n_estimators = st.sidebar.slider("Number of Trees", 50, 200, 100)
    threshold = st.sidebar.slider("Confidence Threshold", 0.5, 0.9, 0.6, 0.05)
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📈 Overview", "🎯 Predictions", "📊 Performance"])
    
    with tab1:
        st.header("Stock Overview")
        
        # Current metrics
        col1, col2, col3, col4 = st.columns(4)
        
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
        change = current_price - prev_price
        change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Current Price</h3>
                <h2>${current_price:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            color = "#38ef7d" if change >= 0 else "#fc4a1a"
            st.markdown(f"""
            <div class="metric-card" style="background: {color};">
                <h3>Daily Change</h3>
                <h2>{change:+.2f} ({change_pct:+.2f}%)</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            high_52w = data['High'].rolling(min(252, len(data))).max().iloc[-1]
            st.markdown(f"""
            <div class="metric-card">
                <h3>Period High</h3>
                <h2>${high_52w:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            low_52w = data['Low'].rolling(min(252, len(data))).min().iloc[-1]
            st.markdown(f"""
            <div class="metric-card">
                <h3>Period Low</h3>
                <h2>${low_52w:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Price chart
        st.subheader("Price Chart")
        fig = go.Figure()
        
        # Show last 6 months or all data if less
        chart_data = data.tail(min(180, len(data)))
        fig.add_trace(go.Scatter(
            x=chart_data.index,
            y=chart_data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#76B900', width=2)
        ))
        
        # Moving averages
        for period in [20, 50]:
            if len(data) > period:
                ma = data['Close'].rolling(period).mean().tail(min(180, len(data)))
                fig.add_trace(go.Scatter(
                    x=ma.index,
                    y=ma,
                    mode='lines',
                    name=f'MA{period}',
                    line=dict(width=1),
                    opacity=0.7
                ))
        
        fig.update_layout(
            title="NVIDIA Stock Price",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Model Predictions")
        
        if len(predictors) == 0:
            st.error("No predictors available")
            return
        
        # Train/test split
        split_idx = max(50, int(len(data) * 0.8))
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        if len(train_data) < 50 or len(test_data) < 5:
            st.error("Insufficient data for training")
            return
        
        # Train model
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=1)
        
        try:
            # Filter predictors that exist in both train and test
            available_predictors = [p for p in predictors if p in train_data.columns and p in test_data.columns]
            
            if len(available_predictors) == 0:
                st.error("No valid predictors available")
                return
            
            # Clean data
            train_clean = train_data[available_predictors + ['Target']].dropna()
            test_clean = test_data[available_predictors].dropna()
            
            if len(train_clean) < 20 or len(test_clean) < 2:
                st.error("Insufficient clean data for training")
                return
            
            model.fit(train_clean[available_predictors], train_clean['Target'])
            
            # Make predictions
            probabilities = model.predict_proba(test_clean[available_predictors])[:, 1]
            predictions = (probabilities >= threshold).astype(int)
            
            # Align test targets with predictions
            test_targets = test_data.loc[test_clean.index, 'Target']
            
            # Calculate metrics
            precision = precision_score(test_targets, predictions) if len(set(predictions)) > 1 else 0
            accuracy = accuracy_score(test_targets, predictions)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Precision</h3>
                    <h2>{precision:.1%}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Accuracy</h3>
                    <h2>{accuracy:.1%}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                buy_signals = sum(predictions)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Buy Signals</h3>
                    <h2>{buy_signals}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Recent predictions
            st.subheader("Recent Predictions")
            
            n_recent = min(10, len(test_clean))
            recent_indices = test_clean.index[-n_recent:]
            
            recent_data = pd.DataFrame({
                'Date': [idx.strftime('%Y-%m-%d') for idx in recent_indices],
                'Close': [f"${data.loc[idx, 'Close']:.2f}" for idx in recent_indices],
                'Prediction': ['🟢 BUY' if p == 1 else '🔴 HOLD' for p in predictions[-n_recent:]],
                'Confidence': [f"{p:.1%}" for p in probabilities[-n_recent:]],
                'Actual': ['📈 UP' if data.loc[idx, 'Target'] == 1 else '📉 DOWN' for idx in recent_indices]
            })
            
            st.dataframe(recent_data, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error training model: {e}")
            st.write("Debug info:")
            st.write(f"Train data shape: {train_data.shape}")
            st.write(f"Test data shape: {test_data.shape}")
            st.write(f"Available predictors: {len(available_predictors) if 'available_predictors' in locals() else 'Not calculated'}")
    
    with tab3:
        st.header("Model Performance")
        
        if len(predictors) == 0:
            st.error("No predictors available")
            return
        
        # Feature importance
        try:
            split_idx = max(50, int(len(data) * 0.8))
            train_data = data.iloc[:split_idx]
            
            available_predictors = [p for p in predictors if p in train_data.columns]
            
            if len(available_predictors) == 0:
                st.error("No valid predictors for feature importance")
                return
            
            train_clean = train_data[available_predictors + ['Target']].dropna()
            
            if len(train_clean) < 20:
                st.error("Insufficient data for feature importance")
                return
            
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=1)
            model.fit(train_clean[available_predictors], train_clean['Target'])
            
            importance_df = pd.DataFrame({
                'Feature': available_predictors,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig = go.Figure(go.Bar(
                x=importance_df['Importance'],
                y=importance_df['Feature'],
                orientation='h',
                marker_color='#76B900'
            ))
            
            fig.update_layout(
                title="Feature Importance",
                xaxis_title="Importance",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error calculating feature importance: {e}")
    
    # Data info
    st.sidebar.markdown("---")
    st.sidebar.info(f"📊 Dataset: {len(data)} records")
    st.sidebar.info(f"📅 Period: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    
    # Disclaimer
    st.markdown("---")
    st.warning("⚠️ **Disclaimer**: This is for educational purposes only. Not financial advice. Trading involves risk.")

if __name__ == "__main__":
    main()