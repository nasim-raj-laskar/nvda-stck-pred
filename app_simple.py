import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score
import warnings
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

@st.cache_data
def load_data():
    periods = ['2y', '1y', '6mo', '3mo']
    
    for period in periods:
        try:
            ticker = yf.Ticker("NVDA")
            data = ticker.history(period=period)
            
            if not data.empty and len(data) > 50:
                # Create target
                data["Tomorrow"] = data["Close"].shift(-1)
                data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
                
                # Add simple technical indicators
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
        except:
            continue
    
    return None, []

def main():
    st.title("🚀 NVIDIA Stock Prediction Dashboard")
    
    # Load data
    with st.spinner("Loading data..."):
        data, predictors = load_data()
    
    if data is None or data.empty:
        st.error("Failed to load data. Please try again later.")
        return
    
    st.success(f"✅ Loaded {len(data)} days of data")
    
    # Sidebar
    st.sidebar.header("Model Settings")
    n_estimators = st.sidebar.slider("Number of Trees", 50, 200, 100)
    threshold = st.sidebar.slider("Confidence Threshold", 0.5, 0.9, 0.6, 0.05)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Overview", "🎯 Predictions", "📊 Performance"])
    
    with tab1:
        st.header("Stock Overview")
        
        # Current metrics
        col1, col2, col3, col4 = st.columns(4)
        
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2]
        change = current_price - prev_price
        change_pct = (change / prev_price) * 100
        
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
            high_52w = data['High'].rolling(252).max().iloc[-1]
            st.markdown(f"""
            <div class="metric-card">
                <h3>52W High</h3>
                <h2>${high_52w:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            low_52w = data['Low'].rolling(252).min().iloc[-1]
            st.markdown(f"""
            <div class="metric-card">
                <h3>52W Low</h3>
                <h2>${low_52w:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Price chart
        st.subheader("Price Chart")
        fig = go.Figure()
        
        # Last 6 months
        chart_data = data.tail(180)
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
                ma = data['Close'].rolling(period).mean().tail(180)
                fig.add_trace(go.Scatter(
                    x=ma.index,
                    y=ma,
                    mode='lines',
                    name=f'MA{period}',
                    line=dict(width=1),
                    opacity=0.7
                ))
        
        fig.update_layout(
            title="NVIDIA Stock Price - Last 6 Months",
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
        
        # Simple train/test split
        split_idx = int(len(data) * 0.8)
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        if len(train_data) < 100 or len(test_data) < 10:
            st.error("Insufficient data for training")
            return
        
        # Train model
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=1)
        
        try:
            model.fit(train_data[predictors], train_data['Target'])
            
            # Make predictions
            probabilities = model.predict_proba(test_data[predictors])[:, 1]
            predictions = (probabilities >= threshold).astype(int)
            
            # Calculate metrics
            precision = precision_score(test_data['Target'], predictions)
            accuracy = accuracy_score(test_data['Target'], predictions)
            
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
            
            recent_data = pd.DataFrame({
                'Date': test_data.index[-10:].strftime('%Y-%m-%d'),
                'Close': test_data['Close'].iloc[-10:].round(2),
                'Prediction': ['🟢 BUY' if p == 1 else '🔴 HOLD' for p in predictions[-10:]],
                'Confidence': [f"{p:.1%}" for p in probabilities[-10:]],
                'Actual': ['📈 UP' if t == 1 else '📉 DOWN' for t in test_data['Target'].iloc[-10:]]
            })
            
            st.dataframe(recent_data, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error training model: {e}")
    
    with tab3:
        st.header("Model Performance")
        
        if len(predictors) == 0:
            st.error("No predictors available")
            return
        
        # Feature importance
        try:
            split_idx = int(len(data) * 0.8)
            train_data = data.iloc[:split_idx]
            
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=1)
            model.fit(train_data[predictors], train_data['Target'])
            
            importance_df = pd.DataFrame({
                'Feature': predictors,
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
    
    # Disclaimer
    st.markdown("---")
    st.warning("⚠️ **Disclaimer**: This is for educational purposes only. Not financial advice. Trading involves risk.")

if __name__ == "__main__":
    main()