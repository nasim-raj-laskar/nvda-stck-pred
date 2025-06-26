import streamlit as st
import pandas as pd
import numpy as np
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
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .warning-card {
        background: linear-gradient(135deg, #fc4a1a 0%, #f7b733 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def create_nvidia_sample_data():
    """Create realistic NVIDIA stock sample data"""
    # Create date range (weekdays only)
    dates = pd.date_range(start='2024-01-01', end='2024-12-01', freq='D')
    dates = dates[dates.weekday < 5]  # Only weekdays
    
    # Generate realistic NVDA-like price data
    np.random.seed(42)
    base_price = 400
    prices = []
    current_price = base_price
    
    for i in range(len(dates)):
        # NVIDIA has had significant growth in 2024, simulate this
        if i < len(dates) * 0.3:  # First 30% - steady growth
            trend = 0.002
        elif i < len(dates) * 0.7:  # Middle 40% - rapid growth
            trend = 0.004
        else:  # Last 30% - some volatility
            trend = 0.001
        
        # Add some cyclical patterns and volatility
        cycle = 0.001 * np.sin(i / 20)  # 20-day cycle
        volatility = np.random.normal(0, 0.025)  # 2.5% daily volatility
        
        current_price *= (1 + trend + cycle + volatility)
        prices.append(max(current_price, 50))  # Minimum price floor
    
    # Create OHLCV data
    data = pd.DataFrame(index=dates)
    data['Close'] = prices
    
    # Generate Open, High, Low based on Close
    data['Open'] = data['Close'].shift(1) * (1 + np.random.normal(0, 0.005, len(data)))
    data['High'] = np.maximum(data['Open'], data['Close']) * (1 + np.abs(np.random.normal(0, 0.015, len(data))))
    data['Low'] = np.minimum(data['Open'], data['Close']) * (1 - np.abs(np.random.normal(0, 0.015, len(data))))
    data['Volume'] = np.random.randint(15000000, 80000000, len(data))
    
    # Fill NaN values
    data['Open'].fillna(data['Close'], inplace=True)
    
    # Create target variable
    data["Tomorrow"] = data["Close"].shift(-1)
    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
    
    # Add technical indicators
    horizons = [2, 5, 20, 60]
    predictors = []
    
    for horizon in horizons:
        rolling_avg = data["Close"].rolling(horizon).mean()
        ratio_col = f"Close_Ratio_{horizon}"
        data[ratio_col] = data["Close"] / rolling_avg
        trend_col = f"Trend_{horizon}"
        data[trend_col] = data.shift(1).rolling(horizon).sum()["Target"]
        predictors += [ratio_col, trend_col]
    
    # Add RSI
    data['RSI'] = calculate_rsi(data['Close'])
    
    # Add MACD
    data['MACD'], data['MACD_Signal'] = calculate_macd(data['Close'])
    
    data = data.dropna()
    return data, predictors

def calculate_rsi(prices, window=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    return macd, macd_signal

def main():
    st.title("🚀 NVIDIA Stock Prediction Dashboard")
    
    # Info about data source
    st.info("📊 **Note**: Using simulated NVIDIA stock data for demonstration due to API limitations. The model and features are fully functional.")
    
    # Load sample data
    with st.spinner("Loading stock data..."):
        data, predictors = create_nvidia_sample_data()
    
    st.success(f"✅ Loaded {len(data)} days of sample data")
    
    # Sidebar
    st.sidebar.header("🔧 Model Settings")
    n_estimators = st.sidebar.slider("Number of Trees", 50, 300, 150)
    threshold = st.sidebar.slider("Confidence Threshold", 0.5, 0.9, 0.6, 0.05)
    test_size = st.sidebar.slider("Test Data %", 10, 40, 20)
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"📊 Dataset: {len(data)} records")
    st.sidebar.info(f"📅 Period: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🎯 Predictions", "📊 Performance", "🔍 Technical Analysis"])
    
    with tab1:
        st.header("📈 Stock Overview")
        
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
            card_class = "success-card" if change >= 0 else "warning-card"
            st.markdown(f"""
            <div class="{card_class}">
                <h3>Daily Change</h3>
                <h2>{change:+.2f} ({change_pct:+.2f}%)</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            high_period = data['High'].rolling(min(252, len(data))).max().iloc[-1]
            st.markdown(f"""
            <div class="metric-card">
                <h3>Period High</h3>
                <h2>${high_period:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            low_period = data['Low'].rolling(min(252, len(data))).min().iloc[-1]
            st.markdown(f"""
            <div class="metric-card">
                <h3>Period Low</h3>
                <h2>${low_period:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Price chart
        st.subheader("📊 Stock Price Chart")
        
        fig = go.Figure()
        
        # Show last 6 months or all data
        chart_data = data.tail(min(130, len(data)))
        
        fig.add_trace(go.Scatter(
            x=chart_data.index,
            y=chart_data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#76B900', width=3)
        ))
        
        # Moving averages
        for period, color in [(20, '#FF6B6B'), (50, '#4ECDC4')]:
            if len(data) > period:
                ma = data['Close'].rolling(period).mean().tail(min(130, len(data)))
                fig.add_trace(go.Scatter(
                    x=ma.index,
                    y=ma,
                    mode='lines',
                    name=f'MA{period}',
                    line=dict(color=color, width=2),
                    opacity=0.8
                ))
        
        fig.update_layout(
            title="NVIDIA Stock Price - Recent Performance",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500,
            template="plotly_white",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance summary
        st.subheader("📈 Performance Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            period_return = ((current_price - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
            st.metric("Period Return", f"{period_return:.1f}%")
        
        with col2:
            volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
            st.metric("Annualized Volatility", f"{volatility:.1f}%")
        
        with col3:
            up_days = (data['Close'].pct_change() > 0).sum()
            win_rate = (up_days / len(data)) * 100
            st.metric("Up Days", f"{win_rate:.1f}%")
    
    with tab2:
        st.header("🎯 Model Predictions")
        
        # Train/test split
        split_idx = int(len(data) * (1 - test_size/100))
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        st.info(f"📊 Training on {len(train_data)} days, testing on {len(test_data)} days")
        
        # Train model
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, min_samples_split=5)
        
        try:
            # Clean data
            train_clean = train_data[predictors + ['Target']].dropna()
            test_clean = test_data[predictors].dropna()
            
            model.fit(train_clean[predictors], train_clean['Target'])
            
            # Make predictions
            probabilities = model.predict_proba(test_clean[predictors])[:, 1]
            predictions = (probabilities >= threshold).astype(int)
            
            # Get corresponding test targets
            test_targets = test_data.loc[test_clean.index, 'Target']
            
            # Calculate metrics
            precision = precision_score(test_targets, predictions) if len(set(predictions)) > 1 else 0
            accuracy = accuracy_score(test_targets, predictions)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
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
            
            with col4:
                avg_confidence = np.mean(probabilities)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Avg Confidence</h3>
                    <h2>{avg_confidence:.1%}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Prediction chart
            st.subheader("📈 Predictions vs Actual Prices")
            
            fig = go.Figure()
            
            # Price line
            fig.add_trace(go.Scatter(
                x=test_data.index,
                y=test_data['Close'],
                mode='lines',
                name='Stock Price',
                line=dict(color='blue', width=2)
            ))
            
            # Buy signals
            buy_indices = test_clean.index[predictions == 1]
            if len(buy_indices) > 0:
                fig.add_trace(go.Scatter(
                    x=buy_indices,
                    y=[test_data.loc[idx, 'Close'] for idx in buy_indices],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(color='green', size=10, symbol='triangle-up')
                ))
            
            fig.update_layout(
                title="Stock Price with Buy Signals",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=400,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recent predictions table
            st.subheader("🔮 Recent Predictions")
            
            n_recent = min(15, len(test_clean))
            recent_indices = test_clean.index[-n_recent:]
            
            recent_data = pd.DataFrame({
                'Date': [idx.strftime('%Y-%m-%d') for idx in recent_indices],
                'Price': [f"${test_data.loc[idx, 'Close']:.2f}" for idx in recent_indices],
                'Signal': ['🟢 BUY' if p == 1 else '🔴 HOLD' for p in predictions[-n_recent:]],
                'Confidence': [f"{p:.1%}" for p in probabilities[-n_recent:]],
                'Actual Next Day': ['📈 UP' if test_data.loc[idx, 'Target'] == 1 else '📉 DOWN' for idx in recent_indices]
            })
            
            st.dataframe(recent_data, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error in prediction model: {e}")
    
    with tab3:
        st.header("📊 Model Performance Analysis")
        
        try:
            # Feature importance
            st.subheader("🔍 Feature Importance")
            
            split_idx = int(len(data) * 0.8)
            train_data = data.iloc[:split_idx]
            train_clean = train_data[predictors + ['Target']].dropna()
            
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            model.fit(train_clean[predictors], train_clean['Target'])
            
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
                title="Feature Importance in Prediction Model",
                xaxis_title="Importance Score",
                height=500,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Model insights
            st.subheader("🧠 Model Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top 3 Most Important Features:**")
                top_features = importance_df.tail(3)
                for _, row in top_features.iterrows():
                    st.write(f"• {row['Feature']}: {row['Importance']:.3f}")
            
            with col2:
                st.write("**Model Configuration:**")
                st.write(f"• Number of Trees: {n_estimators}")
                st.write(f"• Confidence Threshold: {threshold:.1%}")
                st.write(f"• Features Used: {len(predictors)}")
            
        except Exception as e:
            st.error(f"Error in performance analysis: {e}")
    
    with tab4:
        st.header("🔍 Technical Analysis")
        
        # RSI Chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 RSI (Relative Strength Index)")
            
            fig = go.Figure()
            
            rsi_data = data['RSI'].tail(100)
            fig.add_trace(go.Scatter(
                x=rsi_data.index,
                y=rsi_data,
                mode='lines',
                name='RSI',
                line=dict(color='purple', width=2)
            ))
            
            fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
            fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
            fig.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="Neutral (50)")
            
            fig.update_layout(
                yaxis_title="RSI",
                height=300,
                template="plotly_white",
                yaxis=dict(range=[0, 100])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Current RSI status
            current_rsi = data['RSI'].iloc[-1]
            if current_rsi > 70:
                st.warning(f"🔴 RSI: {current_rsi:.1f} - Potentially Overbought")
            elif current_rsi < 30:
                st.success(f"🟢 RSI: {current_rsi:.1f} - Potentially Oversold")
            else:
                st.info(f"🟡 RSI: {current_rsi:.1f} - Neutral Zone")
        
        with col2:
            st.subheader("📈 MACD")
            
            fig = go.Figure()
            
            macd_data = data[['MACD', 'MACD_Signal']].tail(100)
            
            fig.add_trace(go.Scatter(
                x=macd_data.index,
                y=macd_data['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=macd_data.index,
                y=macd_data['MACD_Signal'],
                mode='lines',
                name='Signal Line',
                line=dict(color='red', width=2)
            ))
            
            fig.add_hline(y=0, line_dash="dot", line_color="gray")
            
            fig.update_layout(
                yaxis_title="MACD",
                height=300,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # MACD signal
            current_macd = data['MACD'].iloc[-1]
            current_signal = data['MACD_Signal'].iloc[-1]
            
            if current_macd > current_signal:
                st.success("🟢 MACD above Signal Line - Bullish")
            else:
                st.warning("🔴 MACD below Signal Line - Bearish")
        
        # Volume analysis
        st.subheader("📊 Volume Analysis")
        
        fig = go.Figure()
        
        volume_data = data.tail(100)
        
        # Volume bars
        colors = ['green' if close > open else 'red' 
                 for close, open in zip(volume_data['Close'], volume_data['Open'])]
        
        fig.add_trace(go.Bar(
            x=volume_data.index,
            y=volume_data['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ))
        
        # Volume moving average
        vol_ma = data['Volume'].rolling(20).mean().tail(100)
        fig.add_trace(go.Scatter(
            x=vol_ma.index,
            y=vol_ma,
            mode='lines',
            name='Volume MA(20)',
            line=dict(color='orange', width=2)
        ))
        
        fig.update_layout(
            title="Trading Volume",
            xaxis_title="Date",
            yaxis_title="Volume",
            height=400,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Disclaimer
    st.markdown("---")
    st.warning("⚠️ **Disclaimer**: This dashboard is for educational purposes only. The data is simulated and should not be used for actual trading decisions. Always consult with financial professionals before making investment decisions.")

if __name__ == "__main__":
    main()