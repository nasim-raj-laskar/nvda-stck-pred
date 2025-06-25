import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="NVIDIA Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #76B900;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-positive {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    .prediction-negative {
        background: linear-gradient(135deg, #fc4a1a 0%, #f7b733 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_nvidia_data():
    """Load NVIDIA stock data"""
    periods = ['5y', '2y', '1y', '6mo']
    
    for period in periods:
        try:
            nvidia = yf.Ticker("NVDA")
            nvda = nvidia.history(period=period)
            
            if not nvda.empty and len(nvda) > 100:
                # Create target variable
                nvda["Tomorrow"] = nvda["Close"].shift(-1)
                nvda["Target"] = (nvda["Tomorrow"] > nvda["Close"]).astype(int)
                
                # Add technical indicators
                horizons = [2, 5, 20, 60]
                new_predictors = []
                
                for horizon in horizons:
                    if len(nvda) > horizon:
                        rolling_averages = nvda.rolling(horizon).mean()
                        ratio_column = f"Close_Ratio_{horizon}"
                        nvda[ratio_column] = nvda["Close"] / rolling_averages["Close"]
                        trend_column = f"Trend_{horizon}"
                        nvda[trend_column] = nvda.shift(1).rolling(horizon).sum()["Target"]
                        new_predictors += [ratio_column, trend_column]
                
                nvda = nvda.dropna()
                if not nvda.empty:
                    return nvda, new_predictors
        except:
            continue
    
    st.error("Failed to load stock data. Please check your internet connection.")
    return pd.DataFrame(), []

def create_model():
    """Create and return the Random Forest model"""
    return RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

def predict_with_model(train, test, predictors, model):
    """Make predictions using the trained model"""
    try:
        if train.empty or test.empty or len(predictors) == 0:
            return np.array([]), np.array([])
        
        available_predictors = [p for p in predictors if p in train.columns and p in test.columns]
        if len(available_predictors) == 0:
            return np.array([]), np.array([])
        
        train_clean = train[available_predictors + ['Target']].dropna()
        test_clean = test[available_predictors].dropna()
        
        if train_clean.empty or test_clean.empty:
            return np.array([]), np.array([])
        
        model.fit(train_clean[available_predictors], train_clean["Target"])
        predictions = model.predict_proba(test_clean[available_predictors])[:, 1]
        predictions_binary = (predictions >= 0.6).astype(int)
        
        return predictions, predictions_binary
    
    except Exception as e:
        return np.array([]), np.array([])

def backtest_model(data, model, predictors, start=None, step=250):
    """Perform backtesting on the model"""
    if data.empty or len(predictors) == 0:
        return pd.DataFrame()
    
    if start is None:
        start = max(1000, len(data) // 3)
    
    start = min(start, len(data) - 100)
    all_predictions = []
    
    try:
        for i in range(start, data.shape[0], step):
            train = data.iloc[0:i].copy()
            test = data.iloc[i:(i+step)].copy()
            
            if len(test) == 0 or len(train) < 100:
                break
                
            probs, preds = predict_with_model(train, test, predictors, model)
            
            if len(probs) > 0:
                predictions_df = pd.DataFrame({
                    'Target': test['Target'],
                    'Prediction': preds,
                    'Probability': probs
                }, index=test.index)
                
                all_predictions.append(predictions_df)
        
        return pd.concat(all_predictions) if all_predictions else pd.DataFrame()
    
    except Exception as e:
        st.error(f"Error in backtesting: {str(e)}")
        return pd.DataFrame()

def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 NVIDIA Stock Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Model Configuration")
    
    # Load data
    with st.spinner("Loading NVIDIA stock data..."):
        nvda_data, predictors = load_nvidia_data()
    
    if not nvda_data.empty:
        st.sidebar.success(f"✅ Loaded {len(nvda_data)} days of data")
        try:
            start_date = nvda_data.index[0].strftime('%Y-%m-%d')
            end_date = nvda_data.index[-1].strftime('%Y-%m-%d')
            st.sidebar.info(f"📅 Data range: {start_date} to {end_date}")
        except:
            st.sidebar.info("📅 Data loaded successfully")
    else:
        st.sidebar.error("❌ Failed to load data")
        st.stop()
    
    # Model parameters
    st.sidebar.subheader("🔧 Model Parameters")
    n_estimators = st.sidebar.slider("Number of Trees", 50, 500, 200, 50)
    min_samples_split = st.sidebar.slider("Min Samples Split", 10, 100, 50, 10)
    confidence_threshold = st.sidebar.slider("Prediction Confidence Threshold", 0.5, 0.9, 0.6, 0.05)
    
    # Create model with user parameters
    model = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, random_state=1)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Overview", "🎯 Predictions", "📊 Model Performance", "🔍 Technical Analysis", "📋 Data Explorer"])
    
    with tab1:
        st.header("📈 NVIDIA Stock Overview")
        
        if nvda_data.empty:
            st.error("No data available to display")
            return
        
        # Current stock info
        col1, col2, col3, col4 = st.columns(4)
        
        try:
            current_price = nvda_data['Close'].iloc[-1]
            prev_price = nvda_data['Close'].iloc[-2] if len(nvda_data) > 1 else current_price
            price_change = current_price - prev_price
            price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
        except:
            current_price = 0
            price_change = 0
            price_change_pct = 0
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Current Price</h3>
                <h2>${current_price:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            color = "prediction-positive" if price_change >= 0 else "prediction-negative"
            st.markdown(f"""
            <div class="{color}">
                <h3>Daily Change</h3>
                <h2>{price_change:+.2f} ({price_change_pct:+.2f}%)</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            try:
                high_52w = nvda_data['High'].rolling(252).max().iloc[-1]
            except:
                high_52w = nvda_data['High'].max() if not nvda_data.empty else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>52W High</h3>
                <h2>${high_52w:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            try:
                low_52w = nvda_data['Low'].rolling(252).min().iloc[-1]
            except:
                low_52w = nvda_data['Low'].min() if not nvda_data.empty else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>52W Low</h3>
                <h2>${low_52w:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Price chart
        st.subheader("📊 Stock Price History")
        
        # Create price chart
        fig = go.Figure()
        
        # Add price line
        chart_data = nvda_data.tail(min(252, len(nvda_data)))
        fig.add_trace(go.Scatter(
            x=chart_data.index,
            y=chart_data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#76B900', width=2)
        ))
        
        # Add moving averages
        for period in [20, 50, 200]:
            if len(nvda_data) > period:
                ma = nvda_data['Close'].rolling(period).mean()
                ma_data = ma.tail(min(252, len(ma)))
                fig.add_trace(go.Scatter(
                    x=ma_data.index,
                    y=ma_data,
                    mode='lines',
                    name=f'MA{period}',
                    line=dict(width=1),
                    opacity=0.7
                ))
        
        fig.update_layout(
            title="NVIDIA Stock Price - Last 12 Months",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500,
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("🎯 Model Predictions")
        
        if nvda_data.empty or len(predictors) == 0:
            st.error("No data available for predictions")
            return
        
        # Run backtest
        with st.spinner("Running model predictions..."):
            predictions_df = backtest_model(nvda_data, model, predictors)
        
        if not predictions_df.empty:
            # Calculate metrics
            precision = precision_score(predictions_df['Target'], predictions_df['Prediction'])
            accuracy = accuracy_score(predictions_df['Target'], predictions_df['Prediction'])
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Model Precision</h3>
                    <h2>{precision:.1%}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Model Accuracy</h3>
                    <h2>{accuracy:.1%}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                total_predictions = len(predictions_df[predictions_df['Prediction'] == 1])
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Buy Signals</h3>
                    <h2>{total_predictions}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Prediction timeline
            st.subheader("📈 Prediction Timeline")
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Stock Price & Predictions', 'Prediction Confidence'),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )
            
            # Price chart with predictions
            fig.add_trace(
                go.Scatter(
                    x=predictions_df.index,
                    y=[nvda_data.loc[idx, 'Close'] for idx in predictions_df.index],
                    mode='lines',
                    name='Stock Price',
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )
            
            # Buy signals
            buy_signals = predictions_df[predictions_df['Prediction'] == 1]
            if not buy_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buy_signals.index,
                        y=[nvda_data.loc[idx, 'Close'] for idx in buy_signals.index],
                        mode='markers',
                        name='Buy Signal',
                        marker=dict(color='green', size=8, symbol='triangle-up')
                    ),
                    row=1, col=1
                )
            
            # Prediction confidence
            fig.add_trace(
                go.Scatter(
                    x=predictions_df.index,
                    y=predictions_df['Probability'],
                    mode='lines',
                    name='Prediction Confidence',
                    line=dict(color='orange', width=1)
                ),
                row=2, col=1
            )
            
            # Add confidence threshold line
            fig.add_hline(
                y=confidence_threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold ({confidence_threshold})",
                row=2, col=1
            )
            
            fig.update_layout(
                height=700,
                template="plotly_dark",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recent predictions
            st.subheader("🔮 Recent Predictions")
            recent_predictions = predictions_df.tail(10).copy()
            recent_predictions['Date'] = recent_predictions.index.strftime('%Y-%m-%d')
            recent_predictions['Confidence'] = recent_predictions['Probability'].apply(lambda x: f"{x:.1%}")
            recent_predictions['Signal'] = recent_predictions['Prediction'].apply(lambda x: "🟢 BUY" if x == 1 else "🔴 HOLD")
            recent_predictions['Actual'] = recent_predictions['Target'].apply(lambda x: "📈 UP" if x == 1 else "📉 DOWN")
            
            st.dataframe(
                recent_predictions[['Date', 'Signal', 'Confidence', 'Actual']],
                use_container_width=True
            )
    
    with tab3:
        st.header("📊 Model Performance Analysis")
        
        if not predictions_df.empty:
            # Performance metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Confusion Matrix")
                
                # Calculate confusion matrix values
                tp = len(predictions_df[(predictions_df['Target'] == 1) & (predictions_df['Prediction'] == 1)])
                tn = len(predictions_df[(predictions_df['Target'] == 0) & (predictions_df['Prediction'] == 0)])
                fp = len(predictions_df[(predictions_df['Target'] == 0) & (predictions_df['Prediction'] == 1)])
                fn = len(predictions_df[(predictions_df['Target'] == 1) & (predictions_df['Prediction'] == 0)])
                
                confusion_matrix = np.array([[tn, fp], [fn, tp]])
                
                fig = px.imshow(
                    confusion_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="Actual"),
                    x=['Hold', 'Buy'],
                    y=['Down', 'Up']
                )
                fig.update_layout(title="Confusion Matrix")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📊 Performance Metrics")
                
                # Calculate additional metrics
                if tp + fp > 0:
                    precision = tp / (tp + fp)
                else:
                    precision = 0
                
                if tp + fn > 0:
                    recall = tp / (tp + fn)
                else:
                    recall = 0
                
                if precision + recall > 0:
                    f1_score = 2 * (precision * recall) / (precision + recall)
                else:
                    f1_score = 0
                
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                
                metrics_df = pd.DataFrame({
                    'Metric': ['Precision', 'Recall', 'F1-Score', 'Accuracy'],
                    'Value': [precision, recall, f1_score, accuracy]
                })
                
                fig = px.bar(
                    metrics_df,
                    x='Metric',
                    y='Value',
                    color='Value',
                    color_continuous_scale="Viridis",
                    title="Model Performance Metrics"
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            st.subheader("🔍 Feature Importance")
            
            # Train model to get feature importance
            train_data = nvda_data.iloc[:-100]
            model.fit(train_data[predictors], train_data['Target'])
            
            importance_df = pd.DataFrame({
                'Feature': predictors,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Feature Importance in Prediction Model",
                color='Importance',
                color_continuous_scale="Plasma"
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("🔍 Technical Analysis")
        
        # Technical indicators
        st.subheader("📊 Technical Indicators")
        
        # Calculate additional technical indicators
        nvda_data['RSI'] = calculate_rsi(nvda_data['Close'])
        nvda_data['MACD'], nvda_data['MACD_Signal'] = calculate_macd(nvda_data['Close'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # RSI Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=nvda_data.index[-252:],
                y=nvda_data['RSI'].iloc[-252:],
                mode='lines',
                name='RSI',
                line=dict(color='purple')
            ))
            fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
            fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
            fig.update_layout(
                title="RSI (Relative Strength Index)",
                yaxis_title="RSI",
                height=400,
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # MACD Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=nvda_data.index[-252:],
                y=nvda_data['MACD'].iloc[-252:],
                mode='lines',
                name='MACD',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=nvda_data.index[-252:],
                y=nvda_data['MACD_Signal'].iloc[-252:],
                mode='lines',
                name='Signal',
                line=dict(color='red')
            ))
            fig.update_layout(
                title="MACD (Moving Average Convergence Divergence)",
                yaxis_title="MACD",
                height=400,
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Volume analysis
        st.subheader("📊 Volume Analysis")
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Price', 'Volume'),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Price
        fig.add_trace(
            go.Scatter(
                x=nvda_data.index[-252:],
                y=nvda_data['Close'].iloc[-252:],
                mode='lines',
                name='Price',
                line=dict(color='#76B900')
            ),
            row=1, col=1
        )
        
        # Volume
        fig.add_trace(
            go.Bar(
                x=nvda_data.index[-252:],
                y=nvda_data['Volume'].iloc[-252:],
                name='Volume',
                marker_color='rgba(118, 185, 0, 0.6)'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            template="plotly_dark",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.header("📋 Data Explorer")
        
        # Data summary
        st.subheader("📊 Dataset Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", len(nvda_data))
        with col2:
            st.metric("Features", len(predictors))
        with col3:
            up_days = (nvda_data['Target'] == 1).sum()
            st.metric("Up Days", f"{up_days} ({up_days/len(nvda_data):.1%})")
        
        # Raw data
        st.subheader("📈 Raw Data")
        
        # Display options
        col1, col2 = st.columns(2)
        with col1:
            show_rows = st.selectbox("Rows to display", [10, 25, 50, 100], index=1)
        with col2:
            data_period = st.selectbox("Period", ["Recent", "All"], index=0)
        
        if data_period == "Recent":
            display_data = nvda_data.tail(show_rows)
        else:
            display_data = nvda_data.head(show_rows)
        
        # Format the data for display
        display_data_formatted = display_data.copy()
        for col in ['Open', 'High', 'Low', 'Close', 'Tomorrow']:
            if col in display_data_formatted.columns:
                display_data_formatted[col] = display_data_formatted[col].apply(lambda x: f"${x:.2f}")
        
        display_data_formatted['Volume'] = display_data_formatted['Volume'].apply(lambda x: f"{x:,.0f}")
        
        st.dataframe(display_data_formatted, use_container_width=True)
        
        # Download data
        st.subheader("💾 Download Data")
        
        csv = nvda_data.to_csv()
        st.download_button(
            label="📥 Download Full Dataset (CSV)",
            data=csv,
            file_name=f"nvidia_stock_data_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

def calculate_rsi(prices, window=14):
    """Calculate RSI (Relative Strength Index)"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    return macd, macd_signal

if __name__ == "__main__":
    main()