"""
Utility functions for the NVIDIA stock prediction model
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score, classification_report
import yfinance as yf

def load_and_prepare_data(symbol="NVDA", start_date="2000-01-01"):
    """
    Load and prepare stock data with technical indicators
    
    Args:
        symbol (str): Stock symbol
        start_date (str): Start date for data
    
    Returns:
        tuple: (prepared_data, predictor_columns)
    """
    # Load data
    ticker = yf.Ticker(symbol)
    data = ticker.history(period="max")
    
    # Create target variable
    data["Tomorrow"] = data["Close"].shift(-1)
    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
    
    # Filter data from start_date onwards
    data = data.loc[start_date:].copy()
    
    # Add technical indicators
    horizons = [2, 5, 60, 250, 1000]
    new_predictors = []
    
    for horizon in horizons:
        rolling_averages = data.rolling(horizon).mean()
        
        # Price ratio to moving average
        ratio_column = f"Close_Ratio_{horizon}"
        data[ratio_column] = data["Close"] / rolling_averages["Close"]
        
        # Trend indicator (sum of targets over horizon)
        trend_column = f"Trend_{horizon}"
        data[trend_column] = data.shift(1).rolling(horizon).sum()["Target"]
        
        new_predictors += [ratio_column, trend_column]
    
    # Remove rows with NaN values
    data = data.dropna()
    
    return data, new_predictors

def create_prediction_model(n_estimators=200, min_samples_split=50, random_state=1):
    """
    Create a Random Forest model for stock prediction
    
    Args:
        n_estimators (int): Number of trees in the forest
        min_samples_split (int): Minimum samples required to split a node
        random_state (int): Random state for reproducibility
    
    Returns:
        RandomForestClassifier: Configured model
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        random_state=random_state
    )

def predict_with_probability(train_data, test_data, predictors, model, threshold=0.6):
    """
    Make predictions with probability threshold
    
    Args:
        train_data (pd.DataFrame): Training data
        test_data (pd.DataFrame): Test data
        predictors (list): List of predictor column names
        model: Trained model
        threshold (float): Probability threshold for positive prediction
    
    Returns:
        tuple: (probabilities, binary_predictions)
    """
    # Train the model
    model.fit(train_data[predictors], train_data["Target"])
    
    # Get probabilities
    probabilities = model.predict_proba(test_data[predictors])[:, 1]
    
    # Convert to binary predictions based on threshold
    binary_predictions = (probabilities >= threshold).astype(int)
    
    return probabilities, binary_predictions

def backtest_strategy(data, model, predictors, start=2500, step=250, threshold=0.6):
    """
    Perform backtesting on the prediction strategy
    
    Args:
        data (pd.DataFrame): Stock data
        model: Machine learning model
        predictors (list): List of predictor columns
        start (int): Starting index for backtesting
        step (int): Step size for rolling window
        threshold (float): Probability threshold
    
    Returns:
        pd.DataFrame: Backtest results with predictions and probabilities
    """
    all_predictions = []
    
    for i in range(start, data.shape[0], step):
        # Define train and test sets
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        
        if len(test) == 0:
            break
        
        # Make predictions
        probabilities, predictions = predict_with_probability(
            train, test, predictors, model, threshold
        )
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'Target': test['Target'],
            'Prediction': predictions,
            'Probability': probabilities,
            'Close': test['Close'],
            'Date': test.index
        }, index=test.index)
        
        all_predictions.append(results_df)
    
    return pd.concat(all_predictions) if all_predictions else pd.DataFrame()

def calculate_performance_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate comprehensive performance metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional)
    
    Returns:
        dict: Dictionary of performance metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix components
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    metrics['true_positives'] = tp
    metrics['true_negatives'] = tn
    metrics['false_positives'] = fp
    metrics['false_negatives'] = fn
    
    # Additional metrics
    if tp + fn > 0:
        metrics['recall'] = tp / (tp + fn)
    else:
        metrics['recall'] = 0
    
    if metrics['precision'] + metrics['recall'] > 0:
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
    else:
        metrics['f1_score'] = 0
    
    return metrics

def calculate_technical_indicators(data):
    """
    Calculate additional technical indicators
    
    Args:
        data (pd.DataFrame): Stock data with OHLCV columns
    
    Returns:
        pd.DataFrame: Data with additional technical indicators
    """
    data = data.copy()
    
    # RSI (Relative Strength Index)
    data['RSI'] = calculate_rsi(data['Close'])
    
    # MACD (Moving Average Convergence Divergence)
    data['MACD'], data['MACD_Signal'] = calculate_macd(data['Close'])
    
    # Bollinger Bands
    data['BB_Upper'], data['BB_Lower'], data['BB_Middle'] = calculate_bollinger_bands(data['Close'])
    
    # Moving Averages
    for period in [20, 50, 200]:
        data[f'MA_{period}'] = data['Close'].rolling(period).mean()
    
    # Volume indicators
    data['Volume_MA'] = data['Volume'].rolling(20).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
    
    return data

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

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    rolling_mean = prices.rolling(window).mean()
    rolling_std = prices.rolling(window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band, rolling_mean

def generate_trading_signals(predictions_df, confidence_threshold=0.6):
    """
    Generate trading signals based on predictions
    
    Args:
        predictions_df (pd.DataFrame): DataFrame with predictions and probabilities
        confidence_threshold (float): Minimum confidence for signal generation
    
    Returns:
        pd.DataFrame: DataFrame with trading signals
    """
    signals = predictions_df.copy()
    
    # Generate signals
    signals['Signal'] = 'HOLD'
    signals.loc[
        (signals['Prediction'] == 1) & (signals['Probability'] >= confidence_threshold),
        'Signal'
    ] = 'BUY'
    
    # Calculate signal strength
    signals['Signal_Strength'] = signals['Probability']
    
    # Add signal description
    signals['Signal_Description'] = signals.apply(
        lambda row: f"{'Strong' if row['Probability'] > 0.8 else 'Moderate'} {row['Signal']} signal",
        axis=1
    )
    
    return signals

def calculate_strategy_returns(signals_df, initial_capital=10000):
    """
    Calculate returns from trading strategy
    
    Args:
        signals_df (pd.DataFrame): DataFrame with trading signals
        initial_capital (float): Initial capital for trading
    
    Returns:
        pd.DataFrame: DataFrame with strategy performance
    """
    strategy = signals_df.copy()
    strategy['Position'] = 0
    strategy['Holdings'] = 0
    strategy['Cash'] = initial_capital
    strategy['Total_Value'] = initial_capital
    
    cash = initial_capital
    holdings = 0
    
    for i, row in strategy.iterrows():
        if row['Signal'] == 'BUY' and cash > 0:
            # Buy signal - invest all cash
            shares_to_buy = cash / row['Close']
            holdings += shares_to_buy
            cash = 0
        elif row['Signal'] == 'SELL' and holdings > 0:
            # Sell signal - sell all holdings
            cash += holdings * row['Close']
            holdings = 0
        
        strategy.loc[i, 'Position'] = 1 if holdings > 0 else 0
        strategy.loc[i, 'Holdings'] = holdings
        strategy.loc[i, 'Cash'] = cash
        strategy.loc[i, 'Total_Value'] = cash + (holdings * row['Close'])
    
    # Calculate returns
    strategy['Strategy_Return'] = strategy['Total_Value'].pct_change()
    strategy['Cumulative_Return'] = (strategy['Total_Value'] / initial_capital - 1) * 100
    
    # Buy and hold benchmark
    strategy['BuyHold_Value'] = (strategy['Close'] / strategy['Close'].iloc[0]) * initial_capital
    strategy['BuyHold_Return'] = (strategy['BuyHold_Value'] / initial_capital - 1) * 100
    
    return strategy