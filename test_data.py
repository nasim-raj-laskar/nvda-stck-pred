import yfinance as yf
import pandas as pd
import time

print('Testing different approaches to load NVDA data...')

approaches = [
    {'period': '1y'},
    {'period': '6mo'}, 
    {'period': '3mo'},
    {'start': '2023-01-01', 'end': '2024-12-31'},
    {'start': '2024-01-01', 'end': '2024-12-31'},
]

for i, params in enumerate(approaches):
    try:
        print(f'Approach {i+1}: {params}')
        ticker = yf.Ticker('NVDA')
        
        if 'period' in params:
            data = ticker.history(period=params['period'])
        else:
            data = ticker.history(start=params['start'], end=params['end'])
        
        if not data.empty:
            print(f'  ✅ Success: {len(data)} records from {data.index[0].date()} to {data.index[-1].date()}')
            print(f'  Latest price: ${data["Close"].iloc[-1]:.2f}')
            break
        else:
            print('  ❌ Empty data')
            
    except Exception as e:
        print(f'  ❌ Error: {e}')
    
    time.sleep(0.5)