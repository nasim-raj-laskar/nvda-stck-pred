import sys
sys.path.append('.')
from app_working import create_nvidia_sample_data

print('Testing sample data generation...')
data, predictors = create_nvidia_sample_data()
print(f'✅ Generated {len(data)} records')
print(f'📊 Features: {len(predictors)}')
print(f'📅 Date range: {data.index[0].date()} to {data.index[-1].date()}')
print(f'💰 Price range: ${data["Close"].min():.2f} - ${data["Close"].max():.2f}')
print(f'📈 Latest price: ${data["Close"].iloc[-1]:.2f}')
print('✅ Sample data generation successful!')