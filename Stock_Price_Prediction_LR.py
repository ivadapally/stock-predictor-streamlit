import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def predict_stock_price(csv_file):
    stock_df = pd.read_csv(csv_file)
    stock_df['Datetime'] = pd.to_datetime(stock_df['Datetime'])

    stock_df['Hour'] = stock_df['Datetime'].dt.hour
    stock_df['Minute'] = stock_df['Datetime'].dt.minute
    stock_df['Day'] = stock_df['Datetime'].dt.day
    stock_df['DayOfWeek'] = stock_df['Datetime'].dt.dayofweek

    y = stock_df['Close']
    X = stock_df[['Hour', 'Minute', 'Day', 'DayOfWeek']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    os.makedirs('static', exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label='Actual', marker='o')
    plt.plot(y_pred, label='Predicted', marker='x')
    plt.legend()
    plt.title("Linear Regression: Actual vs Predicted")
    plt.savefig('static/plot.png')
    plt.close()

    return mse, r2
