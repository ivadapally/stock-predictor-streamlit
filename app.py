import streamlit as st
from Stock_Price_Prediction_LR import predict_stock_price

st.title("📈 Stock Price Prediction (Linear Regression)")

uploaded_file = st.file_uploader("Upload your stock CSV", type="csv")

if uploaded_file:
    mse, r2 = predict_stock_price(uploaded_file)
    st.success(f"MSE: {round(mse, 4)} | R² Score: {round(r2, 4)}")
    st.image("static/plot.png", caption="Actual vs Predicted Plot")
