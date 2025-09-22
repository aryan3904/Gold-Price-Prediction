import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Gold Price Prediction",
    page_icon="💰",
    layout="wide"
)

# Title
st.title("Gold Price Prediction App 💰")
st.write("This app predicts the price of Gold based on various market indicators.")

# Sidebar
st.sidebar.header("Input Parameters")

def get_user_input():
    SPX = st.sidebar.number_input("SPX (S&P 500)", value=2000.0)
    USO = st.sidebar.number_input("USO (United States Oil Fund)", value=30.0)
    SLV = st.sidebar.number_input("SLV (Silver Trust)", value=15.0)
    EUR_USD = st.sidebar.number_input("EUR/USD Exchange Rate", value=1.2)
    
    # Create a dictionary with user inputs
    user_data = {
        'SPX': SPX,
        'USO': USO,
        'SLV': SLV,
        'EUR/USD': EUR_USD
    }
    
    return pd.DataFrame(user_data, index=[0])

# Get user input
user_input = get_user_input()

# Load the model and make prediction
try:
    with open('gold_price_model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # Make prediction
    if st.sidebar.button("Predict Gold Price"):
        prediction = model.predict(user_input)
        
        # Display prediction
        st.subheader("Prediction Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Predicted Gold Price", f"${prediction[0]:.2f}")
        
        with col2:
            st.write("Input Parameters:")
            st.write(user_input)
        
        # Add some context
        st.info("""
        Note: This prediction is based on historical data and market indicators.
        Actual gold prices may vary due to numerous external factors.
        """)

except FileNotFoundError:
    st.error("""
    Model file not found! Please make sure to train and save the model first.
    Run the notebook to generate the model file.
    """)

# Add information about the project
st.markdown("""
## About This Project
This gold price prediction model uses a Random Forest Regressor trained on historical data including:
- S&P 500 (SPX)
- United States Oil Fund (USO)
- Silver Trust (SLV)
- EUR/USD Exchange Rate

The model considers these market indicators to predict the price of gold.

### How to Use
1. Adjust the input parameters in the sidebar
2. Click the "Predict Gold Price" button
3. View the prediction and analysis

### Model Performance
The model has been trained on historical data and evaluated using:
- R-squared Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
""")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")