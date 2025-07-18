import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, timedelta

# Set up the app
st.set_page_config(page_title="Gold Price Forecaster", layout="wide")
st.title("💰 Gold Price Forecasting App")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    model_choice = st.radio("Select Model", ["ARIMA", "SARIMA"])
    weeks_to_forecast = st.slider("Weeks to Forecast", 4, 26, 12)
    show_confidence = st.checkbox("Show Confidence Interval", True)
    st.markdown("---")
    st.markdown("*Models trained on weekly gold price data*")

# Load data function
@st.cache_data
def load_gold_data():
    # Create proper DataFrame with datetime index
    return pd.DataFrame({'Price': [1845.50, 1832.30, 1820.10]}, 
                       index=pd.to_datetime(['2025-06-16', '2025-06-23', '2025-06-30']))

# Function to calculate confidence intervals for ARIMA
def get_arima_confidence(model, steps):
    forecast = model.get_forecast(steps=steps)
    if hasattr(forecast, 'conf_int'):
        return forecast.conf_int()
    else:
        # Manually calculate approximate CI for ARIMA
        pred_mean = forecast.predicted_mean
        stdev = np.sqrt(model.params[-1])  # Using the variance parameter
        lower = pred_mean - 1.96 * stdev
        upper = pred_mean + 1.96 * stdev
        return np.column_stack((lower, upper))

# Model loading function
def load_model(model_type, gold_data):
    try:
        model_data = load(f"{model_type.lower()}_gold_weekly_optimized.joblib")
        
        # Convert to proper 2D array for statsmodels
        endog = gold_data[['Price']].values
        
        if model_type == "ARIMA":
            model = ARIMA(endog, order=model_data['order'])
        else:
            model = SARIMAX(endog, 
                          order=model_data['order'],
                          seasonal_order=model_data['seasonal_order'])
        
        model_fit = model.fit()
        model_fit.params = model_data['params']
        return model_fit, model_data['model_summary']
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Main app function
def main():
    gold_data = load_gold_data()
    
    # Display last observed price
    st.subheader(f"Last Observed Price: ${gold_data['Price'].iloc[-1]:.2f}")
    st.caption(f"Date: {gold_data.index[-1].strftime('%Y-%m-%d')}")
    
    # Load model
    model, model_summary = load_model(model_choice, gold_data)
    
    # Generate forecast
    forecast_dates = pd.date_range(start=gold_data.index[-1] + timedelta(weeks=1),
                                 periods=weeks_to_forecast,
                                 freq='W')
    
    # Get predictions and confidence intervals
    if model_choice == "ARIMA":
        pred_mean = model.forecast(steps=weeks_to_forecast)
        if show_confidence:
            conf_int = get_arima_confidence(model, weeks_to_forecast)
    else:  # SARIMA
        forecast = model.get_forecast(steps=weeks_to_forecast)
        pred_mean = forecast.predicted_mean
        if show_confidence:
            conf_int = forecast.conf_int()
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecast': pred_mean
    })
    
    if show_confidence:
        forecast_df['Lower'] = conf_int[:, 0]
        forecast_df['Upper'] = conf_int[:, 1]
    
    # Display results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"{weeks_to_forecast}-Week Forecast ({model_choice})")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical data
        ax.plot(gold_data.index, gold_data['Price'], 
                label='Historical', marker='o', markersize=6, 
                color='#1f77b4', linestyle='-', linewidth=2)
        
        # Plot forecast
        forecast_color = '#d62728' if model_choice == "ARIMA" else '#2ca02c'
        ax.plot(forecast_df['Date'], forecast_df['Forecast'], 
                label=f'{model_choice} Forecast', 
                color=forecast_color, 
                linestyle='--', marker='o', markersize=5, linewidth=2)
        
        # Plot confidence interval if available
        if show_confidence and 'Lower' in forecast_df.columns:
            ci_color = '#ff9896' if model_choice == "ARIMA" else '#98df8a'
            ax.fill_between(forecast_df['Date'],
                          forecast_df['Lower'],
                          forecast_df['Upper'],
                          color=ci_color, alpha=0.3,
                          label='95% Confidence')
        
        # Customize plot
        ax.set_title(f"Gold Price Forecast", fontsize=16, pad=20)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Price (USD)", fontsize=12)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Forecast Values")
        display_df = forecast_df.set_index('Date')
        if show_confidence and 'Lower' in display_df.columns:
            display_df = display_df[['Forecast', 'Lower', 'Upper']]
            st.dataframe(display_df.style.format("{:.2f}"), height=400)
        else:
            st.dataframe(display_df[['Forecast']].style.format("{:.2f}"), height=400)
        
        with st.expander("Model Summary"):
            st.text(model_summary)

if __name__ == "__main__":
    main()
