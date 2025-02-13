import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

st.set_page_config(page_title="Black-Scholes Pricing Model", layout="wide")

# Title and creator info
st.title("📊 Black-Scholes Model")

# Sidebar input parameters
st.sidebar.header("Input Parameters")
S = st.sidebar.number_input("Current Asset Price", value=100.00, step=0.01, format="%.4f")
K = st.sidebar.number_input("Strike Price", value=100.00, step=0.01, format="%.4f")
T = st.sidebar.number_input("Time to Maturity (Years)", value=1.00, step=0.01, format="%.4f")
sigma = st.sidebar.number_input("Volatility (σ)", value=0.20, step=0.01, format="%.4f")
r = st.sidebar.number_input("Risk-Free Interest Rate", value=0.05, step=0.01, format="%.4f")

# Add "Created by" section in the sidebar
st.sidebar.markdown("Created by:")
st.sidebar.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?style=flat&logo=linkedin&labelColor=blue)](https://linkedin.com)")

def black_scholes(S, K, T, sigma, r, option_type='call'):
    d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return round(option_price, 2)

# Calculate and display option values
call_value = black_scholes(S, K, T, sigma, r, 'call')
put_value = black_scholes(S, K, T, sigma, r, 'put')

col_call, col_put = st.columns(2)
with col_call:
    st.markdown(f"""
    <div style='background-color: rgba(144, 238, 144, 0.3); padding: 20px; border-radius: 10px;'>
        <h4 style='text-align: center;'>CALL Value</h4>
        <h2 style='text-align: center;'>${call_value}</h2>
    </div>
    """, unsafe_allow_html=True)

with col_put:
    st.markdown(f"""
    <div style='background-color: rgba(255, 182, 193, 0.3); padding: 20px; border-radius: 10px;'>
        <h4 style='text-align: center;'>PUT Value</h4>
        <h2 style='text-align: center;'>${put_value}</h2>
    </div>
    """, unsafe_allow_html=True)

# Interactive Heatmaps
st.header("Options Price - Interactive Heatmap")
st.markdown("""
<div style='background-color: rgba(173, 216, 230, 0.3); padding: 20px; border-radius: 10px;'>
    <p style='text-align: center;'>Explore how option prices fluctuate with varying 'Spot Prices and Volatility' levels using interactive heatmap parameters, all while maintaining a constant 'Strike Price'.</p>
</div>
""", unsafe_allow_html=True)

# Generate heatmap data
spot_prices = np.linspace(K*0.5, K*1.5, 20)
volatilities = np.linspace(0.1, 0.5, 20)
X, Y = np.meshgrid(spot_prices, volatilities)

Z_call = np.zeros_like(X)
Z_put = np.zeros_like(X)

for i in range(len(volatilities)):
    for j in range(len(spot_prices)):
        Z_call[i,j] = black_scholes(spot_prices[j], K, T, volatilities[i], r, 'call')
        Z_put[i,j] = black_scholes(spot_prices[j], K, T, volatilities[i], r, 'put')

# Plot heatmaps using Seaborn
col_heat1, col_heat2 = st.columns(2)

with col_heat1:
    st.subheader("Call Price Heatmap")
    fig_call, ax_call = plt.subplots(figsize=(10, 8))
    sns.heatmap(Z_call, 
                xticklabels=[f'{x:.1f}' for x in spot_prices],
                yticklabels=[f'{y:.2f}' for y in volatilities],
                cmap='YlOrRd',
                ax=ax_call,
                cbar_kws={'label': 'Price'})
    ax_call.set_xlabel('Spot Price')
    ax_call.set_ylabel('Volatility')
    ax_call.set_title('Call Option Prices')
    plt.tight_layout()
    st.pyplot(fig_call)

with col_heat2:
    st.subheader("Put Price Heatmap")
    fig_put, ax_put = plt.subplots(figsize=(10, 8))
    sns.heatmap(Z_put, 
                xticklabels=[f'{x:.1f}' for x in spot_prices],
                yticklabels=[f'{y:.2f}' for y in volatilities],
                cmap='YlOrRd',
                ax=ax_put,
                cbar_kws={'label': 'Price'})
    ax_put.set_xlabel('Spot Price')
    ax_put.set_ylabel('Volatility')
    ax_put.set_title('Put Option Prices')
    plt.tight_layout()
    st.pyplot(fig_put)

# Parameter table display
st.header("Current Parameters")
param_df = pd.DataFrame({
    'Parameter': ['Current Asset Price', 'Strike Price', 'Time to Maturity', 'Volatility', 'Risk-Free Rate'],
    'Value': [S, K, T, sigma, r]
})
st.table(param_df)