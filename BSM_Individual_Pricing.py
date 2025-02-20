import streamlit as st
import numpy as np
from scipy.stats import norm
import streamlit as st

def bsm_model(S, K, r, t, sigma):
    if t == 0:
        return max(S - K, 0), max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
    put_price = K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return call_price, put_price

st.title("Black - Sholes & Merton Option Pricing Calculator")

col1, col2 = st.columns(2)

with col1:
    st.header("Input Parameters")
    S = st.number_input("Current Stock Price: ", min_value=float(0), step=0.01)
    K = st.number_input("Strike Price: ", min_value=float(0), step=0.01)
    r = st.number_input("Risk-Free Rate(%): ", min_value=float(0), max_value=float(100), step=0.01)
    t = st.number_input("Time Until Maturity (years): ", min_value=float(0), step=0.01)
    sigma = st.number_input("Volatility: ", min_value=float(0), step=0.01)

r_decimal = r / 100
call_price, put_price = bsm_model(S, K, r, t, sigma)
rounded_call_price = round(call_price, 2)
rounded_put_price = round(put_price, 2)

with col2:
    st.header("Output")
    st.write("### Option Prices")
    st.write("Call Price: ", rounded_call_price)
    st.write("Put Price: ", rounded_put_price)