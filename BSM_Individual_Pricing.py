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
    S = st.number_input("Current Stock Price: ", min_value=float(0), step=0.01, key="S_input")
    K = st.number_input("Strike Price: ", min_value=float(0), step=0.01, key="K_input")
    r = st.number_input("Risk-Free Rate(%): ", min_value=float(0), max_value=float(100), value=1.5, step=0.01, key="r_input")
    t = st.number_input("Time Until Maturity (years): ", min_value=float(0), value=1,0, step=0.01, key="t_input")
    sigma = st.number_input("Volatility: ", min_value=float(0), value=0.2,  step=0.01, key="V_input")

r_decimal = r / 100
call_price, put_price = bsm_model(S, K, r, t, sigma)

with col2:
    st.header("Output")
    st.write("### Option Prices")
    st.write("Call Price: ", round(call_price, 2))
    st.write("Put Price: ", round(put_price, 2))