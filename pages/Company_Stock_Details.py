import yfinance as yf
import streamlit as st
import numpy as np
import time

st.sidebar.markdown("Stock Details")

st.title("Stock Details")

def load_stock_data(ticker):
    return yf.Ticker(ticker)

def stock_info(ticker):
    stock_data = load_stock_data(ticker)
    info = stock_data.info

    st.subheader(f"{ticker} Stock Info")
    st.write(f"**Sector:** {info['sector']}")
    st.write(f"**Industry:** {info['industry']}")
    st.write(f"**Country:** {info['country']}")
    st.write(f"**Website:** {info['website']}")
    st.write(f"**Market Cap:** ${info['marketCap']:.2f}")
    st.write(f"**Forward P/E:** {info['forwardPE']:.2f}")
    st.write(f"**Beta:** {info['beta']:.2f}")
    st.write(f"**Dividend Yield:** {info['dividendYield']*100:.2f}%" if info['dividendYield'] is not None else "**Dividend Yield:** N/A")

def stock_history(ticker):
    stock_data = load_stock_data(ticker)
    period = st.selectbox('Select a time period:', ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'))
    interval = '1d' if period in ('1d', '5d') else '1wk'
    history = stock_data.history(period=period, interval=interval)
    st.subheader(f'{ticker} Stock History')
    st.line_chart(history.Close)

def stock_recommendations(ticker):
    stock_data = load_stock_data(ticker)
    recommendations = stock_data.recommendations

    st.subheader(f"{ticker} Stock Recommendations")
    st.write(recommendations)


def capital_allocation(ticker):
    stock_data = load_stock_data(ticker)
    info = stock_data.info
    market_cap = info['marketCap']
    debt_to_equity = info['debtToEquity'] if 'debtToEquity' in info else np.nan
    enterprise_value = info['enterpriseValue'] if 'enterpriseValue' in info else np.nan
    total_cash = info['totalCash']
    total_debt = info['totalDebt'] if 'totalDebt' in info else np.nan
    equity_value = market_cap + total_debt - total_cash
    debt_percentage = 100 * total_debt / equity_value if not np.isnan(total_debt) else np.nan
    cash_percentage = 100 * total_cash / equity_value
    equity_percentage = 100 - debt_percentage - cash_percentage

    st.subheader(f"{ticker} Capital Allocation")
    st.write(f"**Market Cap:** ${market_cap:.2f}")
    st.write(f"**Total Debt:** ${total_debt:.2f}" if not np.isnan(total_debt) else "**Total Debt:** N/A")
    st.write(f"**Total Cash:** ${total_cash:.2f}")
    st.write(f"**Enterprise Value:** ${enterprise_value:.2f}" if not np.isnan(enterprise_value) else "**Enterprise Value:** N/A")
    st.write(f"**Debt to Equity Ratio:** {debt_to_equity:.2f}" if not np.isnan(debt_to_equity) else "**Debt to Equity Ratio:** N/A")
    st.write(f"**Debt Percentage:** {debt_percentage:.2f}%" if not np.isnan(debt_percentage) else "**Debt Percentage:** N/A")
    
menu = ['Stock Info', 'Stock History', 'Stock Recommendations','Capital Allocation']

st.subheader('choose stock ticker')
ticker = st.text_input('Enter a stock ticker (e.g. AAPL):', 'AAPL')

st.subheader('choose an option')
choice = st.selectbox('Select an option:', menu)
if choice == 'Stock Info':
    stock_info(ticker)
elif choice == 'Stock History':
    stock_history(ticker)
elif choice == 'Stock Recommendations':
    stock_recommendations(ticker)
elif choice == 'Capital Allocation':
    capital_allocation(ticker)