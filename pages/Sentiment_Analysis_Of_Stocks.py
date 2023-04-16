import streamlit as st
import yfinance as yf
import pandas as pd
from textblob import TextBlob
import time


st.sidebar.markdown("Sentiment Analysis of Stock")

st.title("sentiment Analysis of Stock")

# Define function to get stock data

def load_stock_data(ticker):
    stock_data = yf.Ticker(ticker)
    return stock_data



# Define function to display stock news
def stock_news(ticker):
    stock_data = load_stock_data(ticker)
    news = pd.DataFrame(stock_data.news)
    #news['Date'] = news['Date'].astype(str)
    news = news[['title', 'publisher', 'link', 'providerPublishTime']]
    st.header(f'{ticker} Stock News')
    for _, row in news.iterrows():
        st.divider()
        st.markdown(f"- [{row['title']}]({row['link']})")
        st.markdown("- "  + row['publisher'])
        sentiment = TextBlob(row['title']).sentiment.polarity
        st.write(f"Sentiment Score:")
        st.write(sentiment)

# Define app layout
st.subheader('choose stock ticker')
ticker = st.text_input('Enter a ticker symbol:', 'AAPL')
stock_news(ticker)