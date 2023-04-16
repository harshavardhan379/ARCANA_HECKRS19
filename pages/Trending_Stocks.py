import streamlit as st
import time


st.sidebar.markdown("Trending Stock")

st.title("Trending Stock")


tab1, tab2, tab3 = st.tabs(["Top Gainers", "Top Losers", "most-active"])
import pandas as pd
gainers = pd.read_html('https://finance.yahoo.com/gainers')
gainers =pd.DataFrame(gainers[0])
losers = pd.read_html('https://finance.yahoo.com/losers')
losers =pd.DataFrame(losers[0])
mostatv = pd.read_html('https://finance.yahoo.com/most-active')
mostatv =pd.DataFrame(mostatv[0])


with tab1:
   st.header("Top Gainers")
   for i in range(len(gainers)):
       with st.expander(gainers['Name'][i]):
           st.dataframe(gainers.iloc[i].to_frame())
       col1, col2, col3 = st.columns(3)
       col1.metric("Price", gainers['Price (Intraday)'][i], gainers["% Change"][i])
       col2.metric("Volume", gainers['Volume'][i], gainers['Avg Vol (3 month)'][i])
       col3.metric("Market Cap", gainers['Market Cap'][i], gainers['Market Cap'][i])

with tab2:
   st.header("Top Losers")
   for i in range(len(losers)):
       with st.expander(losers['Name'][i]):
           st.dataframe(losers.iloc[i].to_frame())
       col1, col2, col3= st.columns(3)
       col1.metric("Price", losers['Price (Intraday)'][i], losers["% Change"][i])
       col2.metric("Volume", losers['Volume'][i], losers['Avg Vol (3 month)'][i])
       col3.metric("Market Cap", losers['Market Cap'][i], losers['Market Cap'][i])

with tab3:
   st.header("Most Active")
   for i in range(len(mostatv)):
       with st.expander(mostatv['Name'][i]):
           st.dataframe(mostatv.iloc[i].to_frame())
       col1, col2, col3 = st.columns(3)
       col1.metric("Price", mostatv['Price (Intraday)'][i], mostatv["% Change"][i])
       col2.metric("Price", mostatv['Volume'][i], mostatv['Avg Vol (3 month)'][i])
       col3.metric("Market Cap", mostatv['Market Cap'][i], mostatv['Market Cap'][i])