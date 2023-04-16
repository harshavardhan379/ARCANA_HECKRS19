import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import plotly.express as px
import plotly.graph_objs as go

import time

base="dark"
primaryColor="purple"
st.title('STOCK PREDICTION')
xt = []
yt = []
close = []
tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN','A','AA','AACG','AAON','AAPL','AAT','AAWW']
input_text = st.multiselect('PICK YOUR ASSESTS', tech_list)
trash = input_text
for i in range(len(trash)):
    input_text = trash[i]
    st.write('The current' + trash[i] + ' stocks is')
    sns.set_style('whitegrid')
    plt.style.use("fivethirtyeight")

    # For reading stock data from yahoo
    from pandas_datareader.data import DataReader
    import yfinance as yf
    from pandas_datareader import data as pdr

    yf.pdr_override()

    # For time stamps
    from datetime import datetime
    end = datetime.now()
    start = datetime(end.year - 10, end.month, end.day)
    # Get the stock quote
    @st.cache_data
    def get_data(input_text):
        df = pdr.get_data_yahoo(input_text, start='2010-01-01', end=datetime.now())
        return df

    df=get_data(input_text)
    df['Daily Return'] = df['Adj Close'].pct_change()
    col1, col2, col3 = st.columns(3)

    col1.metric("Close", df['Close'][-1])
    col2.metric("Volume", df["Volume"][-1])
    col3.metric("Open", df["Open"][-1])
    
    pro = st.progress(0)
    for i in range(100):
       pro.progress(i+1)
       time.sleep(0.01)
    # col1.metric("close", df['Close'][-1])
    # col2.metric("Volume", df["Volume"][-1])
    # col3.metric("Open",df["Open"][-1])

    # Grab all the closing prices for the tech stock list into one DataFrame

    closing_df = pdr.get_data_yahoo(input_text, start=start, end=end)['Adj Close']

    # Make a new tech returns DataFrame
    tech_rets = closing_df.pct_change()
    rets = tech_rets.dropna()

    area = np.pi * 20

    #for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
        #plt.annotate(label,xy=(x, y),xytext=(50, 50),textcoords='offset points',ha='right',va='bottom',arrowprops=dict(arrowstyle='-',color='blue',connectionstyle='arc3,rad=-0.3'))
    close.append(df['Close'])
    # Create a new dataframe with only the 'Close column 
    data = df.filter(['Close'])
    # Convert the dataframe to a numpy array
    dataset = data.values
    # Get the number of rows to train the model on
    training_data_len = int(np.ceil( len(dataset) * .95 ))

    # Scale the data
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    # Create the training data set 
    # Create the scaled training data set
    train_data = scaled_data[0:int(training_data_len), :]
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
        if i<= 61:
            print(x_train)
            print(y_train)
            print()
            
    # Convert the x_train and y_train to numpy arrays 
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Create the testing data set
    # Create a new array containing scaled values from index 1543 to 2002 
    test_data = scaled_data[training_data_len - 60: , :]
    # Create the data sets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
        
    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

    model=load_model('LSTM3.h5')


    # Get the models predicted price values 
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    xt.append(predictions)
    yt.append(y_test)
    # Get the root mean squared error (RMSE)
    # rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    # rmse

    # Plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    # Visualize the data
    # fig2=plt.figure(figsize=(16,6))
    # plt.title('Model')
    # plt.xlabel('Date', fontsize=18)
    # plt.ylabel('Close Price USD ($)', fontsize=18)
    # plt.plot(train['Close'])
    # plt.plot(valid[['Close', 'Predictions']])
    # plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    # st.pyplot(fig2)
    # stocks = np.array(train['Close'])
    # st.header("GDP per Capita over time")
    # fig = px.line(stocks)

    df1 = {'train':train['Close'], 'test':valid['Close'], 'pred':valid['Predictions']}
    fig = px.line(df1, y=['train', 'test','pred'], title='Cose Price Prediction')
    st.plotly_chart(fig)
    #x = 'Date', y = 'Close Price USD ($)', title = 'Model')
    num = valid['Predictions'][-1]
    st.metric(label="Current Day Predicted",value= num)
    

l = []


   
for i in range(len(trash)):
    l.append(trash[i])

class my_dictionary(dict):
 
  # _init_ function
  def _init_(self):
    self = dict()
 
  # Function to add key:value
  def add(self, key, value):
    self[key] = value
df2 = {}
# fig=plt.figure(figsize=(16,6))
# plt.title('Close Price History')
df2 = my_dictionary()
for i in range(len(trash)):
    # plt.plot(close[i])
    df2.add(trash[i], close[i])
    
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Close Price USD ($)', fontsize=18)
# plt.legend(l, loc='lower right')
list_of_the_keys = list(df2.keys())
print(list_of_the_keys)
fig2 = px.line(df2, y=list(df2.keys()), title='Comparision Plots')
st.plotly_chart(fig2)