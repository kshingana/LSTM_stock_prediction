import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

start = "2010-01-01"
end = "2021-11-01"

st.title('stock Trend Prediction')
user_input = st.text_input("Enter Stock Ticker as per YahooFin", "AAPL")
df = data.DataReader("user_input","yahoo",start,end)

st.subheader('Data from 2010 - 2021 Oct')
st.write(df.describe())

st.subheader("closing Price Vs Time Chart")
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader("closing Price Vs Time Chart with 100MA")
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader("Moving Average Crossover Strategy")
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100 ,'g')
plt.plot(ma200,'r')
plt.plot(df.Close,'b')
st.pyplot(fig)

#splitting data into trainig & testing
data_training = pd.DataFrame(df["Close"][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df["Close"][int(len(df)*0.70):int(len(df))])
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)
data_testing_array = scaler.fit_transform(data_testing)


#load model

model = load_model("keras_model.h5")
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing,ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)
my_scaler = scaler.scale_

y_predicted = y_predicted*my_scaler[0]
y_test = y_test*my_scaler[0]

#Final Prediction Graph

st.subheader("LSTM Predictions")
fig2 = plt.figure(figsize = (12,6))
plt.plot(y_test,'b',label='original price')
plt.plot(y_predicted,'r',label='predicted price')
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig2)
