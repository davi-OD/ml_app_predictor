import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
# import pandas_datareader.tsp as tsp
from pandas_datareader import data as pdr
from datetime import datetime, timedelta
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

st.title("Prediction")

user_input = st.text_input('Enter Company', 'AAPL')

start = st.date_input("Start")
st.markdown(
    """
    <style>
  div[class="stDateInput"] div[class="st-b8"] input {
        color: white;
        }
    div[role="presentation"] div{
    color: white;
    }

    div[class="st-b3 st-d0"] button {
        color:white
        };
        </style>
""",
    unsafe_allow_html=True,
)

# start = "2010-01-01"

# end = "2022-04-30"

end = st.date_input("End")
st.markdown(
    """
    <style>
  div[class="stDateInput"] div[class="st-b8"] input {
        color: white;
        }
    div[role="presentation"] div{
    color: white;
    }

    div[class="st-b3 st-d0"] button {
        color:white
        };
        </style>
""",
    unsafe_allow_html=True,
)

yf.pdr_override()
df = pdr.get_data_yahoo(user_input, start, end)
df.head()

st.subheader('Data from date selected')
st.write(df.describe())


st.subheader('Close Price ')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Close Price 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Close Price 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])

# 30
data_test = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

# Sklearn
scaler = MinMaxScaler(feature_range=(0, 1))

data_train_array = scaler.fit_transform(data_train)

# Splitting data into x and y train
# x_train = []
# y_train = []

# for i in range(100, data_train_array.shape[0]):
#     x_train.append(data_train_array[i-100: i])
#     y_train.append(data_train_array[i, 0])

# x_train, y_train = np.array(x_train), np.array(y_train)


# Loan model
model = load_model('keras_model.h5')

# Testing
past_100_days = data_test.tail(100)

final_df = pd.concat([past_100_days, data_test])

input_data = scaler.fit_transform(final_df)


x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])


x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


st.subheader('Predictions Vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
