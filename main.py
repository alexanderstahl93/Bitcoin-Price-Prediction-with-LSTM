# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# Load and preprocess data
# Load data
df = pd.read_csv('data\BTC-USD.csv') # Replace with your filename

# Select only the "Close" column
data = df.filter(['Close'])
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)


# Convert the dataframe to a numpy array
dataset = data.values

# Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# Split the data into training set and test set
train_ratio = 0.8
training_data_len = int(np.round(train_ratio * len(dataset)))

# Create sequences of data points
# Define the lookback period and split into samples
lookback = 60
x_train, y_train = [], []

for i in range(lookback, len(scaled_data)):
    x_train.append(scaled_data[i-lookback:i, 0])
    y_train.append(scaled_data[i, 0])
    
# Convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data to 3-D so it's suitable for LSTM model
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build / Define the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Test model
# Create the testing data set
test_data = scaled_data[training_data_len - lookback: , :]

# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(lookback, len(test_data)):
    x_test.append(test_data[i-lookback:i, 0])
    
# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

# Get the models predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Evaluate model
# Calculate RMSE
rmse = np.sqrt(np.mean(predictions - y_test)**2)
print(f'RMSE: {rmse}')

# Plot the data
train = df[:training_data_len]
valid = df[training_data_len:]
valid['Predictions'] = predictions

# Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train.index, train['Close'])
plt.plot(valid.index, valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()