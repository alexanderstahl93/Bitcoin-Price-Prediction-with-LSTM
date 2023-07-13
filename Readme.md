# Bitcoin Price Prediction with LSTM

This project uses a Long Short-Term Memory (LSTM) model, a type of recurrent neural network, to predict the future price of Bitcoin.

## Setup
* Clone the repository to your local machine.
* Install the required packages using pip:
pip install pandas numpy sklearn tensorflow matplotlib

## Data
The model uses daily closing price data for Bitcoin. This data is stored in the file 'BTC-USD.csv'. <br> Each row in the file represents one day, and the 'Close' column is used for training and prediction.

## Usage
To run the model, simply execute the main.py file in the root directory of the project: python main.py <br>
The model will train on the data, make predictions, and then display a plot of the actual vs. predicted prices.

Please note that while this model can make predictions, it should not be used as a sole decision-maker for making investments. Always do your own research and consider multiple sources of information before making investment decisions.

## Model
The LSTM model used in this project is defined in the file main.py. The model consists of two LSTM layers and two Dense layers. The model is compiled with the 'adam' optimizer and the 'mean_squared_error' loss function.

The model is trained on 80% of the available data. The remaining 20% is used for testing the model. The model makes predictions for the testing data and then evaluates the predictions using the root mean squared error (RMSE) metric.

The model also makes future predictions for the Bitcoin price. The start date for future predictions is set to the 13th of July, 2023 and the end date is set to the end of 2024.

## Visualization
The project uses matplotlib to visualize the actual vs. predicted prices. The visualization includes the training data, testing data, and future predictions.

## Explanation
### Data Processing and Feature Scaling
The script begins by loading historical Bitcoin price data from a CSV file. It's using only the 'Close' column, which represents the closing price of Bitcoin each day.

After the data is loaded, it's transformed using a MinMaxScaler from the sklearn.preprocessing package. This scales the price data to a range between 0 and 1, which can make it easier for the model to learn from the data.

### Creating Sequences of Data
The script then creates sequences of 60 days of price data. This means that for each sample, the model will look at the prices from the past 60 days to make a prediction about the price on the next day.

### Building the LSTM Model
Next, the script builds an LSTM (Long Short-Term Memory) model using TensorFlow's Keras API. LSTM is a type of recurrent neural network (RNN) that can learn and remember patterns over sequences of data. This makes it well-suited to predicting time-series data like Bitcoin prices.

The model has two LSTM layers and two dense (fully connected) layers. The 'adam' optimizer is used, which is a popular choice for training neural networks because it's efficient and requires little memory.

### Training the Model
The model is trained on 80% of the data, using the mean squared error loss function. This function calculates the difference between the model's predictions and the actual prices, squares these differences, and averages them. The model's goal is to minimize this value.

### Making Predictions
After training, the model is used to make predictions on the remaining 20% of the data. It also makes future predictions for Bitcoin prices until the end of 2024.

### Visualization
Finally, the script plots the actual and predicted prices using matplotlib. This can help visualize how well the model is performing.

### Potential Improvements
There are several ways the model could potentially be improved:
* Hyperparameter tuning: The script uses a basic LSTM model with a fixed architecture. The number of layers, neurons per layer, and lookback period could be tuned to potentially improve performance.
* Additional features: The model is only using the closing price to make predictions. Including additional features, like trading volume or other technical indicators, could potentially improve the model's predictions.
* Different model architectures: While LSTM models are a popular choice for time series forecasting, there are other types of models that could also be effective. For example, GRU (Gated Recurrent Unit) models are similar to LSTMs but often faster to train. Transformer models, which use self-attention mechanisms, have also shown strong performance on time series data.

### Contact
If you have any questions or suggestions, feel free to open an issue or make a pull request.
