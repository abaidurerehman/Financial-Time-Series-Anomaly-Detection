# Financial Time-Series Anomaly Detection

## Overview
This project aims to build a tool that detects anomalies in stock price trends to identify unusual market activities or potential manipulations. The tool analyzes historical stock price data, calculates financial indicators, performs unsupervised anomaly detection, and visualizes detected anomalies on stock price trends.

## Objective
The goal of this project is to create an anomaly detection system for financial time-series data, using unsupervised machine learning models. The system detects unusual trends in stock price data and can help identify potential market manipulations.

## Key Features
- **Stock Price Analysis**: Analyze stock price trends over time using various financial indicators.
- **Anomaly Detection**: Use the **Isolation Forest** model for unsupervised anomaly detection to identify outliers in stock price data.
- **Time-Series Forecasting**: Use **Prophet** (or optionally **LSTM**) to forecast future stock prices and detect deviations from predicted values.
- **Data Visualization**: Visualize stock price trends along with detected anomalies using **matplotlib** in an interactive **Streamlit** dashboard.
- **Downloadable Report**: Export detected anomalies in a downloadable CSV format.

## Tools & Libraries
This project uses the following tools and libraries:
- **Python**: Programming language used for the analysis and model building.
- **Streamlit**: Web application framework to deploy the tool in an interactive dashboard.
- **Pandas**: Data manipulation library to handle and preprocess the stock price data.
- **Matplotlib**: Visualization library for plotting stock price trends and anomalies.
- **Scikit-learn**: Machine learning library used for **Isolation Forest** anomaly detection.
- **Prophet**: Time-series forecasting library for predicting future stock prices (can optionally be replaced with **LSTM**).
- **Numpy**: Used for numerical operations.

## Dataset
The dataset used in this project contains historical stock price data for companies, which is sourced from **Yahoo Finance**. The dataset includes columns like:
- `Date`: Date of the stock price.
- `Close/Last`: The closing price of the stock on a given day.
- `Open`: The opening price of the stock on a given day.
- `High`: The highest price reached by the stock on a given day.
- `Low`: The lowest price reached by the stock on a given day.
- `Volume`: The trading volume for the stock on a given day.

## Steps Taken in the Project

### 1. Data Preprocessing
- Cleaned and preprocessed the stock price data to handle missing values and ensure correct data types.
- Removed unnecessary characters like `$` and `,` from the price data.
- Sorted the data by date to ensure correct chronological order.

### 2. Calculating Financial Indicators
- **Simple Moving Average (SMA)**: A moving average that helps smooth the stock price over a specified window.
- **Exponential Moving Average (EMA)**: Similar to SMA, but gives more weight to recent prices.
- **Bollinger Bands**: Upper and lower bands that indicate overbought or oversold conditions in the market.
- **Relative Strength Index (RSI)**: A momentum oscillator that measures the speed and change of price movements.

### 3. Anomaly Detection
- Used **Isolation Forest**, an unsupervised machine learning algorithm, to detect anomalies in the stock price data.
- Scaled the data using **StandardScaler** for better performance of the model.
- Anomalies are flagged and visualized on the stock price trends.

### 4. Time-Series Forecasting
- Used **Prophet** (from Facebook) to predict future stock prices based on historical data.
- The predicted values are compared against real values to detect significant deviations (anomalies).
  
### 5. Data Visualization
- Visualized the stock price trends and anomalies using **matplotlib**.
- Anomalies are marked in **red** on the stock price graph for easy identification.

### 6. Anomaly Report
- Detected anomalies are saved to a CSV file for easy export and further analysis.

## How to Use


1. Clone the repository:
   ```bash
git clone https://github.com/abaidurerehman/Financial-Time-Series-Anomaly-Detection.git


   cd stock-price-anomaly-detection
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Upload your stock price dataset (`.csv` file) and the tool will perform anomaly detection and forecasting.

5. Visualize the detected anomalies in the interactive graph and download the CSV file containing the anomalies.

## Expected Output
The app will generate:
- A time-series graph showing stock price trends.
- Anomalies detected in the stock price data will be marked as red dots.
- A downloadable CSV file containing the anomalies detected in the dataset.

## Future Improvements
- **LSTM Model**: As an enhancement, an **LSTM (Long Short-Term Memory)** model can be added to improve forecasting accuracy, especially for high-frequency stock price data.
- **Real-time Data**: Integrating real-time stock price feeds to detect anomalies as they happen.
- **Additional Anomaly Detection Algorithms**: Implementing other unsupervised algorithms like **DBSCAN** for anomaly detection.

## License
This project is licensed under the MIT License.

## Acknowledgments
- **Prophet** by Facebook for time-series forecasting.
- **Isolation Forest** and other scikit-learn models.
- **Yahoo Finance** for providing the stock price data.
