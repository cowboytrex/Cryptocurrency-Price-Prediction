# Cryptocurrency Price Prediction Using LSTM

A Python-based deep learning project that explores cryptocurrency price forecasting using a Long Short-Term Memory (LSTM) neural network. The application processes historical market data, trains a time-series prediction model, evaluates prediction accuracy, and visualizes predicted versus actual cryptocurrency prices.

## Project Overview

Cryptocurrency markets exhibit high volatility and complex price movements, making forecasting a challenging machine learning problem.

This project investigates the application of LSTM neural networks to historical cryptocurrency price data, with a focus on Bitcoin (BTC-USD).

The project includes an original synthetic-data workflow and an updated historical Bitcoin dataset obtained through Yahoo Finance.

## Key Features

- **Historical Data Collection:** Retrieve historical Bitcoin market data using the `yfinance` Python library.
- **Data Preprocessing:** Clean, transform, and normalize cryptocurrency price datasets using Pandas, NumPy, and Scikit-learn.
- **Time-Series Modeling:** Transform sequential price observations into sliding windows suitable for LSTM training.
- **Deep Learning:** Implement a stacked LSTM neural network using TensorFlow and Keras.
- **Model Evaluation:** Measure prediction accuracy using Root Mean Squared Error (RMSE).
- **Data Visualization:** Compare predicted and actual cryptocurrency prices using Matplotlib.
- **Experimental Analysis:** Explore model performance and limitations in cryptocurrency price forecasting.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core application development |
| TensorFlow / Keras | Deep learning and LSTM architecture |
| Pandas | Financial data processing |
| NumPy | Numerical operations |
| Scikit-learn | Feature scaling and model evaluation |
| Matplotlib | Financial data visualization |
| yfinance | Historical Bitcoin market data retrieval |

## Project Structure

```text
Cryptocurrency-Price-Prediction/
├── Price-Prediction.py
├── synthetic_cryptocurrency_data.csv
├── bitcoin_prices_2020_2026.csv
├── test_tensorflow.py
├── .gitignore
└── README.md
```

The repository includes both synthetic cryptocurrency data and historical Bitcoin market data. The prediction script must be configured to use the desired dataset.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/cowboytrex/Cryptocurrency-Price-Prediction.git
cd Cryptocurrency-Price-Prediction
```

### 2. Create a Virtual Environment

For macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

For Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install tensorflow pandas numpy scikit-learn matplotlib yfinance
```

## Usage

Run the cryptocurrency prediction script:

```bash
python Price-Prediction.py
```

The application processes the dataset selected in the script, prepares time-series sequences, trains the LSTM model, and evaluates its predictions.

**Note:** The historical Bitcoin CSV is included as a data source, but the prediction script must explicitly load that file to train on real market prices.

## LSTM Model Architecture

The original model uses the following architecture:

| Layer | Configuration |
|---|---|
| LSTM Layer 1 | 50 units, return sequences enabled |
| Dropout Layer 1 | 20% |
| LSTM Layer 2 | 50 units |
| Dropout Layer 2 | 20% |
| Dense Layer | 25 units |
| Output Layer | 1 unit |

**Training configuration**

- Optimizer: Adam
- Loss function: Mean Squared Error (MSE)
- Sequence length: 60 observations
- Training/testing split: 80% / 20%
- Original training epochs: 1
- Original batch size: 1

These are experimental settings and have not been established as optimal.

## Model Evaluation

The model uses Root Mean Squared Error (RMSE) to measure the difference between predicted and actual prices.

RMSE is defined as:

\[
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
\]

A lower RMSE indicates smaller prediction errors on the evaluated dataset.

For a reliable assessment of forecasting performance, training and testing data should be separated chronologically, and preprocessing parameters should be fitted using only training data.

## Data Sources

**Yahoo Finance — Bitcoin (BTC-USD)**

Historical Bitcoin market data is retrieved through the Python `yfinance` library.

The historical dataset covers available observations from 2020 through 2026 and may include opening, closing, high, and low prices, as well as trading volume.

The project also includes a synthetic cryptocurrency dataset for experimentation and initial model development.

## Current Limitations

- Cryptocurrency markets are highly volatile and difficult to forecast.
- The original prediction workflow uses synthetic price data.
- The original model uses closing prices rather than multiple market indicators.
- The original preprocessing may introduce data leakage because scaling occurs before the training/testing split.
- A single training epoch is insufficient to establish stable forecasting performance.
- Historical predictive accuracy does not guarantee future performance.

## Future Improvements

- Integrate historical Bitcoin data directly into the model training pipeline.
- Prevent data leakage through training-only normalization.
- Add technical indicators such as RSI, MACD, and moving averages.
- Implement walk-forward validation.
- Compare LSTM performance against baseline forecasting methods.
- Tune hyperparameters to improve model performance.
- Expand forecasting experiments to Ethereum and other cryptocurrencies.
- Add reproducible evaluation metrics and prediction charts.

## Disclaimer

This project is intended for educational and research purposes only. It is not a production trading system and does not constitute financial or investment advice.
