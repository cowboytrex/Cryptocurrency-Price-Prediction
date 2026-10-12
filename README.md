# Cryptocurrency Price Prediction Using LSTM

A Python-based machine learning project that uses a Long Short-Term Memory (LSTM) neural network to analyze historical cryptocurrency prices and predict price movements.

The project uses TensorFlow and Keras for deep learning, Pandas and NumPy for data processing, Scikit-learn for preprocessing and evaluation, and Matplotlib for visualization.

## Project Overview

Cryptocurrency markets are highly volatile, making price forecasting a challenging time-series prediction problem.

This project explores whether an LSTM neural network can learn patterns from historical cryptocurrency prices and estimate subsequent closing prices.

The original implementation uses synthetic price data. The updated workflow supports downloading real historical Bitcoin (BTC-USD) market data through Yahoo Finance.

### Features

- Download historical Bitcoin market data from Yahoo Finance.
- Process and normalize historical closing prices.
- Create time-series sequences for LSTM training.
- Train a stacked LSTM neural network using TensorFlow/Keras.
- Generate predictions for training and testing datasets.
- Evaluate prediction accuracy using Root Mean Squared Error (RMSE).
- Visualize actual prices against model predictions.

## Technologies Used

- **Python** — Core programming language
- **TensorFlow / Keras** — LSTM neural network development
- **Pandas** — Data manipulation
- **NumPy** — Numerical computing
- **Scikit-learn** — Data normalization and model evaluation
- **Matplotlib** — Data visualization
- **yfinance** — Historical cryptocurrency data retrieval

## Project Structure

```text
Cryptocurrency-Price-Prediction/
├── Price-Prediction.py
├── download_data.py
├── bitcoin_historical_data.csv
├── synthetic_cryptocurrency_data.csv
├── test_tensorflow.py
└── README.md
```

**Note:** `download_data.py` and `bitcoin_historical_data.csv` are part of the updated data workflow. The CSV is generated when the download script runs. The main prediction script must be configured to read this real dataset; otherwise, the original implementation will continue generating synthetic prices.

## Installation

### 1. Clone the Repository

```bash
git clone YOUR_REPOSITORY_URL
cd Cryptocurrency-Price-Prediction
```

Replace `YOUR_REPOSITORY_URL` with the actual GitHub repository URL.

### 2. Create a Virtual Environment

Python 3.11 or 3.12 is recommended, subject to compatibility with the installed TensorFlow version.

On macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib scikit-learn tensorflow yfinance
```

### 4. Verify Installation

```bash
python -c "import numpy, pandas, sklearn, matplotlib, tensorflow, yfinance; print('All dependencies installed successfully')"
```

## How to Run

### Step 1 — Download Historical Bitcoin Data

If you have added `download_data.py` using the updated download script:

```bash
python download_data.py
```

This downloads available daily Bitcoin market data beginning January 1, 2020, and saves it as:

```text
bitcoin_historical_data.csv
```

The dataset contains historical market information including opening, high, low, and closing prices, as well as trading volume where available.

### Step 2 — Run the Prediction Model

```bash
python Price-Prediction.py
```

**Important:** The original `Price-Prediction.py` generates and reads synthetic data. To train on real Bitcoin prices, first replace that synthetic-data section with code that loads `bitcoin_historical_data.csv` and selects the closing-price column.

### Step 3 — Review the Results

The model trains on historical sequences and outputs training and testing RMSE values.

Example output format:

```text
Train RMSE: ...
Test RMSE: ...
```

A Matplotlib chart displays actual closing prices alongside the model's training and testing predictions.

The exact results vary depending on the dataset, training configuration, and model initialization.

## Model Architecture

The model uses a stacked LSTM architecture:

1. LSTM layer with 50 units and sequence output
2. Dropout layer with a rate of 0.2
3. LSTM layer with 50 units
4. Dropout layer with a rate of 0.2
5. Dense layer with 25 units
6. Dense output layer with 1 unit

**Training configuration:**

- Optimizer: Adam
- Loss function: Mean Squared Error
- Sequence length: 60 observations
- Original train/test split: 80% / 20%
- Original batch size: 1
- Original epochs: 1

These settings are a starting point for experimentation rather than an optimized production model.

## Model Evaluation

Prediction performance is evaluated using Root Mean Squared Error (RMSE).

Lower RMSE indicates that predicted prices are closer to actual prices on the evaluated dataset.

For reliable out-of-sample evaluation, preprocessing should be fitted using only training data, and test sequences should use historical context without incorporating future target values.

## Limitations

- Cryptocurrency prices are volatile and difficult to forecast.
- Historical price patterns do not guarantee future performance.
- The original model uses only closing prices.
- The original preprocessing fits the scaler before splitting the dataset, which introduces data leakage.
- A single training epoch may be insufficient for stable model performance.
- The project is an educational machine-learning experiment, not a financial trading system.

## Future Improvements

- Incorporate additional features such as trading volume and technical indicators.
- Compare Bitcoin and Ethereum predictions.
- Improve time-series validation and prevent data leakage.
- Tune LSTM hyperparameters.
- Compare LSTM performance with simpler baseline models.
- Add prediction charts and evaluation results to the repository.
- Automate dataset updates.

## Disclaimer

This project is intended for educational and research purposes only. It does not provide financial or investment advice.
