# 📈 Microsoft Stock Price Prediction using Stacked LSTM

A deep learning project that predicts Microsoft (MSFT) closing stock prices using a stacked Long Short-Term Memory (LSTM) neural network. The model learns from 60-day historical price windows to forecast the next day's closing price, built with TensorFlow/Keras.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Demo / Results](#-demo--results)
- [How It Works](#-how-it-works)
- [Model Architecture](#-model-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Data Pipeline Explained](#-data-pipeline-explained)
- [Results & Evaluation](#-results--evaluation)
- [Limitations & Future Work](#-limitations--future-work)
- [License](#-license)

---

## 🔍 Overview

This project tackles **time-series forecasting** on financial data. Stock prices are sequential by nature — each day's price depends on what came before — which makes them a natural fit for recurrent neural networks.

The core idea is simple: **given the last 60 days of Microsoft's closing prices, predict the price on day 61.** The model slides a 60-day window across years of historical data, learning the temporal patterns that precede each price movement.

Key features of this project:

- End-to-end pipeline from raw CSV to trained model to visualized predictions
- Exploratory data analysis with trend, volume, and correlation visualizations
- A **two-layer stacked LSTM** architecture for capturing both short- and long-term patterns
- Proper sequence preprocessing (scaling + sliding windows)
- Train/test split with prediction visualization against actual prices

---

## 📊 Demo / Results

The final output overlays three series on a single chart:

- **Blue** — Training data (actual historical prices)
- **Orange** — Test data (actual prices the model never saw during training)
- **Red** — Model predictions on the test period

When the red line tracks the orange line closely, the model has successfully learned the price dynamics.

> 📌 *Add a screenshot of your output plot here once you run it:*
> `![Stock Prediction Results](images/predictions.png)`

---

## ⚙️ How It Works

The pipeline follows these stages:

1. **Load & explore** the historical stock data
2. **Visualize** trends, trading volume, and feature correlations
3. **Isolate** the closing price as the prediction target
4. **Scale** the data to a standard range for stable training
5. **Window** the series into 60-day input sequences
6. **Train** a stacked LSTM on these sequences
7. **Predict** on held-out test data and **inverse-transform** back to real prices
8. **Visualize** predictions against actual values

---

## 🧠 Model Architecture

The model is a `Sequential` stack of five layers:

| Layer | Type | Output Shape | Purpose |
|-------|------|--------------|---------|
| 1 | `LSTM(64, return_sequences=True)` | (60, 64) | Reads the 60-day sequence, outputs a hidden state at every timestep |
| 2 | `LSTM(64, return_sequences=False)` | (64,) | Condenses the sequence into a single 64-dim summary vector |
| 3 | `Dense(128, activation='relu')` | (128,) | Non-linear processing of extracted features |
| 4 | `Dropout(0.5)` | (128,) | Regularization — randomly drops 50% of neurons during training |
| 5 | `Dense(1)` | (1,) | Outputs the final predicted price |

**Why two stacked LSTMs?** The first layer learns low-level temporal patterns (short-term momentum, local fluctuations). The second learns higher-level abstractions over those patterns, giving the model more representational depth.

**Compilation settings:**

- **Optimizer:** Adam (adaptive learning rate)
- **Loss:** Mean Absolute Error (MAE) — robust to price outliers
- **Metric:** Root Mean Squared Error (RMSE) — for monitoring

The "64" in each LSTM layer refers to the **number of units** — meaning the hidden and cell states are 64-dimensional vectors. Think of it as 64 parallel memory cells, each learning to track a different pattern in the price series.

---

## 🛠 Tech Stack

| Tool | Role |
|------|------|
| **Python 3.8+** | Core language |
| **TensorFlow / Keras** | Building and training the LSTM |
| **Pandas** | Data loading and manipulation |
| **NumPy** | Numerical operations and array reshaping |
| **scikit-learn** | `StandardScaler` for feature scaling |
| **Matplotlib** | Plotting trends and predictions |
| **Seaborn** | Correlation heatmap visualization |

---

## 📁 Project Structure

```
Microsoft-Stock-Prediction/
│
├── MicrosoftStock.csv        # Historical stock data (date, open, high, low, close, volume)
├── stock_prediction.py       # Main script (data prep, model, training, prediction)
├── requirements.txt          # Python dependencies
├── images/                   # Output plots and screenshots
│   └── predictions.png
└── README.md                 # This file
```

---

## 💾 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Microsoft-Stock-Prediction.git
cd Microsoft-Stock-Prediction
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` yet, create one with:

```
tensorflow
pandas
numpy
scikit-learn
matplotlib
seaborn
```

---

## 🚀 Usage

Make sure `MicrosoftStock.csv` is in the project root, then run:

```bash
python stock_prediction.py
```

The script will:
1. Print dataset info and summary statistics
2. Generate exploratory plots (open/close, volume, correlation heatmap)
3. Train the LSTM for 20 epochs
4. Display the final predictions-vs-actual chart

> 💡 The exploratory `plt.show()` calls are commented out by default so training isn't interrupted. Uncomment them if you want to inspect the EDA plots step by step.

---

## 🔬 Data Pipeline Explained

### Scaling

```python
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset)
```

Stock prices can range from tens to hundreds of dollars. Large raw values cause unstable gradients, so the data is standardized to **mean 0, standard deviation 1** before training. Predictions are later converted back to real dollars with `scaler.inverse_transform()`.

### Sliding Window

```python
for i in range(60, len(training_data)):
    X_train.append(training_data[i-60:i, 0])  # past 60 days
    y_train.append(training_data[i, 0])       # the 61st day
```

This transforms a flat price series into supervised learning pairs:

```
Window 1:  days [0..59]  → predict day 60
Window 2:  days [1..60]  → predict day 61
Window 3:  days [2..61]  → predict day 62
...
```

### Reshaping for the LSTM

Keras LSTMs require **3D input** of shape `(samples, timesteps, features)`:

```python
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#                              (num_windows,      60 timesteps,      1 feature)
```

---

## 📈 Results & Evaluation

- **Train/Test split:** 95% training, 5% testing
- **Loss function:** MAE on scaled data
- **Monitoring metric:** RMSE

The model is evaluated visually by overlaying predictions against actual test prices. Closer alignment indicates better learning of the underlying price dynamics.

---

## ⚠️ Limitations & Future Work

This project is an excellent **learning pipeline**, but a few honest caveats apply to real-world use:

- **Data leakage in scaling:** The scaler is currently fit on the entire dataset before splitting, which leaks future statistics into training. *Best practice:* fit the scaler on training data only.
- **The "lag trap":** Predicting next-day price from past prices often produces a model that essentially outputs "tomorrow ≈ today." The prediction line can look impressive while really just lagging the actual by one day — a well-known pitfall in stock-prediction tutorials.
- **Single feature:** Only the closing price is used. Real forecasting benefits from additional features (volume, technical indicators, sentiment).

**Potential improvements:**

- [ ] Fit the scaler on training data only to eliminate leakage
- [ ] Add evaluation metrics like **directional accuracy** (did it predict up/down correctly?)
- [ ] Incorporate multiple features (volume, moving averages, RSI)
- [ ] Experiment with GRU layers and compare performance
- [ ] Add hyperparameter tuning (units, window size, dropout rate)
- [ ] Implement a proper validation split and early stopping

> **Disclaimer:** This project is for educational purposes only and is **not financial advice**. Stock markets are influenced by countless factors that historical price alone cannot capture. Do not use this model for actual trading decisions.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements

Built as a hands-on exploration of LSTM networks for time-series forecasting. Contributions, issues, and suggestions are welcome!

⭐ If you found this project helpful, consider giving it a star!
