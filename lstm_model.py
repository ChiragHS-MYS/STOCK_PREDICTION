"""
LSTM Model Module for Stock Price Prediction
Provides PyTorch LSTM neural network architecture, dataset sliding windows,
online CPU training, autoregressive forecasting, and ensemble weighting.
"""
import numpy as np
import pandas as pd
import logging
from typing import Tuple, List, Dict, Any, Optional
logger = logging.getLogger(__name__) #logging to debug and track the code 

# Check if PyTorch is available dynamically
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch is not available. Falling back to mathematical trend estimation for LSTM model.")


class PyTorchStockLSTM(nn.Module if TORCH_AVAILABLE else object):
    """
    Standard PyTorch LSTM model for single-variable time-series forecasting.
    """
    def __init__(self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 1, output_size: int = 1):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        out, _ = self.lstm(x)
        # Take the output of the last time step
        out = self.fc(out[:, -1, :])
        return out


class StockScaler:
    """Simple MinMaxScaler for 1D/2D price sequences without external dependencies."""
    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)):
        self.min_val = 0.0
        self.max_val = 1.0
        self.range_min, self.range_max = feature_range

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.min_val = np.min(data)
        self.max_val = np.max(data)
        if self.max_val == self.min_val:
            self.max_val += 1e-8
        scaled = (data - self.min_val) / (self.max_val - self.min_val)
        return scaled * (self.range_max - self.range_min) + self.range_min

    def transform(self, data: np.ndarray) -> np.ndarray:
        scaled = (data - self.min_val) / (self.max_val - self.min_val + 1e-8)
        return scaled * (self.range_max - self.range_min) + self.range_min

    def inverse_transform(self, scaled_data: np.ndarray) -> np.ndarray:
        unscaled = (scaled_data - self.range_min) / (self.range_max - self.range_min + 1e-8)
        return unscaled * (self.max_val - self.min_val) + self.min_val


def create_sliding_windows(data: np.ndarray, lookback: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """Generates (X, y) sequence windows for training sequence models."""
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)


def train_and_predict_lstm(
    close_prices: np.ndarray,
    lookback: int = 30,
    future_days: int = 30,
    epochs: int = 25,
    lr: float = 0.01,
    weight_lstm: float = 0.60
) -> Dict[str, Any]:
    """
    Trains PyTorch LSTM on historical close prices and generates future multi-step forecasts.
    Returns dictionary with predicted prices, lower/upper bands, and metrics.
    """
    prices = np.array(close_prices, dtype=np.float32).flatten()
    n_samples = len(prices)

    # Adjust lookback if series is short
    if n_samples < lookback + 5:
        lookback = max(5, n_samples // 3)

    scaler = StockScaler(feature_range=(0.0, 1.0))
    scaled_prices = scaler.fit_transform(prices)

    X_train, y_train = create_sliding_windows(scaled_prices, lookback=lookback)

    if len(X_train) == 0 or not TORCH_AVAILABLE:
        # Fallback to smooth exponential trend prediction if insufficient data or torch unavailable
        return _fallback_lstm_predict(prices, future_days)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)  # (N, lookback, 1)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)  # (N, 1)

    # Model training
    model = PyTorchStockLSTM(input_size=1, hidden_size=32, num_layers=1, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()

    # Calculate residual error (RMSE) on historical data to build confidence intervals
    with torch.no_grad():
        # Detach tensor and move to CPU before converting to NumPy
        fitted_scaled = model(X_tensor).detach().cpu().numpy().flatten()
    fitted_prices = scaler.inverse_transform(fitted_scaled)
    actual_prices = prices[lookback:]
    rmse = float(np.sqrt(np.mean((actual_prices - fitted_prices) ** 2)))

    # Autoregressive future multi-step forecasting
    current_window = scaled_prices[-lookback:].tolist()
    future_scaled_predictions = []

    with torch.no_grad():
        for _ in range(future_days):
            inp_seq = torch.tensor(current_window[-lookback:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            pred_val = model(inp_seq).item()
            future_scaled_predictions.append(pred_val)
            current_window.append(pred_val)

    future_lstm_predictions = scaler.inverse_transform(np.array(future_scaled_predictions))
    
    # Confidence bounds (1.5 * RMSE scaling dynamically over time horizon)
    horizon_factors = np.sqrt(np.linspace(1.0, 2.0, future_days))
    future_upper = future_lstm_predictions + (1.5 * rmse * horizon_factors)
    future_lower = future_lstm_predictions - (1.5 * rmse * horizon_factors)

    return {
        "lstm_predictions": np.round(future_lstm_predictions, 2).tolist(),
        "lstm_upper": np.round(future_upper, 2).tolist(),
        "lstm_lower": np.round(future_lower, 2).tolist(),
        "rmse": round(rmse, 2),
        "lookback_used": lookback,
        "model_type": "PyTorch LSTM" if TORCH_AVAILABLE else "Fallback Exponential Trend"
    }


def compute_weighted_ensemble(
    linear_preds: List[float],
    lstm_preds: List[float],
    linear_std: float,
    lstm_rmse: float,
    weight_lstm: float = 0.60
) -> Dict[str, List[float]]:
    """
    Combines Linear Regression and LSTM predictions into a weighted ensemble forecast.
    Ensemble = weight_lstm * LSTM + (1 - weight_lstm) * LinearReg
    """
    weight_linear = 1.0 - weight_lstm
    n = len(linear_preds)

    ensemble_preds = []
    ensemble_upper = []
    ensemble_lower = []

    # Combined standard deviation / uncertainty estimate
    combined_std = np.sqrt((weight_lstm * (lstm_rmse ** 2)) + (weight_linear * (linear_std ** 2)))

    for i in range(n):
        l_pred = linear_preds[i]
        lstm_p = lstm_preds[i]
        ens_p = (weight_lstm * lstm_p) + (weight_linear * l_pred)
        
        # Horizon scaling factor for confidence band width expansion over future days
        scale = np.sqrt(1.0 + (i / n))
        ens_u = ens_p + (combined_std * scale)
        ens_l = ens_p - (combined_std * scale)

        ensemble_preds.append(round(float(ens_p), 2))
        ensemble_upper.append(round(float(ens_u), 2))
        ensemble_lower.append(round(float(ens_l), 2))

    return {
        "ensemble_predictions": ensemble_preds,
        "ensemble_upper": ensemble_upper,
        "ensemble_lower": ensemble_lower,
        "combined_rmse": round(float(combined_std), 2)
    }


def _fallback_lstm_predict(prices: np.ndarray, future_days: int) -> Dict[str, Any]:
    """Fallback predictor if data is insufficient or torch fails."""
    last_price = prices[-1]
    returns = np.diff(prices) / (prices[:-1] + 1e-8)
    avg_return = np.mean(returns[-20:]) if len(returns) >= 20 else np.mean(returns)
    std_return = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns) + 0.01

    future_preds = []
    curr = last_price
    for _ in range(future_days):
        curr = curr * (1.0 + avg_return)
        future_preds.append(curr)

    future_preds = np.array(future_preds)
    rmse = float(last_price * std_return * 2.0)
    horizon_factors = np.sqrt(np.linspace(1.0, 2.0, future_days))

    return {
        "lstm_predictions": np.round(future_preds, 2).tolist(),
        "lstm_upper": np.round(future_preds + (rmse * horizon_factors), 2).tolist(),
        "lstm_lower": np.round(future_preds - (rmse * horizon_factors), 2).tolist(),
        "rmse": round(rmse, 2),
        "lookback_used": 10,
        "model_type": "Fallback Exponential Trend"
    }
