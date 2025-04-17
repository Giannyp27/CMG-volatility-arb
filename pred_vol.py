import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from datetime import datetime, timedelta

# Enable importing config and data from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def train_vol_model(data_path, use_log_target=True, output_predictions=True, model_type="xgb"):
    print(" Loading feature-engineered data...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # Set up features
    feature_cols = [
        'log_return', 'rolling_mean_5', 'rolling_std_5',
        'rolling_mean_10', 'rolling_std_10', 'return_squared',
        'abs_return', 'log_return_lag1', 'rv_5d_lag1',
        'vol_of_vol', 'abs_return_x_vol'
    ]
    
    target_col = 'target_rv_5d_log' if use_log_target else 'target_rv_5d'

    # Training target is now generated in the feature engineering script.
    df.dropna(subset=feature_cols + [target_col], inplace=True)

    X = df[feature_cols]
    y = df[target_col]
    y_raw = y.copy()

    # Scale the features
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    # Reshape for LSTM with a lookback window of 5
    lookback = 5
    X_lstm, y_lstm = [], []
    for i in range(lookback, len(X_scaled)):
        X_lstm.append(X_scaled[i-lookback:i])
        y_lstm.append(y_scaled[i])
    X_reshaped = np.array(X_lstm)
    y_scaled = np.array(y_lstm)
    y_raw = y_raw[lookback:]

    # Date-based train/test split using 1 year ago from today
    cutoff_date = pd.Timestamp(datetime.today() - timedelta(days=365)).tz_localize(None)
    X_df = pd.DataFrame(index=df.index[lookback:], data=X_reshaped.tolist())
    y_df = pd.Series(index=df.index[lookback:], data=y_scaled[lookback:])
    y_raw.index = df.index[lookback:]

    train_mask = X_df.index < cutoff_date
    test_mask = X_df.index >= cutoff_date

    X_train = np.array(X_df[train_mask].tolist())
    X_test = np.array(X_df[test_mask].tolist())
    y_train = np.array(y_df[train_mask].tolist())
    y_test = np.array(y_df[test_mask].tolist())
    y_train_raw = y_raw[train_mask]
    y_test_raw = y_raw[test_mask]

    print(f" Training LSTM model on {len(X_train)} samples...")

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)

    # Predict and inverse scale
    y_pred_scaled = model.predict(X_test)
    y_pred_actual = scaler_y.inverse_transform(y_pred_scaled)

    y_test_actual = scaler_y.inverse_transform(y_test)

    # === XGBoost Ensembling ===
    X_train_flat = X_scaled[:len(y_train)]
    X_test_flat = X_scaled[len(y_train):]

    xgb_model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train_flat, y_train.ravel())
    y_pred_xgb_scaled = xgb_model.predict(X_test_flat).reshape(-1, 1)
    y_pred_xgb_actual = scaler_y.inverse_transform(y_pred_xgb_scaled)

    # === Meta-Ensembling ===
    min_len = min(len(y_pred_actual), len(y_pred_xgb_actual), len(y_test_actual), len(y_test_raw))
    df_preds = pd.DataFrame({
        "lstm": y_pred_actual.flatten()[-min_len:],
        "xgb": y_pred_xgb_actual.flatten()[-min_len:],
        "actual": y_test_actual.flatten()[-min_len:]
    }, index=y_test_raw.iloc[-min_len:].index)

    df_preds.dropna(inplace=True)
    meta_X = df_preds[['lstm', 'xgb']]
    meta_y = df_preds['actual']
    meta_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    meta_model.fit(meta_X, meta_y)
    y_pred_ensemble = meta_model.predict(meta_X).reshape(-1, 1)

    print(f"\n📊 Meta-Ensemble Model Feature Importances:")
    print(dict(zip(meta_X.columns, meta_model.feature_importances_)))

    # Evaluation
    print(" Ensemble Model Evaluation:")
    y_test_trimmed = y_test_actual[-len(y_pred_ensemble):]
    print(f"MAE:  {mean_absolute_error(y_test_trimmed, y_pred_ensemble):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test_trimmed, y_pred_ensemble)):.4f}")
    print(f"R^2:  {r2_score(y_test_trimmed, y_pred_ensemble):.4f}")

    print("\n Last 10 Predictions vs Actual:")
    for i in range(1, 11):
        print(f"{y_test_raw.index[-i].date()} | Actual: {y_test_trimmed[-i][0]:.4f} | Predicted: {y_pred_ensemble[-i][0]:.4f}")

    # Plot actual vs predicted
    plt.figure(figsize=(12, 5))
    plt.plot(y_test_trimmed, label="Actual", linewidth=2)
    plt.plot(y_pred_ensemble, label="Predicted (Ensemble)", linewidth=2)
    plt.legend()
    plt.title("Predicted vs Actual Realized Volatility")
    plt.tight_layout()
    plt.show()

    # Save predictions to CSV (optional)
    if output_predictions:
        output_df = pd.DataFrame({
            "actual_vol": y_test_trimmed.flatten(),
            "predicted_vol": y_pred_ensemble.flatten()
        }, index=y_test_raw.index[-len(y_pred_ensemble):])

        output_path = "../data/vol_predictions.csv"
        output_df.to_csv(output_path)
        print(f"📄 Predictions saved to: {output_path}")


if __name__ == "__main__":
    DATA_PATH = "/Users/gianlucapannozzo/Desktop/university/VS Code/dltr-volatility-arb/data/cmg_features.csv"
    train_vol_model(DATA_PATH, use_log_target=False, model_type="lstm")