import os
import sys
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from xgboost import plot_importance

# Allow import of root project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def predict_vol_spikes(data_path):
    print(" Loading CMG engineered features...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df.dropna(inplace=True)

    # Features and target
    feature_cols = [
        'log_return', 'rolling_mean_5', 'rolling_std_5',
        'rolling_mean_10', 'rolling_std_10', 'return_squared',
        'abs_return', 'log_return_lag1', 'rv_5d_lag1',
        'vol_of_vol', 'abs_return_x_vol'
    ]
    X = df[feature_cols].astype(float)
    df['is_spike'] = (df['rv_5d_lag1'] > df['rv_5d_lag1'].quantile(0.85)).astype(int)
    y = df['is_spike']

    # Time-based split
    split_index = int(len(df) * 0.6)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    if len(X_test) == 0 or len(y_test) == 0:
        print(" No test data available after split. Check dataset size.")
        return

    # Adjust for class imbalance
    weight_ratio = (y_train == 0).sum() / (y_train == 1).sum()

    # Model
    model = XGBClassifier(
        objective="binary:logistic",
        base_score=0.5,
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        scale_pos_weight=weight_ratio
    )
    model.fit(X_train, y_train)

    # Feature importance plot
    plt.figure(figsize=(10, 6))
    plot_importance(model, max_num_features=10)
    plt.title("Top Feature Importances (XGBoost)")
    plt.tight_layout()
    plt.show()

    # Predictions
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Save spike probability into df
    df.loc[X_test.index, 'spike_prob'] = y_prob

    from sklearn.metrics import f1_score

    best_thresh = 0.0
    best_f1 = 0.0
    for t in np.linspace(0.01, 0.99, 100):
        preds = (y_prob > t).astype(int)
        score = f1_score(y_test, preds)
        if score > best_f1:
            best_f1 = score
            best_thresh = t

    print(f"\n Optimal threshold based on F1 score: {best_thresh:.2f} (F1 = {best_f1:.2f})")
    y_pred = (y_prob > best_thresh).astype(int)

    # Evaluation
    print("\n Spike Prediction Report:")

    if len(np.unique(y_test)) < 2:
        print(" Warning: Only one class present in y_test. Skipping detailed report.")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
    else:
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

        from sklearn.metrics import precision_recall_curve, average_precision_score

        precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
        avg_precision = average_precision_score(y_test, y_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'XGBoost (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # === Meta Model: Filter False Positives ===
    false_positives = (y_pred == 1) & (y_test == 0)
    true_spikes = (y_test == 1)

    meta_X = X_test.copy()
    meta_X['is_fp'] = false_positives.astype(int)

    if meta_X['is_fp'].sum() > 10:
        print("\n Training meta-model to filter false positives...")

        from sklearn.ensemble import RandomForestClassifier

        meta_features = meta_X.drop(columns=['is_fp'])
        meta_labels = meta_X['is_fp']

        meta_train_X, meta_test_X, meta_train_y, meta_test_y = train_test_split(
            meta_features, meta_labels, test_size=0.3, random_state=42, stratify=meta_labels
        )

        meta_model = RandomForestClassifier(n_estimators=100, random_state=42)
        meta_model.fit(meta_train_X, meta_train_y)
        meta_preds = meta_model.predict(meta_test_X)

        print("\n Meta-Model (False Positive Filter) Report:")
        print(classification_report(meta_test_y, meta_preds))

        importances = pd.Series(meta_model.feature_importances_, index=meta_features.columns)
        importances.nlargest(10).plot(kind='barh', title="Meta-Model Feature Importances")
        plt.tight_layout()
        plt.show()
    else:
        print("\n Not enough false positives to train a reliable meta-model.")

    # Save updated feature set with spike_prob column (excluding is_spike)
    df.drop(columns=["is_spike"]).to_csv(data_path)
    print(" spike_prob column added to feature CSV.")

if __name__ == "__main__":
    DATA_PATH = "../data/cmg_features.csv"
    predict_vol_spikes(DATA_PATH)
