"""
train_model.py
1. Generates a SYNTHETIC (simulated) dataset  -> dataset.csv
2. Trains a Decision Tree Classifier
3. Prints accuracy and classification report
4. Saves the model with Pickle              -> irrigation_model.pkl

Run with:  python train_model.py
"""
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Column names are defined ONCE here and reused by app.py (same names, same order)
FEATURE_COLUMNS = ["Soil Moisture (%)", "Temperature (°C)", "Humidity (%)"]
TARGET_COLUMN = "Irrigation Required"

SEED = 42            # fixed seed -> same dataset and same results every run
N_ROWS = 500         # more than the required 300 rows
DATASET_FILE = "dataset.csv"
MODEL_FILE = "irrigation_model.pkl"


def label_rule(moisture, temperature, humidity):
    """Transparent rules used to create the target label (1 = irrigate)."""
    if moisture < 30:                                   # Rule 1: soil is dry
        return 1
    if moisture < 45 and temperature > 30 and humidity < 50:
        return 1                                        # Rule 2: hot, dry air, soil getting low
    return 0                                            # Otherwise: no irrigation needed


def generate_dataset(n_rows=N_ROWS, seed=SEED):
    """Create simulated sensor readings and label them using label_rule()."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        FEATURE_COLUMNS[0]: np.clip(rng.normal(40, 18, n_rows), 0, 100).round(1),
        FEATURE_COLUMNS[1]: np.clip(rng.normal(28, 7, n_rows), 10, 45).round(1),
        FEATURE_COLUMNS[2]: np.clip(rng.normal(60, 18, n_rows), 10, 100).round(1),
    })
    df[TARGET_COLUMN] = [label_rule(m, t, h) for m, t, h in df[FEATURE_COLUMNS].values]

    # Flip 5% of labels to mimic real-world noise (sensor errors / farmer decisions)
    flip = rng.random(n_rows) < 0.05
    df.loc[flip, TARGET_COLUMN] = 1 - df.loc[flip, TARGET_COLUMN]
    return df


def main():
    # 1. Generate and save the dataset (simulated data, NOT real field data)
    df = generate_dataset()
    df.to_csv(DATASET_FILE, index=False, encoding="utf-8")
    print(f"Dataset saved to {DATASET_FILE} ({len(df)} rows)")
    print(df[TARGET_COLUMN].value_counts().rename({0: "Not required (0)", 1: "Required (1)"}))

    # 2. Separate input features (X) and target label (y)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # 3. Split BEFORE training so the model never sees test data (prevents data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # 4. Train the Decision Tree (max_depth limits tree size to reduce overfitting)
    model = DecisionTreeClassifier(max_depth=4, random_state=SEED)
    model.fit(X_train, y_train)

    # 5. Evaluate on the unseen test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Not Required", "Required"]))

    # 6. Save model + accuracy + feature order together with Pickle
    bundle = {"model": model, "accuracy": accuracy, "features": FEATURE_COLUMNS}
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Model saved to {MODEL_FILE}")


if __name__ == "__main__":
    main()
