# College Submission Materials
**Project:** Smart Irrigation Prediction Using Machine Learning

> Numbers marked **(my run)** came from running `train_model.py` once while building this project.
> The seed is fixed, so you should see the same values, but library versions can change them slightly.
> **Always replace them with the values printed on YOUR computer.**

---

## 1. Project Report

### Abstract
This project builds a simple IoT/Edge Machine Learning application that predicts whether a field needs irrigation. Soil moisture, temperature and humidity readings are simulated in Python instead of using physical sensors. A Decision Tree Classifier is trained on a synthetic dataset and saved with Pickle. A Streamlit dashboard accepts sensor values and shows the prediction.

### Problem Statement
Over-irrigation wastes water and under-irrigation harms crops. Farmers often decide by guesswork. A small ML model that runs on a local/edge device can suggest whether irrigation is needed from basic sensor data.

### Objectives
- Create a synthetic dataset of at least 300 sensor readings with transparent labelling rules.
- Train and evaluate a Decision Tree Classifier.
- Simulate IoT sensor data without hardware.
- Build a dashboard for live prediction.
- Save and reload the trained model with Pickle.

### Methodology
1. **Data generation:** 500 readings are generated with NumPy using normal distributions, clipped to realistic ranges. Labels follow two rules: (1) soil moisture < 30% -> irrigate; (2) soil moisture < 45% AND temperature > 30 °C AND humidity < 50% -> irrigate; otherwise no irrigation. 5% of labels are flipped randomly to imitate real-world noise.
2. **Preprocessing:** Features and label are separated; data is split 80% train / 20% test (stratified, seed 42) **before** training to avoid data leakage.
3. **Training:** `DecisionTreeClassifier(max_depth=4, random_state=42)`.
4. **Evaluation:** accuracy and classification report on the unseen test set.
5. **Deployment:** model saved to `irrigation_model.pkl` and loaded by the Streamlit app.

### System Architecture
```
+------------------+     +------------------+     +-------------------+
| sensor_simulator |---->|  Streamlit app   |---->| Decision Tree     |
| (fake IoT node)  |     |  (app.py inputs) |     | (loaded from .pkl)|
+------------------+     +------------------+     +---------+---------+
                                  ^                         |
                                  |   Prediction + history  |
                                  +-------------------------+

Offline training:  train_model.py -> dataset.csv -> Decision Tree -> irrigation_model.pkl
```

### Implementation
- **Language/libraries:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Streamlit, Pickle.
- **`train_model.py`:** dataset generation, training, evaluation, saving.
- **`sensor_simulator.py`:** `get_sensor_reading()` returns new random readings.
- **`app.py`:** dashboard with inputs, prediction, accuracy, history table and feature-importance chart.
- The feature names and order are defined once (`FEATURE_COLUMNS`) in `train_model.py` and imported by `app.py`, so both always match.

### Results
- Dataset: 500 rows; 336 "not required" and 164 "required" **(my run)**.
- Test accuracy: **89.00%** on 100 test rows **(my run)**.
- Classification report **(my run)**:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Not Required | 0.92 | 0.91 | 0.92 | 67 |
| Required | 0.82 | 0.85 | 0.84 | 33 |

The accuracy is below 100% mainly because of the 5% random label noise added on purpose.
The dashboard was tested with dry soil (15% -> "Irrigation Required") and wet soil (70% -> "Irrigation Not Required").

### Limitations
- Data is synthetic, so accuracy shows how well the tree learned the made-up rules, not real-world performance.
- Only three features; real irrigation also depends on crop type, soil type, rainfall forecast and growth stage.
- Pickle is only simple persistence: no versioning, tracking or registry.
- Sensors are simulated; real sensors have drift, noise and connectivity issues.
- Only a small test set (100 rows), so accuracy can vary with a different split.

### Conclusion
The project shows a complete, small ML pipeline: data creation, training, evaluation, saving, and serving through a dashboard with simulated IoT input. It can be extended with real sensors (e.g. ESP32 + soil moisture sensor), real field data, more features and MLflow for model tracking.

---

## 2. Screenshots to Capture (after running the project)
1. VS Code showing the folder structure with all files (after training).
2. Terminal output of `python train_model.py` showing accuracy and the classification report.
3. Terminal output of `python sensor_simulator.py` (5 different readings).
4. Terminal showing `streamlit run app.py` running.
5. Dashboard home page (title, description, sliders).
6. Dashboard after clicking **Generate Simulated Sensor Readings** (changed values).
7. A prediction showing **Irrigation Required** (e.g. soil moisture 15).
8. A prediction showing **Irrigation Not Required** (e.g. soil moisture 70).
9. The Recent Predictions table with several rows, plus the Model Accuracy section.
10. The Feature Importance chart.
11. Opened `dataset.csv` (first rows).

---

## 3. Five-Minute Demonstration Script

**0:00 - 0:40 Introduction.** "Our project predicts whether a field needs irrigation using Machine Learning. Since we have no physical sensors, we simulate IoT sensor data in Python. The data is synthetic, not real field data."

**0:40 - 1:30 Folder walkthrough.** Show the files in VS Code. "`train_model.py` creates the dataset and trains the model, `sensor_simulator.py` fakes the sensor, `app.py` is the dashboard."

**1:30 - 2:30 Training.** Run `python train_model.py`. Point out the 500 rows, the labelling rules in `label_rule()`, the train/test split, the accuracy and the report. "We split before training to avoid data leakage, and we only quote the accuracy that the program calculated."

**2:30 - 3:00 Sensor simulator.** Run `python sensor_simulator.py`. "Each call gives new readings, just like a real sensor sending data."

**3:00 - 4:30 Dashboard.** Run `streamlit run app.py`.
- Click **Generate Simulated Sensor Readings** and then **Predict Irrigation**.
- Set soil moisture to 15 and predict -> *Irrigation Required*.
- Set soil moisture to 70 and predict -> *Irrigation Not Required*.
- Show the history table, model accuracy and feature-importance chart.

**4:30 - 5:00 Conclusion.** "Pickle saves the model, but it is not a model registry; MLflow could be added for that. Limitations are synthetic data and only three features. In future we can connect real sensors like an ESP32."

---

## 4. Ten Basic Viva Questions

1. **What is the aim of your project?** To predict whether irrigation is needed from soil moisture, temperature and humidity using ML.
2. **Why is the data simulated?** We have no physical sensors or real field dataset; simulation lets us demonstrate the full pipeline.
3. **What is IoT?** A network of connected devices (like sensors) that collect and send data.
4. **How do you simulate IoT sensors?** `sensor_simulator.py` generates new random readings each time it is called.
5. **Why a Decision Tree?** It is simple, easy to explain, works well on small tabular data and runs on low-power edge devices.
6. **How were the labels created?** By rules: moisture below 30% means irrigate; moisture below 45% with high temperature and low humidity also means irrigate.
7. **What is data leakage and how did you avoid it?** Leakage is when test data influences training. We split the data first and trained only on the training part.
8. **What does the accuracy mean?** The percentage of test rows predicted correctly. It is measured on synthetic data, so it does not prove real-world performance.
9. **What is Pickle? Is it a model registry?** Pickle saves a Python object to a file so we can reload it. It is not a registry because it does not track versions or metrics. MLflow can do that.
10. **What is Edge ML?** Running ML models near the data source (on the device) instead of the cloud, giving faster response and less data transfer.

---

## 5. Explanation of Every File

### `sensor_simulator.py`
- `_rng = np.random.default_rng()` is a random generator **without** a fixed seed, so each call gives new values.
- `get_sensor_reading()` draws moisture, temperature and humidity from normal distributions, uses `np.clip` to keep values in a realistic range and returns a dictionary.
- `if __name__ == "__main__"` lets you run the file alone to test it, without running that code when `app.py` imports it.

### `train_model.py`
- `FEATURE_COLUMNS` / `TARGET_COLUMN`: column names defined once and reused in `app.py` so names and order always match.
- `label_rule()`: the transparent rules that decide the label (1 = irrigate).
- `generate_dataset()`: creates 500 rows using a seeded generator (reproducible), applies the rules, then flips 5% of labels to add noise.
- `main()`:
  1. Generates and saves `dataset.csv`.
  2. Separates `X` (features) and `y` (target).
  3. `train_test_split(..., test_size=0.2, random_state=42, stratify=y)` splits before training (prevents leakage).
  4. `DecisionTreeClassifier(max_depth=4)` limits the tree depth to reduce overfitting; `.fit()` trains it.
  5. `accuracy_score` and `classification_report` evaluate on the test set.
  6. `pickle.dump` saves a dictionary containing the model, accuracy and feature names into `irrigation_model.pkl`.

### `app.py`
- `load_model()`: opens `irrigation_model.pkl` with `pickle.load`; trains first if the file is missing. `@st.cache_resource` loads it only once.
- `st.session_state`: remembers slider values and prediction history between button clicks (Streamlit reruns the script on every click).
- `use_simulated_reading()`: callback for the **Generate Simulated Sensor Readings** button; fills the sliders with a reading from `sensor_simulator.py`.
- **Predict Irrigation** button: builds a one-row DataFrame using `FEATURE_COLUMNS`, calls `model.predict`, displays "Irrigation Required" or "Irrigation Not Required" and stores the result in history.
- The remaining sections show the accuracy, the last 10 predictions, and a Matplotlib bar chart of `feature_importances_`.

### `dataset.csv`
Generated data: 3 sensor columns plus `Irrigation Required` (0 or 1). Synthetic, not real.

### `irrigation_model.pkl`
The pickled trained model (with accuracy and feature names).

### `requirements.txt`
The five libraries needed: numpy, pandas, scikit-learn, streamlit, matplotlib.
