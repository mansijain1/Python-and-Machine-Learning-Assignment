# Python-and-Machine-Learning-Assignment

# 🩺 Diabetes Prediction Project

This project involves building a machine learning pipeline to predict diabetes based on patient health data. The dataset contains a mix of categorical and numerical features, and the goal is to accurately classify whether a patient is diabetic.

## 📁 Dataset Description

- **File name:** `diabetes_prediction_dataset.csv`
- **Target column:** `diabetes` (0: Non-diabetic, 1: Diabetic)
- **Features:**
  - `gender`
  - `age`
  - `hypertension`
  - `heart_disease`
  - `smoking_history`
  - `bmi`
  - `HbA1c_level`
  - `blood_glucose_level`

---

## 🧪 1. Data Loading & Exploration

### ✅ Steps:
- Loaded dataset using `pandas`.
- Displayed the first few rows and checked data types and summary statistics.
- Checked for class imbalance in the `diabetes` target column.

### 📊 Visualization:
- Used `matplotlib` and `seaborn` to:
  - Plot histograms for numerical feature distributions.
  - Create a correlation heatmap.
  - Visualize `bmi` and `blood_glucose_level` against `diabetes`.

---

## 🧹 2. Preprocessing

### ✅ Steps Implemented:
- **Missing Values:** Removed rows with nulls using `dropna()` (could also be imputed).
- **Encoding:**
  - Used `OneHotEncoder` for `gender` and `smoking_history`.
- **Scaling:**
  - Applied `StandardScaler` to numerical features.
- **Train/Test Split:**
  - Split data into training and test sets using `train_test_split` with stratification.

---

## 🤖 3. Model Selection

- Chose **Random Forest Classifier** for initial modeling due to:
  - Strong performance with mixed data types
  - Robustness to outliers and non-linearity

### ✅ Created a full pipeline using a Python class:
- Combined preprocessing (encoding + scaling) and modeling in one `Pipeline`.

---

## 🏋️‍♂️ 4. Model Training & Evaluation

- Trained the model on the training set.
- Evaluated using classification metrics:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1 Score**

### 📋 Baseline Results:

| Metric     | Score    |
|------------|----------|
| Accuracy   | 0.9697   |
| Precision  | 0.9376   |
| Recall     | 0.6888   |
| F1 Score   | 0.7942   |

- Observed high accuracy but lower recall, indicating room for improvement.

---

## 🔍 5. Hyperparameter Tuning

### ✅ Used `GridSearchCV` to tune:
- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

### ✅ Best Parameters Found:
```python
{
  'classifier__n_estimators': 200,
  'classifier__max_depth': 20,
  'classifier__min_samples_split': 2,
  'classifier__min_samples_leaf': 1
}
