# Predicting Diabetes using Pima Indians Diabetes Database

## 1. Introduction to Problem & Data

### Problem Statement
The goal of this project is to predict whether a patient has diabetes based on health metrics such as glucose level, BMI, and age. The dataset is specifically focused on females of Pima Indian heritage who are at least 21 years old.

### Dataset Description
The dataset used for this classification task is the **Pima Indians Diabetes Database** sourced from the UCI Machine Learning Repository. The dataset contains the following features:

- **Pregnancies**: Number of times the patient has been pregnant
- **Glucose**: Plasma glucose concentration (mg/dL)
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: A function which scores the likelihood of diabetes based on family history
- **Age**: Age of the patient (years)
- **Outcome**: Target variable (0 = No diabetes, 1 = Diabetes)

## 2. Data Pre-Processing & Preliminary Examination

### Load and Inspect Data
```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# Load dataset
data = pd.read_csv("pima_indians_diabetes.csv")

# Display the first few rows of the dataset
print(data.head())

# Basic information about the dataset
data.info()

# Check for missing values
print(data.isnull().sum())

# Descriptive statistics
print(data.describe())
```

### Handling Missing or Erroneous Data
Some of the features, such as Glucose, BloodPressure, SkinThickness, Insulin, and BMI, may contain zero values which are not realistic for medical data. These values will be treated as missing and imputed appropriately.
```python
# Replace zero values with NaN for specific columns
columns_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in columns_with_zero:
    data[col].replace(0, np.nan, inplace=True)

# Check missing values again
print(data.isnull().sum())

# Impute missing values with the median
data.fillna(data.median(), inplace=True)
```

### Exploratory Data Analysis (EDA)
```python
# Plot histograms for each feature
data.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
```

## 3. Model Preparation

### Data Splitting
Split the dataset into training and testing sets.
```python
# Define features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

## 4. Model Selection & Training

### Logistic Regression
```python
from sklearn.linear_model import LogisticRegression

# Initialize and train Logistic Regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Predict on the test set
y_pred_logreg = logreg.predict(X_test)

# Evaluate Logistic Regression
print("Logistic Regression Performance:")
print(classification_report(y_test, y_pred_logreg))
```

### Random Forest
```python
from sklearn.ensemble import RandomForestClassifier

# Initialize and train Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf.predict(X_test)

# Evaluate Random Forest
print("Random Forest Performance:")
print(classification_report(y_test, y_pred_rf))
```

### Gradient Boosting
```python
from sklearn.ensemble import GradientBoostingClassifier

# Initialize and train Gradient Boosting model
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)

# Predict on the test set
y_pred_gb = gb.predict(X_test)

# Evaluate Gradient Boosting
print("Gradient Boosting Performance:")
print(classification_report(y_test, y_pred_gb))
```

## 5. Model Evaluation

### Compare Performance Metrics
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
models = ["Logistic Regression", "Random Forest", "Gradient Boosting"]
preds = [y_pred_logreg, y_pred_rf, y_pred_gb]

for i, (model, pred) in enumerate(zip(models, preds)):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i])
    axes[i].set_title(f"{model} Confusion Matrix")
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")

plt.tight_layout()
plt.show()
```

## 6. Discussion of Results
- Compare the models' performance based on precision, recall, F1-score, and ROC-AUC.
- Discuss which model performed best and why.

## 7. Conclusion & Future Work
- Summarize the findings of the project.
- Discuss potential improvements or future work.

## 8. References
- UCI Machine Learning Repository: Pima Indians Diabetes Database
- Scikit-learn Documentation