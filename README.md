# German Credit Risk Prediction with SVM

A machine learning project that predicts whether a bank customer is a **good** or **bad** credit risk, using the German Credit dataset and a Support Vector Machine (SVM) classifier.

## Dataset

- **Source:** [German Credit Data with Risk (Kaggle)](https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk)
- **Size:** 1,000 customers, 10 original columns
- **Target:** `Risk` — `good` (700 customers) or `bad` (300 customers)
- **Features:** Age, Sex, Job, Housing, Saving accounts, Checking account, Credit amount, Duration, Purpose

## Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Checked data types, shape, and summary statistics with `.info()` and `.describe()`
- Found missing values in `Saving accounts` (183) and `Checking account` (394) — filled with `"none"`, since a missing value here likely means the customer has no such account
- Found class imbalance in the target: 70% good vs. 30% bad
- Visualized age distribution and credit amount by risk group using histograms and boxplots

### 2. Outlier Detection
- Used the IQR (Interquartile Range) method on `Credit amount`
- Found **72 outliers** (customers with unusually high credit amounts)
- Decided to keep them (they represent real customers) and handle their impact later with a robust scaler instead of removing them

### 3. Feature Engineering
- **Label Encoding** for binary/ordinal columns: `Sex`, `Risk`, `Saving accounts`, `Checking account`
- **One-Hot Encoding** for nominal columns: `Housing`, `Purpose`
- Split data into training (80%) and test (20%) sets, stratified by `Risk` to preserve the class balance
- Scaled features with `RobustScaler` (chosen over `StandardScaler` because it's less sensitive to outliers)

### 4. Modeling
- Trained a baseline **SVM (SVC)** with `class_weight="balanced"` to account for class imbalance
- Tuned hyperparameters with **GridSearchCV** (`C`, `kernel`, `gamma`), optimizing for F1-score
- Best parameters found: `C=10, kernel='rbf', gamma='auto'`

## Results

| Metric (bad class) | Baseline SVM | Tuned SVM |
|---|---|---|
| Precision | 0.50 | 0.51 |
| Recall | 0.67 | 0.70 |
| F1-score | 0.57 | 0.59 |

The tuned model reduced the number of risky customers wrongly classified as "good" from 20 to 18 — the most costly type of error for a bank.

## Tools Used

`Python`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn` (SVC, GridSearchCV, RobustScaler)

## Possible Next Steps

- Try `SMOTE` to better balance the classes
- Compare SVM against Random Forest or Logistic Regression
- Add more engineered features (e.g., credit amount per month of duration)# german-credit-svm
