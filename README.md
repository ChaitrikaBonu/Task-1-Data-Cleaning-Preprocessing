# Task 1: Data Cleaning & Preprocessing

## Objective
To clean and preprocess the Titanic dataset for Machine Learning applications.

## Dataset
Titanic Dataset

## Tools Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

## Steps Performed

### 1. Data Exploration
- Loaded the dataset using Pandas.
- Checked dataset shape, columns, and data types.
- Identified missing values.

### 2. Handling Missing Values
- Filled missing Age values using median.
- Filled missing Embarked values using mode.
- Replaced missing Cabin values with "Unknown".

### 3. Encoding Categorical Variables
- Converted Sex and Embarked columns into numerical values using Label Encoding.

### 4. Outlier Detection and Removal
- Visualized outliers using boxplots.
- Removed outliers using the IQR method.

### 5. Feature Scaling
- Standardized Age and Fare columns using StandardScaler.

### 6. Data Visualization
- Created boxplots for outlier detection.
- Generated a correlation heatmap.

### 7. Final Output
- Saved the cleaned dataset as cleaned_titanic.csv.

## Files Included
- Titanic-Dataset.csv
- cleaned_titanic.csv
- task1.py
- README.md

## Outcome
Successfully cleaned and preprocessed the dataset, making it suitable for Machine Learning models.