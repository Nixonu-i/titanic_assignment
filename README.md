# Titanic Survival Prediction - Machine Learning Project

This project implements a machine learning pipeline to predict passenger survival on the Titanic using Python, pandas, and scikit-learn.

## 📋 Project Overview

The goal is to build a predictive model that determines whether a passenger survived the Titanic disaster based on features like age, gender, ticket class, fare, and other attributes.

## 📁 Project Structure

```
titanic_assignment/
├── data/
│   ├── train.csv              # Training dataset (891 passengers)
│   ├── test.csv               # Test dataset for predictions
│   ├── train_features.csv     # Processed training features
│   └── train_selected.csv     # Selected features after feature selection
├── scripts/
│   ├── data_cleaning.py       # Handle missing values, outliers, duplicates
│   ├── feature_engineering.py # Create derived features (FamilySize, Title, etc.)
│   └── feature_selection.py   # Feature importance analysis using Random Forest
├── notebooks/
│   └── Titanic_Feature_Engineering.ipynb  # Jupyter notebook for data exploration and analysis
├── jupyter_env/               # Python virtual environment for Jupyter
├── requirements.txt           # Project dependencies
└── README.md                  # This file
```

## 🚀 Features Implemented

### Data Cleaning (`scripts/data_cleaning.py`)
- **Missing Value Handling**:
  - Age: Imputed with median + created `Age_missing` indicator
  - Cabin: Extracted Deck from cabin number
  - Embarked: Filled with mode value
  - Fare: Imputed with median
- **Outlier Management**: Capped Fare at 99th percentile
- **Data Quality**: Removed duplicates and standardized categories

### Feature Engineering (`scripts/feature_engineering.py`)
- **FamilySize**: Total family members (SibSp + Parch + 1)
- **IsAlone**: Binary flag for passengers traveling alone
- **Title**: Extracted from names (Mr, Mrs, Miss, Master, etc.)
- **AgeGroup**: Categorized into Child, Teen, Adult, Senior
- **FarePerPerson**: Fare divided by family size

### Feature Selection (`scripts/feature_selection.py`)
- Random Forest-based feature importance ranking
- Identifies most predictive features for survival

## 🛠️ Installation

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### Setup Instructions

1. **Navigate to project directory**:
   ```bash
   cd titanic_assignment
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Set up virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   # or
   .\venv\Scripts\activate  # On Windows
   pip install -r requirements.txt
   ```

## 📊 Usage

### Using the Scripts

```python
import pandas as pd
from scripts.data_cleaning import clean_data
from scripts.feature_engineering import engineer_features
from scripts.feature_selection import feature_importance

# Load data
df = pd.read_csv('data/train.csv')

# Clean data
df_clean = clean_data(df)

# Engineer features
df_features = engineer_features(df_clean)

# Analyze feature importance
importance = feature_importance(df_features, 'Survived')
print(importance.sort_values(ascending=False))
```

### Using the Jupyter Notebook

Open the notebook for interactive exploration:
```bash
jupyter notebook notebooks/Titanic_Feature_Engineering.ipynb
```

## 📈 Dataset Information

The Titanic dataset contains the following key features:
- **PassengerId**: Unique identifier
- **Survived**: Target variable (0 = No, 1 = Yes)
- **Pclass**: Ticket class (1, 2, 3)
- **Name**: Passenger name (contains title)
- **Sex**: Gender
- **Age**: Age in years
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Ticket**: Ticket number
- **Fare**: Passenger fare
- **Cabin**: Cabin number
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## 🔧 Pipeline Workflow

1. **Load Data** → Read CSV files
2. **Clean Data** → Handle missing values and outliers
3. **Engineer Features** → Create new predictive features
4. **Encode Categoricals** → Convert text to numerical values
5. **Select Features** → Identify most important predictors
6. **Train Model** → Build classification model
7. **Evaluate** → Assess model performance
8. **Predict** → Generate predictions on test data

## 📝 Next Steps

- Train and evaluate multiple ML models (Logistic Regression, Random Forest, XGBoost)
- Perform hyperparameter tuning
- Create ensemble models
- Generate final predictions for test dataset
- Analyze model performance metrics (accuracy, precision, recall, F1-score)

## 🤝 Contributing

This is an educational project. Feel free to fork and improve!

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- Dataset source: Kaggle Titanic Competition
- Virtual environment: Jupyter Lab setup in `jupyter_env/`
