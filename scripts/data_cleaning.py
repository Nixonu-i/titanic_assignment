# scripts/data_cleaning.py

import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the Titanic dataset:
    - Handles missing values
    - Standardizes categories
    - Removes duplicates
    - Caps outliers
    """

    # --- Missing Values ---
    # Age
    df['Age_missing'] = df['Age'].isnull().astype(int)
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # Cabin → Deck
    df['Deck'] = df['Cabin'].str[0].fillna('U')

    # Embarked
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Fare
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # --- Outliers ---
    # Cap Fare at 99th percentile
    fare_cap = df['Fare'].quantile(0.99)
    df['Fare'] = df['Fare'].apply(lambda x: fare_cap if x > fare_cap else x)

    # --- Consistency ---
    df['Sex'] = df['Sex'].str.lower().replace({'male':'male','female':'female'})

    # --- Remove duplicates ---
    df.drop_duplicates(inplace=True)

    return df