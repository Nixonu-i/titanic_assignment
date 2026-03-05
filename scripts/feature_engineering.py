# scripts/feature_engineering.py

import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates derived features for Titanic dataset:
    - FamilySize, IsAlone
    - Title from Name
    - AgeGroup
    - Fare per person
    """

    # Family Size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # Is Alone
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Title extraction
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    # Age groups
    def age_group(age):
        if age < 13: return 'Child'
        elif age < 20: return 'Teen'
        elif age < 60: return 'Adult'
        else: return 'Senior'
    df['AgeGroup'] = df['Age'].apply(age_group)

    # Fare per person
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']

    # Optional: One-hot encode simple nominal features here (or leave for notebook)
    return df