import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def feature_importance(df: pd.DataFrame, target: str) -> pd.Series:
    """
    Returns feature importance using Random Forest classifier.

    Parameters:
        df (pd.DataFrame): The dataframe with features and target
        target (str): Name of the target column (e.g., 'Survived')

    Returns:
        pd.Series: Features sorted by importance
    """
    X = df.drop(columns=[target])
    y = df[target]

    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Feature importance
    importance = pd.Series(model.feature_importances_, index=X.columns)
    importance = importance.sort_values(ascending=False)
    return importance