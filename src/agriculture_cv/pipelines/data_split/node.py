import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(df: pd.DataFrame):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    test_df, val_df = train_test_split(test_df, test_size=0.2, random_state=42)
    return train_df, test_df, val_df
