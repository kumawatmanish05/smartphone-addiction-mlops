import pandas as pd
from pathlib import Path

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load raw dataset from given file path
    """
    try:
        df = pd.read_csv(file_path)
        print("✅ Data loaded successfully")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise



def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop useless columns
    df = df.drop(columns=['transaction_id', 'user_id'])

    # Fill addiction_level (example logic)
    df['addiction_level'] = df['addiction_level'].fillna(
        df['addicted_label'].map({
            0: 'Mild',
            1: 'Severe'
        })
    )

    return df

