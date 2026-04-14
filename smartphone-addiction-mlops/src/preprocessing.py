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