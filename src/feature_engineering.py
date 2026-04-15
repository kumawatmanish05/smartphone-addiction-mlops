import pandas as pd

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df['total_usage'] = (
        df['social_media_hours'] +
        df['gaming_hours'] +
        df['work_study_hours']
    )

    df['usage_efficiency'] = df['work_study_hours'] / (df['daily_screen_time_hours'] + 1)

    df['high_stress_flag'] = (df['stress_level'] == 'High').astype(int)

    # 🔥 NEW FEATURES
    df['social_ratio'] = df['social_media_hours'] / (df['daily_screen_time_hours'] + 1)
    df['gaming_ratio'] = df['gaming_hours'] / (df['daily_screen_time_hours'] + 1)
    df['sleep_deficit'] = 8 - df['sleep_hours']
    df['weekend_spike'] = df['weekend_screen_time'] - df['daily_screen_time_hours']

    return df