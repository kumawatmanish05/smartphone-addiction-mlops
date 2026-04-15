def create_features(df):
    # Convert stress_level
    df['stress_level'] = df['stress_level'].map({
        'Low': 3,
        'Medium': 6,
        'High': 9
    })

    df['total_usage'] = (
        df['social_media_hours'] +
        df['gaming_hours'] +
        df['work_study_hours']
    )

    df['usage_efficiency'] = df['work_study_hours'] / (df['daily_screen_time_hours'] + 1)

    df['high_stress_flag'] = (df['stress_level'] > 7).astype(int)

    # 🔥 NEW (VERY IMPORTANT)
    df['addiction_score'] = (
        df['daily_screen_time_hours'] * 2 +
        df['social_media_hours'] * 1.5 +
        df['gaming_hours'] * 1.2 -
        df['sleep_hours'] * 1.5
    )

    df['engagement_intensity'] = df['app_opens_per_day'] / (df['daily_screen_time_hours'] + 1)

    df['notification_pressure'] = df['notifications_per_day'] / (df['daily_screen_time_hours'] + 1)

    return df