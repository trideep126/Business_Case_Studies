import pandas as pd
import numpy as np

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    #Create new features
    df['distance_risk'] = np.where(df['distance_from_home'] > df['distance_from_home'].quantile(0.75),1,0)
    df['high_value'] = np.where(df['ratio_to_median_purchase_price']>3,1,0)
    df['suspicious_pattern'] = (
        (df['online_order'] == 1) &
        (df['repeat_retailer'] == 0) &
        (df['high_value'] == 1)
    ).astype(int)
    df['security_score'] = df['used_chip'] + df['used_pin_number']
    df['high_velocity'] = np.where(df['distance_from_last_transaction'] > df['distance_from_last_transaction'].quantile(0.9),1,0)

    return df 
