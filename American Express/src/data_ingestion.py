import pandas as pd

def load_amex_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df