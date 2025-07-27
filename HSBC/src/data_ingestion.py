import pandas as pd

def load_hsbc_data(path:str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df 