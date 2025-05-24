import pandas as pd

def load_excel(file) -> pd.DataFrame:
    return pd.read_excel(file)
