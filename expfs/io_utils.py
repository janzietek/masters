import os
import pandas as pd

def append_csv(path: str, df: pd.DataFrame):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)
