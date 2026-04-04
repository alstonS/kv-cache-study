import os
import pandas as pd

def append_result(csv_path: str, row: dict):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    df = pd.DataFrame([row])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)
