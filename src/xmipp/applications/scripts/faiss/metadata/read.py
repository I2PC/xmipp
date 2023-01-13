import starfile
import pandas as pd

def read(path: str) -> pd.DataFrame:
    return starfile.read(path)