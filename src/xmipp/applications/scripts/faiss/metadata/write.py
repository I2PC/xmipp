import starfile
import pandas as pd

def write(data: pd.DataFrame, path: str, overwrite=True):
    return starfile.write(data, path, overwrite=overwrite)