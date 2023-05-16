"""
This file contains an assortment
of helper functions being used
throughout the project.
"""

import os
import numpy as np
import pandas as pd
from settings import CELEBA_ATTRS_CSV

def save_to_csv(csv_file_path:str, file_names:list[str],
                save_celeb_attrs:bool=True, **kwargs):
    """
    Save the given data in kwargs
    to a csv file. This function will
    overwrite any existing file at that
    path. It is assumed that data in
    kwargs are flat lists or numpy arrays.

    'filename' is col 0, so data in kwargs
    is inserted in columns right after that.
    """ 
    celeba_df = pd.read_csv(CELEBA_ATTRS_CSV)
    df_for_paths = celeba_df[celeba_df['filename'].isin(file_names)]
    for col_idx, (key, list_val) in enumerate(kwargs.items()):
        df_for_paths.insert(col_idx+1, key, list_val)
    if save_celeb_attrs:
        df_for_paths.to_csv(csv_file_path)
    else:
        df_for_paths[['filename', *kwargs.keys()]].to_csv(csv_file_path)
