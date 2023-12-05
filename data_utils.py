import numpy as np
import pandas as pd

def load_diabetes_dataset(data_path):
    return pd.read_csv(data_path)

def get_data_stats(dataset):
    return dataset.shape, dataset.describe(), dataset['Outcome'].value_counts(), dataset.groupby('Outcome').mean()
