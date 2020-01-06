import pandas as pd
import numpy as np

def wrangle_data():
    print("Removing None values")
    data = pd.read_csv('data/interim/output.csv')

    data = data.replace(to_replace='None', value=np.nan).dropna()
    data.isna().contenu.value_counts()

    print('Writing cleaned data to data/processed/output.csv')
    to_save_data = pd.DataFrame(data)
    to_save_data.to_csv('data/processed/output.csv', index=False)

wrangle_data()