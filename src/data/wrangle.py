import pandas as pd
import numpy as np

def wrangle_data():
    data = pd.read_csv('data/interim/output.csv')
    print("Removing None values")
    data = data.replace(to_replace='None', value=np.nan).dropna()
    print(data.isna().contenu.value_counts())
    
    print('Removing unwanted formations')
    unwanted_formations = ['COMMISSION_REPARATION_DETENTION', 'CHAMBRE_CIVILE', 'ASSEMBLEE_PLENIERE', 'AVIS', 'COUR_REVISION', 'CHAMBRES_REUNIES', 'CHAMBRE_MIXTE', 'COMMISSION_REVISION']
    for unwanted_formation in unwanted_formations:
        print('dropping:', unwanted_formation)
        data = data.drop(data[data['formation'] == unwanted_formation].index)
    
    print('Writing cleaned data to data/processed/output.csv')
    to_save_data = pd.DataFrame(data)
    to_save_data.to_csv('data/processed/output.csv', index=False)

wrangle_data()