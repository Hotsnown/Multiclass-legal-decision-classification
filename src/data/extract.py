import xmltodict
import json
import pandas as pd
from pathlib import Path

def extract_data():
    print('Parsing xml')
    parsed_data = []
    def parse_XML(file):
        
        with open(file) as fd:
            doc = xmltodict.parse(fd.read())

        test = json.dumps(doc)
        data = json.loads(test)
        try: 
            ID = data["TEXTE_JURI_JUDI"]["META"]["META_COMMUN"]["ID"]
        except:
            ID = None
        try: 
            formation = data["TEXTE_JURI_JUDI"]["META"]["META_SPEC"]["META_JURI_JUDI"]["FORMATION"]
        except :
            formation = None
        try: 
            president = data["TEXTE_JURI_JUDI"]["META"]["META_SPEC"]["META_JURI_JUDI"]["PRESIDENT"]
        except :
            president = None
        try:
            contenu = data["TEXTE_JURI_JUDI"]["TEXTE"]["BLOC_TEXTUEL"]["CONTENU"]["#text"]
        except :
            contenu = None

        cleaned_json = json.dumps({"ID":ID, "formation":formation, "president":president, "contenu":contenu})
        parsed_cleaned_json = json.loads(cleaned_json)
        
        parsed_data.append(parsed_cleaned_json)
    
    countdown = len(list(Path('data/raw/extract').rglob('*.xml')))
    for file in Path('data/raw/extract').rglob('*.xml'):
        countdown = countdown - 1
        print('items left : ', countdown, 'parsing : ', file)
        parse_XML(file)

    print('writing parsed data to data/interim/output.csv')
    to_save_data = pd.DataFrame(parsed_data)
    to_save_data.to_csv('data/interim/output.csv', index=False)

extract_data()