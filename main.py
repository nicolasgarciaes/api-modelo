from fastapi import FastAPI
from pydantic import BaseModel
import fasttext
import pandas as pd
import unicodedata
import re

app = FastAPI()

model = fasttext.load_model('fasttext_ext.bin')

def preprocess_string(text, filters):
    for f in filters:
        text = f(text)
    return text

def strip_non_alphanum(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def strip_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

def strip_multiple_whitespaces(text):
    return re.sub(r'\s+', ' ', text).strip()

CUSTOM_FILTERS = [
    lambda x: x.lower(),
    lambda x: unicodedata.normalize('NFKD', x).encode('ASCII', 'ignore').decode('utf-8'),
    strip_non_alphanum, 
    strip_punctuation, 
    strip_multiple_whitespaces
]

data_path = 'data.xlsx'
prov_path = 'provision.xlsx'

data = pd.read_excel(data_path)
prov = pd.read_excel(prov_path)

prov = prov.rename(columns={'specialty': 'provision_specialty', 'name': 'provision'})
data_code = pd.merge(data, prov, how='left', on=['provision', 'provision_specialty'])
data_code = data_code.drop(['id', 'dm_id'], axis=1)  

data_code['provider_provision'] = data_code['provider_provision'].apply(lambda x: preprocess_string(x, CUSTOM_FILTERS))
data_code['provider_provision_specialty'] = data_code['provider_provision_specialty'].apply(lambda x: preprocess_string(x, CUSTOM_FILTERS))

data_code['code'] = '__label__' + data_code['code'].astype(str)
data_code['provider_provision'] = data_code['code'] + ' ' + data_code['provider_provision'] + ' ' + data_code['provider_provision_specialty']

label_mapping = dict(zip(data_code['code'], data_code['provider_provision']))

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input_data: TextInput):
    labels, probabilities = model.predict(input_data.text, k=3)
    
    readable_labels = [label_mapping.get(label, label) for label in labels]
    
    response = {
        "predictions": [
            {"label": readable_label, "confidence": prob}
            for readable_label, prob in zip(readable_labels, probabilities)
        ]
    }
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
