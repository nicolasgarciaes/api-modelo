from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import fasttext
import pandas as pd
import unicodedata
import re
from gensim.parsing.preprocessing import strip_non_alphanum, strip_punctuation, strip_multiple_whitespaces

app = FastAPI()

API_KEY = "mysecretapikey123"
API_KEY_NAME = "Authorization"

api_key_header = APIKeyHeader(name=API_KEY_NAME)

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )
                 
models = {
    "CL": fasttext.load_model('CL_model.bin'),
    "COL": fasttext.load_model('COL_model.bin'),
    "MX": fasttext.load_model('MX_model.bin')
}

CUSTOM_FILTERS = [
    lambda x: x.lower(),
    lambda x: unicodedata.normalize('NFKD', x).encode('ASCII', 'ignore').decode('utf-8'),
    strip_non_alphanum, 
    strip_punctuation, 
    strip_multiple_whitespaces
]

def preprocess_text(text, filters):
    for f in filters:
        text = f(text)
    return text

data_path = 'data.xlsx'
prov_path = 'provision.xlsx'

data = pd.read_excel(data_path)
prov = pd.read_excel(prov_path)

prov = prov.rename(columns={'specialty': 'provision_specialty', 'name': 'provision'})
data_code = pd.merge(data, prov, how='left', on=['provision', 'provision_specialty'])
data_code = data_code.drop(['id', 'dm_id'], axis=1)

data_code['provider_provision'] = data_code['provider_provision'].apply(lambda x: preprocess_text(x, CUSTOM_FILTERS))
data_code['provider_provision_specialty'] = data_code['provider_provision_specialty'].apply(lambda x: preprocess_text(x, CUSTOM_FILTERS))

data_code['code'] = '__label__' + data_code['code'].astype(str)
data_code['provider_provision'] = data_code['code'] + ' ' + data_code['provider_provision'] + ' ' + data_code['provider_provision_specialty']

label_mapping = dict(zip(data_code['code'], data_code['provider_provision']))

class TextInput(BaseModel):
    provider_provision: str
    country_code: str

@app.post("/predict")
def predict(input_data: TextInput, api_key: str = Depends(verify_api_key)):
    if input_data.country_code not in models:
        return {"error": "Invalid country code. Must be 'CL', 'COL', or 'MX'."}
    
    model = models[input_data.country_code]

    preprocessed_text = preprocess_text(input_data.provider_provision, CUSTOM_FILTERS)
    
    labels, probabilities = model.predict(preprocessed_text, k=3)
    
    readable_labels = [label_mapping.get(label, label).split(' ', 1)[1] for label in labels]
    
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
