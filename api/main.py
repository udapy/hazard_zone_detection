# api/main.py
from fastapi import FastAPI
import joblib
from pydantic import BaseModel
from src.data_preprocessing import preprocess

app = FastAPI()

model = joblib.load('models/hazard_kmeans.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

class Report(BaseModel):
    text: str

@app.post("/predict")
def predict_zone(report: Report):
    processed_text = preprocess(report.text)
    vector = vectorizer.transform([processed_text])
    cluster = model.predict(vector)[0]
    return {"zone": int(cluster)}