# src/inference.py

import joblib
import sys
import logging
from data_preprocessing import preprocess

logging.basicConfig(level=logging.INFO)

def load_model_and_vectorizer(model_path='models/hazard_kmeans.pkl', vectorizer_path='models/vectorizer.pkl'):
    logging.info("Loading model and vectorizer...")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def predict_hazard_zone(report_text, model, vectorizer):
    logging.info("Preprocessing input text...")
    processed_text = preprocess(report_text)
    vector = vectorizer.transform([processed_text])
    logging.info("Predicting hazard zone...")
    cluster = model.predict(vector)[0]
    return cluster

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logging.error("Usage: python inference.py \"Your hazard report text here\"")
        sys.exit(1)

    report_text = sys.argv[1]
    model, vectorizer = load_model_and_vectorizer()
    cluster = predict_hazard_zone(report_text, model, vectorizer)
    
    logging.info(f"The predicted hazard zone is: {cluster}")