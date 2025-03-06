# src/train.py

import joblib
import logging
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import mlflow
import mlflow.sklearn
from data_preprocessing import preprocess_reports, vectorize_text
from mlflow.models.signature import infer_signature

logging.basicConfig(level=logging.INFO)

def train_model(n_clusters=5):
    logging.info("Loading dataset...")
    df = pd.read_csv('data/processed_data.csv')

    logging.info("Vectorizing text...")
    vectors, vectorizer = vectorize_text(df)

    logging.info(f"Training KMeans model with {n_clusters} clusters...")
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(vectors)

    logging.info("Saving model and vectorizer...")
    joblib.dump(model, 'models/hazard_kmeans.pkl')
    joblib.dump(vectorizer, 'models/vectorizer.pkl')

    logging.info("Logging experiment with MLflow...")
    mlflow.set_experiment("Hazard Zone Detection")

    # Prepare input example and signature
    input_example = vectors[0].toarray()
    signature = infer_signature(vectors, model.predict(vectors))

    with mlflow.start_run():
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="kmeans-model",
            signature=signature,
            input_example=input_example
        )
        mlflow.log_metric("inertia", model.inertia_)

if __name__ == '__main__':
    train_model(n_clusters=10)