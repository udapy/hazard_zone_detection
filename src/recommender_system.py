# src/recommender_system.py

import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO)

class HazardRecommender:
    def __init__(self, vectorizer_path='models/vectorizer.pkl', data_path='data/processed_data.csv'):
        logging.info("Loading vectorizer and processed data...")
        self.vectorizer = joblib.load(vectorizer_path)
        self.data = pd.read_csv(data_path)
        self.vectors = self.vectorizer.transform(self.data['cleaned_reports'])
        self.similarity_matrix = cosine_similarity(self.vectors)

    def recommend_similar_reports(self, report_id, top_n=5):
        logging.info(f"Fetching recommendations for report_id={report_id}")
        report_index = self.data[self.data['report_id'] == report_id].index[0]
        similarity_scores = list(enumerate(self.similarity_matrix[report_index]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        similar_indices = [idx for idx, _ in similarity_scores]
        return self.data.iloc[similar_indices][['report_id', 'report_text', 'disaster_type', 'location']]

if __name__ == '__main__':
    recommender = HazardRecommender()
    recommendations = recommender.recommend_similar_reports(report_id=10, top_n=3)
    print(recommendations)