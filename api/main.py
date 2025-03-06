# api/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from src.recommender_system import HazardRecommender

app = FastAPI()
recommender = HazardRecommender()

class RecommendationRequest(BaseModel):
    report_id: int
    top_n: int = 5

@app.post("/predict")
def recommend(request: RecommendationRequest):
    recommendations = recommender.recommend_similar_reports(request.report_id, request.top_n)
    return recommendations.to_dict(orient='records')