import pytest
from src.inference import predict_hazard_zone, load_model_and_vectorizer

@pytest.fixture
def model_and_vectorizer():
    return load_model_and_vectorizer()

def test_predict_hazard_zone(model_and_vectorizer):
    model, vectorizer = model_and_vectorizer
    report_text = "Severe flooding around New Orleans."
    cluster = predict_hazard_zone(report_text, model, vectorizer)
    assert isinstance(cluster, int)