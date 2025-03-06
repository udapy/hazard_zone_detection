import pytest
from src.train import train_model

def test_train_model():
    # Basic smoke test to ensure no exceptions are raised
    train_model(n_clusters=2)