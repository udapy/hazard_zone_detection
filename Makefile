.PHONY: setup preprocess train serve lint test

setup:
	uv venv hazards-env
	source hazards-env/bin/activate && uv pip install -r requirements.txt
	make spacy

spacy:
	uv venv --seed
	.venv\Scripts\activate
	pip install uv
	uv run pip install -U pip setuptools wheel
	uv run pip install -U spacy
	uv run python -m spacy download en_core_web_sm

preprocess:
	uv run python src/data_preprocessing.py

train:
	uv run python src/train.py

serve:
	uv run uvicorn api.main:app --reload

recommend:
	uv run python src/recommender_system.py

lint:
	uv run pre-commit run --all-files

test:
	uv run pytest tests/