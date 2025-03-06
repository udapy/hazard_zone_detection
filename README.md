# Hazard Zone Detection and Recommendation System

## Project Overview
This project leverages unsupervised machine learning techniques to automatically detect hazard zones based on user-generated textual reports. Additionally, it provides a content-based recommender system to suggest related hazard reports. The solution employs a comprehensive end-to-end pipeline, including data preprocessing, model training, experiment tracking, and deployment.

## Key Objectives
- **Unsupervised Clustering:** Automatically identify and group hazard reports into hazard zones.
- **Content-based Recommendation:** Recommend similar hazard reports based on textual content.
- **Scalable Deployment:** Utilize FastAPI and AWS SageMaker to deploy a scalable and performant RESTful API.
- **Experiment Tracking:** Track model experiments using MLflow for reproducibility.

## Technologies
- **Programming Language:** Python
- **Package Management:** uv
- **Data Processing:** Pandas, spaCy, NLTK
- **Machine Learning Frameworks:** Scikit-learn, TensorFlow, TensorFlow Hub
- **Experiment Tracking:** MLflow
- **Deployment:** FastAPI, Docker, AWS SageMaker

## Project Structure
```
hazard_zone_detection/
├── data/
│   ├── raw_reports.csv
│   └── processed_data.csv
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── model_experiments.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── hazard_detection.py
│   ├── recommender_system.py
│   ├── train.py
│   └── inference.py
├── models/
│   └── model.pkl
├── api/
│   └── main.py
├── Dockerfile
├── pyproject.toml
├── pre-commit-config.yaml
├── Makefile
└── mlflow/
    └── tracking
```

## How to Set Up and Run the Project

### 1. Setup Environment with `uv`
Install dependencies using `uv`:
```bash
uv venv create env
source env/bin/activate
uv pip install -r requirements.txt
make spacy-model
```

### 2. Data Preprocessing
Run preprocessing scripts to clean, tokenize, and vectorize textual data:
```bash
make preprocess
```

### Data Overview
The dataset contains simulated hazard reports with fields:
- `report_id`
- `disaster_type`
- `location`
- `latitude`, `longitude`
- `report_timestamp`
- `textual description`

## Training Models
Run training scripts for hazard clustering:
```bash
make train
```

### Hazard Detection
Uses clustering methods (e.g., KMeans, DBSCAN) to detect hazard zones from report embeddings.

### Content-based Recommendations
Uses cosine similarity to recommend hazard reports with similar textual content.

## Inference
Run inference script for new hazard reports:
```bash
make inference text="New hazard report description here"
```

## Experiment Tracking with MLflow
To track model parameters and metrics, MLflow is configured:
- Start MLflow tracking:
```bash
uv run mlflow ui
```
- Access MLflow experiments at `http://localhost:5000`.

## Deployment with FastAPI
The trained model is deployed via FastAPI. To run locally:
```bash
make serve
```

API endpoint for hazard predictions:
```http
POST /predict
Body:
{
  "text": "Reported hazard description here"
}
```

### Querying the FastAPI Service

Using `curl`:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Severe flooding reported near Miami."}'
```

Using HTTPie:
```bash
http POST http://localhost:8000/predict text="Severe flooding reported near Miami."
```

## Containerization and AWS Deployment
- Build the Docker container:
```bash
docker build -t hazard-api:latest .
aws ecr get-login-password --region <your-region> | docker login --username AWS --password-stdin <your-account-id>.dkr.ecr.<your-region>.amazonaws.com

docker tag hazard-api:latest <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/hazard-api:latest

docker push <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/hazard-api:latest
```
- Push Docker image to AWS ECR.
- Deploy to AWS SageMaker as an endpoint for scalable inference.
  - Use the AWS console or AWS CLI to create an inference endpoint pointing to this image.

## CI/CD and MLOps
Continuous integration and deployment can be set up using GitHub Actions for automated testing, building, and deployment to AWS SageMaker. Utilize AWS CloudWatch for monitoring performance and logging.

## Local Development and Pre-commit Hooks
Set up pre-commit hooks to maintain code quality:
```bash
uv pip install pre-commit
pre-commit install
```

### Makefile Targets
Use the provided Makefile for convenience:
```bash
make setup          # Sets up environment and dependencies
make lint           # Runs pre-commit hooks and linting
make spacy-model    # Downloads spaCy model
make preprocess     # Runs data preprocessing
make train          # Trains the model
make serve          # Runs local API server
make inference text="Your text here"  # Runs inference with specified text
```


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributions
Contributions are welcome! Please open an issue first to discuss changes or improvements.

