# src/data_preprocessing.py
import logging
import spacy
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO)

def load_spacy_model(model_name="en_core_web_sm"):
    try:
        nlp = spacy.load(model_name)
        logging.info(f"SpaCy model '{model_name}' loaded successfully.")
        return nlp
    except OSError as e:
        logging.error(f"SpaCy model '{model_name}' not found. Run 'make spacy-model' to download it.")
        raise e

nlp = load_spacy_model()

def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(tokens)

def preprocess_reports(df):
    logging.info("Preprocessing reports...")
    df['cleaned_reports'] = df['report_text'].apply(preprocess)
    return df

def vectorize_text(df):
    logging.info("Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    vectors = vectorizer.fit_transform(df['cleaned_reports'])
    joblib.dump(vectorizer, 'models/vectorizer.pkl')
    logging.info("Vectorizer saved to 'models/vectorizer.pkl'.")
    return vectors, vectorizer

def main(input_csv='data/Synthetic_FEMA_Hazard_Reports.csv', output_csv='data/processed_data.csv'):
    logging.info("Starting preprocessing pipeline...")
    df = pd.read_csv(input_csv)
    df = preprocess_reports(df)
    vectors, vectorizer = vectorize_text(df)

    logging.info(f"Saving processed data to '{output_csv}'...")
    df.to_csv(output_csv, index=False)
    logging.info("Preprocessing complete.")

if __name__ == '__main__':
    main()