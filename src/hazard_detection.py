from sklearn.cluster import KMeans
import joblib

def train_kmeans(vectors, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(vectors)
    joblib.dump(kmeans, 'models/hazard_kmeans.pkl')
    return kmeans