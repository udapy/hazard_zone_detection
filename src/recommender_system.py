from sklearn.metrics.pairwise import cosine_similarity

def recommend_similar_reports(vectors, report_index, top_n=5):
    similarity_matrix = cosine_similarity(vectors)
    similar_indices = similarity_matrix[report_index].argsort()[::-1][1:top_n+1]
    return similar_indices