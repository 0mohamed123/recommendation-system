import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class UserBasedCF:
    def __init__(self, k=5):
        self.k = k
        self.ratings = None
        self.user_similarity = None
        self.users = None
        self.items = None

    def fit(self, ratings_matrix):
        self.ratings = np.array(ratings_matrix, dtype=float)
        self.user_similarity = cosine_similarity(self.ratings)
        np.fill_diagonal(self.user_similarity, 0)
        return self

    def predict(self, user_idx, item_idx):
        sim_scores = self.user_similarity[user_idx]
        top_k = np.argsort(sim_scores)[-self.k:]
        
        numerator = 0
        denominator = 0
        for u in top_k:
            if self.ratings[u][item_idx] > 0:
                numerator += sim_scores[u] * self.ratings[u][item_idx]
                denominator += abs(sim_scores[u])
        
        return numerator / denominator if denominator > 0 else 0

    def recommend(self, user_idx, n=5):
        user_ratings = self.ratings[user_idx]
        unrated = np.where(user_ratings == 0)[0]
        
        predictions = []
        for item in unrated:
            score = self.predict(user_idx, item)
            predictions.append((item, score))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]


class ItemBasedCF:
    def __init__(self, k=5):
        self.k = k
        self.ratings = None
        self.item_similarity = None

    def fit(self, ratings_matrix):
        self.ratings = np.array(ratings_matrix, dtype=float)
        self.item_similarity = cosine_similarity(self.ratings.T)
        np.fill_diagonal(self.item_similarity, 0)
        return self

    def predict(self, user_idx, item_idx):
        sim_scores = self.item_similarity[item_idx]
        top_k = np.argsort(sim_scores)[-self.k:]
        
        numerator = 0
        denominator = 0
        for i in top_k:
            if self.ratings[user_idx][i] > 0:
                numerator += sim_scores[i] * self.ratings[user_idx][i]
                denominator += abs(sim_scores[i])
        
        return numerator / denominator if denominator > 0 else 0

    def recommend(self, user_idx, n=5):
        user_ratings = self.ratings[user_idx]
        unrated = np.where(user_ratings == 0)[0]
        
        predictions = []
        for item in unrated:
            score = self.predict(user_idx, item)
            predictions.append((item, score))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]