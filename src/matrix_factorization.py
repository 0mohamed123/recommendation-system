import numpy as np


class MatrixFactorization:
    def __init__(self, n_factors=10, lr=0.01, reg=0.01, n_epochs=50):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.loss_history = []

    def fit(self, ratings_matrix):
        self.R = np.array(ratings_matrix, dtype=float)
        n_users, n_items = self.R.shape

        np.random.seed(42)
        self.P = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)
        self.mu = np.mean(self.R[self.R > 0])

        for epoch in range(self.n_epochs):
            loss = 0
            for u in range(n_users):
                for i in range(n_items):
                    if self.R[u][i] > 0:
                        pred = self._predict(u, i)
                        err = self.R[u][i] - pred
                        loss += err ** 2

                        self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                        self.bi[i] += self.lr * (err - self.reg * self.bi[i])

                        p_update = self.lr * (err * self.Q[i] - self.reg * self.P[u])
                        q_update = self.lr * (err * self.P[u] - self.reg * self.Q[i])
                        self.P[u] += p_update
                        self.Q[i] += q_update

            self.loss_history.append(loss)

        return self

    def _predict(self, u, i):
        return self.mu + self.bu[u] + self.bi[i] + self.P[u].dot(self.Q[i])

    def predict(self, user_idx, item_idx):
        return np.clip(self._predict(user_idx, item_idx), 1, 5)

    def recommend(self, user_idx, n=5):
        user_ratings = self.R[user_idx]
        unrated = np.where(user_ratings == 0)[0]

        predictions = [(i, self.predict(user_idx, i)) for i in unrated]
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]

    def rmse(self):
        errors = []
        for u in range(self.R.shape[0]):
            for i in range(self.R.shape[1]):
                if self.R[u][i] > 0:
                    errors.append((self.R[u][i] - self._predict(u, i)) ** 2)
        return np.sqrt(np.mean(errors))