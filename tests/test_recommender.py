import sys
sys.path.append('../src')

import numpy as np
import pytest
from collaborative_filtering import UserBasedCF, ItemBasedCF
from matrix_factorization import MatrixFactorization


@pytest.fixture
def ratings():
    return np.array([
        [5, 3, 0, 1, 4],
        [4, 0, 4, 1, 2],
        [1, 1, 0, 5, 4],
        [0, 0, 4, 4, 0],
        [2, 1, 5, 4, 0],
        [0, 3, 4, 0, 3],
    ], dtype=float)


# ===== User-Based CF =====
def test_ucf_fit(ratings):
    model = UserBasedCF(k=3)
    model.fit(ratings)
    assert model.user_similarity is not None
    assert model.user_similarity.shape == (6, 6)


def test_ucf_similarity_diagonal(ratings):
    model = UserBasedCF(k=3)
    model.fit(ratings)
    assert np.all(np.diag(model.user_similarity) == 0)


def test_ucf_predict(ratings):
    model = UserBasedCF(k=3)
    model.fit(ratings)
    pred = model.predict(0, 2)
    assert pred >= 0


def test_ucf_recommend(ratings):
    model = UserBasedCF(k=3)
    model.fit(ratings)
    recs = model.recommend(0, n=3)
    assert len(recs) <= 3
    scores = [r[1] for r in recs]
    assert scores == sorted(scores, reverse=True)


# ===== Item-Based CF =====
def test_icf_fit(ratings):
    model = ItemBasedCF(k=3)
    model.fit(ratings)
    assert model.item_similarity.shape == (5, 5)


def test_icf_recommend(ratings):
    model = ItemBasedCF(k=3)
    model.fit(ratings)
    recs = model.recommend(0, n=3)
    assert len(recs) <= 3


# ===== Matrix Factorization =====
def test_mf_fit(ratings):
    model = MatrixFactorization(n_factors=3, n_epochs=10)
    model.fit(ratings)
    assert model.P.shape == (6, 3)
    assert model.Q.shape == (5, 3)


def test_mf_predict_range(ratings):
    model = MatrixFactorization(n_factors=3, n_epochs=10)
    model.fit(ratings)
    for u in range(6):
        for i in range(5):
            pred = model.predict(u, i)
            assert 1 <= pred <= 5


def test_mf_loss_decreases(ratings):
    model = MatrixFactorization(n_factors=3, n_epochs=50)
    model.fit(ratings)
    assert model.loss_history[0] > model.loss_history[-1]


def test_mf_rmse(ratings):
    model = MatrixFactorization(n_factors=3, n_epochs=100)
    model.fit(ratings)
    assert model.rmse() < 1.0


def test_mf_recommend(ratings):
    model = MatrixFactorization(n_factors=3, n_epochs=10)
    model.fit(ratings)
    recs = model.recommend(0, n=3)
    assert len(recs) <= 3