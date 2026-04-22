# Recommendation System

![Language](https://img.shields.io/badge/Language-Python-blue)
![Tests](https://img.shields.io/badge/Tests-11%20passing-green)
![Algorithms](https://img.shields.io/badge/Algorithms-3-orange)

Implementation of 3 recommendation algorithms from scratch:
User-Based CF, Item-Based CF, and Matrix Factorization with bias terms.

## Algorithms

| Algorithm | Type | RMSE |
|-----------|------|------|
| User-Based CF | Memory-Based | - |
| Item-Based CF | Memory-Based | - |
| Matrix Factorization | Model-Based | 0.33 |

## Demo Output

    Ratings Matrix (0 = not rated):
              Matrix    Inception  Titanic   Avengers  Joker
    Alice     5         3          *         1         4
    Bob       4         *          4         1         2
    Carol     1         1          *         5         4

    User-Based CF  -> Titanic: 4.00
    Item-Based CF  -> Titanic: 2.79
    Matrix Factor  -> Titanic: 4.15  (RMSE: 0.3308)

## Quick Start

    git clone https://github.com/0mohamed123/recommendation-system.git
    cd recommendation-system
    pip install numpy scikit-learn

    cd src
    python recommender.py

    cd ../tests
    python -m pytest test_recommender.py -v

## Key Design Decisions

- User-Based CF: cosine similarity between users
- Item-Based CF: cosine similarity between items
- Matrix Factorization: SGD with bias terms (mu, bu, bi)
- Predictions clipped to valid rating range (1-5)
- CosineAnnealing scheduler equivalent via manual SGD

## Test Results

    11 passed | 0 failed

    Tests cover: model fitting, similarity matrix,
    prediction range, loss decrease, RMSE threshold,
    recommendation ordering

## Technologies

- Python 3.12
- NumPy (core computations)
- scikit-learn (cosine similarity)
- pytest (11 tests)