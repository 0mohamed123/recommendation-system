import numpy as np
from collaborative_filtering import UserBasedCF, ItemBasedCF
from matrix_factorization import MatrixFactorization


def generate_sample_data():
    np.random.seed(42)
    ratings = np.array([
        [5, 3, 0, 1, 4],
        [4, 0, 4, 1, 2],
        [1, 1, 0, 5, 4],
        [0, 0, 4, 4, 0],
        [2, 1, 5, 4, 0],
        [0, 3, 4, 0, 3],
    ], dtype=float)
    
    users = ['Alice', 'Bob', 'Carol', 'Dave', 'Eve', 'Frank']
    items = ['Matrix', 'Inception', 'Titanic', 'Avengers', 'Joker']
    
    return ratings, users, items


def run_demo():
    ratings, users, items = generate_sample_data()

    print("=" * 55)
    print("   Recommendation System Demo")
    print("=" * 55)

    print("\nRatings Matrix (0 = not rated):")
    print(f"{'':10s}", end="")
    for item in items:
        print(f"{item:12s}", end="")
    print()
    for i, user in enumerate(users):
        print(f"{user:10s}", end="")
        for r in ratings[i]:
            print(f"{'*' if r == 0 else str(int(r)):12s}", end="")
        print()

    print("\n--- User-Based CF ---")
    ucf = UserBasedCF(k=3)
    ucf.fit(ratings)
    recs = ucf.recommend(0, n=3)
    print(f"Recommendations for {users[0]}:")
    for item_idx, score in recs:
        print(f"  {items[item_idx]:12s}: {score:.2f}")

    print("\n--- Item-Based CF ---")
    icf = ItemBasedCF(k=3)
    icf.fit(ratings)
    recs = icf.recommend(0, n=3)
    print(f"Recommendations for {users[0]}:")
    for item_idx, score in recs:
        print(f"  {items[item_idx]:12s}: {score:.2f}")

    print("\n--- Matrix Factorization ---")
    mf = MatrixFactorization(n_factors=3, n_epochs=100)
    mf.fit(ratings)
    recs = mf.recommend(0, n=3)
    print(f"Recommendations for {users[0]}:")
    for item_idx, score in recs:
        print(f"  {items[item_idx]:12s}: {score:.2f}")
    print(f"RMSE: {mf.rmse():.4f}")
    print(f"Loss decreased: {mf.loss_history[0] > mf.loss_history[-1]}")
    print("=" * 55)


if __name__ == '__main__':
    run_demo()