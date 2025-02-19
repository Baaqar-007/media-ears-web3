import numpy as np

def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02, debug=True):
    """
    Perform matrix factorization using gradient descent.
    
    Parameters:
        R (numpy array): User-item rating matrix.
        P (numpy array): User-feature matrix.
        Q (numpy array): Item-feature matrix.
        K (int): Number of latent features.
        steps (int): Number of iterations for training.
        alpha (float): Learning rate.
        beta (float): Regularization parameter.
        debug (bool): If True, prints debugging information.
        
    Returns:
        P, Q.T: Factorized matrices where P @ Q.T approximates R.
    """
    Q = Q.T  # Transpose Q for easier calculations

    for step in range(steps):
        total_error = 0

        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:  # Only update for existing ratings
                    eij = R[i][j] - np.dot(P[i, :], Q[:, j])  # Compute error
                    
                    # Update user and item feature matrices
                    P[i, :] += alpha * (2 * eij * Q[:, j] - beta * P[i, :])
                    Q[:, j] += alpha * (2 * eij * P[i, :] - beta * Q[:, j])
                    
                    total_error += eij ** 2  # Accumulate squared error

        total_error += (beta / 2) * (np.sum(P ** 2) + np.sum(Q ** 2))  # Regularization

        if debug and step % 500 == 0:
            print(f"Iteration {step} - Total Error: {total_error:.4f}")

        if total_error < 0.001:  # Early stopping condition
            break

    return P, Q.T  # Return the updated matrices

# Sample user-item rating matrix
R = np.array([
    [5, 3, 0, 1, 4],
    [4, 0, 0, 1, 2],
    [1, 1, 0, 5, 0],
    [1, 0, 0, 4, 3],
    [0, 1, 5, 4, 0],
])  # Zeroes represent unrated items

N, M = R.shape  # Number of users (N) and items (M)
K = 3  # Number of latent features

# Initialize user and item matrices with small random values
np.random.seed(42)  # For reproducibility
P = np.random.rand(N, K)
Q = np.random.rand(M, K)

print("Original Ratings Matrix (R):")
print(R)

# Perform matrix factorization
nP, nQ = matrix_factorization(R, P, Q, K, steps=10000, alpha=0.0002, beta=0.02, debug=True)

# Compute the reconstructed matrix (predicted ratings)
nR = np.dot(nP, nQ.T)

print("\nPredicted Ratings Matrix (nR):")
print(np.round(nR, 2))  # Round for readability

# Debugging: Compare known and predicted ratings
print("\nComparing Actual and Predicted Ratings:")
for i in range(N):
    for j in range(M):
        if R[i, j] > 0:
            print(f"User {i+1} -> Item {j+1}: Actual {R[i, j]} | Predicted {nR[i, j]:.2f}")
