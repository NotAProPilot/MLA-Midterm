import sys
import os
from scipy.linalg import sqrtm

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *

def svd_reconstruct(matrix, k):
    """ Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, fill in the missing values (assume missing values are NaNs or zeroes).
    new_matrix = matrix.copy()

    # Fill missing values with 0 (could be based on NaN or a placeholder like 0).
    new_matrix[np.isnan(new_matrix)] = 0  # if NaN, this fills them with 0 (POTENTIAL BUG)
    # If missing values are zeros, no need to change anything further

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix (user factors)
    :param z: 2D matrix (item factors)
    :return: (u, z)
    """
    # Randomly select a pair (user_id, question_id)
    i = np.random.choice(len(train_data["question_id"]), 1)[0]
    
    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]
    
    # 1. Compute the prediction: dot product between u[n] and z[q]
    pred = np.dot(u[n], z[q])
    
    # 2. Compute the error
    error = c - pred
    
    # 3. Update u[n] and z[q] using the gradient of the squared error loss
    u[n] += lr * error * z[q]
    z[q] += lr * error * u[n]
    
    return u, z



def als(train_data, k, lr, num_iteration):
    """ Performs ALS algorithm, here we use the iterative solution - SGD 
    rather than the direct solution.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))

    # Iterate for the number of specified iterations
    for iteration in range(num_iteration):
        # Update U matrix
        for i in range(len(train_data["is_correct"])):
            user_id = train_data["user_id"][i]
            question_id = train_data["question_id"][i]
            # Get the correct answer
            correct_answer = train_data["is_correct"][i]
            # Update U and Z based on the current pair
            u, z = update_u_z(train_data, lr, u, z)
        
    # Reconstruct the matrix
    mat = np.dot(u, z.T)
    return mat


def main():
    # Load data
    train_data = load_train_csv(r"starter_code\data")
    sparse_matrix = load_train_sparse(r"starter_code\data").toarray()
    val_data = load_valid_csv(r"starter_code\data")
    test_data = load_public_test_csv(r"starter_code\data")

    # Experiment with SVD
    best_svd_k = None
    best_svd_performance = float('inf')
    k_values = [5, 10, 20, 50, 100]
    for k in k_values:
        reconst_matrix = svd_reconstruct(sparse_matrix, k)
        val_loss = squared_error_loss(val_data, reconst_matrix, sparse_matrix)
        if val_loss < best_svd_performance:
            best_svd_k = k
            best_svd_performance = val_loss

    # Experiment with ALS
    best_als_k = None
    best_als_performance = float('inf')
    k_values = [5, 10, 20, 50, 100]
    for k in k_values:
        reconst_matrix = als(train_data, k, lr=0.01, num_iteration=100)
        val_loss = squared_error_loss(val_data, reconst_matrix, sparse_matrix)
        if val_loss < best_als_performance:
            best_als_k = k
            best_als_performance = val_loss

    # Print results
    print("Best SVD k:", best_svd_k, "Performance:", best_svd_performance)
    print("Best ALS k:", best_als_k, "Performance:", best_als_performance)


if __name__ == "__main__":
    main()
