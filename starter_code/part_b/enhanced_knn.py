import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, List, Tuple
import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *

def load_data() -> Tuple[np.ndarray, Dict, Dict]:
    """
    Loads the sparse matrix, validation data, and test data.

    Returns:
        Tuple containing:
            - sparse_matrix: np.ndarray
            - val_data: Dict
            - test_data: Dict
    """
    sparse_matrix = load_train_sparse(r"starter_code\data").toarray()
    val_data = load_valid_csv(r"starter_code\data")
    test_data = load_public_test_csv(r"starter_code\data")
    return sparse_matrix, val_data, test_data

def knn_impute_by_user(matrix: np.ndarray, data: Dict, k: int) -> np.ndarray:
    """
    Performs user-based KNN imputation.

    Args:
        matrix: np.ndarray, the sparse matrix of user-question responses
        data: Dict, containing user_id and question_id
        k: int, number of neighbors

    Returns:
        np.ndarray of imputed values
    """
    nbrs = KNNImputer(n_neighbors=k)
    imputed_matrix = nbrs.fit_transform(matrix)
    return np.array([imputed_matrix[u, q] for u, q in zip(data["user_id"], data["question_id"])])

def knn_impute_by_item(matrix: np.ndarray, data: Dict, k: int) -> np.ndarray:
    """
    Performs item-based KNN imputation.

    Args:
        matrix: np.ndarray, the sparse matrix of user-question responses
        data: Dict, containing user_id and question_id
        k: int, number of neighbors

    Returns:
        np.ndarray of imputed values
    """
    nbrs = KNNImputer(n_neighbors=k)
    imputed_matrix = nbrs.fit_transform(matrix.T).T
    return np.array([imputed_matrix[u, q] for u, q in zip(data["user_id"], data["question_id"])])

def weighted_knn_impute(matrix: np.ndarray, data: Dict, k: int, alpha: float) -> Tuple[float, float, float, float]:
    """
    Performs weighted KNN imputation combining user-based and item-based approaches.

    Args:
        matrix: np.ndarray, the sparse matrix of user-question responses
        data: Dict, containing user_id, question_id, and is_correct
        k: int, number of neighbors
        alpha: float, weighting factor for user-based vs item-based predictions

    Returns:
        Tuple of (accuracy, precision, recall, f1)
    """
    user_pred = knn_impute_by_user(matrix, data, k)
    item_pred = knn_impute_by_item(matrix, data, k)
    
    weighted_pred = alpha * user_pred + (1 - alpha) * item_pred
    y_pred = (weighted_pred >= 0.5).astype(int)
    y_true = data["is_correct"]
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    
    return accuracy, precision, recall, f1

def find_optimal_alpha(matrix: np.ndarray, data: Dict, k: int, alpha_range: np.ndarray) -> float:
    """
    Finds the optimal alpha value using cross-validation.

    Args:
        matrix: np.ndarray, the sparse matrix of user-question responses
        data: Dict, containing user_id, question_id, and is_correct
        k: int, number of neighbors
        alpha_range: np.ndarray, range of alpha values to test

    Returns:
        float, the optimal alpha value
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_alpha = 0
    best_score = 0
    
    for alpha in alpha_range:
        cv_scores = []
        for train_index, val_index in kf.split(data["user_id"]):
            train_data = {key: np.array(data[key])[train_index] for key in data}
            val_data = {key: np.array(data[key])[val_index] for key in data}
            
            accuracy, _, _, _ = weighted_knn_impute(matrix, val_data, k, alpha)
            cv_scores.append(accuracy)
        
        mean_score = np.mean(cv_scores)
        if mean_score > best_score:
            best_score = mean_score
            best_alpha = alpha
    
    return best_alpha

def evaluate_knn(matrix: np.ndarray, data: Dict, k_values: List[int], alpha_range: np.ndarray) -> List[Tuple[int, float, float, float, float, float]]:
    """
    Evaluates the weighted KNN model for different k values.

    Args:
        matrix: np.ndarray, the sparse matrix of user-question responses
        data: Dict, containing user_id, question_id, and is_correct
        k_values: List[int], list of k values to evaluate
        alpha_range: np.ndarray, range of alpha values to test

    Returns:
        List of tuples (k, best_alpha, accuracy, precision, recall, f1)
    """
    results = []
    for k in k_values:
        print(f"Testing k = {k} for Weighted KNN...")
        best_alpha = find_optimal_alpha(matrix, data, k, alpha_range)
        accuracy, precision, recall, f1 = weighted_knn_impute(matrix, data, k, best_alpha)
        results.append((k, best_alpha, accuracy, precision, recall, f1))
    return results

def plot_results(results: List[Tuple[int, float, float, float, float, float]]):
    """
    Plots the results of the KNN evaluation.

    Args:
        results: List of tuples (k, best_alpha, accuracy, precision, recall, f1)
    """
    k_values, alphas, accuracies, precisions, recalls, f1s = zip(*results)
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(k_values, accuracies, marker='o')
    plt.title("Accuracy vs k")
    plt.xlabel("k (Number of Neighbors)")
    plt.ylabel("Accuracy")
    
    plt.subplot(2, 2, 2)
    plt.plot(k_values, precisions, marker='o')
    plt.title("Precision vs k")
    plt.xlabel("k (Number of Neighbors)")
    plt.ylabel("Precision")
    
    plt.subplot(2, 2, 3)
    plt.plot(k_values, recalls, marker='o')
    plt.title("Recall vs k")
    plt.xlabel("k (Number of Neighbors)")
    plt.ylabel("Recall")
    
    plt.subplot(2, 2, 4)
    plt.plot(k_values, f1s, marker='o')
    plt.title("F1-Score vs k")
    plt.xlabel("k (Number of Neighbors)")
    plt.ylabel("F1-Score")
    
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, alphas, marker='o')
    plt.title("Optimal Alpha vs k")
    plt.xlabel("k (Number of Neighbors)")
    plt.ylabel("Optimal Alpha")
    plt.show()

def main():
    """
    Main function to run the enhanced KNN algorithm.
    """
    sparse_matrix, val_data, test_data = load_data()
    
    # Initialize values of k
    k_values = [1, 6, 11, 16, 21, 26]
    alpha_range = np.arange(0, 1.1, 0.1)
    
    results = evaluate_knn(sparse_matrix, val_data, k_values, alpha_range)
    
    print("\nResults (k, best_alpha, accuracy, precision, recall, f1):")
    for result in results:
        print(result)
    
    plot_results(results)

if __name__ == "__main__":
    main()