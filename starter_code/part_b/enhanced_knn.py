import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, List, Tuple
import sys
import os
from tqdm import tqdm

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
    
    for alpha in tqdm(alpha_range, desc=f"Finding optimal alpha for k={k}", leave=False):
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
    print(f"Starting KNN imputation for k={k}, alpha={alpha}")
    user_pred = knn_impute_by_user(matrix, data, k)
    print("User-based imputation completed")
    item_pred = knn_impute_by_item(matrix, data, k)
    print("Item-based imputation completed")
    
    weighted_pred = alpha * user_pred + (1 - alpha) * item_pred
    y_pred = (weighted_pred >= 0.5).astype(int)
    y_true = data["is_correct"]
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    
    print(f"Imputation completed. Accuracy: {accuracy:.4f}")
    return accuracy, precision, recall, f1

def evaluate_knn(matrix: np.ndarray, data: Dict, k_values: List[int], alpha_range: np.ndarray) -> Dict[int, List[Tuple[float, float, float, float, float]]]:
    """
    Evaluates the weighted KNN model for different k and alpha values, storing accuracy, precision, recall, and f1.

    Args:
        matrix: np.ndarray, the sparse matrix of user-question responses
        data: Dict, containing user_id, question_id, and is_correct
        k_values: List[int], list of k values to evaluate
        alpha_range: np.ndarray, range of alpha values to test

    Returns:
        Dictionary where keys are k values and values are lists of tuples 
        (alpha, accuracy, precision, recall, f1).
    """
    results = {}
    for k in tqdm(k_values, desc="Evaluating k values"):
        results[k] = []
        for alpha in tqdm(alpha_range, desc=f"Evaluating alpha for k={k}", leave=False):
            accuracy, precision, recall, f1 = weighted_knn_impute(matrix, data, k, alpha)
            results[k].append((alpha, accuracy, precision, recall, f1))
    return results


def plot_results(results: Dict[int, List[Tuple[float, float, float, float, float]]], alpha_range: np.ndarray, k_values: List[int]):
    """
    Plots how accuracy, precision, recall, and f1 change for different alpha and k values.

    Args:
        results: Dict[int, List[Tuple[float, float, float, float, float]]], results from the KNN evaluation
        alpha_range: np.ndarray, range of alpha values tested
        k_values: List[int], list of k values tested
    """
    plt.figure(figsize=(15, 10))
    
    # Accuracy vs alpha for different k
    plt.subplot(2, 2, 1)
    for k in k_values:
        alphas, accuracies, _, _, _ = zip(*results[k])
        plt.plot(alphas, accuracies, marker='o', label=f'k={k}')
    plt.title("Accuracy vs Alpha")
    plt.xlabel("Alpha")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    # Precision vs alpha for different k
    plt.subplot(2, 2, 2)
    for k in k_values:
        alphas, _, precisions, _, _ = zip(*results[k])
        plt.plot(alphas, precisions, marker='o', label=f'k={k}')
    plt.title("Precision vs Alpha")
    plt.xlabel("Alpha")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True)

    # Recall vs alpha for different k
    plt.subplot(2, 2, 3)
    for k in k_values:
        alphas, _, _, recalls, _ = zip(*results[k])
        plt.plot(alphas, recalls, marker='o', label=f'k={k}')
    plt.title("Recall vs Alpha")
    plt.xlabel("Alpha")
    plt.ylabel("Recall")
    plt.legend()
    plt.grid(True)

    # F1-Score vs alpha for different k
    plt.subplot(2, 2, 4)
    for k in k_values:
        alphas, _, _, _, f1_scores = zip(*results[k])
        plt.plot(alphas, f1_scores, marker='o', label=f'k={k}')
    plt.title("F1-Score vs Alpha")
    plt.xlabel("Alpha")
    plt.ylabel("F1-Score")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to run the enhanced KNN algorithm and plot accuracy, precision, recall, and f1 changes.
    """
    sparse_matrix, val_data, test_data = load_data()
    
    # Initialize values of k and alpha range
    k_values = [1, 5, 10, 15, 20, 25, 30]
    alpha_range = np.arange(0, 1.01, 0.25)
    
    print("Starting KNN evaluation...")
    results = evaluate_knn(sparse_matrix, val_data, k_values, alpha_range)
    
    print("\nResults for different k and alpha values:")
    for k in results:
        print(f"k={k}: {results[k]}")
    
    plot_results(results, alpha_range, k_values)

if __name__ == "__main__":
    main()
