import random
import sys
import os
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.metrics import precision_score, recall_score, f1_score

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *

def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return accuracy, precision, recall, and F1-score.
    """
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix)
    
    # Predictions for the validation data
    y_pred = [mat[user_id, question_id] >= 0.5 for user_id, question_id in zip(valid_data["user_id"], valid_data["question_id"])]
    y_true = valid_data["is_correct"]
    
    # Calculate accuracy
    accuracy = sparse_matrix_evaluate(valid_data, mat)
    
    # Calculate precision, recall, and F1-score
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    
    return accuracy, precision, recall, f1

def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return accuracy, precision, recall, and F1-score.
    """
    nbrs = KNNImputer(n_neighbors=k)
    matrix_T = matrix.T
    mat_T = nbrs.fit_transform(matrix_T).T
    
    # Predictions for the validation data
    y_pred = [mat_T[user_id, question_id] >= 0.5 for user_id, question_id in zip(valid_data["user_id"], valid_data["question_id"])]
    y_true = valid_data["is_correct"]
    
    # Calculate accuracy
    accuracy = sparse_matrix_evaluate(valid_data, mat_T)
    
    # Calculate precision, recall, and F1-score
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    
    return accuracy, precision, recall, f1

def main():
    sparse_matrix = load_train_sparse(r"starter_code\data").toarray()
    val_data = load_valid_csv(r"starter_code\data")
    test_data = load_public_test_csv(r"starter_code\data")
    
    # Initialize values of k in range [1, 100] in increments of 5
    k_values = [1] + list(range(5, 100, 5))
    
    # Empty lists to append results
    user_accuracies = []
    user_precisions = []
    user_recalls = []
    user_f1s = []
    
    item_accuracies = []
    item_precisions = []
    item_recalls = []
    item_f1s = []
    
    for k in k_values:
        print(f"Testing k = {k} for KNN by user and item...")
        
        # Accuracy, precision, recall, and F1 for KNN by user
        user_acc, user_prec, user_rec, user_f1 = knn_impute_by_user(sparse_matrix, val_data, k)
        user_accuracies.append(user_acc)
        user_precisions.append(user_prec)
        user_recalls.append(user_rec)
        user_f1s.append(user_f1)
        
        # Print user metrics
        print(f"KNN by User: Accuracy = {user_acc:.4f}, Precision = {user_prec:.4f}, Recall = {user_rec:.4f}, F1 = {user_f1:.4f}")
        
        # Accuracy, precision, recall, and F1 for KNN by item
        item_acc, item_prec, item_rec, item_f1 = knn_impute_by_item(sparse_matrix, val_data, k)
        item_accuracies.append(item_acc)
        item_precisions.append(item_prec)
        item_recalls.append(item_rec)
        item_f1s.append(item_f1)
        
        # Print item metrics
        print(f"KNN by Item: Accuracy = {item_acc:.4f}, Precision = {item_prec:.4f}, Recall = {item_rec:.4f}, F1 = {item_f1:.4f}")
        print("-" * 60)  # Separator for better readability
    
    # Plot the metrics against k for KNN by user and by item
    plt.figure(figsize=(12, 8))
    
    # Plot accuracy
    plt.subplot(2, 2, 1)
    plt.plot(k_values, user_accuracies, marker='o', label='KNN by User - Accuracy', color='b')
    plt.plot(k_values, item_accuracies, marker='o', label='KNN by Item - Accuracy', color='r')
    plt.title("Accuracy vs k")
    plt.xlabel("k (Number of Neighbors)")
    plt.ylabel("Accuracy")
    plt.legend()
    
    # Plot precision
    plt.subplot(2, 2, 2)
    plt.plot(k_values, user_precisions, marker='o', label='KNN by User - Precision', color='b')
    plt.plot(k_values, item_precisions, marker='o', label='KNN by Item - Precision', color='r')
    plt.title("Precision vs k")
    plt.xlabel("k (Number of Neighbors)")
    plt.ylabel("Precision")
    plt.legend()
    
    # Plot recall
    plt.subplot(2, 2, 3)
    plt.plot(k_values, user_recalls, marker='o', label='KNN by User - Recall', color='b')
    plt.plot(k_values, item_recalls, marker='o', label='KNN by Item - Recall', color='r')
    plt.title("Recall vs k")
    plt.xlabel("k (Number of Neighbors)")
    plt.ylabel("Recall")
    plt.legend()
    
    # Plot F1-Score
    plt.subplot(2, 2, 4)
    plt.plot(k_values, user_f1s, marker='o', label='KNN by User - F1-Score', color='b')
    plt.plot(k_values, item_f1s, marker='o', label='KNN by Item - F1-Score', color='r')
    plt.title("F1-Score vs k")
    plt.xlabel("k (Number of Neighbors)")
    plt.ylabel("F1-Score")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
