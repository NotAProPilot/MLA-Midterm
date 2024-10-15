import numpy as np
from sklearn.utils import resample
import random
import sys
import numpy as np
import os
import matplotlib.pyplot as plt 

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *

def bagging_ensemble(train_data, valid_data, test_data, base_models, n_bootstrap=3):
    """
    Perform bagging ensemble using KNN User, KNN Item, and IRT models.
    
    :param train_data: The original training data.
    :param valid_data: The validation data.
    :param test_data: The test data.
    :param base_models: A list of base model functions (KNN and IRT).
    :param n_bootstrap: Number of bootstrap datasets to generate.
    :return: Validation and test accuracy of the ensemble model.
    """
    # Bootstrap sampling: create multiple training datasets
    bootstrapped_datasets = []
    for _ in range(n_bootstrap):
        bootstrapped_data = resample(train_data, replace=True)
        bootstrapped_datasets.append(bootstrapped_data)
    
    # Store predictions from each model for validation and test data
    val_preds = np.zeros((len(valid_data["user_id"]), len(base_models)))
    test_preds = np.zeros((len(test_data["user_id"]), len(base_models)))
    
    # Train each model on each bootstrapped dataset
    for i, model_func in enumerate(base_models):
        for b_data in bootstrapped_datasets:
            # Train the model on the bootstrapped dataset
            if model_func.__name__ == 'knn_impute_by_user' or model_func.__name__ == 'knn_impute_by_item':
                # Train the KNN models
                accuracy, precision, recall, f1 = model_func(b_data, valid_data, k=5)  # Use the existing KNN function
            elif model_func.__name__ == 'irt':
                # Train the IRT model
                theta, beta, val_acc_list = irt(b_data, valid_data, lr=0.01, iterations=100)
            
            # Get the model predictions for the validation set
            if model_func.__name__ == 'irt':
                val_pred = [theta[u] - beta[q] >= 0.5 for u, q in zip(valid_data["user_id"], valid_data["question_id"])]
                test_pred = [theta[u] - beta[q] >= 0.5 for u, q in zip(test_data["user_id"], test_data["question_id"])]
            else:
                val_pred = [mat[u, q] >= 0.5 for u, q in zip(valid_data["user_id"], valid_data["question_id"])]
                test_pred = [mat[u, q] >= 0.5 for u, q in zip(test_data["user_id"], test_data["question_id"])]
            
            # Store the predictions for later aggregation
            val_preds[:, i] = val_pred
            test_preds[:, i] = test_pred
    
    # Aggregate predictions (majority vote)
    final_val_preds = np.mean(val_preds, axis=1) >= 0.5  # Averaging the predictions
    final_test_preds = np.mean(test_preds, axis=1) >= 0.5
    
    # Evaluate the ensemble performance on validation and test datasets
    val_accuracy = np.mean(valid_data["is_correct"] == final_val_preds)
    test_accuracy = np.mean(test_data["is_correct"] == final_test_preds)
    
    print(f"Ensemble Validation Accuracy: {val_accuracy:.4f}")
    print(f"Ensemble Test Accuracy: {test_accuracy:.4f}")
    
    return val_accuracy, test_accuracy


# Main function to run the ensemble
def main():
    train_data = load_train_csv(r"starter_code\data")
    sparse_matrix = load_train_sparse(r"starter_code\data").toarray()
    val_data = load_valid_csv(r"starter_code\data")
    test_data = load_public_test_csv(r"starter_code\data")
    
    # List of base models to use in the ensemble (KNN by user, KNN by item, IRT)
    base_models = [knn_impute_by_user, knn_impute_by_item, irt]
    
    # Run bagging ensemble with 3 base models and 3 bootstrapped datasets
    val_acc, test_acc = bagging_ensemble(train_data, val_data, test_data, base_models, n_bootstrap=3)

if __name__ == "__main__":
    main()
