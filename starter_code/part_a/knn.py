import random
import sys
import os
import matplotlib.pyplot as plt


# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from sklearn.impute import KNNImputer
from utils import *


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    
    # By default, the matrix args here 
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO: In this function, you'll need to fill in missing values based on question similarity. 
    # TODO: You’ll use the same technique as the user-based function but treat questions as the nearest neighbors.                  
    # ALSO TODO: IGNORE STRANGE WORDS
    # 
    # CAUTION: THIS APPROACH IS SUGGESTED BY CHATGPT. NO OFFICIAL DOCUMENTATION EXISTED FOR
    # THIS APPROACH. IMPLEMENT WITH CAUTION. ASK DR. KHANH AND TA IMMEDIATELY. 
    # Implement the function as described in the docstring.             #
    
    
    # Initialize the KNN imputer with k neighbors:
    nbrs = KNNImputer(n_neighbors=k)
    
    # Transpose the matrix so that the questions (items) are treated as rows
    matrix_T = matrix.T
    
    # Fit to data, then transform the TRANSPOSED matrix:
    mat_T = nbrs.fit_transform(matrix_T).T
    
    # Evaluate said transposed matrix:
    acc = sparse_matrix_evaluate(valid_data, mat_T)
    print("Validation Accuracy: {}".format(acc))
    return acc
    
    #TODO: DELETE THIS NOTE (2203 2OCT24)




def main():
    #TODO: JESUS WHO THE FUCK DON'T ACTUALLY ACCEPT THE ACTUAL FILE
    sparse_matrix = load_train_sparse(r"starter_code\data").toarray()
    val_data = load_valid_csv(r"starter_code\data")
    test_data = load_public_test_csv(r"starter_code\data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    
    
    user_accuracies = []
    item_accuracies = []
    
    for k in k_values:
        print(f"Testing k = {k} for KNN by user (first line) and item (second line)...")
        
        # Accuracy for KNN by user
        user_acc = knn_impute_by_user(sparse_matrix, val_data, k)
        user_accuracies.append(user_acc)
        
        # Accuracy for KNN by item
        item_acc = knn_impute_by_item(sparse_matrix, val_data, k)
        item_accuracies.append(item_acc)
    
    # Plot the accuracies against k
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, user_accuracies, marker='o', label='KNN by User', color='b')
    plt.plot(k_values, item_accuracies, marker='o', label='KNN by Item', color='r')
    plt.title("KNN Accuracy vs k (Number of Neighbors)")
    plt.xlabel("k (Number of Neighbors)")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example k values: [1, 6, 11, 16, 21, 26]
k_values = [1, 6, 11, 16, 21, 26]   




if __name__ == "__main__":
    main()
