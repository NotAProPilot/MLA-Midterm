import sys
import numpy as np
import os

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *

def sigmoid(x):
    """ Apply sigmoid function.
    
    Computes the probability of a correct response for a student i to a question j.
    
    :param theta: Ability of student i (array of abilities)
    :param beta: Difficulty of question j (array of difficulties)
    :return: Probability of a correct response (array of probabilities)
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param theta: Vector of student abilities
    :param beta: Vector of question difficulties
    :return: Negative log-likelihood (float)
    """
    log_likelihood = 0.0

    # Loop through each data point
    for i in range(len(data['user_id'])):
        user_id = data['user_id'][i]       # Student ID
        question_id = data['question_id'][i]  # Question ID
        is_correct = data['is_correct'][i]  # Actual response (1 if correct, 0 if incorrect)
        
        # Compute the probability of the correct response using sigmoid
        probability_correct = sigmoid(theta[user_id] - beta[question_id])
        
        # Update log likelihood using the formula:
        # L = y_ij * log(p_ij) + (1 - y_ij) * log(1 - p_ij)
        log_likelihood += is_correct * np.log(probability_correct) + (1 - is_correct) * np.log(1 - probability_correct)

    # Return negative log-likelihood
    return -log_likelihood
    
    
def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.
    This function is where you use the derivative of log likelihood formula. 

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    
    
    # Initialize the gradients for theta and beta
    # In the form of an array
    theta_grad = np.zeros_like(theta) 
    beta_grad = np.zeros_like(beta)
    
     # Loop through each data point (the length of user_id column)
    for i in range(len(data['user_id'])):
        user_id = data['user_id'][i]       # Student ID
        question_id = data['question_id'][i]  # Question ID
        is_correct = data['is_correct'][i]  # Actual response (1 if correct, 0 if incorrect)
        
        # Compute the probability of correct response using sigmoid
        prob_correct = sigmoid(theta[user_id] - beta[question_id])
        
        # Compute the error (actual - predicted probability)
        error = is_correct - prob_correct
        
        # Update the gradients
        theta_grad[user_id] += error  # Gradient for theta
        beta_grad[question_id] -= error  # Gradient for beta

    # Perform the gradient descent updates for theta and beta
    theta += lr * theta_grad
    beta += lr * beta_grad
    
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
      # Get the number of unique users and questions
    num_users = len(set(data['user_id']))  # Total number of students
    num_questions = len(set(data['question_id']))  # Total number of questions

    # Initialize theta and beta with random small values
    theta = np.random.normal(0, 0.1, num_users)  # Abilities of students
    beta = np.random.normal(0, 0.1, num_questions)  # Difficulties of questions

    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    # Load the training, validation, and test datasets
    train_data = load_train_csv(r"starter_code\data")
    sparse_matrix = load_train_sparse(r"starter_code\data").toarray()
    val_data = load_valid_csv(r"starter_code\data")
    test_data = load_public_test_csv(r"starter_code\data")

    # Initialize theta and beta
    num_users = sparse_matrix.shape[0]  # Number of students
    num_questions = sparse_matrix.shape[1]  # Number of questions

    # Initialize theta and beta (random values or zeros)
    theta = np.random.normal(0, 0.1, num_users)  # Abilities of students
    beta = np.random.normal(0, 0.1, num_questions)  # Difficulties of questions

    # Set the learning rate and number of iterations
    lr = 0.01  # Adjust the learning rate if necessary
    iterations = 1000  # Number of iterations

    # Train the model using the IRT function
    print("Training the IRT model...")
    theta, beta, val_acc_lst = irt(train_data, val_data, lr, iterations)

    # After training, evaluate the model on the validation and test sets
    val_acc = evaluate(val_data, theta, beta)
    test_acc = evaluate(test_data, theta, beta)

    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Plot the validation accuracy over iterations if you want to visualize it
    import matplotlib.pyplot as plt
    plt.plot(val_acc_lst)
    plt.xlabel("Iteration")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy over Iterations")
    plt.show()


if __name__ == "__main__":
    main()