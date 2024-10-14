import random
import sys
import numpy as np
import os
import matplotlib.pyplot as plt

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *

def sigmoid(x):
    """ Apply sigmoid function in part 4.2. 
    
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
    
    # Initialize the gradients for theta and beta
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


def plot_probability_vs_theta(theta_range, beta, question_indices):
    """
    Plot the probability of correct responses for different questions as a 
    function of student ability (theta).

    :param theta_range: Array of student abilities
    :param beta: Array of question difficulties
    :param question_indices: List of question indices to plot
    """
    plt.figure()

    for question_id in question_indices:
        probabilities = sigmoid(theta_range - beta[question_id])
        plt.plot(theta_range, probabilities, label=f"Question {question_id + 1}")

    plt.xlabel("Student Ability (θ)")
    plt.ylabel("Probability of Correct Response")
    plt.title("Probability of Correct Response vs. Student Ability")
    plt.legend()
    plt.show()


def main():
    """
    Main function to train the IRT model, evaluate it, and plot results.

    This function performs the following tasks:
    1. Loads the training, validation, and test datasets.
    2. Initializes the learning rates and iterates over them to train the IRT model with each rate.
    3. For each learning rate, it computes the validation accuracy over iterations.
    4. Plots the validation accuracy for different learning rates on the same graph for comparison.
    5. Evaluates the final model on the test set and prints the accuracy.
    6. Chooses three distinct questions and plots their probability of a correct response
       as a function of student ability (theta).
    """
    # Load the training, validation, and test datasets
    train_data = load_train_csv(r"starter_code\data")
    sparse_matrix = load_train_sparse(r"starter_code\data").toarray()
    val_data = load_valid_csv(r"starter_code\data")
    test_data = load_public_test_csv(r"starter_code\data")

    # Initialize parameters
    learning_rates = [0.005, 0.01, 0.02]  # List of learning rates to test
    iterations = 100  # Number of iterations
    val_acc_lists = {}  # Dictionary to store validation accuracy lists for each learning rate

    # Loop through each learning rate
    for lr in learning_rates:
        print(f"Training the IRT model with learning rate {lr}...")

        # Train the model using the IRT function
        theta, beta, val_acc_lst = irt(train_data, val_data, lr, iterations)

        # Store the validation accuracy list for this learning rate
        val_acc_lists[lr] = val_acc_lst

        # After training, evaluate the model on the validation and test sets
        val_acc = evaluate(val_data, theta, beta)
        test_acc = evaluate(test_data, theta, beta)

        print(f"Validation Accuracy for lr={lr}: {val_acc:.4f}")
        print(f"Test Accuracy for lr={lr}: {test_acc:.4f}")

    # Plot the validation accuracy over iterations for all learning rates
    plt.figure()
    for lr, val_acc_lst in val_acc_lists.items():
        plt.plot(val_acc_lst, label=f"lr={lr}")

    plt.xlabel("Iteration")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy over Iterations for Different Learning Rates")
    plt.legend()
    plt.show()

    # Step 4: Choose three distinct questions and plot their probability of a correct response
    theta_range = np.linspace(-3, 3, 100)  # Range of student abilities
    question_indices = [random.randint(1, 100), random.randint(1, 100), random.randint(1, 100)]  # Choose three distinct questions
    plot_probability_vs_theta(theta_range, beta, question_indices)


if __name__ == "__main__":
    main()


