import sys
import os
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(r"starter_code\data").toarray()
    valid_data = load_valid_csv(r"starter_code\data")
    test_data = load_public_test_csv(r"starter_code\data")

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data

# Step 1: Implement the AutoEncoder class.
class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        out = inputs
        hidden = torch.sigmoid(self.g(inputs))  # Hidden layer with sigmoid activation
        out = torch.sigmoid(self.h(hidden))     # Output layer with sigmoid activation
        return out
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
# Step 2: Implement the train function. 

def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function. 
    
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    for epoch in range(0, num_epoch):
        train_loss = 0
        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]  # TODO: THIS LINE IS ERROR PRONE

            loss = torch.sum((output - target) ** 2.)
            loss.backward()
            
            # Compute the loss (squared error) + L2 regularization
            loss = torch.sum((output - target) ** 2.) + (lamb / 2) * model.get_weight_norm()
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    # Set optimization hyperparameters.
    lr = 0.01
    num_epoch = 30
    lamb = 0.01

    best_k = 0
    best_acc = 0

    # Test different values of latent dimension k
    k_values = [10, 50, 100, 200, 500]
    for k in k_values:
        print(f"Training with k={k}")
        
        # Initialize the AutoEncoder model with latent dimension k
        model = AutoEncoder(num_question=train_matrix.shape[1], k=k)

        # Train the model
        train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
        
        # Evaluate validation accuracy
        valid_acc = evaluate(model, zero_train_matrix, valid_data)

        print(f"k={k}, Validation Accuracy: {valid_acc}")

        # Track the best model based on validation accuracy
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_k = k
    
    lamb_values = [0.001, 0.01, 0.1, 1]
    for lamb in lamb_values:
        print(f"Training with lambda={lamb}")
        # Train the model as before, and track the results.


    print(f"Best k: {best_k} with Validation Accuracy: {best_acc}")



if __name__ == "__main__":
    main()
