import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode
from sklearn.impute import KNNImputer
from sklearn.utils import resample
import torch
import torch.nn as nn
from torch.autograd import Variable
from neural_network import AutoEncoder, load_data, train
import os
import sys
from item_response import irt, sigmoid

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from part_a.item_response import irt
from utils import *

# Neural Network model definition
class AutoEncoder(nn.Module):
    """AutoEncoder model for reconstructing user responses."""
    
    def __init__(self, num_question, k=100):
        """Initializes the AutoEncoder with specified input size and hidden layer size.
        
        Args:
            num_question: Number of questions (input size).
            k: Size of the hidden layer.
        """
        super(AutoEncoder, self).__init__()
        self.g = nn.Linear(num_question, k)  # Linear transformation to hidden layer
        self.h = nn.Linear(k, num_question)  # Linear transformation to output layer

    def get_weight_norm(self):
        """Calculates and returns the L2 norm of the weights."""
        g_w_norm = torch.norm(self.g.weight, 2)
        h_w_norm = torch.norm(self.h.weight, 2)
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """Defines the forward pass of the model.
        
        Args:
            inputs: Input tensor for the model.
        
        Returns:
            Output tensor after passing through the model.
        """
        out = torch.sigmoid(self.g(inputs))
        out = torch.sigmoid(self.h(out))
        return out


def train_neural_network(model, train_data, zero_train_data, valid_data, lr=0.1, lamb=0.01, num_epoch=20):
    """Trains the neural network model.
    
    Args:
        model: The neural network model to train.
        train_data: Training data.
        zero_train_data: Data with NaNs replaced by zeros.
        valid_data: Validation data.
        lr: Learning rate for the optimizer.
        lamb: Regularization parameter.
        num_epoch: Number of training epochs.
    
    Returns:
        The trained model.
    """
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    for epoch in range(num_epoch):
        train_loss = 0.0
        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            nan_mask = nan_mask.reshape(-1)
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.0) + lamb * 0.5 * model.get_weight_norm()
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

    return model


def neural_network_pred(model, data, zero_train_data):
    """Generates predictions using the trained neural network model.
    
    Args:
        model: The trained neural network model.
        data: Data for generating predictions.
        zero_train_data: Data with NaNs replaced by zeros.
    
    Returns:
        List of predictions.
    """
    model.eval()
    pred = []
    for i, u in enumerate(data["user_id"]):
        inputs = Variable(zero_train_data[u]).unsqueeze(0)
        output = model(inputs)
        pred.append(output[0][data["question_id"][i]].item() >= 0.5)
    return pred


def resample(data):
    """Resamples the given dataset with replacement.
    
    Args:
        data: Input data to be resampled.
    
    Returns:
        Resampled data dictionary.
    """
    n_samples = len(data["user_id"])
    indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
    resampled_data = {
        "user_id": np.array(data["user_id"])[indices],
        "question_id": np.array(data["question_id"])[indices],
        "is_correct": np.array(data["is_correct"])[indices]
    }
    return resampled_data


def irt_pred(data, theta, beta):
    """Generates predictions using Item Response Theory (IRT).
    
    Args:
        data: Input data for predictions.
        theta: User parameters.
        beta: Item parameters.
    
    Returns:
        List of predictions.
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return pred


def ensemble(train_matrix, train_data, val_data, test_data, hp):
    """Combines predictions from KNN, IRT, and Neural Network models.
    
    Args:
        train_matrix: Sparse matrix of training data.
        train_data: Training data.
        val_data: Validation data.
        test_data: Test data.
        hp: Hyperparameters for the models.
    
    Returns:
        Validation and test predictions from the ensemble model.
    """
    val_pred = []  # Initialize a list to store validation predictions
    test_pred = []  # Initialize a list to store test predictions

    # KNN (user-based)
    knn_train_data = resample(train_data)  # Resample the training data for KNN
    knn_sparse_matrix = np.empty(train_matrix.shape)  # Create an empty sparse matrix
    knn_sparse_matrix[:] = np.nan  # Fill the sparse matrix with NaNs
    for i in range(len(knn_train_data["is_correct"])):
        # Populate the KNN sparse matrix with user responses
        knn_sparse_matrix[knn_train_data["user_id"][i], knn_train_data["question_id"][i]] \
            = knn_train_data["is_correct"][i]
    
    # Initialize KNN imputer with specified number of neighbors
    nbrs = KNNImputer(n_neighbors=hp["knn_user_k"])
    # Fit the imputer on the sparse matrix and transform it
    knn_mat = nbrs.fit_transform(knn_sparse_matrix)
    
    # Evaluate the KNN model on the validation set
    knn_acc = sparse_matrix_evaluate(val_data, knn_mat)
    print(f"KNN: Validation Accuracy: {knn_acc}")
    
    # Store the KNN predictions for validation and test data
    val_pred.append(sparse_matrix_predictions(val_data, knn_mat))
    test_pred.append(sparse_matrix_predictions(test_data, knn_mat))

    # IRT (Item Response Theory)
    irt_train_data = resample(train_data)  # Resample the training data for IRT
    # Apply IRT to estimate user and item parameters
    theta, beta, _ = irt(irt_train_data, val_data, hp["irt_lr"], hp["irt_iter"])
    
    # Evaluate the IRT model on the validation set
    irt_acc = evaluate(val_data, irt_pred(val_data, theta, beta))
    print(f"IRT: Validation Accuracy: {irt_acc}")
    
    # Store the IRT predictions for validation and test data
    val_pred.append(irt_pred(val_data, theta, beta))
    test_pred.append(irt_pred(test_data, theta, beta))

    # Neural Network
    zero_train_matrix = torch.FloatTensor(train_matrix.copy())  # Convert training matrix to a PyTorch tensor
    zero_train_matrix[np.isnan(train_matrix)] = 0  # Replace NaNs with zeros
    
    # Initialize the AutoEncoder model
    nn_model = AutoEncoder(train_matrix.shape[1], hp["nn_k"])
    # Train the neural network model on the training data
    nn_model = train_neural_network(nn_model, torch.FloatTensor(train_matrix), zero_train_matrix, val_data,
                                    lr=hp["nn_lr"], lamb=hp["nn_lamb"], num_epoch=hp["nn_epoch"])
    
    # Evaluate the neural network model on the validation set
    nn_acc = evaluate(val_data, neural_network_pred(nn_model, val_data, zero_train_matrix))
    print(f"Neural Network: Validation Accuracy: {nn_acc}")
    
    # Store the neural network predictions for validation and test data
    val_pred.append(neural_network_pred(nn_model, val_data, zero_train_matrix))
    test_pred.append(neural_network_pred(nn_model, test_data, zero_train_matrix))

    return val_pred, test_pred  # Return the combined validation and test predictions


def main():
    """Main function to load data and execute the ensemble prediction."""
    # Load the training, validation, and test datasets
    train_data = load_train_csv(r"starter_code\data")
    sparse_matrix = load_train_sparse(r"starter_code\data").toarray()
    val_data = load_valid_csv(r"starter_code\data")
    test_data = load_public_test_csv(r"starter_code\data")

    # Hyper-parameters
    hp = {
        # KNN
        "knn_user_k": 11,
        # IRT
        "irt_lr": 0.006,
        "irt_iter": 100,
        # Neural Network
        "nn_k": 10,
        "nn_lr": 0.1,
        "nn_lamb": 0.01,
        "nn_epoch": 20
    }

    # Ensemble predictions
    val_pred, test_pred = ensemble(sparse_matrix, train_data, val_data, test_data, hp)
    # The mean of predictions from the 3 models
    mean_val_pred = np.mean(np.array(val_pred), axis=0)
    mean_test_pred = np.mean(np.array(test_pred), axis=0)
    # Accuracy of combined prediction
    val_acc = evaluate(val_data, mean_val_pred)
    test_acc = evaluate(test_data, mean_test_pred)
    print("Final Ensembled Results:")
    print(f"Validation Accuracy: {val_acc}")
    print(f"Test Accuracy: {test_acc}")
    
    # Plotting the Validation and Test Accuracies
    val_accuracies = {
        'KNN': knn_acc,
        'IRT': irt_acc,
        'Neural Network': nn_acc,
        'Ensemble': val_acc
    }
    test_accuracies = {
        'KNN': knn_acc,
        'IRT': irt_acc,
        'Neural Network': nn_acc,
        'Ensemble': test_acc
    }
    
    models = list(val_accuracies.keys())
    val_scores = list(val_accuracies.values())
    test_scores = list(test_accuracies.values())

    x = np.arange(len(models))  # the label locations
    width = 0.35  # width of the bars

    fig, ax = plt.subplots()

    # Plotting bars for validation and test accuracies
    rects1 = ax.bar(x - width/2, val_scores, width, label='Validation Accuracy')
    rects2 = ax.bar(x + width/2, test_scores, width, label='Test Accuracy')

    # Add labels, title, and custom x-axis tick labels
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation and Test Accuracy by Model')
    ax.set_xticks(x)
    ax.set_xticklabels(models)

    # Adding the values on top of the bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    # Move the legend below the graph
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
