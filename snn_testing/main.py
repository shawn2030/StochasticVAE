import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import mlflow
from vanilla_snn import StochasticNN
import pandas as pd
import os
from torch import optim 
from torchviz import make_dot

EPOCHS = 100
LR_RATE = 0.01

INPUT_DIM = 4     
OUTPUT_DIM = 3

def import_data(path):
    """
    This function is used to import the training data.
    :return: X: X1 and X2 features of the dataset which becomes our input variable for the model.
             Y: Y is our target variable.
    """
    path = os.getcwd() + path
    data = pd.read_csv(path, header=None, names=['X1', 'X2', 'X3', 'X4', 'Y'])
    X = data[['X1', 'X2', 'X3', 'X4']].to_numpy()
    Y = np.array(data['Y'])

    return torch.Tensor(X), torch.Tensor(Y)

def one_hot_encode(Y):
    """
    performing one hot encoding over target variable Y
    :param Y:
    :return: one hot encoded Y
    """
    num_classes_k = len(np.unique(Y))
    new_y = np.zeros((Y.shape[0], num_classes_k))
    for i in range(len(Y)):
        if Y[i] == 0:
            new_y[i][0] = 1
        else:
            new_y[i][1] = 1

    return new_y


def train(snn, x, y):
    """
    This is the train method to train the model and returns it.
    :param snn: SNN model
            x : input data
            y : target variables
    :return: one hot encoded Y
    """
    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()  # since this is a classification problem
    optimizer = optim.Adam(snn.parameters(), lr=LR_RATE)

    snn.train()  # put the model in training mode

    for epoch in range(EPOCHS):
        optimizer.zero_grad()  # zero the gradients from the last iteration

        # Reparameterize weights and biases
        output = snn(x.T)  # forward pass

        # Compute loss
        loss = loss_fn(output.T, y)
        
        # Backprop
        loss.backward()

        # batch gradient descent
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    print("Training complete.")

    # create a computation graph
    make_dot(output, params=dict(list(snn.named_parameters()))).render("snn_testing/computation_graph", format="png")

    return snn


def test(snn, X_test, Y_test):
    """
    Model testing on test dataset.
    """
    snn.eval()

    
    with torch.no_grad():  # No need to calculate gradients during testing
        predictions = snn(X_test.T)

    # Convert predictions to class labels (taking the index of the max probability)
    predicted_labels = torch.argmax(predictions.T, dim=1)
    true_labels = torch.argmax(Y_test, dim=1)

    # Calculate accuracy
    correct_predictions = torch.sum(predicted_labels == true_labels).item()
    accuracy = correct_predictions / len(true_labels)

    print(f'Test Accuracy: {accuracy * 100:.2f}%')



def main():
    snn = StochasticNN(input_dim= INPUT_DIM, z_dim=OUTPUT_DIM)
    train_data_path = '/snn_testing/data/iris_train.dat'
    test_data_path = '/snn_testing/data/iris_test.dat'

    X, Y = import_data(train_data_path)
    Y = torch.Tensor(one_hot_encode(Y))

    snn_trained = train(snn, X, Y)

    X_TEST, Y_TEST = import_data(test_data_path)
    Y_TEST= torch.Tensor(one_hot_encode(Y_TEST))

    test(snn_trained, X_TEST, Y_TEST)

    

if __name__ == "__main__":
    main()

