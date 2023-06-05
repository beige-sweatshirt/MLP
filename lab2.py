# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 11:07:47 2023

@author: Matt
"""
import torch
import torch.nn as nn
import torch.optim as optim
import math as m
import numpy as np
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self, hidden_layers, activation):
        super(MLP, self).__init__()
        self.input_size = 2
        self.output_size = 1
        self.hidden_layers = hidden_layers
        self.layers = nn.ModuleList()
        self.activation = activation

        # Create Layers
        in_nodes = self.input_size
        for i, out_nodes in enumerate(self.hidden_layers):
            self.layers.append(nn.Linear(in_nodes, out_nodes))
            in_nodes = out_nodes
        self.layers.append(nn.Linear(in_nodes, self.output_size))
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers)-1: x = self.activation(x)
        
        return x

def fun(x,y):
    return m.sin((m.pi*10*x+10) / (1+y**2)) + m.log(x**2+y**2)

def train(num_epochs, batch_size, model, learning_rate, criterion, optimizer, train_inputs, train_targets):
    num_batches = int(train_inputs.shape[0] / batch_size)
    train_loss = []
    test_loss = []

    for epoch in range(num_epochs):
        for i in range(num_batches):
            x_batch = torch.tensor(train_inputs[i*batch_size:(i+1)*batch_size], dtype=torch.float32)
            y_batch = torch.tensor(train_targets[i*batch_size:(i+1)*batch_size], dtype=torch.float32).unsqueeze(1)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            train_loss.append(criterion(model(torch.tensor(train_inputs, dtype=torch.float32)), torch.tensor(train_targets, dtype=torch.float32).unsqueeze(1)).item())
            test_loss.append(criterion(model(torch.tensor(test_inputs, dtype=torch.float32)), torch.tensor(test_targets, dtype=torch.float32).unsqueeze(1)).item())

def eval(test_inputs, test_targets):
    with torch.no_grad():
        approx_y = model(torch.tensor(test_inputs, dtype=torch.float32)).numpy()
        actual_y = test_targets
    mse = round(np.mean((actual_y - approx_y)**2), 4)
    return(mse)

def show_plots():
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    x = np.linspace(1, 100, 100)
    y = np.linspace(1, 100, 100)
    X, Y = np.meshgrid(x, y)

    def plot(f, a, title):
        f = f.reshape(X.shape)
        a.contourf(X, Y, f, cmap='plasma')
        a.set_title(title)
    
    f = np.array([fun(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
    plot(f, axs[0], "Actual Function")
    f = model(torch.tensor(np.vstack((X.ravel(), Y.ravel())).T, dtype=torch.float32)).detach().numpy()
    plot(f, axs[1], "Approximate Function")

    plt.show()

# Generate training / test data
train_set_size = 10000
test_set_size = 8000
train_inputs = np.random.uniform(1, 100, (train_set_size, 2)) 
train_targets = np.array([fun(x, y) for x, y in train_inputs])
test_inputs = np.random.uniform(1, 100, (test_set_size, 2))
test_targets = np.array([fun(x, y) for x, y in test_inputs])
print("Training Set Size:", train_set_size)
print("Testing Set Size:", test_set_size)

# Define
hidden_layers = [10,5]
activation = nn.Sigmoid()
print("Hidden Layers:", hidden_layers)
print("Activation Function: Sigmoid")
model = MLP(hidden_layers, activation)

# Train
num_epochs = 100
batch_size = 32
learning_rate = 0.01
criterion = nn.MSELoss() # MSE
optimizer = optim.Adam(model.parameters(), learning_rate)
print("Number of Epochs:", num_epochs)
print("Batch Size:", batch_size)
print("Learning Rate:", learning_rate)
print("Loss Function: Mean Squared Error")
print("Optimizer Function: Adam")
train(num_epochs, batch_size, model, learning_rate, criterion, optimizer, train_inputs, train_targets)

# Evaluate
error = eval(test_inputs, test_targets)
print("MSE:", error)

show_plots()