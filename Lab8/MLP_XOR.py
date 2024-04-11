import numpy as np
from Lab8 import MLP_TF as ml

# XOR inputs
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# XOR outputs
y_train = np.array([[0], [1], [1], [0]])

# Number of inputs
n_x = 2
# Number of neurons in the output layer
n_y = 1
# Number of neurons in the hidden layer
n_h = 2

# Create an instance of the MLP_TF class
model = ml.MLP_TF(n_x, n_h, n_y)

# Print weights before training
print("Weights before training:", model.get_weights())

# Train the model
model.train(x_train.T, y_train.T, iterations=10, lr=0.1)

# Print weights after training
print("Weights after training:", model.get_weights())

# Test the trained model with XOR inputs
newdata=np.array([[1, 1]])
prediction = model.predict(newdata)
print(f"Input: {newdata}, Predicted Output: {prediction}")

