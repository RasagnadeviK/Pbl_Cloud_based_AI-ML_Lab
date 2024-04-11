import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class MLP_TF:
    def __init__(self, input_nodes, n_hidden_nodes, n_output_nodes):
        self.w1 = tf.Variable(tf.random.normal([n_hidden_nodes, input_nodes]), dtype=tf.float32)
        self.w2 = tf.Variable(tf.random.normal([n_output_nodes, n_hidden_nodes]), dtype=tf.float32)
        self.b1 = tf.Variable(tf.random.normal([n_hidden_nodes]), dtype=tf.float32)
        self.b2 = tf.Variable(tf.random.normal([n_output_nodes]), dtype=tf.float32)

    def sigmoid(self, z):
        return 1 / (1 + tf.exp(-z))

    def get_weights(self):
       return self.w1,self.w2,self.b1,self.b2

    def forward_prop(self, x):
        z1 = tf.matmul(x,self.w1, transpose_b=True) + self.b1
        a1 = self.sigmoid(z1)
        z2 = tf.matmul(a1,self.w2, transpose_b=True) + self.b2
        a2 = self.sigmoid(z2)
        return z1, a1, z2, a2


    def predict(self, x):
        x = tf.constant(x, dtype=tf.float32)
        z1, a1, z2, a2 = self.forward_prop(x)
        a2 = tf.squeeze(a2)
        return 1 if a2.numpy() >= 0.5 else 0, a2.numpy()

    def train(self, x, y, iterations, lr):
        losses = []
        x = tf.constant(x, dtype=tf.float32)
        y = tf.constant(y, dtype=tf.float32)
        #print("x shape is ",x)
        #print("y shape is ", y)
        for i in range(iterations):
            with tf.GradientTape() as tape:
                z1, a1, z2, a2 = self.forward_prop(tf.transpose(x))
                loss = -tf.reduce_mean(y * tf.math.log(a2) + (1 - y) * tf.math.log(1 - a2))

                losses.append(loss.numpy())

            gradients = tape.gradient(loss, [self.w1, self.w2, self.b1, self.b2])
            dz2, dw2, dz1, dw1 = gradients[0], gradients[1], gradients[2], gradients[3]
            #print("Shape of w1:", self.w1.shape)
            #print("Shape of dw1:", dw1.shape)
            #print(lr* dw1)
#updates the weights of the first layer in your neural network by subtracting a fraction of the gradient of the loss with respect to those weights
            self.w2.assign_sub(lr * dw2)
            self.w1=self.w1-(lr* dw1)
            self.b2.assign_sub(lr * gradients[3])
            self.b1.assign_sub(lr * gradients[2])

        plt.plot(losses)
        plt.xlabel("EPOCHS")
        plt.ylabel("Loss value")
        plt.show()
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
model = MLP_TF(n_x, n_h, n_y)

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
