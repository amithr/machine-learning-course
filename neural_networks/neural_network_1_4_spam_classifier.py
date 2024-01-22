
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Neural Network Class Definition
class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights_1 = 2 * np.random.random((feature_count, 4)) - 1
        self.synaptic_weights_2 = 2 * np.random.random((4, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        for iteration in range(training_iterations):
            hidden_layer = self.sigmoid(np.dot(training_inputs, self.synaptic_weights_1))
            output = self.sigmoid(np.dot(hidden_layer, self.synaptic_weights_2))
            error = training_outputs - output
            adjustments_output = error * self.sigmoid_derivative(output)
            error_hidden_layer = adjustments_output.dot(self.synaptic_weights_2.T)
            adjustments_hidden = error_hidden_layer * self.sigmoid_derivative(hidden_layer)
            self.synaptic_weights_2 += hidden_layer.T.dot(adjustments_output)
            self.synaptic_weights_1 += training_inputs.T.dot(adjustments_hidden)

    def think(self, inputs):
        hidden_layer = self.sigmoid(np.dot(inputs, self.synaptic_weights_1))
        output = self.sigmoid(np.dot(hidden_layer, self.synaptic_weights_2))
        return output

# Load and preprocess the dataset
data = pd.read_csv('spam_dataset.csv')
X = data.drop('label', axis=1).values
y = data['label'].values.reshape(-1, 1)

# Feature count (number of input neurons)
feature_count = X.shape[1]

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Standardize the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the neural network
neural_network = NeuralNetwork()
neural_network.train(X_train, y_train, training_iterations=10000)

# Evaluate the neural network
predictions = neural_network.think(X_test)
predictions = (predictions > 0.5).astype(int)
accuracy = np.mean(predictions == y_test)
print(f"Accuracy on test data: {accuracy * 100:.2f}%")

# new_email_features = np.array([0.05, 300, 0.02, 1, 0.1])
# The frequency of the word "free" is 5% (0.05).
# The total number of words in the email is 300.
# The frequency of exclamation marks is 2% (0.02).
# A domain associated with spam is present (1).
# The ratio of uppercase to lowercase letters is 10% (0.1).

# Example: Classifying a new email
new_email_features = np.array([...])  # Replace with actual features of a new email
classification = "Spam" if neural_network.think(new_email_features.reshape(1, -1)) > 0.5 else "Not Spam"
print("New email classification:", classification)
