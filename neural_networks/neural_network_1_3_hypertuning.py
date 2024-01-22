import numpy as np

class NeuralNetwork():

    def __init__(self, input_neurons, hidden_neurons):
        # Seed the random number generator
        np.random.seed(1)
        
         # Initialize weights randomly with mean 0 for synapse 1 (input layer to hidden layer)
        self.synaptic_weights_1 = 2 * np.random.random((input_neurons, hidden_neurons)) - 1

        # Initialize weights randomly for synapse 2 (hidden layer to output layer)
        self.synaptic_weights_2 = 2 * np.random.random((hidden_neurons, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        for iteration in range(training_iterations):
            # Forward pass
            input_layer = training_inputs
            hidden_layer = self.sigmoid(np.dot(input_layer, self.synaptic_weights_1))
            output = self.sigmoid(np.dot(hidden_layer, self.synaptic_weights_2))

            # Backpropagation
            error = training_outputs - output
            adjustments_output = error * self.sigmoid_derivative(output)
            error_hidden_layer = adjustments_output.dot(self.synaptic_weights_2.T)
            adjustments_hidden = error_hidden_layer * self.sigmoid_derivative(hidden_layer)

            # Adjust weights
            self.synaptic_weights_2 += hidden_layer.T.dot(adjustments_output)
            self.synaptic_weights_1 += input_layer.T.dot(adjustments_hidden)

    def think(self, inputs):
        # Pass inputs through the neural network
        inputs = inputs.astype(float)
        hidden_layer = self.sigmoid(np.dot(inputs, self.synaptic_weights_1))
        output = self.sigmoid(np.dot(hidden_layer, self.synaptic_weights_2))
        return output

if __name__ == "__main__":

    hidden_neurons = 4

    # Experiment with different numbers of iterations for training
    training_iterations = 10000

    # Initialize the neural network
    neural_network = NeuralNetwork(3, hidden_neurons)
    

    print("Beginning Randomly Generated Weights: ")
    print("Layer 1 weights: \n", neural_network.synaptic_weights_1)
    print("Layer 2 weights: \n", neural_network.synaptic_weights_2)

    # Training data consisting of 4 examples -- 3 input values and 1 output
    training_inputs = np.array([[0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,1,1]])

    training_outputs = np.array([[0,1,1,0]]).T

    # Training taking place
    neural_network.train(training_inputs, training_outputs, 15000)

    print("Ending Weights After Training: ")
    print("Layer 1 weights: \n", neural_network.synaptic_weights_1)
    print("Layer 2 weights: \n", neural_network.synaptic_weights_2)

    # User input for making a prediction
    user_input_one = float(input("User Input One: "))
    user_input_two = float(input("User Input Two: "))
    user_input_three = float(input("User Input Three: "))

    print("Considering New Situation: ", user_input_one, user_input_two, user_input_three)
    print("New Output data: ")
    print(neural_network.think(np.array([user_input_one, user_input_two, user_input_three])))
    print("Wow, we did it!")