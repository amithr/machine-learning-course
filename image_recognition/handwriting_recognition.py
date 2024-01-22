'''
How to Use the Script:

Install Dependencies: Ensure TensorFlow (and optionally Matplotlib) is installed in your Python environment.
Run the Script: This script will download the MNIST dataset, preprocess the data, build a simple neural network model, and train this model to recognize handwritten digits.
Model Evaluation and Prediction: After training, the script evaluates the model on the test dataset and displays its accuracy. It also shows a sample image from the test set and predicts the digit.

Notes:

MNIST Dataset: The MNIST dataset consists of 28x28 pixel grayscale images of handwritten digits (0-9). The script automatically splits this dataset into training and testing sets.
Neural Network Architecture: This example uses a simple Sequential model with two Dense layers. The Flatten layer converts the 2D image data into a 1D array for input into the neural network.
Training: The model is trained for 5 epochs. You can adjust the number of epochs or the batch size as needed.

'''


import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the images
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build the model
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images
    Dense(128, activation='relu'),
    Dense(10, activation='softmax') # Output layer with 10 classes (digits 0-9)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {accuracy * 100:.2f}%")

# Optional: Visualize one of the images
plt.imshow(x_test[0], cmap='gray')
plt.show()

# Predict the first image in the test set
predicted = model.predict(x_test[:1])
print(f"Predicted label: {predicted.argmax()}")
