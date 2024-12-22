import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt

# x_train : Training images (60000, 28, 28)
# y_train : Training labels (60000)
# x_test : Testing images (10000, 28, 28)
# y_test : Test Labels (10000)

(x_train, y_train),(x_test,y_test) = mnist.load_data()

# Normalise data values
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build neural network 

model = Sequential([
    # Input layer of 28x28 pixels (784 total pixels / inputs)
    Flatten(input_shape =(28,28)),

    # Hidden layer of 128 neurons with relu activation function.
    Dense(128, activation='relu'),

    # Output layer of 10 neurons making use of the softmax activation function to assign probabilities to values.
    Dense(10, activation='softmax')
])


model.compile(
    # Optimiser value for Stochastic Gradient Descent with learning rate of 0.01.
    optimizer = 'adam',#SGD(learning_rate = 0.01),

    # Make use of cross entropy loss.
    loss = 'sparse_categorical_crossentropy',

    metrics= ['accuracy']
)

# Train the model

history = model.fit(x_train, y_train, epochs = 10, validation_split=0.2, batch_size = 32)

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

predictions = model.predict(x_test)

# Display a test image and its prediction

for i in range(10,15):
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f'Predicted: {np.argmax(predictions[i])}, True: {y_test[i]}')
    plt.axis('off')
    plt.show()

