# ==========================================
# HANDWRITTEN DIGIT RECOGNIZER (CNN)
# Uses MNIST dataset
# ==========================================

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import random

print("Loading MNIST dataset...")

# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape for CNN
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

print("Building CNN model...")

# Build CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Training Model (2–4 minutes)...")
model.fit(x_train, y_train, epochs=3, validation_split=0.1)

# Evaluate
loss, accuracy = model.evaluate(x_test, y_test)
print("\nTest Accuracy:", accuracy)

# Predict Random Digit
index = random.randint(0,9999)

prediction = model.predict(x_test[index].reshape(1,28,28,1))
predicted_digit = np.argmax(prediction)

print("Predicted Digit:", predicted_digit)

plt.imshow(x_test[index].reshape(28,28), cmap="gray")
plt.title(f"Predicted Digit: {predicted_digit}")
plt.show()
