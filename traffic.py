import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Normalize the input
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []

    # Walk the directory tree
    for root, dirs, files in os.walk(data_dir):
        # Skip the directory if its name is not a number
        if not os.path.basename(root).isdigit():
            continue

        label = int(os.path.basename(root))

        for file in files:
            if not file.endswith('.ppm'):
                continue

            # load the image
            img = cv2.imread(os.path.join(root, file))

            # Resize the image
            img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

            images.append(img_resized)
            labels.append(label)

    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    model = tf.keras.models.Sequential([
        # Convolutional layer 1
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        # Max-pooling layer 1
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Convolutional layer 2
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        # Max-pooling layer 2
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Convolutional layer 3
        tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
        # Max-pooling layer 3
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten the units
        tf.keras.layers.Flatten(),

        # Hidden layer with 0.6 dropout
        tf.keras.layers.Dense(400, activation="relu"),
        tf.keras.layers.Dropout(0.6),

        # Output layer
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
