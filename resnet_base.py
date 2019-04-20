"""
Created Date: Apr 20, 2019

Created By: varunravivarma
-------------------------------------------------------------------------------

resnet_base.py :
    Imports the base resnet50 model from Tensorflow.Keras for transfer learning,
    updating the classification layers with user defined layers.
"""

import tensorflow as tf

# Image variables (size, color channels and flattened image array size)

IMG_SIZE = 32

RGB_CHANNELS = 3

FLAT_IMG_SIZE = IMG_SIZE * IMG_SIZE * RGB_CHANNELS

# Size allocation variables

NUM_CLASSES = 10

NUM_TRAIN_FILES = 5

IMG_PER_FILE = 10000

# Functions defining DNN model

def _base_model():
    """
    Creates the base ResNet50 model from keras models, initializes the model
    with ImageNet weights.

    Arguments:
        None
    Returns:
        base_model: Keras model, base ResNet50 model with ImageNet weights
    """
    _img_shape = (IMG_SIZE, IMG_SIZE, RGB_CHANNELS)
    base_model = tf.keras.applications.ResNet50(input_shape=_img_shape,
                                                include_top=False,
                                                weights='imagenet')

    return base_model

# TODO: Create Image generator methods (what does this do and is it necessary)

def generate_model(fine_tune_from=25):
    """
    Creates the complete classifier model using Keras Sequential and
    the base model created by _base_model function.

    Arguments:
        fine_tune_from: int, default=25, the number of layers (from the top)
            of the neural network that can be retrained
            ResNet-50 has 175 layers (excluding top layers)

    Returns:
        model: Keras model, complete model for classification, built using the
            base model and custom layers for fitting the model to classify the
            CIFAR-10 dataset
    """
    base_model = _base_model()
    if 0 < fine_tune_from < len(base_model.layers):
        print("Finetuning the top {} layers of the network, all other layers \
              will be preserved with ImageNet weights.".format(fine_tune_from))
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_from]:
            layer.trainable = False
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(units=10, activation='sigmoid')
        ])

    else:
        print("Fine tune set to false, the base network will be preserved \
              with ImageNet weights.")
        base_model.trainable = False
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(units=10, activation='sigmoid')
        ])

    return model
