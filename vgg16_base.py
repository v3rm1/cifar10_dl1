"""
Created Date: May 16, 2019

Created By: varunravivarma, tparisotto
-------------------------------------------------------------------------------

vgg16_base.py :
    Imports the base VGG16 model from Tensorflow.Keras for transfer learning,
    updating the classification layers with user defined layers.
"""

import tensorflow as tf

# Image variables (size, color channels and flattened image array size)

IMG_SIZE = 32

RGB_CHANNELS = 3

FLAT_IMG_SIZE = IMG_SIZE * IMG_SIZE * RGB_CHANNELS

# Size allocation variables

NUM_CLASSES = 10

# Functions defining DNN model

def _base_model():
    """
    Creates the base VGG16 model from keras models, initializes the model
    with ImageNet weights.

    Arguments:
        None
    Returns:
        base_model: Keras model, base VGG16 model with ImageNet weights
    """
    _img_shape = (IMG_SIZE, IMG_SIZE, RGB_CHANNELS)
    base_model = tf.keras.applications.vgg16.VGG16(input_shape=_img_shape,
                                                include_top=False,
                                                weights='imagenet')

    return base_model


def baseline_model(fine_tune_from=5, fcl_activation='sigmoid'):
    """
    Creates the baseline classifier model using Keras Sequential and
    the base model created by _base_model function.

    Arguments:
        fine_tune_from: int, default=5, the number of layers (from the top)
            of the neural network that can be retrained
            VGG16 has 16 layers
        fcl_activation: string, default='sigmoid', specifies activation
            function for the final dense layer (classification/output layer)

    Returns:
        model: Keras model, baseline model for classification, built using the
            base model and custom layers for fitting the model to classify the
            CIFAR-10 dataset
    """
    base_model = _base_model()
    if 0 < fine_tune_from < len(base_model.layers):
        print("Finetuning the top {} layers of the network, all other layers will be preserved with ImageNet weights.".format(fine_tune_from))
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_from]:
            layer.trainable = False
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(units=10, activation=fcl_activation)
        ])

    else:
        print("Fine tune set to false, the base network will be preserved with ImageNet weights.")
        base_model.trainable = False
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(units=NUM_CLASSES, activation=fcl_activation)
        ])

    return model

def dropout_model(fine_tune_from=5, dropout_rate=0.25, fcl_activation='sigmoid'):
    """
    Creates the classifier model with a single dropout layer using Keras Sequential and the base model created by _base_model function.

    Arguments:
        fine_tune_from: int, default=5, the number of layers (from the top)
            of the neural network that can be retrained
            VGG16 has 16 layers
        dropout_rate: int, default=0.25, fraction of inputs to be dropped
        fcl_activation: string, default='sigmoid', specifies activation
            function for the final dense layer (classification/output layer)

    Returns:
        model: Keras model, complete model for classification, built using the
            base model and custom layers for fitting the model to classify the
            CIFAR-10 dataset
    """
    base_model = _base_model()
    if 0 < fine_tune_from < len(base_model.layers):
        print("Finetuning the top {} layers of the network, all other layers will be preserved with ImageNet weights.".format(fine_tune_from))
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_from]:
            layer.trainable = False
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.Dropout(rate=dropout_rate),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(units=10, activation=fcl_activation)
        ])

    else:
        print("Fine tune set to false, the base network will be preserved with ImageNet weights.")
        base_model.trainable = False
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(units=NUM_CLASSES, activation=fcl_activation)
        ])

    return model

def batch_norm_model(fine_tune_from=5, fcl_activation='sigmoid'):
    """
    Creates the classifier model with batch normalization using Keras Sequential and the base model created by _base_model function.

    Arguments:
        fine_tune_from: int, default=5, the number of layers (from the top)
            of the neural network that can be retrained
            VGG16 has 16 layers
        fcl_activation: string, default='sigmoid', specifies activation
            function for the final dense layer (classification/output layer)

    Returns:
        model: Keras model, complete model for classification, built using the
            base model and custom layers for fitting the model to classify the
            CIFAR-10 dataset
    """
    base_model = _base_model()
    if 0 < fine_tune_from < len(base_model.layers):
        print("Finetuning the top {} layers of the network, all other layers will be preserved with ImageNet weights.".format(fine_tune_from))
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_from]:
            layer.trainable = False
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(units=10, activation=fcl_activation)
        ])

    else:
        print("Fine tune set to false, the base network will be preserved with ImageNet weights.")
        base_model.trainable = False
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(units=NUM_CLASSES, activation=fcl_activation)
        ])

    return model

def weight_decay_model(fine_tune_from=5, fcl_activation='sigmoid', regularizer='l1_l2'):
    """
    Creates the classifier model with a weight decay in the dense (classification/output) layer using Keras Sequential and the base model created by _base_model function.

    Arguments:
        fine_tune_from: int, default=5, the number of layers (from the top)
            of the neural network that can be retrained
            VGG16 has 16 layers
        fcl_activation: string, default='sigmoid', specifies activation
            function for the final dense layer (classification/output layer)
        regularizer: string, default='l1_l2', specifies kernel regularizer to be applied on the dense layer

    Returns:
        model: Keras model, complete model for classification, built using the
            base model and custom layers for fitting the model to classify the
            CIFAR-10 dataset
    """
    base_model = _base_model()
    if 0 < fine_tune_from < len(base_model.layers):
        print("Finetuning the top {} layers of the network, all other layers will be preserved with ImageNet weights.".format(fine_tune_from))
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_from]:
            layer.trainable = False
            if regularizer == 'l1_l2':
                model = tf.keras.Sequential([
                    base_model,
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dense(units=10, activation=fcl_activation, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01))
                ])

            elif regularizer == 'l1':
                model = tf.keras.Sequential([
                    base_model,
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dense(units=10, activation=fcl_activation, kernel_regularizer=tf.keras.regularizers.l1(0.01))
                ])

            elif regularizer == 'l2':
                model = tf.keras.Sequential([
                    base_model,
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dense(units=10, activation=fcl_activation, kernel_regularizer=tf.keras.regularizers.l2(0.01))
                ])

    else:
        print("Fine tune set to false, the base network will be preserved with ImageNet weights.")
        base_model.trainable = False
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(units=NUM_CLASSES, activation=fcl_activation)
        ])

    return model

    
    