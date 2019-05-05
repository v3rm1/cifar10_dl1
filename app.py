"""
Created Date: Apr 25, 2019

Created By: varunravivarma
-------------------------------------------------------------------------------

app.py :
    1) Instantiates and compiles model defined in resnet_base.py
    2) Trains model on CIFAR-10 dataset and saves model and weights
    3) Predicts classes for CIFAR-10 test dataset
    4) Experiments on various training variables and architectures
"""

import tensorflow as tf
import parse_cifar as pc
import resnet_base as base
import pandas as pd
import numpy as np
from time import strftime as timestamp

BATCH_SIZE = 500

def gs_params():
    """
    Defines a dictionary of parameters that can be changed on each model and tested
    Arguments:
        None
    Returns:
        param_grid: dict of hyper parameters to be tested on each model
    """
    adam = tf.keras.optimizers.Adam(lr=0.001)
    adagrad = tf.keras.optimizers.Adagrad(lr=0.001)
    nadam = tf.keras.optimizers.Nadam(lr=0.001)
    param_grid = dict(
        epochs = [5, 10, 15, 20, 25],
        batch_size = [50, 100, 150, 200, 250],
        optimizers = [adam, adagrad, nadam],
        fcl_activations = ['sigmoid', 'relu', 'softmax', 'selu', 'softsign']
       )
    return param_grid


def main():
    
    train_data, train_classes, train_classes_ohc = pc.load_train_data()
    test_data, test_classes, test_classes_ohc = pc.load_test_data()
    class_names = pc.load_class_names()
    param_grid = gs_params()

    # Baseline model
    baseline_model_path = './models/baseline/model_' + timestamp("%d%b_%H%M") + 'baseline.h5'
    baseline_wts_path = './models/baseline/model_wts' + timestamp("%d%b_%H%M") + 'baseline.h5'
    baseline_call_back = tf.keras.callbacks.TensorBoard(log_dir='./logs/baseline', histogram_freq=0, write_graph=True, write_images=True)
    baseline_model = base.baseline_model(fine_tune_from=100, fcl_activation=param_grid['fcl_activations'][0])
    compiled_baseline = baseline_model.compile(param_grid['optimizers'][0], loss='categorical_crossentropy', metrics=['accuracy'])
    baseline_model.fit(x=train_data, y=train_classes_ohc, batch_size=BATCH_SIZE, epochs=10, verbose=2, validation_split=0.3, callbacks=[baseline_call_back])
    baseline_model.save(baseline_model_path)
    baseline_model.save_weights(baseline_wts_path)

    # Dropout model
    dropout_model_path = './models/dropout/model_' + timestamp("%d%b_%H%M") + 'dropout.h5'
    dropout_wts_path = './models/dropout/model_wts' + timestamp("%d%b_%H%M") + 'dropout.h5'
    dropout_call_back = tf.keras.callbacks.TensorBoard(log_dir='./logs/dropout', histogram_freq=0, write_graph=True, write_images=True)
    dropout_model = base.dropout_model(fine_tune_from=100, fcl_activation=param_grid['fcl_activations'][0])
    compiled_dropout = dropout_model.compile(param_grid['optimizers'][0], loss='categorical_crossentropy', metrics=['accuracy'])
    dropout_model.fit(x=train_data, y=train_classes_ohc, batch_size=BATCH_SIZE, epochs=10, verbose=2, validation_split=0.3, callbacks=[dropout_call_back])
    dropout_model.save(dropout_model_path)
    dropout_model.save_weights(dropout_wts_path)

    # Batch Normalization model
    batchnorm_model_path = './models/batchnorm/model_' + timestamp("%d%b_%H%M") + 'batchnorm.h5'
    batchnorm_wts_path = './models/batchnorm/model_wts' + timestamp("%d%b_%H%M") + 'batchnorm.h5'
    batchnorm_call_back = tf.keras.callbacks.TensorBoard(log_dir='./logs/batchnorm', histogram_freq=0, write_graph=True, write_images=True)
    batchnorm_model = base.batchnorm_model(fine_tune_from=100, fcl_activation=param_grid['fcl_activations'][0])
    compiled_batchnorm = batchnorm_model.compile(param_grid['optimizers'][0], loss='categorical_crossentropy', metrics=['accuracy'])
    batchnorm_model.fit(x=train_data, y=train_classes_ohc, batch_size=BATCH_SIZE, epochs=10, verbose=2, validation_split=0.3, callbacks=[batchnorm_call_back])
    batchnorm_model.save(batchnorm_model_path)
    batchnorm_model.save_weights(batchnorm_wts_path)

    # Weight decay model
    w_decay_model_path = './models/w_decay/model_' + timestamp("%d%b_%H%M") + 'w_decay.h5'
    w_decay_wts_path = './models/w_decay/model_wts' + timestamp("%d%b_%H%M") + 'w_decay.h5'
    w_decay_call_back = tf.keras.callbacks.TensorBoard(log_dir='./logs/w_decay', histogram_freq=0, write_graph=True, write_images=True)
    w_decay_model = base.w_decay_model(fine_tune_from=100, fcl_activation=param_grid['fcl_activations'][0])
    compiled_w_decay = w_decay_model.compile(param_grid['optimizers'][0], loss='categorical_crossentropy', metrics=['accuracy'])
    w_decay_model.fit(x=train_data, y=train_classes_ohc, batch_size=BATCH_SIZE, epochs=10, verbose=2, validation_split=0.3, callbacks=[w_decay_call_back])
    w_decay_model.save(w_decay_model_path)
    w_decay_model.save_weights(w_decay_wts_path)
    
    test_y = baseline_model.predict_classes(x=test_data)
    test_out = pd.DataFrame(columns=['op'], data=test_y)
    test_out.to_csv('./test_out/test_baseline' + timestamp("%d%b_%H%M") + '_out.csv')
    test_y = dropout_model.predict_classes(x=test_data)
    test_out = pd.DataFrame(columns=['op'], data=test_y)
    test_out.to_csv('./test_out/test_dropout' + timestamp("%d%b_%H%M") + '_out.csv')
    test_y = batchnorm_model.predict_classes(x=test_data)
    test_out = pd.DataFrame(columns=['op'], data=test_y)
    test_out.to_csv('./test_out/test_batchnorm' + timestamp("%d%b_%H%M") + '_out.csv')
    test_y = w_decay_model.predict_classes(x=test_data)
    test_out = pd.DataFrame(columns=['op'], data=test_y)
    test_out.to_csv('./test_out/test_w_decay' + timestamp("%d%b_%H%M") + '_out.csv')

    return

if __name__ == "__main__":
    main()


