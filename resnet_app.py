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
import resnet_base as res_base
import pandas as pd
import numpy as np
from time import strftime as timestamp

def gs_params():
    """
    Defines a dictionary of parameters that can be changed on each model and tested
    Arguments:
        None
    Returns:
        param_grid: dict of hyper parameters to be tested on each model
    """
    param_grid = dict(
        epochs = [10],
        batch_size = [250],
        optimizers = ['adam', 'adagrad'],
        fcl_activations = ['sigmoid', 'relu', 'softmax']
       )
    return param_grid


def main():
    
    train_data, train_classes, train_classes_ohc = pc.load_train_data()
    test_data, test_classes, test_classes_ohc = pc.load_test_data()
    class_names = pc.load_class_names()
    param_grid = gs_params()
    models = ['baseline', 'batchnorm', 'dropout', 'weightdecay']

    for model in models:
        for activation_fn in param_grid['fcl_activations']:
            for epoch in param_grid['epochs']:
                for batch in param_grid['batch_size']:
                    for opt in param_grid['optimizers']:
                        if opt == 'adam':
                            optimizer = tf.keras.optimizers.Adam(lr=0.001)
                        elif opt == 'nadam':
                            optimizer = tf.keras.optimizers.Nadam(lr=0.001)
                        elif opt == 'adagrad':
                            optimizer = tf.keras.optimizers.Adagrad(lr=0.001)
        
                        if model == 'baseline':                         
                            # Baseline model
                            baseline_model_path = './models/baseline/res_model_' + str(epoch) + activation_fn + opt + str(batch) + '_' + timestamp("%d%b_%H%M") + 'baseline.h5'
                            baseline_wts_path = './models/baseline/res_model_wts' + str(epoch) + activation_fn + opt + str(batch) + '_' + timestamp("%d%b_%H%M") + 'baseline.h5'
                            baseline_call_back = tf.keras.callbacks.TensorBoard(log_dir='./logs/res/baseline', histogram_freq=0, write_graph=True, write_images=True)
                            baseline_model = res_base.baseline_model(fine_tune_from=100, fcl_activation=activation_fn)
                            compiled_baseline = baseline_model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
                            baseline_model.fit(x=train_data, y=train_classes_ohc, batch_size=batch, epochs=epoch, verbose=2, validation_split=0.3, callbacks=[baseline_call_back])
                            baseline_model.save(baseline_model_path)
                            baseline_model.save_weights(baseline_wts_path)
                            print("Saved model file to:{}".format(baseline_model_path))
                            print("Saved weights to:{}".format(baseline_wts_path))
                        
                        elif model == 'batchnorm':
                            # Batch Normalization model
                            batchnorm_model_path = './models/batchnorm/res_model_' + str(epoch) + activation_fn + opt + str(batch) + '_' + timestamp("%d%b_%H%M") + 'batchnorm.h5'
                            batchnorm_wts_path = './models/batchnorm/res_model_wts' + str(epoch) + activation_fn + opt + str(batch) + '_' + timestamp("%d%b_%H%M") + 'batchnorm.h5'
                            batchnorm_call_back = tf.keras.callbacks.TensorBoard(log_dir='./logs/res/batchnorm', histogram_freq=0, write_graph=True, write_images=True)
                            batchnorm_model = res_base.batch_norm_model(fine_tune_from=100, fcl_activation=activation_fn)
                            compiled_batchnorm = batchnorm_model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
                            batchnorm_model.fit(x=train_data, y=train_classes_ohc, batch_size=batch, epochs=epoch, verbose=2, validation_split=0.3, callbacks=[batchnorm_call_back])
                            batchnorm_model.save(batchnorm_model_path)
                            batchnorm_model.save_weights(batchnorm_wts_path)
                            print("Saved model file to:{}".format(batchnorm_model_path))
                            print("Saved weights to:{}".format(batchnorm_wts_path))

                        elif model == 'dropout':
                            # Dropout model
                            dropout_model_path = './models/dropout/res_model_' + str(epoch) + activation_fn + opt + str(batch) + '_' + timestamp("%d%b_%H%M") + 'dropout.h5'
                            dropout_wts_path = './models/dropout/res_model_wts' + str(epoch) + activation_fn + opt + str(batch) + '_' + timestamp("%d%b_%H%M") + 'dropout.h5'
                            dropout_call_back = tf.keras.callbacks.TensorBoard(log_dir='./logs/res/dropout', histogram_freq=0, write_graph=True, write_images=True)
                            dropout_model = res_base.dropout_model(fine_tune_from=100, fcl_activation=activation_fn)
                            compiled_dropout = dropout_model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
                            dropout_model.fit(x=train_data, y=train_classes_ohc, batch_size=batch, epochs=epoch, verbose=2, validation_split=0.3, callbacks=[dropout_call_back])
                            dropout_model.save(dropout_model_path)
                            dropout_model.save_weights(dropout_wts_path)
                            print("Saved model file to:{}".format(dropout_model_path))
                            print("Saved weights to:{}".format(dropout_wts_path))

                        elif model == 'weightdecay': 
                            # Weight decay model
                            w_decay_model_path = './models/w_decay/res_model_' + str(epoch) + activation_fn + opt + str(batch) + '_' + timestamp("%d%b_%H%M") + 'w_decay.h5'
                            w_decay_wts_path = './models/w_decay/res_model_wts'+ str(epoch) + activation_fn + opt + str(batch) + '_' + timestamp("%d%b_%H%M") + 'w_decay.h5'
                            w_decay_call_back = tf.keras.callbacks.TensorBoard(log_dir='./logs/res/w_decay', histogram_freq=0, write_graph=True, write_images=True)
                            w_decay_model = res_base.weight_decay_model(fine_tune_from=100, fcl_activation=activation_fn)
                            compiled_w_decay = w_decay_model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
                            w_decay_model.fit(x=train_data, y=train_classes_ohc, batch_size=batch, epochs=epoch, verbose=2, validation_split=0.3, callbacks=[w_decay_call_back])
                            w_decay_model.save(w_decay_model_path)
                            w_decay_model.save_weights(w_decay_wts_path)
                            print("Saved model file to:{}".format(w_decay_model_path))
                            print("Saved weights to:{}".format(w_decay_wts_path))
    
    return

if __name__ == "__main__":
    main()


