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

MODEL_PATH = './models/model_' + timestamp("%d%b_%H%M") + '.h5'
MODEL_WTS_PATH = './models/model_wts' + timestamp("%d%b_%H%M") + '.h5'

# def decode(data):
#     decoded = []
#     for i in range(data.shape[0]):
#         decoded[i] = np.argmax(data[i])
#     return decoded



def gs_params(model):
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


def main(param_grid):
    tb_call_back = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    train_data, train_classes, train_classes_ohc = pc.load_train_data()
    test_data, test_classes, test_classes_ohc = pc.load_test_data()
    class_names = pc.load_class_names()
    model = base.generate_model()
    compiled_model = model.compile(param_grid.optimizers[0], loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=train_data, y=train_classes_ohc, batch_size=BATCH_SIZE, epochs=10, verbose=2, validation_split=0.3, callbacks=[tb_call_back])
    # model.load_weights('models/model_wts28Apr_1834.h5')
    model.save(MODEL_PATH)
    model.save_weights(MODEL_WTS_PATH)

    
    test_y = model.predict_classes(x=test_data)
    #test_y = decode(test_y)
    test_out = pd.DataFrame(columns=['0'], data=test_y)
    test_out.to_csv('./test_' + timestamp("%d%b_%H%M") + '_out.csv')
    return

if __name__ == "__main__":
    main()


