import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.layers import Input, Dense, Flatten, Concatenate, concatenate, Dropout, Lambda
from keras import models,layers
import tensorflow as tf
import warnings
from keras.callbacks import EarlyStopping
from keras.models import Model
import sklearn
import keras.backend as K
import numpy as np
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically

# Soft Thresholding
def softthresholding(b, lam):
    soft_thresh = np.sign(b) * max(abs(b) - lam//2, 0)
    return soft_thresh


def F1_score(y_true,y_pred):
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    F1score=2*precision*recall/(precision+recall)
    return F1score


def conv_pool(x_input):
    result1 = tf.keras.layers.Conv1D(x_input.shape[-1], 1, activation='relu', padding="valid")(x_input)
    result2 = tf.keras.layers.Conv1D(x_input.shape[-1], 1, activation="selu", padding="valid")(x_input)
    resultcombined = K.concatenate([result1, result2], axis=1)
    result = tf.keras.layers.AveragePooling1D(pool_size=resultcombined.shape[-2],strides=resultcombined.shape[-2],padding='valid')(resultcombined)
    result = tf.keras.layers.Conv1D(x_input.shape[-1], 1, activation='relu', padding="valid")(x_input)

    return result


def themodel(input_shape,hidden_size,ac):
    X_input = tf.keras.Input(input_shape)
    result = tf.keras.layers.Reshape((1, X_input.shape[1]), input_shape= X_input.shape)(X_input)
    result = Dense(hidden_size, activation=ac)(result)
    result = tf.keras.layers.BatchNormalization()(result)
    result = conv_pool(result)
    result = Dense(hidden_size, activation=ac)(result)
    result = tf.keras.layers.Flatten(input_shape=result.shape)(result)
    out = Dense(1, activation='sigmoid')(result)

    model = Model(inputs=X_input, outputs=out)

    return model

def trainmodel(x_train,y_train,x_test, y_test,learnrate,hidden_size,Epoch,batch_size,ac):
    model = themodel(x_train.values[1,:].shape,hidden_size,ac)

    model.compile(loss=tf.keras.losses.binary_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(lr=learnrate, decay = learnrate/Epoch),
                  metrics=["accuracy",
                        tf.keras.metrics.Precision(),
                        tf.keras.metrics.Recall(),
                        tf.keras.metrics.AUC(curve='ROC', name = 'auc'),
                        tf.keras.metrics.AUC(curve='PR', name = 'aupr'),
                        F1_score]
)

    callbacks_list = [EarlyStopping(monitor='val_auc',min_delta=0.0001,patience=500,mode='max',restore_best_weights=True)],

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=Epoch, validation_data=(x_test, y_test) ,validation_freq = 1,
              callbacks=callbacks_list),

    model.summary()

    return history,model

