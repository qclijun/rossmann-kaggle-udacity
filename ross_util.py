import os

import numpy as np
import pandas as pd

import keras.backend as K

MIN_LOG_SALES = 3.8286
MAX_LOG_SALES = 10.6347

MEAN_LOG_SALES = 8.7576
STD_LOG_SALES = 0.4253

USE_LOG_Y = True

def mspe(y_true, y_pred):
    err = ((y_true - y_pred) /y_true)**2
    err = np.mean(err, axis=-1)
    return err


def rmspe(y_true, y_pred):
    return mspe(y_true, y_pred)**0.5


def mape(y_true, y_pred):
    err = np.abs((y_true - y_pred)/y_true)
    err = np.mean(err, axis=-1)
    return err


def submit_to_kaggle(filename, message=""):
    print('submit {} to kaggle...'.format(filename))
    cmd = 'kaggle competitions submit -c rossmann-store-sales -f "{}" -m "{}"'.format(filename, message)
    output = os.popen(cmd)
    print(output.read())


def transform_y(y):
    if USE_LOG_Y:
        return (np.log(y) - MIN_LOG_SALES)/ (MAX_LOG_SALES - MIN_LOG_SALES)
    else:
        return y

def inverse_transform_y(transformed_y):
    if USE_LOG_Y:
        return np.exp(transformed_y * (MAX_LOG_SALES - MIN_LOG_SALES) + MIN_LOG_SALES)
    else:
        return transformed_y

def mspe_keras(y_true, y_pred):
    if USE_LOG_Y:
        y_true = K.exp(y_true * (MAX_LOG_SALES - MIN_LOG_SALES) + MIN_LOG_SALES)
        y_pred = K.exp(y_pred * (MAX_LOG_SALES - MIN_LOG_SALES) + MIN_LOG_SALES)
    diff = K.pow((y_true - y_pred) / y_true, 2)
    return K.mean(diff, axis=-1)

def mape_keras(y_true, y_pred):
    if USE_LOG_Y:
        y_true = K.exp(y_true * (MAX_LOG_SALES - MIN_LOG_SALES) + MIN_LOG_SALES)
        y_pred = K.exp(y_pred * (MAX_LOG_SALES - MIN_LOG_SALES) + MIN_LOG_SALES)
    diff = K.abs((y_true - y_pred) / y_true)
    return K.mean(diff, axis=-1)


def rmspe_lgb(y_true, y_pred):
    #print('y_true:',type(y_true))
    #print('y_pred', type(y_pred))
    y_pred = y_pred.get_label()
    y_true = inverse_transform_y(y_true)
    y_pred = inverse_transform_y(y_pred)
    err = rmspe(y_true, y_pred)
    return 'rmspe', err, False


def rmspe_xgb(y_true, y_pred):
    y_pred = y_pred.get_label()
    y_true = inverse_transform_y(y_true)
    y_pred = inverse_transform_y(y_pred)
    err = rmspe(y_true, y_pred)
    return 'rmspe', err

def predict_with_models(models, X_test):
    y_pred = np.mean([model.predict(X_test) for model in models], axis=0)
    return y_pred


def write_submission(models, X_test, filename):
    print('write submission file:', filename)
    y_pred = predict_with_models(models, X_test)
    submit_df = pd.DataFrame({'Id': range(1, len(y_pred)+1), 'Sales': y_pred})
    submit_df.to_csv(filename, index=False)


