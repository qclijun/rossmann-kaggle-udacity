import numpy as np
import h5py
import pandas as pd

from sklearn.model_selection import train_test_split

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.models import  load_model
from pathlib import Path

from .ross_model import NN_Embedding
from .preprocess_data import load_data

N_NETWORKS = 1
TRAIN_RATIO = 0.97
EPOCHS = 20
BATCH_SIZE = 4096
INIT_EPOCH = 0

FILTER_STORES_TRAIN = 1
FILTER_STORES_VALID = 1

saved_model_file = './output/checkpt2/weights.20-0.0001.hdf5'
submission_file = './output/pred_2018-2-27_97%_19.csv'

def predict_with_models(models, X_test):
    y_pred = np.mean([model.predict(X_test) for model in models], axis=0)
    return y_pred


def write_submission(models, X_test,  filename):
    y_pred = predict_with_models(models, X_test)
    submit_df = pd.DataFrame({'Id': range(1, len(y_pred)+1), 'Sales': y_pred})
    submit_df.to_csv(filename, index=False)


def split_dataset(train_set):
    # train_set is a dict
    result = train_test_split(*tuple(train_set.values()), test_size=1 - TRAIN_RATIO, shuffle=False)
    train = []
    valid = []
    for i, arr in enumerate(result):
        if i%2==0:
            train.append(arr)
        else:
            valid.append(arr)
    train = dict(zip(train_set.keys(), train))
    valid = dict(zip(train_set.keys(), valid))
    return train, valid


def get_stores_in_test(test_set):
    return np.unique(test_set['Store'])


def filter_stores(val_set, stores_in_test):
    mask = [s in stores_in_test for s in val_set['Store'].reshape(-1)]
    result = {k: v[mask] for k,v in val_set.items()}
    return result


def main():
    train, test = load_data()
    stores_in_test = set(get_stores_in_test(test))

    #train = train[:10000,:]

    if TRAIN_RATIO < 1:
        train_set, valid_set = split_dataset(train)
        if FILTER_STORES_TRAIN:
            train_set = filter_stores(train_set, stores_in_test)
        if FILTER_STORES_VALID:
            valid_set = filter_stores(valid_set, stores_in_test)
        X_train = train_set
        y_train = train_set['Sales']
        X_valid = valid_set
        y_valid = valid_set['Sales']
        train_set = (X_train, y_train)
        valid_set = (X_valid, y_valid)
    else:
        train_set = train
        if FILTER_STORES_TRAIN:
            train_set = filter_stores(train_set, stores_in_test)
        valid_set = None
        X_train = train_set
        y_train = train_set['Sales']
        train_set = (X_train, y_train)

    #print('X[0]:',X[0])
    #print('y[0]:', y[0])
    X_test = test

    #stores_in_test = np.unique(X_test[:, 1]).astype(np.int32)
    #print(X_test[:10, 0])

    models = []
    for i in range(N_NETWORKS):
         model = NN_Embedding(print_model_summary=True, save_checkpt=TRAIN_RATIO < 1)
         if INIT_EPOCH>0:
            model.model.load_weights(saved_model_file)
         model.fit(train_set, valid_set, batch_size=BATCH_SIZE, epochs=EPOCHS + INIT_EPOCH, init_epoch=INIT_EPOCH)
         model.eval()
         #print('max 0.01% error in train set:')
         #model.display_max_error_samples(0.0001)
         #eval_out = model.model.evaluate(model.X_valid, model.y_valid, batch_size=len(model.y_valid))
         #print(eval_out)
         model.model.save('./models_'+str(i+1)+'.hdf5')
         models.append(model)

    print('write submission file:')
    write_submission(models, X_test, submission_file)


if __name__=='__main__':
    main()





