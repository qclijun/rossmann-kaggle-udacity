
import numpy as np
import pandas as pd
import keras
import xgboost as xgb
from sklearn.model_selection import train_test_split

from .ross_model import NN_Embedding, rmspe
from .preprocess_data import load_data


model_file = './output/checkpt2/weights.20-0.0117.hdf5'

def get_embed_features(model_file):
    nn_model = NN_Embedding()
    model = nn_model.model
    model.load_weights(model_file)
    for layer in model.layers:
        if layer=='concatenate_1':
            concat_layer = layer

    model2 = keras.models.Model(inputs=model.inputs, outputs=concat_layer.output)
    train, test = load_data()
    train_sales = train['Sales'].reshape(-1)
    train_embed = model2.predict(nn_model.preprocess(train), batch_size=4096)
    test_embed = model2.predict(nn_model.preprocess(test), batch_size=4096)
    print('train_embed.shape:', train_embed.shape)
    print('test_embed.shape:', test_embed.shape)

    return train_embed, train_sales, test_embed


def main():
    param = {
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'max_depth': 7,
        'gamma': 0,
        'min_child_weight': 1,
        'eta': 0.3,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'tree_method': 'hist'
    }
    train_embed, train_sales, test_embed = get_embed_features(model_file)
    train_sales = np.log(train_sales)
    X_train, X_val, y_train, y_val = train_test_split(train_embed, train_sales, test_size=0.03, shuffle=False)
    X_test = test_embed
    dtrain = xgb.DMatrix(X_train, y_train)
    dval = xgb.DMatrix(X_val, y_val)
    dtest = xgb.DMatrix(X_test)

    nrounds = 200
    watchlist = [(dtrain, 'train'), (dval, 'eval')]
    bst = xgb.train(param, dtrain, nrounds, watchlist, early_stopping_rounds=50)

    bst.save_model('xgb_1.model')

    y_val_pred = bst.predict(dval)
    y_train_pred = bst.predict(dtrain)
    print('rmspe(train):', rmspe(np.exp(y_train), np.exp(y_train_pred)))
    print('rmspe(validation):', rmspe(np.exp(y_val), np.exp(y_val_pred)))

    y_test_pred = bst.predict(dtest)
    df = pd.DataFrame({'Id': range(1, len(y_test_pred) + 1), 'Sales': np.exp(y_test_pred)})
    df.to_csv('./output/xgb_embed_1.csv', index=False)