import numpy as np
import pandas as pd
import xgboost as xgb

from ross_main import get_dataset


train_set, valid_set, X_test = get_dataset()
X_train, y_train = train_set
X_valid, y_valid = valid_set

train_df = pd.DataFrame({name: val.reshape(-1) for name, val in X_train.items()})
y_train = train_df['Sales']
X_train = train_df.drop('Sales', axis=1)

valid_df = pd.DataFrame({name: val.reshape(-1) for name, val in X_valid.items()})
y_valid = valid_df['Sales']
X_valid = valid_df.drop('Sales', axis=1)

X_test = pd.DataFrame({name: val.reshape(-1) for name, val in X_test.items()})

dtrain = xgb.DMatrix(X_train, np.log(y_train))
dvalid = xgb.DMatrix(X_valid, np.log(y_valid))
dtest = xgb.DMatrix(X_test)

params = {
    'objective':'reg:linear',
    'eta': 0.02,
    'max_depth': 12,
    'subsample': 0.9,
    'colsample_bytree': 0.7,
    'verbose': 0
}

def rmspe_xgb(y_pred, dtrain: xgb.DMatrix):
    y_true = dtrain.get_label()
    y_true = np.exp(y_true)
    y_pred = np.exp(y_pred)
    err = np.mean(((y_true-y_pred)/y_true)**2)
    err = err**0.5
    return 'RMSPE', err

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
nrounds = 200
# bst = xgb.train(params, dtrain, nrounds, watchlist,
#                 early_stopping_rounds=50,
#                 feval=rmspe_xgb,xgb_model='./models/xgb2.model')
#
# bst.save_model('./models/xgb3.model')
bst = xgb.Booster()
bst.load_model('./models/xgb3.model')
y_pred = bst.predict(dtest)
sales = np.exp(y_pred)
df=pd.DataFrame({'Id': range(1, len(sales)+1), 'Sales': sales})
df.to_csv('xgb_predict.csv', index=False)
