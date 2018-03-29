import os
import pandas as pd
import h5py
import keras
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split

from ross_util import mspe, rmspe_lgb, rmspe_xgb, transform_y, inverse_transform_y, submit_to_kaggle
from ross_data import get_dataset, load_data
from ross_main import load_models_from_hdf

MODEL_FILE = './models/model_1.hdf5'
EMBED_FILE = './embedding_features.h5'

DO_EVAL = False
DO_PREDICT = True


embedding_feature_names={
   'Store': 50,
   'DayOfWeek': 6,
   'Promo': 1,
   'Year': 2,
   'Month': 6,
   'Day': 16,
   'StateHoliday': 3,
   'SchoolHoliday': 1,
   'CompeteMonths': 13,
   'Promo2Weeks': 13,
   'LatestPromo2Months': 2,
   'Distance': 1,
   'StoreType': 2,
   'Assortment': 2,
   'PromoInterval': 3,
   'CompetitionOpenSinceYear': 9,
   'Promo2SinceYear': 4,
   'State': 6,
   'WeekOfYear': 27,
   'Temperature': 3,
   'Humidity': 3,
   'Wind': 2,
   'Cloud': 5,
   'WeatherEvents': 11,
   'Promo_Forward': 4,
   'Promo_Backward': 4,
   'StateHoliday_Forward': 4,
   'StateHoliday_Backward': 4,
   'SchoolHoliday_Forward': 4,
   'SchoolHoliday_Backward': 4,
   'StateHoliday_Count_Forward': 2,
   'StateHoliday_Count_Backward': 2,
   'SchoolHoliday_Count_Forward': 4,
   'SchoolHoliday_Count_Backward': 4,
   'Promo_Count_Forward': 3,
   'Promo_Count_Backward': 3,
   'GoogleTrend_DE': 1,
   'GoogleTrend_State': 1,
    'Sales_Per_Day': 1,
    'Customers_Per_Day': 1,
    'Sales_Per_Customer': 1,
    'Sales_Promo': 1,
    'Sales_Holiday': 1,
    'Sales_Saturday': 1,
    'Open_Ratio': 1,
    'SchoolHoliday_Ratio': 1,
    'Before_Long_Closed': 3,
    'After_Long_Closed': 3
}


def get_embed_fea_names():
    names = []
    for name, length in embedding_feature_names.items():
        if length==1:
            names.append(name)
        else:
            names.extend([name+str(i+1) for i in range(length)])
    return names


def write_embed_features(model_file, embed_file):
    nn_model = load_models_from_hdf([model_file])[0]
    model = nn_model.model
    #model.summary()
    concat_layer = None
    for layer in model.layers:
        if layer.name == 'concatenate_1':
            concat_layer = layer

    assert concat_layer is not None
    model2 = keras.models.Model(inputs=model.inputs, outputs=concat_layer.output)

    train_set, valid_set, test = get_dataset(validation_weeks=0, filt_stores_for_train=True,
                                             filt_stores_for_valid=True)
    X_train, y_train = train_set
    train_sales = transform_y(y_train)

    train_embed = model2.predict(nn_model.preprocess(X_train), batch_size=4096)
    test_embed = model2.predict(nn_model.preprocess(test), batch_size=4096)
    print('train_embed.shape:', train_embed.shape)
    print('test_embed.shape:', test_embed.shape)
    with h5py.File(embed_file, 'w') as h5f:
        h5f.create_dataset('train_embed', data=train_embed)
        h5f.create_dataset('train_sales', data=train_sales)
        h5f.create_dataset('test_embed', data=test_embed)
    return train_embed, train_sales, test_embed


def load_embed_features(h5filename):
    with h5py.File(h5filename) as h5f:
        train_embed = h5f['train_embed'][...]
        train_sales = h5f['train_sales'][...]
        test_embed = h5f['test_embed'][...]
        return train_embed, train_sales, test_embed

def get_embed_features():
    if not os.path.exists(EMBED_FILE):
        return write_embed_features(model_file=MODEL_FILE, embed_file=EMBED_FILE)
    return load_embed_features(EMBED_FILE)


def lgb_predict(lgb_model, X_test, submission_file):
    bst = lgb.Booster(model_file=lgb_model)
    y_pred = bst.predict(X_test)
    y_pred = inverse_transform_y(y_pred)

    df = pd.DataFrame({'Id': range(1, len(y_pred) + 1), 'Sales': y_pred})
    df.to_csv(submission_file, index=False)

def lgb_train(lgb_model_output, init_model=None):
    train_embed, train_sales, test_embed = get_embed_features()
    train_sales = train_sales.reshape(-1)
    feature_names = get_embed_fea_names()
    assert len(feature_names) == train_embed.shape[1]

    if DO_EVAL:
        X_train, X_val, y_train, y_val = train_test_split(train_embed, train_sales, test_size=0.048, shuffle=False)
    else:
        X_train = train_embed
        y_train = train_sales
    X_test = test_embed
    print('X_train:', X_train.shape)
    print('y_train:', y_train.shape)

    lgb_train = lgb.Dataset(X_train, y_train, feature_name=feature_names)
    valid_sets = [lgb_train]
    valid_names = ['train']
    if DO_EVAL:
        lgb_eval = lgb.Dataset(X_val, y_val, feature_name=feature_names)
        valid_sets.append(lgb_eval)
        valid_names.append('eval')

    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression_l1',
        'num_leaves': 31,
        'min_data_in_leaf': 100,
        'max_depth': -1,
        'learning_rate': 0.1,
        'feature_fraction': 0.95,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_bin': 255,
        'verbose': 0,
        'tree_learner': 'data',
        #'nthread': 4
    }
    bst = lgb.train(params,
                    lgb_train,
                    num_boost_round=5000,
                    valid_sets=valid_sets,
                    valid_names=valid_names,
                    feval=rmspe_lgb,
                    #early_stopping_rounds=100,
                    init_model=init_model,
                    verbose_eval=True,
                    )
    bst.save_model(lgb_model_output)
    importance = bst.feature_importance(importance_type='gain')
    importance_df = pd.DataFrame({'name': feature_names, 'importance': importance})
    importance_df.to_csv('importance.csv', index=False)

    if DO_PREDICT:
        submission_file = 'lgb_pred.csv'
        lgb_predict(lgb_model_output, X_test, submission_file)
        submit_to_kaggle(submission_file)


def xgb_train():

    train_embed, train_sales, test_embed = get_embed_features()
    train_sales = train_sales.reshape(-1)
    feature_names = get_embed_fea_names()
    assert len(feature_names) == train_embed.shape[1]

    if DO_EVAL:
        X_train, X_val, y_train, y_val = train_test_split(train_embed, train_sales, test_size=0.048, shuffle=False)
    else:
        X_train = train_embed
        y_train = train_sales
    X_test = test_embed
    print('X_train:', X_train.shape)
    print('y_train:', y_train.shape)

    dtrain = xgb.DMatrix(X_train, y_train, feature_names=feature_names)
    watchlist = [(dtrain, 'train')]
    if DO_EVAL:
        dval = xgb.DMatrix(X_val, y_val, feature_names=feature_names)
        watchlist.append((dval, 'eval'))
    #dtest = xgb.DMatrix(X_test)

    param = {
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'max_depth': 10,
        'gamma': 0,
        'min_child_weight': 1,
        'eta': 0.02,
        'subsample': 0.8,
        'colsample_bytree': 0.95,
        'tree_method': 'hist'
    }
    nrounds = 5000
    bst = xgb.train(param, dtrain, nrounds, watchlist,
                    early_stopping_rounds=100,
                    feval=rmspe_xgb,
                    #model_file='xgb_1.model'
                    )
    #bst = xgb.Booster(model_file='xgb_1.model')
    bst.save_model('xgb_1.model')
    if not DO_EVAL:
        dtest = xgb.DMatrix(X_test)
        y_pred = bst.predict(dtest)
        y_pred = inverse_transform_y(y_pred)
        df = pd.DataFrame({'Id': range(1, len(y_pred) + 1), 'Sales': y_pred})
        df.to_csv('./output/xgb_pred1.csv', index=False)
        submit_to_kaggle('./output/xgb_pred1.csv')

def main():
    #xgb_train()
    lgb_train(lgb_model_output='lgb_model.txt', init_model=None)



if __name__ == '__main__':
    main()
