
import pandas as pd
import xgboost as xgb
import numpy as np

from ross_util import submit_to_kaggle, rmspe_xgb, inverse_transform_y, transform_y
from ross_data import get_dataset, load_data


VALIDATION_WEEKS = 0
FILT_STORES_FOR_TRAIN = True
FILT_STORES_FOR_VALID = True

features_base = ['Store', 'DayOfWeek', 'Promo', 'Year', 'Month',
        'Day', 'WeekOfYear', 'DayOfYear', 'StateHolidayN', 'SchoolHoliday', 'CompeteOpenMonths', 'Promo2OpenWeeks',
        'Latest_Promo2_Start_Month', 'CompetitionDistance', 'StoreTypeN', 'AssortmentN', 'Promo2IntervalN',
        'CompetitionOpenSinceYear', 'Promo2SinceYear', 'StateN']

features_weather = ['Max_TemperatureC',
                     'Mean_TemperatureC', 'Min_TemperatureC', 'Max_Humidity', 'Mean_Humidity', 'Min_Humidity',
                     'Max_Wind_SpeedKm_h', 'Mean_Wind_SpeedKm_h', 'CloudCover', 'Events']

features_googletrend = ['Trend_Val_DE', 'Trend_Val_State']

features_forward_backward = ['Promo_Backward', 'Promo_Forward', 'SchoolHoliday_Backward', 'SchoolHoliday_Forward',
                'StateHoliday_Backward', 'StateHoliday_Forward']
features_count_fb = ['Promo_Count_BW', 'Promo_Count_FW', 'SchoolHoliday_Count_BW', 'SchoolHoliday_Count_FW',
                      'StateHoliday_Count_BW', 'StateHoliday_Count_FW']

#features = features_base + features_weather + features_googletrend + features_forward_backward + features_count_fb
features = features_base


def extract_X(dataset):
    assert isinstance(dataset, dict)
    X = pd.DataFrame({name: arr.reshape(-1) for (name, arr) in dataset.items()})
    X = X[features]
    return X


def log_cosh_obj(preds, dtrain):
    x = preds - dtrain.get_label()
    grad = np.tanh(x)
    hess = 1 / np.cosh(x)**2
    return grad, hess


def fair_obj(preds, dtrain):
    """y = c * abs(x) - c * np.log(abs(abs(x) + c))"""
    x = preds -dtrain.get_label()
    c = 1
    den = abs(x) + c
    grad = c*x / den
    hess = c*c / den ** 2
    return grad, hess


def xgb_train(model_file):
    train, valid, test = get_dataset(validation_weeks=VALIDATION_WEEKS,
                                     filt_stores_for_train=FILT_STORES_FOR_TRAIN,
                                     filt_stores_for_valid=FILT_STORES_FOR_VALID)

    X_train, y_train = train
    # X_train is a dict
    X_train = extract_X(X_train)
    y_train = transform_y(y_train.reshape(-1))

    print('X_train.shape:', X_train.shape)
    print('y_train.shape:', y_train.shape)
    dtrain = xgb.DMatrix(X_train, y_train)
    watchlist = [(dtrain, 'train')]

    if valid is not None:
        X_valid, y_valid = valid
        X_valid = extract_X(X_valid)
        y_valid = transform_y(y_valid.reshape(-1))

        print('X_valid.shape:', X_valid.shape)
        print('y_valid.shape:', y_valid.shape)
        dvalid = xgb.DMatrix(X_valid, y_valid)
        watchlist.append((dvalid, 'eval'))
    param = {
        #'objective': 'reg:linear',
        'max_depth': 8,
        'gamma': 0,
        'min_child_weight': 1,
        'eta': 0.3,
        'subsample': 0.8,
        'colsample_bytree': 0.97,
        'tree_method': 'hist',
        # 'max_bin': 16
    }
    nrounds = 320
    bst = xgb.train(param, dtrain, nrounds, watchlist,
                    obj=fair_obj,
                    #early_stopping_rounds=50,
                    feval=rmspe_xgb,
                    #xgb_model='xgb_simple.model'
                    )
    #bst = xgb.Booster(model_file='xgb_1.model')
    bst.save_model(model_file)


def xgb_predict(model_file, submission_file):
    bst = xgb.Booster(model_file=model_file)
    train, test = load_data()
    X_test = extract_X(test)
    dtest = xgb.DMatrix(X_test)
    y_pred = bst.predict(dtest)
    y_pred = inverse_transform_y(y_pred)
    pred_df = pd.DataFrame({'Id': range(1, len(y_pred)+1), 'Sales': y_pred})
    pred_df.to_csv(submission_file, index=False)
    submit_to_kaggle(submission_file)

if __name__=='__main__':
    model_file = 'xgb_simple.model'
    submission_file = './output/pred_xgb_simple.csv'

    xgb_train(model_file)
    xgb_predict(model_file, submission_file)
