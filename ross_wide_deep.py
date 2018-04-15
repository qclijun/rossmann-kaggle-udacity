from pathlib import Path
import math

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.python.feature_column.feature_column

import keras
from keras.layers import Embedding, Dense, Input, Reshape, Dropout, Concatenate, Add, Activation
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from keras.losses import logcosh, mae, mse, mape
import matplotlib.pyplot as plt

from ross_util import mspe_keras, mape_keras, transform_y, inverse_transform_y, rmspe
from ross_data import get_dataset



USE_LOG_Y = True
USE_SAMPLE_WEIGHT = False
DISPLAY_METRIC = True

EMBEDDING_DROPOUT = 0.03
DNN_DROPOUT = 0.2
ACTIVATION = 'relu' #relu, elu, selu, PReLu, LeakyReLu, Swish
METRIC = 'mspe' # mape, mspe

FEATURES_TREND = True
FEATURES_WEATHER = True
FEATURES_FB = True
FEATURES_COUNT_FB = True
FEATURES_LONGCLOSED = True
FEATURES_SHOPAVG = True
FEATURES_PROMODECAY = False
FEATURES_SUMMER = False


MODEL_BASELINE = False

VALIDATION_WEEKS = 6
DO_EVAL = VALIDATION_WEEKS > 0

BATCH_SIZE = 2048
EPOCHS = 25

SHUFFLE_BUFFER_SIZE = 10280

_DNN_LEARNING_RATE = 0.001
_LINEAR_LEARNING_RATE = 0.005

train_set, valid_set, test = get_dataset(validation_weeks=VALIDATION_WEEKS)

print(test['After_Long_Closed'].dtype)
print(test['After_Long_Closed'].shape)

N_TRAIN = len(train_set[1])
N_VALID = len(valid_set[1]) if DO_EVAL else 0
N_TEST = len(test)

MODEL_PATH = Path('./models/logs')
MODEL_PATH.mkdir(exist_ok=True)

def build_model_columns():
    base_columns = []
    crossed_columns = []
    deep_columns = []

    # Store
    store_column = tf.feature_column.categorical_column_with_identity(
        key='Store', num_buckets=1115
    )
    store_embedding = tf.feature_column.embedding_column(store_column, 50)
    deep_columns.append(store_embedding)

    # DayOfWeek
    dow_column = tf.feature_column.categorical_column_with_identity(
        key='DayOfWeek', num_buckets=7
    )
    dow_embedding = tf.feature_column.embedding_column(dow_column, 6)
    deep_columns.append(dow_embedding)

    # Promo
    promo_column = tf.feature_column.numeric_column('Promo')
    deep_columns.append(promo_column)

    # Year
    year_column = tf.feature_column.categorical_column_with_identity(
        key='Year', num_buckets=3
    )
    year_embedding = tf.feature_column.embedding_column(year_column, 2)
    deep_columns.append(year_embedding)

    # Month
    month_column = tf.feature_column.categorical_column_with_identity(
        key='Month', num_buckets=12
    )
    month_embedding = tf.feature_column.embedding_column(month_column, 6)
    deep_columns.append(month_embedding)

    # Day
    day_column = tf.feature_column.categorical_column_with_identity(
        key='Day', num_buckets=31
    )
    day_embedding = tf.feature_column.embedding_column(day_column, 16)
    deep_columns.append(day_embedding)

    # StateHoliday
    stateholiday_column = tf.feature_column.categorical_column_with_identity(
        key='StateHolidayN', num_buckets=4
    )
    stateholiday_embedding = tf.feature_column.embedding_column(stateholiday_column, 2)
    deep_columns.append(stateholiday_embedding)

    # SchoolHoliday
    schoolholiday_column = tf.feature_column.numeric_column('SchoolHoliday')
    deep_columns.append(schoolholiday_column)

    # CompeteOpenMonths
    competemonths_column = tf.feature_column.categorical_column_with_identity(
        key='CompeteOpenMonths', num_buckets=25
    )
    competemonths_embedding = tf.feature_column.embedding_column(competemonths_column, 13)
    deep_columns.append(competemonths_embedding)

    # Promo2OpenWeeks
    promo2weeks_column = tf.feature_column.categorical_column_with_identity(
        key='Promo2OpenWeeks', num_buckets=26
    )
    promo2weeks_embedding = tf.feature_column.embedding_column(promo2weeks_column, 13)
    deep_columns.append(promo2weeks_embedding)

    # Latest_Promo2_Start_Month
    lastestpromo2months_column = tf.feature_column.categorical_column_with_identity(
        key='Latest_Promo2_Month', num_buckets=4
    )
    lastestpromo2months_embedding = tf.feature_column.embedding_column(lastestpromo2months_column, 2)
    deep_columns.append(lastestpromo2months_embedding)

    # CompetitionDistance
    distance_column = tf.feature_column.numeric_column('CompetitionDistance')
    deep_columns.append(distance_column)

    # StoreTypeN
    storetype_column = tf.feature_column.categorical_column_with_identity(
        key='StoreTypeN', num_buckets=4
    )
    storetype_embedding = tf.feature_column.embedding_column(storetype_column, 2)
    deep_columns.append(storetype_embedding)

    # AssortmentN
    assortment_column = tf.feature_column.categorical_column_with_identity(
        key='AssortmentN', num_buckets=3
    )
    assortment_embedding = tf.feature_column.embedding_column(assortment_column, 2)
    deep_columns.append(assortment_embedding)

    # Promo2IntervalN
    promointerval_column = tf.feature_column.categorical_column_with_identity(
        key='Promo2IntervalN', num_buckets=4
    )
    promointerval_embedding = tf.feature_column.embedding_column(promointerval_column, 3)
    deep_columns.append(promointerval_embedding)

    # CompetitionOpenSinceYear
    competeyear_column = tf.feature_column.categorical_column_with_identity(
        key='CompetitionOpenSinceYear', num_buckets=17
    )
    competeyear_embedding = tf.feature_column.embedding_column(competeyear_column, 9)
    deep_columns.append(competeyear_embedding)

    # Promo2SinceYear
    promoyear_column = tf.feature_column.categorical_column_with_identity(
        key='Promo2SinceYear', num_buckets=8
    )
    promo2year_embedding = tf.feature_column.embedding_column(promoyear_column, 4)
    deep_columns.append(promo2year_embedding)

    # State
    state_column = tf.feature_column.categorical_column_with_identity(
        key='StateN', num_buckets=12
    )
    state_embedding = tf.feature_column.embedding_column(state_column, 6)
    deep_columns.append(state_embedding)

    # WeekOfYear
    weekofyear_column = tf.feature_column.categorical_column_with_identity(
        key='WeekOfYear', num_buckets=53
    )
    weekofyear_embedding = tf.feature_column.embedding_column(weekofyear_column, 27)
    deep_columns.append(weekofyear_embedding)

    if FEATURES_WEATHER:
        temperature_column = tf.feature_column.numeric_column(
            key='Temperature', shape=(3,)
        )
        deep_columns.append(temperature_column)

        humidity_column = tf.feature_column.numeric_column(
            key='Humidity', shape=(3,)
        )
        deep_columns.append(humidity_column)

        wind_column = tf.feature_column.numeric_column(
            key='Wind', shape=(2,)
        )
        deep_columns.append(wind_column)

        cloud_column = tf.feature_column.categorical_column_with_identity(
            key='CloudCover', num_buckets=9
        )
        cloud_embedding = tf.feature_column.embedding_column(cloud_column, 5)
        deep_columns.append(cloud_embedding)

        weatherevent_column = tf.feature_column.categorical_column_with_identity(
            key='Events', num_buckets=22
        )
        weatherevent_embedding = tf.feature_column.embedding_column(weatherevent_column, 11)
        deep_columns.append(weatherevent_embedding)

    if FEATURES_FB:
        # Promo
        promo_forward_column = tf.feature_column.categorical_column_with_identity(
            key='Promo_Forward', num_buckets=8
        )
        promo_forward_embedding = tf.feature_column.embedding_column(promo_forward_column, 4)
        deep_columns.append(promo_forward_embedding)

        promo_backward_column = tf.feature_column.categorical_column_with_identity(
            key='Promo_Backward', num_buckets=8
        )
        promo_backward_embedding = tf.feature_column.embedding_column(promo_backward_column, 4)
        deep_columns.append(promo_backward_embedding)

        # StateHoliday
        stateholiday_forward_column = tf.feature_column.categorical_column_with_identity(
            key='StateHoliday_Forward', num_buckets=8
        )
        stateholiday_forward_embedding = tf.feature_column.embedding_column(stateholiday_forward_column, 4)
        deep_columns.append(stateholiday_forward_embedding)

        stateholiday_backward_column = tf.feature_column.categorical_column_with_identity(
            key='StateHoliday_Backward', num_buckets=8
        )
        stateholiday_backward_embedding = tf.feature_column.embedding_column(stateholiday_backward_column, 4)
        deep_columns.append(stateholiday_backward_embedding)

        # SchoolHoliday
        schoolholiday_forward_column = tf.feature_column.categorical_column_with_identity(
            key='SchoolHoliday_Forward', num_buckets=8
        )
        schoolholiday_forward_embedding = tf.feature_column.embedding_column(schoolholiday_forward_column, 4)
        deep_columns.append(schoolholiday_forward_embedding)

        schoolholiday_backward_column = tf.feature_column.categorical_column_with_identity(
            key='SchoolHoliday_Backward', num_buckets=8
        )
        schoolholiday_backward_embedding = tf.feature_column.embedding_column(schoolholiday_backward_column, 4)
        deep_columns.append(schoolholiday_backward_embedding)

    if FEATURES_COUNT_FB:
        # StateHoliday_Count
        stateholiday_cf_column = tf.feature_column.categorical_column_with_identity(
            key='StateHoliday_Count_FW', num_buckets=3
        )
        stateholiday_cf_embedding = tf.feature_column.embedding_column(stateholiday_cf_column, 2)
        deep_columns.append(stateholiday_cf_embedding)

        stateholiday_cb_column = tf.feature_column.categorical_column_with_identity(
            key='StateHoliday_Count_BW', num_buckets=3
        )
        stateholiday_cb_embedding = tf.feature_column.embedding_column(stateholiday_cb_column, 2)
        deep_columns.append(stateholiday_cb_embedding)

        # SchoolHoliday_Count
        schoolholiday_cf_column = tf.feature_column.categorical_column_with_identity(
            key='SchoolHoliday_Count_FW', num_buckets=8
        )
        schoolholiday_cf_embedding = tf.feature_column.embedding_column(schoolholiday_cf_column, 4)
        deep_columns.append(schoolholiday_cf_embedding)

        schoolholiday_cb_column = tf.feature_column.categorical_column_with_identity(
            key='SchoolHoliday_Count_BW', num_buckets=8
        )
        schoolholiday_cb_embedding = tf.feature_column.embedding_column(schoolholiday_cb_column, 4)
        deep_columns.append(schoolholiday_cb_embedding)

        # Promo_Count
        promo_cf_column = tf.feature_column.categorical_column_with_identity(
            key='Promo_Count_FW', num_buckets=8
        )
        promo_cf_embedding = tf.feature_column.embedding_column(promo_cf_column, 4)
        deep_columns.append(promo_cf_embedding)

        promo_cb_column = tf.feature_column.categorical_column_with_identity(
            key='Promo_Count_BW', num_buckets=8
        )
        promo_cb_embedding = tf.feature_column.embedding_column(promo_cb_column, 4)
        deep_columns.append(promo_cb_embedding)

    if FEATURES_PROMODECAY:
        promo_decay_column = tf.feature_column.categorical_column_with_identity(
            key='PromoDecay', num_buckets=6
        )
        promo_decay_embedding = tf.feature_column.embedding_column(promo_decay_column, 3)
        deep_columns.append(promo_decay_embedding)

    if FEATURES_TREND:
        googletrend_de_column = tf.feature_column.numeric_column('Trend_Val_DE')
        deep_columns.append(googletrend_de_column)

        googletrend_state_column = tf.feature_column.numeric_column('Trend_Val_State')
        deep_columns.append(googletrend_state_column)

    if FEATURES_SHOPAVG:
        avg_sales_column = tf.feature_column.numeric_column('Avg_Sales', shape=(8,))
        deep_columns.append(avg_sales_column)

    if FEATURES_LONGCLOSED:
        before_longclosed_column = tf.feature_column.categorical_column_with_identity(
            key='Before_Long_Closed', num_buckets=6, default_value=5
        )
        before_longclosed_embedding = tf.feature_column.embedding_column(before_longclosed_column, 3)
        deep_columns.append(before_longclosed_embedding)

        after_longclosed_column = tf.feature_column.categorical_column_with_identity(
            key='After_Long_Closed', num_buckets=6, default_value=5
        )
        after_longclosed_embedding = tf.feature_column.embedding_column(after_longclosed_column, 3)
        deep_columns.append(after_longclosed_embedding)


    wide_columns = base_columns + crossed_columns

    return wide_columns, deep_columns


def process_features(X):
    features = {}

    store_index = X['Store'].astype(np.int32) - 1
    features['Store'] = store_index

    day_of_week = X['DayOfWeek'].astype(np.int32) - 1
    features['DayOfWeek'] = day_of_week

    promo = X['Promo']
    features['Promo'] = promo

    year = X['Year'].astype(np.int32) - 2013
    features['Year'] = year

    month = X['Month'].astype(np.int32) - 1
    features['Month'] = month

    day = X['Day'].astype(np.int32) - 1
    features['Day'] = day

    state_holiday = X['StateHolidayN'].astype(np.int32)
    features['StateHolidayN'] = state_holiday

    school_holiday = X['SchoolHoliday'].astype(np.int32)
    features['SchoolHoliday'] = school_holiday

    has_competition_for_months = X['CompeteOpenMonths'].astype(np.int32)
    has_competition_for_months[has_competition_for_months < 0] = 0
    features['CompeteOpenMonths'] = has_competition_for_months

    has_promo2_for_weeks = X['Promo2OpenWeeks'].astype(np.int32)
    has_promo2_for_weeks[has_promo2_for_weeks < 0] = 0
    features['Promo2OpenWeeks'] = has_promo2_for_weeks

    latest_promo2_for_months = X['Latest_Promo2_Month'].astype(np.int32)
    features['Latest_Promo2_Month'] = latest_promo2_for_months

    log_distance = X['CompetitionDistance']
    features['CompetitionDistance'] = log_distance

    storeType = X['StoreTypeN'].astype(np.int32) - 1
    features['StoreTypeN'] = storeType

    assortment = X['AssortmentN'].astype(np.int32) - 1
    features['AssortmentN'] = assortment

    promoInterval = X['Promo2IntervalN'].astype(np.int32)
    features['Promo2IntervalN'] = promoInterval

    CompetitionOpenSinceYear = X['CompetitionOpenSinceYear'].astype(np.int32) - 1999
    CompetitionOpenSinceYear[CompetitionOpenSinceYear < 0] = 0
    features['CompetitionOpenSinceYear'] = CompetitionOpenSinceYear

    Promo2SinceYear = X['Promo2SinceYear'].astype(np.int32) - 2008
    Promo2SinceYear[Promo2SinceYear < 0] = 0
    features['Promo2SinceYear'] = Promo2SinceYear

    state = X['StateN'].astype(np.int32) - 1
    features['StateN'] = state

    week_of_year = X['WeekOfYear'].astype(np.int32) - 1
    features['WeekOfYear'] = week_of_year

    if FEATURES_WEATHER:
        temperature = np.concatenate((X['Max_TemperatureC'], X['Mean_TemperatureC'], X['Min_TemperatureC']), axis=1)
        features['Temperature'] = temperature

        humidity = np.concatenate((X['Max_Humidity'], X['Mean_Humidity'], X['Min_Humidity']), axis=1)
        features['Humidity'] = humidity

        wind = np.concatenate((X['Max_Wind_SpeedKm_h'], X['Mean_Wind_SpeedKm_h']), axis=1)
        features['Wind'] = wind

        cloud = X['CloudCover'].astype(np.int32)
        features['CloudCover'] = cloud

        weather_event = X['Events'].astype(np.int32) - 1
        features['Events'] = weather_event

    if FEATURES_FB:
        promo_first_forward_looking = X['Promo_Forward'].astype(np.int32)
        features['Promo_Forward'] = promo_first_forward_looking

        promo_last_backward_looking = X['Promo_Backward'].astype(np.int32)
        features['Promo_Backward'] = promo_last_backward_looking

        stateHoliday_first_forward_looking = X['StateHoliday_Forward'].astype(np.int32)
        features['StateHoliday_Forward'] = stateHoliday_first_forward_looking

        stateHoliday_last_backward_looking = X['StateHoliday_Backward'].astype(np.int32)
        features['StateHoliday_Backward'] = stateHoliday_last_backward_looking

        schoolHoliday_first_forward_looking = X['SchoolHoliday_Forward'].astype(np.int32)
        features['SchoolHoliday_Forward'] = schoolHoliday_first_forward_looking

        schoolHoliday_last_backward_looking = X['SchoolHoliday_Backward'].astype(np.int32)
        features['SchoolHoliday_Backward'] = schoolHoliday_last_backward_looking

    if FEATURES_COUNT_FB:
        # 0-2
        stateHoliday_count_fw = X['StateHoliday_Count_FW'].astype(np.int32)
        features['StateHoliday_Count_FW'] = stateHoliday_count_fw
        stateHoliday_count_bw = X['StateHoliday_Count_BW'].astype(np.int32)
        features['StateHoliday_Count_BW'] = stateHoliday_count_bw

        # 0-7
        schoolHoliday_count_fw = X['SchoolHoliday_Count_FW'].astype(np.int32)
        features['SchoolHoliday_Count_FW'] = schoolHoliday_count_fw
        schoolHoliday_count_bw = X['SchoolHoliday_Count_BW'].astype(np.int32)
        features['SchoolHoliday_Count_BW'] = schoolHoliday_count_bw

        # 0-5
        promo_count_fw = X['Promo_Count_FW'].astype(np.int32)
        features['Promo_Count_FW'] = promo_count_fw
        promo_count_bw = X['Promo_Count_BW'].astype(np.int32)
        features['Promo_Count_BW'] = promo_count_bw

    if FEATURES_PROMODECAY:
        promo_decay = X['PromoDecay'].astype(np.int32)
        features['PromoDecay'] = promo_decay

    if FEATURES_TREND:
        googletrend_DE = X['Trend_Val_DE']
        features['Trend_Val_DE'] = googletrend_DE

        googletrend_state = X['Trend_Val_State']
        features['Trend_Val_State'] = googletrend_state

    if FEATURES_SHOPAVG:
        avg_sales = np.concatenate((X['Sales_Per_Day'], X['Customers_Per_Day'], X['Sales_Per_Customer'],
                                    X['Sales_Promo'], X['Sales_Holiday'], X['Sales_Saturday'],
                                    X['Open_Ratio'], X['SchoolHoliday_Ratio']),
                                   axis=1)
        features['Avg_Sales'] = avg_sales

    if FEATURES_LONGCLOSED:
        before_long_closed = X['Before_Long_Closed'].astype(np.int32)
        features['Before_Long_Closed'] = before_long_closed

        after_long_closed = X['After_Long_Closed'].astype(np.int32)
        features['After_Long_Closed'] = after_long_closed

    if FEATURES_SUMMER:
        before_sh_start = X['Before_SummerHoliday_Start'].astype(np.int32)
        features['Before_SummerHoliday_Start'] = before_sh_start
        after_sh_start = X['After_SummerHoliday_Start'].astype(np.int32)
        features['After_SummerHoliday_Start'] = after_sh_start
        before_sh_end = X['Before_SummerHoliday_End'].astype(np.int32)
        features['Before_SummerHoliday_End'] = before_sh_end
        after_sh_end = X['After_SummerHoliday_End'].astype(np.int32)
        features['After_SummerHoliday_End'] = after_sh_end

    numeric_features = ['Promo', 'CompetitionDistance', 'Temperature', 'Humidity', 'Wind', 'Trend_Val_DE', 'Trend_Val_State',
                        'Avg_Sales']
    for fea in numeric_features:
        features[fea] = features[fea].astype(np.float32)
    return features

def make_input_fn(X, y, shuffle, num_epochs, batch_size):
    if y is None:
        inputs = (X,)
    else:
        inputs = (X, y)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    if shuffle:

        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(SHUFFLE_BUFFER_SIZE, num_epochs))

        #dataset = dataset.shuffle(buffer_size=N_TRAIN)
    else:
        dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)

    return dataset

def train_input_fn():
    X_train, y_train = train_set
    y_train = transform_y(y_train).astype(np.float32)
    X_train = process_features(X_train)

    return make_input_fn(X_train, y_train, shuffle=True, num_epochs=EPOCHS, batch_size=BATCH_SIZE)

def eval_input_fn():
    X_eval, y_eval = valid_set
    y_eval = transform_y(y_eval).astype(np.float32)
    X_eval = process_features(X_eval)
    return make_input_fn(X_eval, y_eval, shuffle=False, num_epochs=1, batch_size=BATCH_SIZE)

def test_input_fn():
    X_test = test
    X_test = process_features(X_test)
    return make_input_fn(X_test, y=None, shuffle=False, num_epochs=1, batch_size=BATCH_SIZE)


def model_fn(wide_columns, deep_columns):
    def _model_fn(features, labels, mode, params):
        input_tensor = tf.feature_column.input_layer(features, deep_columns)
        x = input_tensor
        x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='glorot_uniform')(x)
        x = tf.keras.layers.Dense(256, activation='relu', kernel_initializer='glorot_uniform')(x)
        x = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform')(x)
        dnn_logits = x

        dnn_optimizer = tf.train.AdamOptimizer(learning_rate=_DNN_LEARNING_RATE)

        if len(wide_columns) > 0:
            input_tensor = tf.feature_column.input_layer(features, wide_columns)
            x = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform')(input_tensor)
            linear_logits = x
            default_learning_rate = 1. / math.sqrt(wide_columns)
            lr = min(_LINEAR_LEARNING_RATE, default_learning_rate)
            linear_optimizer = tf.train.FtrlOptimizer(learning_rate=lr)
        else:
            linear_logits = None

        if linear_logits is None:
            logits = dnn_logits
        else:
            logits = dnn_logits + linear_logits

        predictions = {
            'y_pred': logits
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions
            )
        labels = tf.squeeze(labels)
        logits = tf.squeeze(logits)
        print('labels.shape:', labels.shape)
        print('logits.shape:', logits.shape)
        loss = tf.keras.losses.mean_absolute_error(labels, logits)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_ops = []
            global_step = tf.train.get_or_create_global_step()
            train_ops.append(dnn_optimizer.minimize(loss))
            if len(wide_columns) > 0:
                train_ops.append(linear_optimizer.minimize(loss))


            train_ops_group = tf.group(train_ops)

            with tf.control_dependencies([train_ops_group]):
                with tf.colocate_with(global_step):
                    train_op = tf.assign_add(global_step, 1)
        else:
            train_op = None

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op
        )

    return _model_fn


def build_estimator():
    wide_columns, deep_columns = build_model_columns()
    run_config = tf.estimator.RunConfig(save_checkpoints_secs=1e7)
    return tf.estimator.Estimator(
        model_fn=model_fn(wide_columns, deep_columns),
        model_dir=str(MODEL_PATH),
        config=run_config
    )

def build_estimator2():
    wide_columns, deep_columns = build_model_columns()
    hidden_units = [100, 75, 50, 25]

    run_config = tf.estimator.RunConfig(save_checkpoints_secs=1e7)
    return tf.estimator.DNNLinearCombinedRegressor(
        model_dir = str(MODEL_PATH),
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        config=run_config
    )

def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    #model = build_estimator()
    model = build_estimator()

    for i in range(EPOCHS):
        print('train at epoch', i+1)
        print('-'*50)
        model.train(train_input_fn)

        results = model.evaluate(eval_input_fn)
        print('eval at epoch', i+1)
        print('-'*50)

        for key in sorted(results):
            print('{}: {}'.format(key, results[key]))


if __name__ == '__main__':
    main()