import datetime
import numpy as np
import pandas as pd
import h5py
import keras
from keras.layers import Embedding, Dense, Input, Reshape, Dropout, Concatenate, Add, LSTM
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from keras.losses import logcosh, mae, mape

from ross_model import residual_layer

h5filename = './output/simple_feature.h5'
seq_data_filename = './output/simple_prev.h5'

VALIDATION_WEEKS = 6
DO_EVAL = VALIDATION_WEEKS > 0
FILT_STORES_FOR_TRAIN = True
FILT_STORES_FOR_VALID = True

MIN_LOG_SALES = 3.8501
MAX_LOG_SALES = 10.6348

USE_LOG_Y = True
USE_SAMPLE_WEIGHT = False
DISPLAY_METRIC = True

EMBEDDING_DROPOUT = 0.03
DNN_DROPOUT = 0.1
ACTIVATION = 'relu'

def transform_y(y):
    if USE_LOG_Y:
        return (np.log1p(y) - MIN_LOG_SALES) / (MAX_LOG_SALES - MIN_LOG_SALES)
    else:
        return y

def inverse_transform_y(transformed_y):
    if USE_LOG_Y:
        return np.expm1(transformed_y * (MAX_LOG_SALES - MIN_LOG_SALES) + MIN_LOG_SALES)
    else:
        return transformed_y

def mspe_keras(y_true, y_pred):
    if USE_LOG_Y:
        y_true = K.exp(y_true * (MAX_LOG_SALES - MIN_LOG_SALES) + MIN_LOG_SALES) - 1
        y_pred = K.exp(y_pred * (MAX_LOG_SALES - MIN_LOG_SALES) + MIN_LOG_SALES) - 1

    diff = K.pow((y_true - y_pred) / y_true, 2)
    return  K.mean(diff, axis=-1)

def mape_keras(y_true, y_pred):
    if USE_LOG_Y:
        y_true = K.exp(y_true * (MAX_LOG_SALES - MIN_LOG_SALES) + MIN_LOG_SALES) - 1
        y_pred = K.exp(y_pred * (MAX_LOG_SALES - MIN_LOG_SALES) + MIN_LOG_SALES) - 1
    diff = K.abs((y_true - y_pred) / y_true)
    return K.mean(diff, axis=-1)

def load_seq_data():
    with h5py.File(seq_data_filename) as f:
        train = {}
        train['open_prev'] = f['/train/open_prev'][:, :].T
        train['promo_prev'] = f['/train/promo_prev'][:, :].T
        train['school_holiday_prev'] = f['/train/school_holiday_prev'][:, :].T
        train['state_holiday_prev'] = f['/train/state_holiday_prev'][:, :].T
        train['sales_prev'] = f['/train/sales_prev'][:, :].T

        test = {}
        test['open_prev'] = f['/test/open_prev'][:, :].T
        test['promo_prev'] = f['/test/promo_prev'][:, :].T
        test['school_holiday_prev'] = f['/test/school_holiday_prev'][:, :].T
        test['state_holiday_prev'] = f['/test/state_holiday_prev'][:, :].T
        test['sales_prev'] = f['/test/sales_prev'][:, :].T

        return train, test

def load_data():
    print('loading data...')
    with h5py.File(h5filename) as f:
        train_ds = f['train']
        test_ds = f['test']
        feature_names_ds = f['features']

        train_ds = train_ds[:, :].T
        test_ds = test_ds[:, :].T
        feature_names = feature_names_ds[:]
        feature_names = [feature_name.decode() for feature_name in feature_names]

        print('train.shape:', train_ds.shape)
        print('test.shape:', test_ds.shape)

        train = {feature_name: train_ds[:, [i]]
                    for i, feature_name in enumerate(feature_names)}
        test = {feature_name: test_ds[:, [i]]
                 for i, feature_name in enumerate(feature_names)}

        train_seq, test_seq = load_seq_data()
        train.update(train_seq)
        test.update(test_seq)

        print('train set:')
        for name, arr in train.items():
            print('{}: {}'.format(name, arr.shape))
        print('min sales:', min(train['Sales']))
        print('max sales:', max(train['Sales']))
        return train, test
        # print(feature_names)

def split_dataset2(train_set, weeks=6, split_pos='last'):
    assert isinstance(train_set, dict)
    last_date_in_train = datetime.date(2015, 7, 31)
    first_date_in_train = datetime.date(2013, 1, 1)

    if split_pos == 'last':
        split_date = last_date_in_train - datetime.timedelta(days=weeks*7)
    else:
        split_date = first_date_in_train + datetime.timedelta(days=weeks*7)

    year = train_set['Year']
    month = train_set['Month']
    day = train_set['Day']
    dates = np.array(
        [datetime.date(int(y), int(m), int(d)) for (y, m, d) in zip(year, month, day)],
        dtype=np.datetime64
    )
    mask = dates > split_date
    part2 = {name: val[mask] for name, val in train_set.items()}
    part1 = {name: val[~mask] for name, val in train_set.items()}
    return part1, part2


def get_stores_in_test(test_set):
    return np.unique(test_set['Store'])

def filter_stores(val_set, stores_in_test):
    mask = [s in stores_in_test for s in val_set['Store'].reshape(-1)]
    result = {k: v[mask] for k,v in val_set.items()}
    return result

def get_dataset():
    train, test = load_data()
    stores_in_test = set(get_stores_in_test(test))

    if DO_EVAL:
        print('split train set...')
        #train_set, valid_set = split_dataset(train)
        train_set, valid_set = split_dataset2(train, VALIDATION_WEEKS)
        if FILT_STORES_FOR_TRAIN:
            train_set = filter_stores(train_set, stores_in_test)
        if FILT_STORES_FOR_VALID:
            valid_set = filter_stores(valid_set, stores_in_test)
        X_train = train_set
        y_train = train_set['Sales']
        X_valid = valid_set
        y_valid = valid_set['Sales']
        train_set = (X_train, y_train)
        valid_set = (X_valid, y_valid)
    else:
        train_set = train
        #_, train_set = split_dataset2(train, VALIDATION_WEEKS, split_pos='first')
        if FILT_STORES_FOR_TRAIN:
            train_set = filter_stores(train_set, stores_in_test)
        valid_set = None
        X_train = train_set
        y_train = train_set['Sales']
        train_set = (X_train, y_train)
    print('train samples: {}, validation samples: {}'.format(len(y_train), 0 if valid_set is None else len(y_valid)))
    return train_set, valid_set, test
# def load_data_with_seq():
#     train, test =  load_data()
#     train_seq, test_seq = load_seq_data()
#     train.update(train_seq)
#     test.update(test_seq)
#
#     print('train set:')
#     for name, arr in train.items():
#         print('{}: {}'.format(name, arr.shape))
#     return train, test

def split_features(X):
    # X is a dict
    assert isinstance(X, dict)

    X_list = []

    store_index = X['Store'] - 1
    X_list.append(store_index)

    day_of_week = X['DayOfWeek'] - 1
    X_list.append(day_of_week)

    promo = X['Promo']
    X_list.append(promo)

    year = X['Year'] - 2013
    X_list.append(year)

    month = X['Month'] - 1
    X_list.append(month)

    day = X['Day'] - 1
    X_list.append(day)

    state_holiday = X['StateHolidayN']
    X_list.append(state_holiday)

    school_holiday = X['SchoolHoliday']
    X_list.append(school_holiday)

    has_competition_for_months = X['CompeteOpenMonths']
    X_list.append(has_competition_for_months)

    has_promo2_for_weeks = X['Promo2OpenWeeks']
    X_list.append(has_promo2_for_weeks)

    latest_promo2_for_months = X['Latest_Promo2_Start_Month']
    X_list.append(latest_promo2_for_months)

    log_distance = X['CompetitionDistance']
    X_list.append(log_distance)

    StoreType = X['StoreTypeN'] - 1
    X_list.append(StoreType)

    Assortment = X['AssortmentN'] - 1
    X_list.append(Assortment)

    PromoInterval = X['Promo2IntervalN']
    X_list.append(PromoInterval)

    CompetitionOpenSinceYear = X['CompetitionOpenSinceYear'] - 1999
    CompetitionOpenSinceYear[CompetitionOpenSinceYear<0] = 0
    X_list.append(CompetitionOpenSinceYear)

    Promo2SinceYear = X['Promo2SinceYear'] - 2008
    Promo2SinceYear[Promo2SinceYear < 0] = 0
    X_list.append(Promo2SinceYear)

    # State = X['StateN'] - 1
    # X_list.append(State)

    week_of_year = X['WeekOfYear'] - 1
    X_list.append(week_of_year)

    open_prev = X['open_prev']
    X_list.append(open_prev)

    promo_prev = X['promo_prev']
    X_list.append(promo_prev)

    school_holiday_prev = X['school_holiday_prev']
    X_list.append(school_holiday_prev)

    state_holiday_prev = X['state_holiday_prev']
    X_list.append(state_holiday_prev)


    sales_prev = X['sales_prev']
    if USE_LOG_Y:
        sales_prev = transform_y(sales_prev)
    X_list.append(sales_prev)

    return X_list


class NN_LSTM:
    def __init__(self, print_model_summary=False, save_checkpt=True):
        self.model = self.__build_keras_model()
        self.callbacks =[]

        if save_checkpt:
            self.callbacks.append(ModelCheckpoint('./output/checkpt/weights.{epoch:02d}-{val_loss:.4f}.hdf5',
                                                 verbose=0, save_best_only=False))
        if print_model_summary:
            self.model.summary()

    def fit(self, train_set, valid_set=None, batch_size = 128, epochs=20, init_epoch=0):
        X_train, y_train = train_set
        self.X_train = self.preprocess(X_train)
        self.y_train = transform_y(y_train.reshape(-1))

        print('min(y_train): {}, max(y_train): {}'.format(min(self.y_train), max(self.y_train)))

        if USE_SAMPLE_WEIGHT:
            year = self.X_train[3].reshape(-1)
            # month = self.X_train[4].reshape(-1)
            # month = month + year*12 #[0, 32]
            #
            # max_weight = 1.3
            # weight = (max_weight - 1) / 32 * month + 1
            weight = np.ones_like(year)
            weight[year==1] = 1.1
            weight[year==2] = 1.2
        else:
            weight=None

        if valid_set is None:
            self.X_valid = None
            self.y_valid = None
        else:
            X_valid, y_valid = valid_set
            self.X_valid = self.preprocess(X_valid)
            self.y_valid = transform_y(y_valid.reshape(-1))
            valid_set = (self.X_valid, self.y_valid)
            print('min(y_valid): {}, max(y_valid): {}'.format(min(self.y_valid), max(self.y_valid)))


        hist = self.model.fit(self.X_train, self.y_train, sample_weight=weight,
                        batch_size=batch_size, epochs=epochs, verbose=1,
                        validation_data=valid_set,
                        callbacks=self.callbacks,
                       initial_epoch=init_epoch)
        self.history = pd.DataFrame(hist.history)
        #self.plot_history(out_file='./output/loss.png')
        #print(hist.history)

    def __build_keras_model(self):
        inputs = []
        embeddings = []

        store_input = Input(shape=(1,))
        x = Embedding(1115, 50, input_length=1)(store_input)
        x = Reshape(target_shape=(50,))(x)
        inputs.append(store_input)
        embeddings.append(x)

        dow_input = Input(shape=(1,))
        x = Embedding(7, 6, input_length=1)(dow_input)
        x = Reshape(target_shape=(6,))(x)
        inputs.append(dow_input)
        embeddings.append(x)

        promo_input = Input(shape=(1,))
        # x = Dense(1)(promo_input)
        x = promo_input
        inputs.append(promo_input)
        embeddings.append(x)

        year_input = Input(shape=(1,))
        x = Embedding(3, 2, input_length=1)(year_input)
        x = Reshape(target_shape=(2,))(x)
        inputs.append(year_input)
        embeddings.append(x)

        month_input = Input(shape=(1,))
        x = Embedding(12, 6, input_length=1)(month_input)
        x = Reshape(target_shape=(6,))(x)
        inputs.append(month_input)
        embeddings.append(x)

        day_input = Input(shape=(1,))
        x = Embedding(31, 16, input_length=1)(day_input)
        x = Reshape(target_shape=(16,))(x)
        inputs.append(day_input)
        embeddings.append(x)

        stateholiday_input = Input(shape=(1,))
        x = Embedding(4, 3, input_length=1)(stateholiday_input)
        x = Reshape(target_shape=(3,))(x)
        inputs.append(stateholiday_input)
        embeddings.append(x)

        school_input = Input(shape=(1,))
        # x = Dense(1)(school_input)
        x = school_input
        inputs.append(school_input)
        embeddings.append(x)

        competemonths_input = Input(shape=(1,))
        x = Embedding(25, 13, input_length=1)(competemonths_input)
        x = Reshape(target_shape=(13,))(x)
        inputs.append(competemonths_input)
        embeddings.append(x)

        promo2weeks_input = Input(shape=(1,))
        x = Embedding(26, 13, input_length=1)(promo2weeks_input)
        x = Reshape(target_shape=(13,))(x)
        inputs.append(promo2weeks_input)
        embeddings.append(x)

        lastestpromo2months_input = Input(shape=(1,))
        x = Embedding(4, 2, input_length=1)(lastestpromo2months_input)
        x = Reshape(target_shape=(2,))(x)
        inputs.append(lastestpromo2months_input)
        embeddings.append(x)

        distance_input = Input(shape=(1,))
        # x = Dense(1)(distance_input)
        x = distance_input
        inputs.append(distance_input)
        embeddings.append(x)

        storetype_input = Input(shape=(1,))
        x = Embedding(4, 2, input_length=1)(storetype_input)
        x = Reshape(target_shape=(2,))(x)
        inputs.append(storetype_input)
        embeddings.append(x)

        assortment_input = Input(shape=(1,))
        x = Embedding(3, 2, input_length=1)(assortment_input)
        x = Reshape(target_shape=(2,))(x)
        inputs.append(assortment_input)
        embeddings.append(x)

        promointerval_input = Input(shape=(1,))
        x = Embedding(4, 3, input_length=1)(promointerval_input)
        x = Reshape(target_shape=(3,))(x)
        inputs.append(promointerval_input)
        embeddings.append(x)

        competyear_input = Input(shape=(1,))
        x = Embedding(17, 9, input_length=1)(competyear_input)
        x = Reshape(target_shape=(9,))(x)
        inputs.append(competyear_input)
        embeddings.append(x)

        promotyear_input = Input(shape=(1,))
        x = Embedding(8, 4, input_length=1)(promotyear_input)
        x = Reshape(target_shape=(4,))(x)
        inputs.append(promotyear_input)
        embeddings.append(x)

        # germanstate_input = Input(shape=(1,))
        # x = Embedding(12, 6, input_length=1)(germanstate_input)
        # x = Reshape(target_shape=(6,))(x)
        # inputs.append(germanstate_input)
        # embeddings.append(x)

        woy_input = Input(shape=(1,))
        x = Embedding(53, 27, input_length=1)(woy_input)
        x = Reshape(target_shape=(27,))(x)
        inputs.append(woy_input)
        embeddings.append(x)


        # promo_decay_input = Input(shape=(1,))
        # x = Embedding(6, 3, input_length=1)(promo_decay_input)
        # x = Reshape(target_shape=(3,))(x)
        # inputs.append(promo_decay_input)
        # embeddings.append(x)
        # #
        # tomorrow_closed_input = Input(shape=(1,))
        # #x = Dense(1)(tomorrow_closed_input)
        # x = tomorrow_closed_input
        # inputs.append(tomorrow_closed_input)
        # embeddings.append(x)
        #
        # avg_sales_input = Input(shape=(3,))
        # # x = Dense(3)(avg_sales_input)
        # x = avg_sales_input
        # inputs.append(avg_sales_input)
        # embeddings.append(x)
        #
        # before_long_closed_input = Input(shape=(1,))
        # x = Embedding(6, 3, input_length=1)(before_long_closed_input)
        # x = Reshape(target_shape=(3,))(x)
        # inputs.append(before_long_closed_input)
        # embeddings.append(x)
        #
        # after_long_closed_input = Input(shape=(1,))
        # x = Embedding(6, 3, input_length=1)(after_long_closed_input)
        # x = Reshape(target_shape=(3,))(x)
        # inputs.append(after_long_closed_input)
        # embeddings.append(x)



        lstm_out = []
        open_prev_input = Input(shape=(60,), name='open_prev_input')
        inputs.append(open_prev_input)
        #x = Reshape(target_shape=(60, 1))(open_prev_input)
        #x = LSTM(32)(x)
        lstm_out.append(x)
        #
        promo_prev_input = Input(shape=(60,), name='promo_prev_input')
        inputs.append(promo_prev_input)
        #x = Reshape(target_shape=(60, 1))(promo_prev_input)
        #x = LSTM(32)(x)
        lstm_out.append(x)
        #
        schoolholiday_prev_input = Input(shape=(60,), name='schoolholiday_prev_input')
        inputs.append(schoolholiday_prev_input)
        #x = Reshape(target_shape=(60, 1))(schoolholiday_prev_input)
        #x = LSTM(32)(x)
        lstm_out.append(x)
        #
        stateholiday_prev_input = Input(shape=(60,), name='stateholiday_prev_input')
        inputs.append(stateholiday_prev_input)
        x = Embedding(4, 3, input_length=60)(stateholiday_prev_input)
        x = Reshape(target_shape=(180,))(x)
        #x = LSTM(32)(x)
        lstm_out.append(x)
        #
        sales_prev_input = Input(shape=(60,), name='sales_prev_input')
        inputs.append(sales_prev_input)
        #x = Reshape(target_shape=(60, 1))(sales_prev_input)
        #x = LSTM(32)(x)
        lstm_out.append(x)

        concat_out = Concatenate()(embeddings + lstm_out)
        # x = Dropout(0.02)(concat_out)
        # x = Dense(1000, kernel_initializer='glorot_uniform', activation='relu')(x)
        # x = Dense(500, kernel_initializer='glorot_uniform', activation='relu')(x)
        # x = Dense(1, activation='sigmoid')(x)

        x = Dropout(EMBEDDING_DROPOUT)(concat_out)
        # x = concat_out
        x = residual_layer(x, 512, dropout=DNN_DROPOUT)
        # x = residual_layer(x, 512)
        x = residual_layer(x, 256, dropout=DNN_DROPOUT)
        # x = residual_layer(x, 128)
        # x = residual_layer(x, 64)
        if USE_LOG_Y:
            x = Dense(1, activation='sigmoid')(x)
        else:
            x = Dense(1)(x)

        model = keras.models.Model(inputs=inputs, outputs=[x])

        # loss = keras.losses.mape
        loss = mae
        # loss = logcosh
        # loss = tukey_loss(0.5)
        # loss = huber_loss(1.0)
        # opt = keras.optimizers.SGD(lr=0.01, momentum=0.9)
        opt = keras.optimizers.Adam(amsgrad=True)
        # opt = keras.optimizers.Nadam()
        if DISPLAY_METRIC:
            metrics = [mspe_keras]
        else:
            metrics = None
        model.compile(loss=loss, optimizer=opt, metrics=metrics)
        return model

    def preprocess(self, X):
        return split_features(X)




def main():
    model = NN_LSTM()

if __name__ == '__main__':
    main()