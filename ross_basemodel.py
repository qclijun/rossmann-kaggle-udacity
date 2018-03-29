from pathlib import Path

import numpy as np
import pandas as pd
import keras
from keras.layers import Embedding, Dense, Input, Reshape, Dropout, Concatenate, Add, Activation
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from keras.losses import logcosh, mae, mse, mape
import matplotlib.pyplot as plt


from ross_util import mspe_keras, mape_keras, transform_y, inverse_transform_y, rmspe

USE_LOG_Y = True
USE_SAMPLE_WEIGHT = False
DISPLAY_METRIC = True

EMBEDDING_DROPOUT = 0.02
DNN_DROPOUT = 0.2
ACTIVATION = 'relu' #relu, elu, selu, PReLu, LeakyReLu, Swish
METRIC = 'mspe' # mape, mspe

FEATURES_TREND = True
FEATURES_WEATHER = True
FEATURES_FB = True
FEATURES_COUNT_FB = True
FEATURES_LONGCLOSED = False
FEATURES_SHOPAVG = False
FEATURES_SUMMER = False
FEATURES_PROMODECAY = False

MODEL_BASELINE = False

def tukey_loss(c=4.6851):
    c1 = c**2/6
    def _tukey_loss(y_true, y_pred):
        diff = 1 - K.pow((y_true - y_pred) / c, 2)
        diff = 1 - K.pow(diff, 3)
        return K.minimum(diff*c1, c1)
    return _tukey_loss

def huber_loss(c=1.0):
    c2 = c**2
    def _huber_loss(y_true, y_pred):
        diff = K.pow(y_true-y_pred, 2) / c2 + 1
        diff = c2*(K.sqrt(diff) - 1)
        return K.mean(diff, axis=-1) * 10
    return _huber_loss

class Swish(keras.layers.Layer):
    def __init__(self):
        super(Swish, self).__init__()

    def call(self, inputs, **kwargs):
        return inputs * K.sigmoid(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape


def activation_layer(x):
    if ACTIVATION == 'PReLu':
        x = keras.layers.PReLU()(x)
    elif ACTIVATION == 'LeakyReLu':
        x = keras.layers.LeakyReLU()(x)
    elif ACTIVATION == 'Swish':
        x = Swish()(x)
    else:
        x = keras.layers.Activation(ACTIVATION)(x)
    return x


def residual_layer(inputs, hidden_units, dropout=0.0):
    input_shape = K.int_shape(inputs)[-1]
    short_cut = inputs
    x = inputs

    x = Dense(hidden_units, kernel_initializer='glorot_uniform', kernel_regularizer=None)(x)
    x = activation_layer(x)
    if dropout > 0.0:
        x = Dropout(dropout)(x)

    x = Dense(input_shape, kernel_initializer='glorot_uniform', kernel_regularizer=None)(x)
    # if dropout > 0.0:
    #     x = Dropout(dropout)(x)
    x = Add()([short_cut, x])
    x = activation_layer(x)

    return x


def residual_layer_bn(inputs, hidden_units, dropout=0.0):
    input_shape = K.int_shape(inputs)[-1]
    short_cut = inputs
    x = inputs

    x = Dense(hidden_units)(x)
    x = keras.layers.BatchNormalization()(x)
    x = activation_layer(x)
    # if dropout > 0.0:
    #     x = Dropout(dropout)(x)
    x = keras.layers.BatchNormalization()(x)
    x = Dense(input_shape)(x)
    # if dropout > 0.0:
    #     x = Dropout(dropout)(x)
    x = Add()([short_cut, x])
    x = activation_layer(x)

    return x


def residual_layer2(inputs, hidden_units, dropout=0.0):
    input_shape = K.int_shape(inputs)[-1]
    short_cut = inputs
    y = [inputs]

    x = inputs
    x = Dense(hidden_units//2)(x)
    x = activation_layer(x)
    if dropout > 0.0:
        x = Dropout(dropout)(x)
    x = Dense(input_shape)(x)
    y.append(x)

    x = inputs
    x = Dense(hidden_units//2)(x)
    x = activation_layer(x)
    if dropout > 0.0:
        x = Dropout(dropout)(x)
    x = Dense(input_shape)(x)
    y.append(x)

    x = Add()(y)
    x = activation_layer(x)
    return x


def residual_layer3(inputs, hidden_units, dropout=0.0):
    input_shape = K.int_shape(inputs)[-1]
    short_cut = inputs
    x = inputs

    x = Dense(input_shape)(x)
    x = activation_layer(x)

    x = Dense(hidden_units)(x)
    x = activation_layer(x)
    if dropout > 0.0:
        x = Dropout(dropout)(x)

    x = Dense(input_shape)(x)
    if dropout > 0.0:
        x = Dropout(dropout)(x)
    x = Add()([short_cut, x])
    x = activation_layer(x)

    return x


class Eval_Callback(keras.callbacks.Callback):
    def __init__(self, train_set):
        self.X, self.y = train_set
        self.metric_name = METRIC + '_keras'
        self.val_metric_name = 'val_' + self.metric_name
        self.history = {metrics_name: [] for metrics_name in ['loss', self.metric_name, 'val_loss', self.val_metric_name]}

    def on_epoch_end(self, epoch, logs=None):
        train_result = self.model.evaluate(self.X, self.y, batch_size=8192,
                 verbose=0)
        self.history['loss'].append(train_result[0])
        self.history[self.metric_name].append(train_result[1])
        self.history['val_loss'].append(logs['val_loss'])
        self.history[self.val_metric_name].append(logs[self.val_metric_name])
        print('\nloss: {:.4f} - {}: {:.4f} - val_loss: {:.4f} - {}: {:.4f}'
              .format(train_result[0], self.metric_name, train_result[1],
                      logs['val_loss'], self.val_metric_name, logs[self.val_metric_name]))


class NN_Embedding_Base:
    def __init__(self, print_model_summary=False, save_checkpt=True):
        self.model = self.__build_keras_model()
        self.callbacks =[]

        if save_checkpt:
            checkpt_dir = './output/checkpt/'
            Path(checkpt_dir).mkdir(parents=True, exist_ok=True)
            self.callbacks.append(ModelCheckpoint(checkpt_dir+'weights.{epoch:02d}.hdf5',
                                                 verbose=0, save_best_only=False))
        if print_model_summary:
            self.model.summary()

        # tb_callback = keras.callbacks.TensorBoard(log_dir='./output/logs',
        #                                           histogram_freq=5,
        #                                           write_graph=False,
        #                                           write_grads=True,
        #                                           embeddings_freq=5)
        # self.callbacks.append(tb_callback)
        self.eval_callback = None

    def fit(self, train_set, valid_set=None, batch_size = 128, epochs=20, init_epoch=0):
        X_train, y_train = train_set
        self.X_train = self.preprocess(X_train)
        self.y_train = transform_y(y_train.reshape(-1))
        if USE_SAMPLE_WEIGHT:
            sales = y_train.reshape(-1)
            #year = self.X_train[3].reshape(-1)
            #month = self.X_train[4].reshape(-1) + 1
            # month = month + year*12 #[0, 32]
            #
            # max_weight = 1.3
            # weight = (max_weight - 1) / 32 * month + 1
            weight = np.ones_like(sales)
            weight[sales < 4000] = 1.2
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
            self.eval_callback = Eval_Callback((self.X_train, self.y_train))
            self.callbacks.append(self.eval_callback)

        hist = self.model.fit(self.X_train, self.y_train, sample_weight=weight,
                        batch_size=batch_size, epochs=epochs, verbose=1,
                        validation_data=valid_set,
                        callbacks=self.callbacks,
                       initial_epoch=init_epoch)
        self.history = pd.DataFrame(hist.history)
        #self.plot_history(out_file='./output/loss.png')
        #print(hist.history)

    def plot_loss(self, out_file=None):
        print('plot loss curve...')
        df = pd.DataFrame(self.eval_callback.history)
        csv_file = Path(out_file).with_suffix('.csv')
        df.to_csv(csv_file)
        df.plot()
        plt.ylim(0, 0.03)
        plt.legend(loc='best')
        if out_file is not None:
            plt.savefig(out_file)
    #
    #
    def plot_history(self, out_file=None):
        print('plot loss curve...')
        self.history.plot()
        plt.ylim(0, 0.03)
        plt.legend(loc='best')
        if out_file is not None:
            plt.savefig(out_file)

    def predict_raw(self, X):
        if not isinstance(X, list):
            X = self.preprocess(X)
        return self.model.predict(X, batch_size=5120).reshape(-1)

    def predict(self, X):
        y_pred = self.predict_raw(X)
        y_pred = inverse_transform_y(y_pred)
        return y_pred # (len(X),)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        y_true = inverse_transform_y(y)
        err = rmspe(y_true, y_pred)
        return err

    def eval(self):
        err_val = 0.0
        if self.X_valid is not None:
            err_val = self.evaluate(self.X_valid, self.y_valid)
            #print('RMSPE(validation):', err_val)
        return err_val

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
        x = promo_input
        x = Dense(1)(x)
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
        x = Embedding(31, 10, input_length=1)(day_input)
        x = Reshape(target_shape=(10,))(x)
        inputs.append(day_input)
        embeddings.append(x)

        stateholiday_input = Input(shape=(1,))
        x = Embedding(4, 3, input_length=1)(stateholiday_input)
        x = Reshape(target_shape=(3,))(x)
        inputs.append(stateholiday_input)
        embeddings.append(x)

        school_input = Input(shape=(1,))
        x = school_input
        x = Dense(1)(x)
        inputs.append(school_input)
        embeddings.append(x)

        competemonths_input = Input(shape=(1,))
        x = Embedding(25, 2, input_length=1)(competemonths_input)
        x = Reshape(target_shape=(2,))(x)
        inputs.append(competemonths_input)
        embeddings.append(x)

        promo2weeks_input = Input(shape=(1,))
        x = Embedding(26, 1, input_length=1)(promo2weeks_input)
        x = Reshape(target_shape=(1,))(x)
        inputs.append(promo2weeks_input)
        embeddings.append(x)

        lastestpromo2months_input = Input(shape=(1,))
        x = Embedding(4, 1, input_length=1)(lastestpromo2months_input)
        x = Reshape(target_shape=(1,))(x)
        inputs.append(lastestpromo2months_input)
        embeddings.append(x)

        distance_input = Input(shape=(1,))
        x = distance_input
        x = Dense(1)(x)
        inputs.append(distance_input)
        embeddings.append(x)

        storetype_input = Input(shape=(1,))
        x = Embedding(5, 2, input_length=1)(storetype_input)
        x = Reshape(target_shape=(2,))(x)
        inputs.append(storetype_input)
        embeddings.append(x)

        assortment_input = Input(shape=(1,))
        x = Embedding(4, 3, input_length=1)(assortment_input)
        x = Reshape(target_shape=(3,))(x)
        inputs.append(assortment_input)
        embeddings.append(x)

        promointerval_input = Input(shape=(1,))
        x = Embedding(4, 3, input_length=1)(promointerval_input)
        x = Reshape(target_shape=(3,))(x)
        inputs.append(promointerval_input)
        embeddings.append(x)

        competyear_input = Input(shape=(1,))
        x = Embedding(18, 4, input_length=1)(competyear_input)
        x = Reshape(target_shape=(4,))(x)
        inputs.append(competyear_input)
        embeddings.append(x)

        promotyear_input = Input(shape=(1,))
        x = Embedding(8, 4, input_length=1)(promotyear_input)
        x = Reshape(target_shape=(4,))(x)
        inputs.append(promotyear_input)
        embeddings.append(x)

        state_input = Input(shape=(1,))
        x = Embedding(12, 6, input_length=1)(state_input)
        x = Reshape(target_shape=(6,))(x)
        inputs.append(state_input)
        embeddings.append(x)

        woy_input = Input(shape=(1,))
        x = Embedding(53, 2, input_length=1)(woy_input)
        x = Reshape(target_shape=(2,))(x)
        inputs.append(woy_input)
        embeddings.append(x)

        if FEATURES_WEATHER:
            temperature_input = Input(shape=(3,))
            x = temperature_input
            x = Dense(3)(x)
            inputs.append(temperature_input)
            embeddings.append(x)

            humidity_input = Input(shape=(3,))
            x = humidity_input
            x = Dense(3)(x)
            inputs.append(humidity_input)
            embeddings.append(x)

            wind_input = Input(shape=(2,))
            x = wind_input
            x = Dense(2)(x)
            inputs.append(wind_input)
            embeddings.append(x)

            cloud_input = Input(shape=(1,))
            x = Dense(1)(cloud_input)
            inputs.append(cloud_input)
            embeddings.append(x)

            weatherevent_input = Input(shape=(1,))
            x = Embedding(22, 4, input_length=1)(weatherevent_input)
            x = Reshape(target_shape=(4,))(x)
            inputs.append(weatherevent_input)
            embeddings.append(x)

        if FEATURES_FB:
            promo_forward_input = Input(shape=(1,))
            x = Embedding(8, 1, input_length=1)(promo_forward_input)
            x = Reshape(target_shape=(1,))(x)
            inputs.append(promo_forward_input)
            embeddings.append(x)

            promo_backward_input = Input(shape=(1,))
            x= Embedding(8, 1, input_length=1)(promo_backward_input)
            x = Reshape(target_shape=(1,))(x)
            inputs.append(promo_backward_input)
            embeddings.append(x)

            stateholiday_forward_input = Input(shape=(1,))
            x = Embedding(8, 1, input_length=1)(stateholiday_forward_input)
            x = Reshape(target_shape=(1,))(x)
            inputs.append(stateholiday_forward_input)
            embeddings.append(x)

            stateholiday_backward_input = Input(shape=(1,))
            x = Embedding(8, 1, input_length=1)(stateholiday_backward_input)
            x = Reshape(target_shape=(1,))(x)
            inputs.append(stateholiday_backward_input)
            embeddings.append(x)

            schoolholiday_forward_input = Input(shape=(1,))
            x = Embedding(8, 1, input_length=1)(schoolholiday_forward_input)
            x = Reshape(target_shape=(1,))(x)
            inputs.append(schoolholiday_forward_input)
            embeddings.append(x)

            schoolholiday_backward_input = Input(shape=(1,))
            x = Embedding(8, 1, input_length=1)(schoolholiday_backward_input)
            x = Reshape(target_shape=(1,))(x)
            inputs.append(schoolholiday_backward_input)
            embeddings.append(x)

        if FEATURES_COUNT_FB:
            stateholiday_count_forward_input = Input(shape=(1,))
            x = Embedding(3, 1, input_length=1)(stateholiday_count_forward_input)
            x = Reshape(target_shape=(1,))(x)
            inputs.append(stateholiday_count_forward_input)
            embeddings.append(x)

            stateholiday_count_backward_input = Input(shape=(1,))
            x = Embedding(3, 1, input_length=1)(stateholiday_count_backward_input)
            x = Reshape(target_shape=(1,))(x)
            inputs.append(stateholiday_count_backward_input)
            embeddings.append(x)

            # schoolholiday_count_forward_input = Input(shape=(1,))
            # x = Embedding(8, 4, input_length=1)(schoolholiday_count_forward_input)
            # x = Reshape(target_shape=(4,))(x)
            # inputs.append(schoolholiday_count_forward_input)
            # embeddings.append(x)
            #
            # schoolholiday_count_backward_input = Input(shape=(1,))
            # x = Embedding(8, 4, input_length=1)(schoolholiday_count_backward_input)
            # x = Reshape(target_shape=(4,))(x)
            # inputs.append(schoolholiday_count_backward_input)
            # embeddings.append(x)
            #
            # promo_count_forward_input = Input(shape=(1,))
            # x = Embedding(6, 3, input_length=1)(promo_count_forward_input)
            # x = Reshape(target_shape=(3,))(x)
            # inputs.append(promo_count_forward_input)
            # embeddings.append(x)
            #
            # promo_count_backward_input = Input(shape=(1,))
            # x = Embedding(6, 3, input_length=1)(promo_count_backward_input)
            # x = Reshape(target_shape=(3,))(x)
            # inputs.append(promo_count_backward_input)
            # embeddings.append(x)

        if FEATURES_PROMODECAY:
            promo_decay_input = Input(shape=(1,))
            x = Embedding(6, 3, input_length=1)(promo_decay_input)
            x = Reshape(target_shape=(3,))(x)
            inputs.append(promo_decay_input)
            embeddings.append(x)

        if FEATURES_TREND:
            googletrend_de_input = Input(shape=(1,))
            x = googletrend_de_input
            x = Dense(1)(x)
            inputs.append(googletrend_de_input)
            embeddings.append(x)

            googletrend_state_input = Input(shape=(1,))
            x = googletrend_state_input
            x = Dense(1)(x)
            inputs.append(googletrend_state_input)
            embeddings.append(x)

        if FEATURES_SHOPAVG:
            avg_sales_input = Input(shape=(8,))
            x = avg_sales_input
            # x = Dense(8)(x)
            inputs.append(avg_sales_input)
            embeddings.append(x)

        if FEATURES_LONGCLOSED:
            before_long_closed_input = Input(shape=(1,))
            x = Embedding(6, 3, input_length=1)(before_long_closed_input)
            x = Reshape(target_shape=(3,))(x)
            inputs.append(before_long_closed_input)
            embeddings.append(x)

            after_long_closed_input = Input(shape=(1,))
            x = Embedding(6, 3, input_length=1)(after_long_closed_input)
            x = Reshape(target_shape=(3,))(x)
            inputs.append(after_long_closed_input)
            embeddings.append(x)

        if FEATURES_SUMMER:
            before_sh_start_input = Input(shape=(1,))
            x = Embedding(16, 3, input_length=1)(before_sh_start_input)
            x = Reshape(target_shape=(3,))(x)
            inputs.append(before_sh_start_input)
            embeddings.append(x)

            after_sh_start_input = Input(shape=(1,))
            x = Embedding(16, 3, input_length=1)(after_sh_start_input)
            x = Reshape(target_shape=(3,))(x)
            inputs.append(after_sh_start_input)
            embeddings.append(x)

            before_sh_end_input = Input(shape=(1,))
            x = Embedding(16, 3, input_length=1)(before_sh_end_input)
            x = Reshape(target_shape=(3,))(x)
            inputs.append(before_sh_end_input)
            embeddings.append(x)

            after_sh_end_input = Input(shape=(1,))
            x = Embedding(16, 3, input_length=1)(after_sh_end_input)
            x = Reshape(target_shape=(3,))(x)
            inputs.append(after_sh_end_input)
            embeddings.append(x)

        x = Concatenate()(embeddings)

        if MODEL_BASELINE:
            x = Dropout(0.02)(x)
            x = Dense(1000, kernel_initializer='random_uniform', activation='relu')(x)
            x = Dense(500, kernel_initializer='random_uniform', activation='relu')(x)
            x = Dense(1, activation='sigmoid')(x)
        else:
            # x = Dense(1, activation='sigmoid')(x)
            # x = Dense(250, activation='relu')(x)
            x = Dropout(EMBEDDING_DROPOUT)(x)
            x = residual_layer(x, 512, dropout=DNN_DROPOUT)
            #x = residual_layer(x, 512, dropout=DNN_DROPOUT)
            x = residual_layer(x, 256, dropout=DNN_DROPOUT)
            #x = residual_layer(x, 128, dropout=DNN_DROPOUT)
            #x = residual_layer(x, 64)
            #x = Dense(100, activation='relu')(x)
            #x = Dense(x, 100)(x)
            if USE_LOG_Y:
                x = Dense(1, activation='sigmoid')(x)
                #x = Dense(1)(x)
            else:
                x = Dense(1)(x)


        model = keras.models.Model(inputs=inputs, outputs=[x])

        #loss = keras.losses.mape
        loss = mae
        #loss = mse
        #loss = logcosh
        #loss = tukey_loss(0.5)
        #loss = huber_loss(1.0)
        # opt = keras.optimizers.SGD(lr=0.01)
        opt = keras.optimizers.Adam(amsgrad=False)
        # opt = keras.optimizers.Nadam()
        if DISPLAY_METRIC:
            metrics = [mape_keras] if METRIC == 'mape' else [mspe_keras]
        else:
            metrics = None
        model.compile(loss=loss, optimizer=opt, metrics=metrics)
        return model

    def split_features(self, X):
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

        has_competition_for_months = X['CompeteOpenMonths'] + 2
        X_list.append(has_competition_for_months)

        has_promo2_for_weeks = X['Promo2OpenWeeks'] + 2
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
        CompetitionOpenSinceYear[CompetitionOpenSinceYear < 0] = 0
        X_list.append(CompetitionOpenSinceYear)

        Promo2SinceYear = X['Promo2SinceYear'] - 2008
        Promo2SinceYear[Promo2SinceYear < 0] = 0
        X_list.append(Promo2SinceYear)

        State = X['StateN'] - 1
        X_list.append(State)

        week_of_year = X['WeekOfYear'] - 1
        X_list.append(week_of_year)


        if FEATURES_WEATHER:

            temperature = np.concatenate((X['Max_TemperatureC'], X['Mean_TemperatureC'], X['Min_TemperatureC']), axis=1)
            X_list.append(temperature)

            humidity = np.concatenate((X['Max_Humidity'], X['Mean_Humidity'], X['Min_Humidity']), axis=1)
            X_list.append(humidity)

            wind = np.concatenate((X['Max_Wind_SpeedKm_h'], X['Mean_Wind_SpeedKm_h']), axis=1)
            X_list.append(wind)

            cloud = X['CloudCover']
            X_list.append(cloud)

            weather_event = X['Events'] - 1
            X_list.append(weather_event)

        if FEATURES_FB:
            promo_first_forward_looking = X['Promo_Forward']
            X_list.append(promo_first_forward_looking)

            promo_last_backward_looking = X['Promo_Backward']
            X_list.append(promo_last_backward_looking)

            stateHoliday_first_forward_looking = X['StateHoliday_Forward']
            X_list.append(stateHoliday_first_forward_looking)

            stateHoliday_last_backward_looking = X['StateHoliday_Backward']
            X_list.append(stateHoliday_last_backward_looking)

            schoolHoliday_first_forward_looking = X['SchoolHoliday_Forward']
            X_list.append(schoolHoliday_first_forward_looking)

            schoolHoliday_last_backward_looking = X['SchoolHoliday_Backward']
            X_list.append(schoolHoliday_last_backward_looking)

        if FEATURES_COUNT_FB:
            # 0-2
            stateHoliday_count_fw = X['StateHoliday_Count_FW']
            X_list.append(stateHoliday_count_fw)
            stateHoliday_count_bw = X['StateHoliday_Count_BW']
            X_list.append(stateHoliday_count_bw)

            # # 0-7
            # schoolHoliday_count_fw = X['SchoolHoliday_Count_FW']
            # X_list.append(schoolHoliday_count_fw)
            # schoolHoliday_count_bw = X['SchoolHoliday_Count_BW']
            # X_list.append(schoolHoliday_count_bw)
            #
            # #0-5
            # promo_count_fw = X['Promo_Count_FW']
            # X_list.append(promo_count_fw)
            # promo_count_bw = X['Promo_Count_BW']
            # X_list.append(promo_count_bw)

        if FEATURES_PROMODECAY:
            promo_decay = X['PromoDecay']
            X_list.append(promo_decay)

        if FEATURES_TREND:
            googletrend_DE = X['Trend_Val_DE']
            X_list.append(googletrend_DE)

            googletrend_state = X['Trend_Val_State']
            X_list.append(googletrend_state)

        if FEATURES_SHOPAVG:

            avg_sales = np.concatenate((X['Sales_Per_Day'], X['Customers_Per_Day'], X['Sales_Per_Customer'],
                                        X['Sales_Promo'], X['Sales_Holiday'], X['Sales_Saturday'],
                                       X['Open_Ratio'], X['SchoolHoliday_Ratio']),
                                       axis=1)
            X_list.append(avg_sales)

        if FEATURES_LONGCLOSED:
            before_long_closed = X['Before_Long_Closed']
            X_list.append(before_long_closed)

            after_long_closed = X['After_Long_Closed']
            X_list.append(after_long_closed)

        if FEATURES_SUMMER:
            before_sh_start = X['Before_SummerHoliday_Start']
            X_list.append(before_sh_start)
            after_sh_start = X['After_SummerHoliday_Start']
            X_list.append(after_sh_start)
            before_sh_end = X['Before_SummerHoliday_End']
            X_list.append(before_sh_end)
            after_sh_end = X['After_SummerHoliday_End']
            X_list.append(after_sh_end)

        return X_list

    def preprocess(self, X):
        return self.split_features(X)


