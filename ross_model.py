
import numpy as np
import pandas as pd
import keras
from keras.layers import Embedding, Dense, Input, Reshape, Dropout, Concatenate, Add, Activation
from keras.callbacks import  ModelCheckpoint
import keras.backend as K
from keras.losses import logcosh, mae, mape


from .preprocess_data import  split_features

MIN_LOG_SALES = 3.8286
MAX_LOG_SALES = 10.6347
USE_LOG_Y = 1
USE_SAMPLE_WEIGHT = 1

EMBEDDING_DROPOUT = 0.03
DNN_DROPOUT = 0.1

def sp_error(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    err = ((y_true - y_pred) / y_true) ** 2
    return err

def mspe(y_true, y_pred):
    #mask = y_true!=0
    #n = len(y_true)
    # y_true = y_true.flatten()
    # y_pred = y_pred.flatten()
    err = ((y_true - y_pred) /y_true)**2
    err = np.mean(err, axis=-1)
    return err

def rmspe(y_true, y_pred):
    return mspe(y_true, y_pred)**0.5


def mspe_keras(y_true, y_pred):
    if USE_LOG_Y:
        y_true = K.exp(y_true * (MAX_LOG_SALES - MIN_LOG_SALES) + MIN_LOG_SALES)
        y_pred = K.exp(y_pred * (MAX_LOG_SALES - MIN_LOG_SALES) + MIN_LOG_SALES)

    diff = K.pow((y_true - y_pred) / y_true, 2)
    return  K.mean(diff, axis=-1)


def mape_keras(y_true, y_pred):
    if USE_LOG_Y:
        y_true = K.exp(y_true * (MAX_LOG_SALES - MIN_LOG_SALES) + MIN_LOG_SALES)
        y_pred = K.exp(y_pred * (MAX_LOG_SALES - MIN_LOG_SALES) + MIN_LOG_SALES)
    diff = K.abs((y_true - y_pred) / y_true)
    return K.mean(diff, axis=-1)


def transform_y(y):
    if USE_LOG_Y:
        return (np.log(y) - MIN_LOG_SALES) / (MAX_LOG_SALES - MIN_LOG_SALES)
    else:
        return y

def inverse_transform_y(transformed_y):
    if USE_LOG_Y:
        return np.exp(transformed_y * (MAX_LOG_SALES - MIN_LOG_SALES) + MIN_LOG_SALES)
    else:
        return transformed_y

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

def residual_layer(inputs, hidden_units, dropout=0.0):
    short_cut = inputs
    x = Dense(hidden_units)(inputs)
    x = keras.layers.PReLU()(x)
    input_shape = K.int_shape(inputs)[-1]
    if dropout>0.0:
        x = Dropout(dropout)(x)
    x = Dense(input_shape)(x)

    x = Add()([short_cut, x])
    return  keras.layers.PReLU()(x)


class NN_Embedding:
    def __init__(self, print_model_summary=False, save_checkpt=True):
        self._build_keras_model()
        if save_checkpt:
            self.checkpointer = [ModelCheckpoint('./output/checkpt2/weights.{epoch:02d}-{val_loss:.4f}.hdf5',
                                                 verbose=0, save_best_only=False)]
        else:
            self.checkpointer = []
        if print_model_summary:
            self.model.summary()

    def fit(self, train_set, valid_set=None, batch_size = 128, epochs=20, init_epoch=0):
        X_train, y_train = train_set
        self.X_train = self.preprocess(X_train)
        self.y_train = transform_y(y_train.reshape(-1))
        if USE_SAMPLE_WEIGHT:
            year = self.X_train[3].reshape(-1)
            #month = self.X_train[4].reshape(-1)
            #month = month + year*12 + 1

            #max_weight = 1.3
            #weight = max_weight / 33 * month
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
        self.batch_size = batch_size
        self.model.fit(self.X_train, self.y_train, sample_weight=weight,
                        batch_size=batch_size, epochs=epochs,verbose=1,
                        validation_data=valid_set,
                        callbacks=self.checkpointer,
                       initial_epoch=init_epoch)
        #print(hist.history)

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
        err = mspe(y_true, y_pred)
        return err

    def eval(self):
        err_tr = self.evaluate(self.X_train, self.y_train)
        print('rmpse on train set:', err_tr**0.5)
        if self.X_valid is not None:
            err_val = self.evaluate(self.X_valid, self.y_valid)
            print('rmpse on valid set:', err_val**0.5)

    def display_max_error_samples(self, sample_ratio=0.0001):
        y_pred = self.predict(self.X_train)
        y_true = inverse_transform_y(self.y_train)
        # print('y_pred.shape:' ,y_pred.shape)
        # print('y_true.shape:', y_true.shape)
        err = sp_error(y_true, y_pred)

        err_thresh = np.percentile(err, 100 - sample_ratio*100)
        mask = err > err_thresh
        max_errors = err[mask]


        samples_store = self.X_train[0].reshape(-1)[mask] + 1
        samples_year = self.X_train[3].reshape(-1)[mask] + 2013
        samples_month = self.X_train[4].reshape(-1)[mask] + 1
        samples_day = self.X_train[5].reshape(-1)[mask] + 1

        result = pd.DataFrame({'Store': samples_store.astype(np.int),
                             'Year': samples_year.astype(np.int),
                             'Month': samples_month.astype(np.int),
                             'Day': samples_day.astype(np.int),
                             'Sales': y_true[mask],
                             'Sales_Pred': y_pred[mask],
                             'Error': max_errors})[['Store', 'Year', 'Month', 'Day', 'Sales', 'Sales_Pred', 'Error']]
        result.to_csv('max_error.csv')
        #print(result)
        return result


    def _build_keras_model(self):
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
        #x = Dense(1)(promo_input)
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
        #x = Dense(1)(school_input)
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
        #x = Dense(1)(distance_input)
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

        germanstate_input = Input(shape=(1,))
        x = Embedding(12, 6, input_length=1)(germanstate_input)
        x = Reshape(target_shape=(6,))(x)
        inputs.append(germanstate_input)
        embeddings.append(x)

        woy_input = Input(shape=(1,))
        x = Embedding(53, 27, input_length=1)(woy_input)
        x = Reshape(target_shape=(27,))(x)
        inputs.append(woy_input)
        embeddings.append(x)

        # open_input = Input(shape=(15,))
        # x20 = Dense(7)(open_input)
        # #x3 = promo_input
        # inputs.append(open_input)
        # embeddings.append(x20)

        temperature_input = Input(shape=(3,))
        #x = Dense(3)(temperature_input)
        x = temperature_input
        inputs.append(temperature_input)
        embeddings.append(x)

        humidity_input = Input(shape=(3,))
        #x = Dense(3)(humidity_input)
        x = humidity_input
        inputs.append(humidity_input)
        embeddings.append(x)

        wind_input = Input(shape=(2,))
        #x = Dense(2)(wind_input)
        x = wind_input
        inputs.append(wind_input)
        embeddings.append(x)

        cloud_input = Input(shape=(1,))
        x = Embedding(9, 5, input_length=1)(cloud_input)
        x = Reshape(target_shape=(5,))(x)
        inputs.append(cloud_input)
        embeddings.append(x)

        weatherevent_input = Input(shape=(1,))
        x = Embedding(22, 11, input_length=1)(weatherevent_input)
        x = Reshape(target_shape=(11,))(x)
        inputs.append(weatherevent_input)
        embeddings.append(x)

        promo_forward_input = Input(shape=(1,))
        x = Embedding(8, 4, input_length=1)(promo_forward_input)
        x = Reshape(target_shape=(4,))(x)
        inputs.append(promo_forward_input)
        embeddings.append(x)

        promo_backward_input = Input(shape=(1,))
        x= Embedding(8, 4, input_length=1)(promo_backward_input)
        x = Reshape(target_shape=(4,))(x)
        inputs.append(promo_backward_input)
        embeddings.append(x)

        stateholiday_forward_input = Input(shape=(1,))
        x = Embedding(8, 4, input_length=1)(stateholiday_forward_input)
        x = Reshape(target_shape=(4,))(x)
        inputs.append(stateholiday_forward_input)
        embeddings.append(x)

        stateholiday_backward_input = Input(shape=(1,))
        x = Embedding(8, 4, input_length=1)(stateholiday_backward_input)
        x = Reshape(target_shape=(4,))(x)
        inputs.append(stateholiday_backward_input)
        embeddings.append(x)

        stateholiday_count_forward_input = Input(shape=(1,))
        x = Embedding(3, 2, input_length=1)(stateholiday_count_forward_input)
        x = Reshape(target_shape=(2,))(x)
        inputs.append(stateholiday_count_forward_input)
        embeddings.append(x)

        stateholiday_count_backward_input = Input(shape=(1,))
        x = Embedding(3, 2, input_length=1)(stateholiday_count_backward_input)
        x = Reshape(target_shape=(2,))(x)
        inputs.append(stateholiday_count_backward_input)
        embeddings.append(x)

        schoolholiday_forward_input = Input(shape=(1,))
        x = Embedding(8, 4, input_length=1)(schoolholiday_forward_input)
        x = Reshape(target_shape=(4,))(x)
        inputs.append(schoolholiday_forward_input)
        embeddings.append(x)

        schoolholiday_backward_input = Input(shape=(1,))
        x = Embedding(8, 4, input_length=1)(schoolholiday_backward_input)
        x = Reshape(target_shape=(4,))(x)
        inputs.append(schoolholiday_backward_input)
        embeddings.append(x)

        googletrend_de_input = Input(shape=(1,))
        #x = Dense(1)(googletrend_de_input)
        x = googletrend_de_input
        inputs.append(googletrend_de_input)
        embeddings.append(x)

        googletrend_state_input = Input(shape=(1,))
        #x = Dense(1)(googletrend_state_input)
        x = googletrend_state_input
        inputs.append(googletrend_state_input)
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
        avg_sales_input = Input(shape=(3,))
        #x = Dense(3)(avg_sales_input)
        x = avg_sales_input
        inputs.append(avg_sales_input)
        embeddings.append(x)

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

        # model_weather = Sequential()
        # model_weather.add(Merge([model_temperature, model_humidity, model_wind, model_weatherevent], mode='concat'))
        # model_weather.add(Dense(1))
        # model_weather.add(Activation('relu'))
        # models.append(model_weather)

        concat_out = Concatenate()(embeddings)

        # x = Dropout(0.02)(concat_out)
        # x = Dense(1000, kernel_initializer='glorot_uniform', activation='relu')(x)
        # x = Dense(500, kernel_initializer='glorot_uniform', activation='relu')(x)
        # x = Dense(1, activation='sigmoid')(x)

        x = Dropout(EMBEDDING_DROPOUT)(concat_out)
        x = residual_layer(x, 512, dropout=DNN_DROPOUT)
        #x = residual_layer(x, 512)
        x = residual_layer(x, 256)
        #x = residual_layer(x, 128)
        #x = residual_layer(x, 64)
        if USE_LOG_Y:
            x = Dense(1, activation='sigmoid')(x)
        else:
            x = Dense(1)(x)
        #x = Dense(1)(x)

        self.model = keras.models.Model(inputs=inputs, outputs=[x])

        #loss = mape
        loss = mae
        #loss = logcosh
        #loss = tukey_loss(0.5)
        #loss = huber_loss(1.0)
        #opt = keras.optimizers.SGD(lr=1e-4, momentum=0.9)
        opt = keras.optimizers.Adam(amsgrad=True)
        self.model.compile(loss=loss, optimizer=opt
                           , metrics=[mspe_keras]
                           )

    def preprocess(self, X):
        return split_features(X)



