import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import Input, Embedding, LSTM, Dense, Reshape, Concatenate, Dropout
from keras.models import Model
from keras.utils import plot_model
from ross_model import residual_layer, NN_Embedding


# inputs = []
# embeddings = []
#
# store_input = Input(shape=(1,), name='Store')
# x = Embedding(1115, 50, input_length=1)(store_input)
# x = Reshape(target_shape=(50,))(x)
# inputs.append(store_input)
# embeddings.append(x)
#
# dow_input = Input(shape=(1,), name='DayOfWeek')
# x = Embedding(7, 6, input_length=1)(dow_input)
# x = Reshape(target_shape=(6,))(x)
# inputs.append(dow_input)
# embeddings.append(x)
#
# promo_input = Input(shape=(1,), name='Promo')
# x = promo_input
# x = Dense(1)(x)
# inputs.append(promo_input)
# embeddings.append(x)
#
# year_input = Input(shape=(1,), name='Year')
# x = Embedding(3, 2, input_length=1)(year_input)
# x = Reshape(target_shape=(2,))(x)
# inputs.append(year_input)
# embeddings.append(x)
#
# month_input = Input(shape=(1,), name='Month')
# x = Embedding(12, 6, input_length=1)(month_input)
# x = Reshape(target_shape=(6,))(x)
# inputs.append(month_input)
# embeddings.append(x)
#
# x = Concatenate()(embeddings)
# x = Dropout(0.02)(x)
#
# # x = Dense(1000, kernel_initializer='random_uniform', activation='relu')(x)
# # x = Dense(500, kernel_initializer='random_uniform', activation='relu')(x)
# x = residual_layer(x, 512, dropout=0.1)
# x = residual_layer(x, 256, dropout=0.1)
# x = Dense(1, activation='sigmoid')(x)
#
#
# model = Model(inputs=inputs, outputs=x)
# model.compile(optimizer='adam', loss='mae')
# #model.summary()
#
#
# plot_model(model, to_file='model_residual.png')

model = NN_Embedding()
plot_model(model.model, to_file='my_model.png', show_shapes=True)