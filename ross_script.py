import datetime

import numpy as np
import pandas as pd

import keras
from ross_model import NN_Embedding
from ross_util import mspe_keras, predict_with_models
from ross_data import load_data
from ross_main import load_models_from_hdf
from ross_ee_xgb import write_embed_features

# def eval_ensemble_models(models, valid_set=None):
#     if valid_set is None:
#         X_val = models[0].X_valid
#         y_val = models[0].y_valid
#         if X_val is None:
#             return
#     else:
#         X_val, y_val = valid_set
#         X_val = models[0].preprocess(X_val)
#         y_val = transform_y(y_val.reshape(-1))
#
#     y_pred = predict_with_models(models, X_val)
#     y_true = inverse_transform_y(y_val)
#     err = rmspe(y_true, y_pred)
#     return err
#
# def fit_ensemble_models(models, train_set=None, valid_set=None):
#     print('model stack...')
#     if train_set is None:
#         return
#
#     X_train, y_train = train_set
#     #X_train = models[0].preprocess(X_train)
#     #y_train = transform_y(y_train.reshape(-1))
#     n_models = len(models)
#     X_train = predict_with_models(models, X_train)
#
#     if valid_set is not None:
#         X_val, y_val = valid_set
#         #X_val = models[0].preprocess(X_val)
#         #y_val = transform_y(y_val.reshape(-1))
#
#         X_val = predict_with_models(models, X_val)
#         valid_set = X_val, y_val
#
#     x = Input(shape=(1,))
#     y = Dense(1, kernel_initializer=keras.initializers.ones(), use_bias=False)(x)
#     stack_model = keras.models.Model(inputs=x, outputs=[y])
#     stack_model.compile(optimizer='adam', loss='mae')
#     stack_model.summary()
#     print('training stack model...')
#     stack_model.fit(X_train, y_train, batch_size=4096, epochs=4, validation_data=valid_set)
#     weights = stack_model.get_weights()
#     print('stack model weights:', weights)
#     return stack_model

# def test_stack():
#     model_files = ['./models/score301/model_{}.hdf5'.format(i + 1) for i in range(10)]
#     models = load_models_from_hdf(model_files)
#     train_set, valid_set, X_test = get_dataset()
#
#     stack_model = fit_ensemble_models(models, train_set, valid_set=valid_set)
#
#     y_pred = predict_with_models(models, X_test)
#
#     y_pred = stack_model.predict(y_pred).reshape(-1)
#     #y_pred  = y_pred * 1.03
#     print('y_pred.shape:', y_pred.shape)
#
#     submission_file_w = Path(submission_file)
#     submission_file_w = submission_file_w.with_name(submission_file_w.stem + '_w' + submission_file_w.suffix)
#     submission_file_w = str(submission_file_w)
#
#     import pandas as pd
#     submit_df = pd.DataFrame({'Id': range(1, len(y_pred)+1), 'Sales': y_pred})
#     submit_df.to_csv(submission_file_w, index=False)

# def model_stack():
#     model_files = ['./models/model_{}.hdf5'.format(i+1) for i in range(10)]
#     models = load_models_from_hdf(model_files)
#     train_set, valid_set, X_test = get_dataset()
#
#     eval_ensemble_models(models, valid_set)
#
#     weights = fit_ensemble_models(models, train_set)

def write_sales_pred(model_files, csv_file):
    models = load_models_from_hdf(model_files)
    train, test = load_data()
    y_pred_tr = predict_with_models(models, train)
    y_pred_te = predict_with_models(models, test)
    train_df = pd.DataFrame({name: arr.reshape(-1) for name, arr in train.items()
                             if name in ['Store', 'Year', 'Month', 'Day', 'Sales']})
    test_df = pd.DataFrame({name: arr.reshape(-1) for name, arr in test.items()
                            if name in ['Store', 'Year', 'Month', 'Day', 'Sales']})
    train_df['Sales_Pred'] = y_pred_tr
    test_df['Sales_Pred'] = y_pred_te
    df = pd.concat([train_df, test_df])
    dates = np.array(
        [datetime.date(int(y), int(m), int(d)) for (y, m, d) in zip(df['Year'], df['Month'], df['Day'])],
        dtype=np.datetime64
    )
    df['Date'] = dates
    df = df[['Store', 'Date', 'Sales', 'Sales_Pred']]
    df.to_csv(csv_file, index=False)

if __name__ == '__main__':
    model_files = ['./models/model_100%.hdf5']
    write_sales_pred(model_files, csv_file='sales_pred_100%.csv')
    #write_embed_features('./models/model_1.hdf5', 'embedding_features.h5')

