import datetime

import numpy as np
import h5py

from ross_util import MEAN_LOG_SALES, STD_LOG_SALES

h5filename = './output/all_data.h5'

def load_data():
    print('loading data: "{}"'.format(h5filename))
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

        # print(feature_names)
        return train, test


# def split_dataset(train_set):
#     # train_set is a dict
#     result = train_test_split(*tuple(train_set.values()), test_size=1 - TRAIN_RATIO, shuffle=False)
#     train = []
#     valid = []
#     for i, arr in enumerate(result):
#         if i%2==0:
#             train.append(arr)
#         else:
#             valid.append(arr)
#     train = dict(zip(train_set.keys(), train))
#     valid = dict(zip(train_set.keys(), valid))
#     return train, valid


def split_dataset2(train_set, weeks=6):
    assert isinstance(train_set, dict)
    last_date_in_train = datetime.date(2015, 7, 31)
    first_date_in_train = datetime.date(2013, 1, 1)

    split_date = last_date_in_train - datetime.timedelta(days=weeks*7)

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


def filter_stores(data_set, stores_in_test):
    mask = [s in stores_in_test for s in data_set['Store'].reshape(-1)]
    result = {k: v[mask] for k,v in data_set.items()}
    return result


def filter_sales(data_set, sales_threshold=800):
    mask = data_set['Sales'].reshape(-1) > sales_threshold
    result = {k: v[mask] for k, v in data_set.items()}
    return result


def filter_sales_sigma(data_set, sigma=4):
    sales_lower = np.exp(MEAN_LOG_SALES - sigma * STD_LOG_SALES)
    sales_upper = np.exp(MEAN_LOG_SALES + sigma * STD_LOG_SALES)
    arr = data_set['Sales'].reshape(-1)
    mask = ((arr >= sales_lower) & (arr <= sales_upper))
    result = {k: v[mask] for k, v in data_set.items()}
    return result


def get_dataset_5_9():
    ### split dataset May to Sep

    train, test = load_data()
    stores_in_test = set(get_stores_in_test(test))
    train_set = filter_stores(train, stores_in_test)

    month = train_set['Month'].reshape(-1)
    mask = (month >= 5) & (month <=9)
    train_set = {name: val[mask] for name, val in train_set.items()}
    X_train = train_set
    y_train = train_set['Sales']
    return (X_train, y_train), None, test


def get_dataset(validation_weeks=6, filt_stores_for_train=True, filt_stores_for_valid=True):

    train, test = load_data()
    stores_in_test = set(get_stores_in_test(test))

    # train = filter_sales_sigma(train, sigma=3)

    if validation_weeks > 0:
        print('split train set...')
        #train_set, valid_set = split_dataset(train)
        train_set, valid_set = split_dataset2(train, validation_weeks)
        if filt_stores_for_train:
            train_set = filter_stores(train_set, stores_in_test)
        if filt_stores_for_valid:
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
        if filt_stores_for_train:
            train_set = filter_stores(train_set, stores_in_test)
        valid_set = None
        X_train = train_set
        y_train = train_set['Sales']
        train_set = (X_train, y_train)
    print('train samples: {}, validation samples: {}'.format(len(y_train), 0 if valid_set is None else len(y_valid)))
    return train_set, valid_set, test


