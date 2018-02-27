
import numpy as np
import pandas as pd
import h5py

h5filename = './output/featured_data4.h5'
seq_data_filename = './output/seq_data.h5'

def load_seq_data():
    with h5py.File(seq_data_filename) as f:
        train = {}
        train['open'] = f['/train/open'][:, :].T
        train['promo'] = f['/train/promo'][:, :].T
        train['schoolholiday'] = f['/train/schoolholiday'][:, :].T
        train['stateholiday'] = f['/train/stateholiday'][:, :].T

        test = {}
        test['open'] = f['/test/open'][:, :].T
        test['promo'] = f['/test/promo'][:, :].T
        test['schoolholiday'] = f['/test/schoolholiday'][:, :].T
        test['stateholiday'] = f['/test/stateholiday'][:, :].T
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

        #train_seq, test_seq = load_seq_data()
        #train.update(train_seq)
        #test.update(test_seq)

        return train, test
        # print(feature_names)


def CompetitionOpenSinceYear2int(since_year_array):
    # since_year_array is numpy array
    since_year_array[since_year_array < 2000] = 0
    since_year_array[since_year_array >= 2000] -= 1999
    return since_year_array


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

    log_distance = X['CompetitionDistance_Log']
    X_list.append(log_distance)

    StoreType = X['StoreTypeN'] - 1
    X_list.append(StoreType)

    Assortment = X['AssortmentN'] - 1
    X_list.append(Assortment)

    PromoInterval = X['Promo2IntervalN']
    X_list.append(PromoInterval)

    CompetitionOpenSinceYear = CompetitionOpenSinceYear2int(X['CompetitionOpenSinceYear'])
    X_list.append(CompetitionOpenSinceYear)

    Promo2SinceYear = X['Promo2SinceYear'] - 2008
    Promo2SinceYear[Promo2SinceYear < 0] = 0
    X_list.append(Promo2SinceYear)

    State = X['StateN'] - 1
    X_list.append(State)

    week_of_year = X['WeekOfYear'] - 1
    X_list.append(week_of_year)

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

    promo_first_forward_looking = X['Promo_Forward'] - 1
    X_list.append(promo_first_forward_looking)

    promo_last_backward_looking = X['Promo_Backward'] - 1
    X_list.append(promo_last_backward_looking)

    stateHoliday_first_forward_looking = X['StateHoliday_Forward'] - 1
    X_list.append(stateHoliday_first_forward_looking)

    stateHoliday_last_backward_looking = X['StateHoliday_Backward'] - 1
    X_list.append(stateHoliday_last_backward_looking)

    stateHoliday_count_forward_looking = X['StateHoliday_Count_Forward']
    X_list.append(stateHoliday_count_forward_looking)

    stateHoliday_count_backward_looking = X['StateHoliday_Count_Backward']
    X_list.append(stateHoliday_count_backward_looking)

    schoolHoliday_first_forward_looking = X['SchoolHoliday_Forward'] - 1
    X_list.append(schoolHoliday_first_forward_looking)

    schoolHoliday_last_backward_looking = X['SchoolHoliday_Backward'] - 1
    X_list.append(schoolHoliday_last_backward_looking)

    googletrend_DE = X['Trend_Val_DE']
    X_list.append(googletrend_DE)

    googletrend_state = X['Trend_Val_State']
    X_list.append(googletrend_state)

    # promo_decay = X['PromoDecay']
    # X_list.append(promo_decay)
    #
    # tomorrow_closed = X['TomorrowClosed']
    # X_list.append(tomorrow_closed)

    avg_sales = np.concatenate((X['Sales_Per_Day'], X['Customers_Per_Day'], X['Sales_Per_Customer']), axis=1)
    X_list.append(avg_sales)

    before_long_closed = X['Before_Long_Closed']
    X_list.append(before_long_closed)

    after_long_closed = X['After_Long_Closed']
    X_list.append(after_long_closed)

    return X_list