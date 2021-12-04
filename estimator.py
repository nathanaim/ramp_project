from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import make_pipeline
from datetime import datetime, date
from astral import LocationInfo
from astral.sun import daylight
from xgboost import XGBRegressor

def _provide_date_info(data):

    data = data.copy()
    data['hour'] = data.date.dt.hour
    data['weekday'] = data.date.dt.weekday
    data['dom'] = data.date.dt.day
    data['week'] = data.date.dt.isocalendar().week
    data['month'] = data.date.dt.month
    data['year'] = data.date.dt.year
    # data['dom_counter'] = data.counter_installation_date.dt.day
    # data['month_counter'] = data.counter_installation_date.dt.month
    # data['year_counter'] = data.counter_installation_date.dt.year
    data['date_datetime'] = data.date.map(lambda x: x.to_pydatetime().date())
    data.drop(columns=['counter_name', 'site_name', 'counter_technical_id', 'counter_installation_date'], inplace=True)

    return data

def _is_daylight(x):

    city=LocationInfo('Paris', timezone='Europe/Paris')
    sun_info = daylight(city.observer, date=x.to_pydatetime().date(), tzinfo='Europe/Paris')
    x = x.tz_localize('Europe/Paris', ambiguous=True, nonexistent='shift_forward')
    return (x > sun_info[0]) & (x < sun_info[1])

def _provide_daylight_info(data):

    data = data.copy()
    data['is_daylight'] = data.date.map(_is_daylight)

    return data

def _clean_and_add_data(data):

    data = data.copy()
    data['origin_index'] = np.arange(data.shape[0])
    file_path = Path(__file__).parent / 'external_data.csv'
    external_data = pd.read_csv(file_path, parse_dates=['date'])
    external_data_grouped = external_data.groupby(by=['date_datetime']).sum()
    jours_feries = external_data_grouped.is_ferie > 0
    holidays = external_data_grouped.is_holiday > 0
    counters_list = data.counter_id.unique()
    down_counters = external_data_grouped[counters_list] > 0
    days_down={}
    for counter in counters_list :
        string_list = down_counters[down_counters[counter] == True].index.values.tolist()
        date_datetime_list = [pd.to_datetime(j).date() for j in string_list]
        days_down[counter] = date_datetime_list

    data['is_down'] = data.apply(lambda x: x.date_datetime in days_down[x.counter_id], axis=1)
    data['is_ferie'] = data.date_datetime.map(lambda x: jours_feries[str(x)])
    data['is_holiday'] = data.date_datetime.map(lambda x: holidays[str(x)])
    data.drop(columns=['date_datetime'], inplace=True)

    external_data.cl = external_data.cl.fillna(value=100)
    external_data.cm = external_data.cm.fillna(value=100)
    external_data.ch = external_data.ch.fillna(value=100)
    external_data.ssfrai = external_data.ssfrai.fillna(value=0.0)
    external_data.perssfrai = external_data.perssfrai.fillna(value=0.0)
    external_data.dropna(axis=1, thresh=3000, inplace=True)
    external_data.fillna(method='ffill', inplace=True)
    external_data.drop(columns=['numer_sta', 'per', 'pres', 'is_ferie', 'is_holiday'], inplace=True)
    ext_index = external_data.set_index('date')
    ext_index.sort_index(inplace=True)
    data_index = data.set_index('date')
    data_index.sort_index(inplace=True)
    merged_data = pd.merge_asof(data_index, ext_index, left_index=True, right_index=True)
    merged_data.sort_values("origin_index", inplace=True)
    merged_data.drop(columns=['origin_index', 'pmer', 'tend', 'cod_tend', 'dd', 'ff', 'td', 'vv', 'ww', 'w1', 'w2', 'tend24', 'ssfrai', 'perssfrai', 'n', 'nbas'], inplace=True)
    merged_data['is_confinement_1'] = (merged_data.date_datetime > '2020-10-30') & (merged_data.date_datetime < '2020-12-15')
    merged_data['is_confinement_2'] = (merged_data.date_datetime > '2021-04-03') & (merged_data.date_datetime < '2021-05-03')

    return merged_data

def get_estimator():
    provide_date_info = FunctionTransformer(_provide_date_info)
    provide_daylight_info = FunctionTransformer (_provide_daylight_info)
    clean_and_add_data = FunctionTransformer (_clean_and_add_data)

    categorical_columns = ['counter_id', 'site_id', 'is_ferie', 'is_holiday', 'is_confinement_1', 'is_confinement_2', 'hour', 'weekday', 'year', 'cl', 'cm', 'ch', 'is_daylight', 'is_down', 'etat_sol']
    numerical_columns = ['latitude',
                        'longitude',
                        'month',
                        'dom',
                        'week',
                        't',
                        'raf10',
                        'rafper',
                        'ht_neige',
                        'rr1',
                        'rr3',
                        'rr6',
                        'rr12',
                        'rr24']


    preprocessor = ColumnTransformer([('one_hot_encoder', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'), categorical_columns),
                                ('standard_scaler', StandardScaler(), numerical_columns)])

    model = XGBRegressor(max_depth=10, reg_lambda=2, eta=0.05, subsample=0.7, min_child_weight=2, max_estimators=200)

    pipe = make_pipeline(provide_date_info, provide_daylight_info, clean_and_add_data, preprocessor, model)

    return pipe