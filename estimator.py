from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

def _provide_date_info(data):

    data = data.copy()
    data['hour'] = data.date.dt.hour
    data['weekday'] = data.date.dt.weekday
    data['dom'] = data.date.dt.day
    data['week'] = data.date.dt.isocalendar().week
    data['month'] = data.date.dt.month
    data['year'] = data.date.dt.year
    data['date_datetime'] = data.date.map(lambda x: x.to_pydatetime().date())
    data.drop(columns=['counter_name', 'site_name', 'counter_technical_id', 'counter_installation_date'], inplace=True)

    return data

def _clean_and_add_data(data):

    data = data.copy()
    data['origin_index'] = np.arange(data.shape[0])
    file_path = Path(__file__).parent / 'external_data.csv'
    external_data = pd.read_csv(file_path, parse_dates=['date'])
    external_data_grouped = external_data.groupby(by=['date_datetime']).sum()
    counters_list = data.counter_id.unique()
    down_counters = external_data_grouped[counters_list] > 0
    days_down={}
    for counter in counters_list :
        string_list = down_counters[down_counters[counter] == True].index.values.tolist()
        date_datetime_list = [pd.to_datetime(j).date() for j in string_list]
        days_down[counter] = date_datetime_list

    data['is_down'] = data.apply(lambda x: x.date_datetime in days_down[x.counter_id], axis=1)
    data.drop(columns=['date_datetime'], inplace=True)

    external_data.cl = external_data.cl.fillna(value=100)
    external_data.cm = external_data.cm.fillna(value=100)
    external_data.ch = external_data.ch.fillna(value=100)
    external_data.drop(columns=counters_list, inplace=True)
    external_data.drop(columns=['numer_sta', 'per', 'pres', 'nnuage1', 'ctype1', 'hnuage1', 'pmer', 'tend', 
                                'cod_tend', 'dd', 'ff', 't', 'td', 'u', 'vv', 'ww', 'w1', 'w2', 'hbas', 'nbas', 'n',
                                'tend24', 'ssfrai', 'niv_bar', 'geop', 'tn12', 'tn24', 'tx12', 'tx24', 'tminsol',
                                'sw', 'tw', 'phenspe1', 'phenspe2', 'phenspe3', 'phenspe4', 'nnuage2', 'ctype2', 'hnuage2',
                                'nnuage3', 'ctype3', 'hnuage3', 'nnuage4', 'ctype4', 'hnuage4',
                                'perssfrai', 'etat_sol'], inplace=True)
    external_data.fillna(method='ffill', inplace=True)
    ext_index = external_data.set_index('date')
    ext_index.sort_index(inplace=True)
    ext_index.drop_duplicates(inplace=True)
    data_index = data.set_index('date')
    data_index.sort_index(inplace=True)
    merged_data = pd.merge_asof(data_index, ext_index, left_index=True, right_index=True)
    merged_data.sort_values("origin_index", inplace=True)
    merged_data.drop(columns=['origin_index'], inplace=True)
    merged_data['is_confinement_1'] = (merged_data.date_datetime > '2020-10-17') & (merged_data.date_datetime < '2020-12-15')
    merged_data['commerce_fermes_20'] = (merged_data.date_datetime > '2020-10-17') & (merged_data.date_datetime < '2020-11-28')
    merged_data['couvre_feu_20'] = (merged_data.date_datetime > '2020-12-14') & (merged_data.date_datetime < '2021-01-17')
    merged_data['couvre_feu_18'] = ((merged_data.date_datetime > '2021-01-16') & (merged_data.date_datetime < '2021-03-20')) | ((merged_data.date_datetime > '2021-05-02') & (merged_data.date_datetime < '2021-05-19'))
    merged_data['is_confinement_2'] = (merged_data.date_datetime > '2021-03-19') & (merged_data.date_datetime < '2021-04-03')
    merged_data['is_confinement_3'] = (merged_data.date_datetime > '2021-04-02') & (merged_data.date_datetime < '2021-05-03')
    merged_data['couvre_feu_21'] = (merged_data.date_datetime > '2021-18-05') & (merged_data.date_datetime < '2021-09-06')
    merged_data['couvre_feu_23'] = (merged_data.date_datetime > '2021-08-06') & (merged_data.date_datetime < '2021-21-06')

    return merged_data

def get_estimator():
    provide_date_info = FunctionTransformer(_provide_date_info)
    clean_and_add_data = FunctionTransformer (_clean_and_add_data)

    categorical_columns = ['counter_id', 'site_id', 'is_ferie', 'is_holiday', 'is_confinement_1', 'is_confinement_2', 'commerce_fermes_20',
                            'couvre_feu_20', 'couvre_feu_18', 'is_confinement_3', 'couvre_feu_21', 'couvre_feu_23', 'weekday',
                            'year', 'is_daylight', 'cl', 'cm', 'ch', 'pluie_intermittente', 'pluie_continue', 'pluie_forte',
                             'pluie_faible', 'pluie_modérée', 'neige', 'bruine', 'brouillard', 'verglas', 'is_down']
    numerical_columns = ['hour',
                        'month',
                        'latitude',
                        'longitude',
                        'dom',
                        'week',
                        'Res.',
                        'Vit.',
                        'Raf.3',
                        'Hum. [%]',
                        'Visi. [Km]',
                        'pluie_direct',
                        'pluie_last_3',
                        'temps_soleil',
                        'pluie_cumul_day',
                        'vent_max',
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

    model = HistGradientBoostingRegressor(max_depth=30, min_samples_leaf=50, l2_regularization=20, max_iter=1000)

    pipe = make_pipeline(provide_date_info, clean_and_add_data, preprocessor, model)

    return pipe