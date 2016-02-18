'''
Copyright (C) 2015 Dmitry Efimov <diefimov@gmail.com>
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re
import psycopg2

creds = pd.read_csv('../data/input/credentials.csv')
REDSHIFT_CONNECTION_STRING = str(creds.loc[creds.field=='REDSHIFT_CONNECTION_STRING', 'value'].values)
S3_CONNECTION_STRING = str(creds.loc[creds.field=='S3_CONNECTION_STRING', 'value'].values)
del creds

RE_FLOAT = re.compile("[0-9]+(?:\.[0-9]+)?")
def parse_float(t):
    t = str(t)
    nums = RE_FLOAT.findall(t)
    if len(nums) > 0:
        return float(nums[0])
    else:
        return float('nan')

### split list l in chunks of size n
def chunks(l, n):
    n = max(1, n)
    return [l[i:i + n] for i in range(0, len(l), n)]

def chunks2(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

### convert string date in year, weekday, month, day and absolute date
def convert_date(df, col):
    df.loc[df[col]=='NA', col] = ''
    df[col] = pd.to_datetime(df[col])
    df[col+'_year'] = pd.DatetimeIndex(df[col]).year
    df[col+'_weekday'] = pd.DatetimeIndex(df[col]).weekday
    df[col+'_month'] = pd.DatetimeIndex(df[col]).month
    df[col+'_day'] = pd.DatetimeIndex(df[col]).day
    df[col+'_abs'] = 365*(df[col+'_year']-1) + \
                     30*(df[col+'_month']-1) + \
                     df[col+'_day']
    df.drop(col, axis=1, inplace=True)
    return df

def convert_dict_based(df, col, dict_file, dict_col_from):
    df.loc[:,col] = [x.strip() for x in df[col].astype(str).values]
    dict_data = pd.read_csv(dict_file,
                            dtype = {dict_col_from: np.str})
    dict_data.reset_index(inplace=True)
    df.reset_index(inplace=True, drop=True)
    df.loc[~df[col].isin(dict_data[dict_col_from].values), col] = ''
    for i in range(len(dict_data)):
        old_value = dict_data.loc[i, dict_col_from]
        new_value = str(dict_data.loc[i, 'coeff'].astype(np.float32) / dict_data.loc[i, 'number_part_final'].astype(np.float32))
        df.loc[df[col]==old_value, col] = new_value
    df.loc[df[col]=='nan', col] = ''
    df.loc[df[col]=='.', col] = ''
    return df

def fill_empty_list(x):
    x = x.tolist()
    for i in range(len(x)-1):
        if x[i] != '' and x[i+1]=='':
            x[i+1] = x[i]
    for i in reversed(range(len(x)-1)):
        if x[i+1] != '' and x[i]=='':
            x[i] = x[i+1]
    return pd.Series(x)

def fill_empty(train, test):
    cols = ['REN_ID','VE_NUMBER','REN_LEASE_LENGTH', 'REN_DATE_EFF_FROM_abs']
    tt = train[cols].append(test[cols], ignore_index=True)
    tt.sort_values(by=['VE_NUMBER', 'REN_DATE_EFF_FROM_abs'], ascending=[True, True], inplace=True)
    tt_group = tt.groupby('VE_NUMBER')
    lease_length = pd.DataFrame()
    lease_length['REN_ID'] = tt_group['REN_ID'].apply(lambda x: pd.Series(x))
    lease_length['REN_LEASE_LENGTH'] = tt_group['REN_LEASE_LENGTH'].apply(fill_empty_list).values
    lease_length.loc[lease_length['REN_LEASE_LENGTH']=='', 'REN_LEASE_LENGTH'] = '1.0'
    lease_length['REN_LEASE_LENGTH'] = lease_length['REN_LEASE_LENGTH'].astype(np.float32)
    train.drop('REN_LEASE_LENGTH', axis=1, inplace=True)
    test.drop('REN_LEASE_LENGTH', axis=1, inplace=True)
    train = pd.merge(train, lease_length, on='REN_ID', how='inner')
    test = pd.merge(test, lease_length, on='REN_ID', how='inner')
    return train, test

### calculates all possible Haversine distances between two arrays
def haversine_distances(loc1, loc2):
    earth_radius = 6378.137

    locs_1 = np.deg2rad(loc1)
    locs_2 = np.deg2rad(loc2)

    lat_dif = (locs_1[:,0][:,None]/2 - locs_2[:,0]/2)
    lon_dif = (locs_1[:,1][:,None]/2 - locs_2[:,1]/2)

    np.sin(lat_dif, out=lat_dif)
    np.sin(lon_dif, out=lon_dif)

    np.power(lat_dif, 2, out=lat_dif)
    np.power(lon_dif, 2, out=lon_dif)

    lon_dif *= ( np.cos(locs_1[:,0])[:,None] * np.cos(locs_2[:,0]) )
    lon_dif += lat_dif

    np.arctan2(np.power(lon_dif,.5), np.power(1-lon_dif,.5), out = lon_dif)
    lon_dif *= ( 2 * earth_radius )

    return lon_dif

### encodes categorical features to dummy features
def encode_onehot(train, test, cols):
    print "One hot encoding..."
    tt = train[cols].append(test[cols], ignore_index=True)
    for col in cols:
        train.drop(col, axis = 1, inplace=True)
        test.drop(col, axis = 1, inplace=True)
    ntrain = len(train)

    cols_to_remove = []
    for col in cols:
        if len(np.unique(tt[col]))<3:
            le = LabelEncoder()
            tt[col] = le.fit_transform(tt[col].values)
            cols_to_remove = cols_to_remove + [col]
    cols = [x for x in cols if not x in cols_to_remove]
    if len(cols)>0:
        tt = pd.get_dummies(tt, columns=cols, dummy_na=False)

    train = pd.concat([train, tt.iloc[0:ntrain]], axis=1)
    test = pd.concat([test, tt.iloc[ntrain:].reset_index(level=0)], axis=1)
    del tt
    return train, test

def connect_to_database(reset_connection=False):
    conn = psycopg2.connect(REDSHIFT_CONNECTION_STRING)
    return conn
