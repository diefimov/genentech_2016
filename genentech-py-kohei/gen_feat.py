# -*- coding: utf-8 -*-
"""
Feature extraction
"""
import os
import sys
import logging as l
import datetime
from collections import Counter

import bloscpack as bp
import scipy.sparse as ss
import scipy.io as sio
import numpy as np
import pandas as pd
import tables as tb
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.feature_extraction.text import CountVectorizer as CV

import genentech_path as g


def _load_base():
    df_base = pd.concat([
        pd.read_csv(g.FILE_TRAIN_OUT, usecols=['patient_id']),
        pd.read_csv(g.FILE_TEST_OUT, usecols=['patient_id']),
    ])
    return df_base


def _savemat(X, filepath):
    X = ss.csc_matrix(X)
    with tb.open_file(filepath, 'w') as f:
        filters = tb.Filters(complevel=5, complib='blosc')
        out_data = f.create_earray(
            f.root, 'data', tb.Float32Atom(), shape=(0,), filters=filters)
        out_indices = f.create_earray(
            f.root, 'indices', tb.Int32Atom(), shape=(0,), filters=filters)
        out_indptr = f.create_earray(
            f.root, 'indptr', tb.Int32Atom(), shape=(0,), filters=filters)
        out_shape = f.create_earray(
            f.root, 'shape', tb.Int32Atom(), shape=(0,), filters=filters)

        out_data.append(X.data)
        out_indices.append(X.indices)
        out_indptr.append(X.indptr)
        out_shape.append(np.array([X.shape[0], X.shape[1]]))


"""
g.FILE_PA
"""


def pa_most_recent_year():
    df_base = _load_base()
    df = pd.read_csv(g.FILE_PA_OUT, usecols=['patient_id', 'activity_year'])
    df = df.groupby('patient_id').agg({'activity_year': np.max}).reset_index()
    df_base = df_base.merge(df, how='left', on='patient_id')
    return np.array(df_base.activity_year.fillna(0)).reshape((-1, 1))


"""
g.FILE_DIAG
"""


def diag_num_record():
    df_base = _load_base()

    df = pd.read_csv(
        g.FILE_DIAG_OUT,
        usecols=['patient_id'])
    df['num_record'] = 1
    df = df.groupby('patient_id').agg({'num_record': np.sum}).reset_index()

    df_base = df_base.merge(df, how='left', on='patient_id')
    return np.array(df_base.num_record.fillna(0)).reshape((-1, 1))


def diag_uniq_claim_id():
    df_base = _load_base()

    df = pd.read_csv(
        g.FILE_DIAG_OUT,
        usecols=['patient_id', 'claim_id'])
    df = df.groupby('patient_id').agg({
        'claim_id': lambda x: len(x.unique())
    }).reset_index()
    df.columns = ['patient_id', 'uniq_claim_id']

    df_base = df_base.merge(df, how='left', on='patient_id')
    return np.array(df_base.uniq_claim_id.fillna(0)).reshape((-1, 1))


def diag_physician_specialty_description_dv():
    df_base = _load_base()
    df_base['idx'] = list(range(len(df_base)))
    print("load trans")
    df = pd.read_csv(
        g.FILE_DIAG_OUT,
        usecols=['patient_id', 'primary_practitioner_id'],
        dtype={'primary_practitioner_id': str},
    ).rename(columns={
        'primary_practitioner_id': 'practitioner_id'})

    df_phy = pd.read_csv(
        g.FILE_PHYS,
        usecols=['specialty_description', 'practitioner_id'],
        dtype={
            'practitioner_id': str,
            'specialty_description': str})

    df['practitioner_id'].fillna('NA', inplace=True)
    df_phy['practitioner_id'].fillna('NA', inplace=True)
    df_phy['specialty_description'].fillna('NA', inplace=True)
    df = df.merge(df_phy, how='left', on='practitioner_id')

    print("load trans ... done")

    patient_list = []
    df_feat = []
    for patient_id, df_part in df.groupby('patient_id'):
        df_feat.append(Counter(df_part.specialty_description))
        patient_list.append(patient_id)
    X = DV(sparse=True).fit_transform(df_feat)
    print(X.shape)

    df_feat_order = pd.DataFrame({
        'patient_id': patient_list,
        'feat_idx': list(range(len(patient_list))),
    })
    df_feat_order = df_feat_order.merge(df_base, how='left', on='patient_id')
    print(df_feat_order.head())

    # Re-ordering
    print("re-ordering")
    sz = len(df_feat_order)
    X_ordered = ss.lil_matrix((len(df_base), X.shape[1]))
    for idx, row in df_feat_order.iterrows():
        if idx % 1000 == 0:
            print(str(datetime.datetime.now()), idx, sz)
        X_ordered[row['idx'], :] = X[row['feat_idx'], :]
    print("re-ordering ... done")

    X_ordered = ss.csr_matrix(X_ordered)
    print(X_ordered.shape)
    return X_ordered


def diag_physician_state_dv():
    df_base = _load_base()
    df_base['idx'] = list(range(len(df_base)))
    print("load trans")
    df = pd.read_csv(
        g.FILE_DIAG_OUT,
        usecols=['patient_id', 'primary_practitioner_id'],
        dtype={'primary_practitioner_id': str},
    ).rename(columns={
        'primary_practitioner_id': 'practitioner_id'})

    df_phy = pd.read_csv(
        g.FILE_PHYS,
        usecols=['state', 'practitioner_id'],
        dtype={
            'practitioner_id': str,
            'state': str})

    df['practitioner_id'].fillna('NA', inplace=True)
    df_phy['practitioner_id'].fillna('NA', inplace=True)
    df_phy['state'].fillna('NA', inplace=True)
    df = df.merge(df_phy, how='left', on='practitioner_id')

    print("load trans ... done")

    patient_list = []
    df_feat = []
    for patient_id, df_part in df.groupby('patient_id'):
        df_feat.append(Counter(df_part.state))
        patient_list.append(patient_id)
    X = DV(sparse=True).fit_transform(df_feat)
    print(X.shape)

    df_feat_order = pd.DataFrame({
        'patient_id': patient_list,
        'feat_idx': list(range(len(patient_list))),
    })
    df_feat_order = df_feat_order.merge(df_base, how='left', on='patient_id')
    print(df_feat_order.head())

    # Re-ordering
    print("re-ordering")
    sz = len(df_feat_order)
    X_ordered = ss.lil_matrix((len(df_base), X.shape[1]))
    for idx, row in df_feat_order.iterrows():
        if idx % 1000 == 0:
            print(str(datetime.datetime.now()), idx, sz)
        X_ordered[row['idx'], :] = X[row['feat_idx'], :]
    print("re-ordering ... done")

    X_ordered = ss.csr_matrix(X_ordered)
    print(X_ordered.shape)
    return X_ordered


def diag_diagnosis_date_unique_count():
    df_base = _load_base()
    df = pd.read_csv(
        g.FILE_DIAG_OUT,
        usecols=['patient_id', 'diagnosis_date'],
        dtype={'diagnosis_date': str},
    )
    df['diagnosis_date'].fillna('NA', inplace=True)
    print("load trans ... done")

    patient_list = []
    df_feat = []
    for patient_id, df_part in df.groupby('patient_id'):
        df_feat.append({
            'patient_id': patient_id,
            'uniq_count': len(df_part.diagnosis_date.unique()),
        })
    df_feat = pd.DataFrame(df_feat)
    df_base = df_base.merge(df_feat, how='left', on='patient_id')
    return np.array(df_base.uniq_count.fillna(0)).reshape((-1, 1))


def diag_diagnosis_code_dv():
    df_base = _load_base()
    df_base['idx'] = list(range(len(df_base)))
    print("load trans")
    df = pd.read_csv(
        g.FILE_DIAG_OUT,
        usecols=['patient_id', 'diagnosis_code'],
        dtype={'diagnosis_code': str},
    )
    df['diagnosis_code'].fillna('NA', inplace=True)
    print("load trans ... done")

    patient_list = []
    df_feat = []
    for patient_id, df_part in df.groupby('patient_id'):
        df_feat.append(Counter(df_part.diagnosis_code))
        patient_list.append(patient_id)
    X = DV(sparse=True).fit_transform(df_feat)
    print(X.shape)

    df_feat_order = pd.DataFrame({
        'patient_id': patient_list,
        'feat_idx': list(range(len(patient_list))),
    })
    df_feat_order = df_feat_order.merge(df_base, how='left', on='patient_id')
    print(df_feat_order.head())

    # Re-ordering
    print("re-ordering")
    sz = len(df_feat_order)
    X_ordered = ss.lil_matrix((len(df_base), X.shape[1]))
    for idx, row in df_feat_order.iterrows():
        if idx % 1000 == 0:
            print(str(datetime.datetime.now()), idx, sz)
        X_ordered[row['idx'], :] = X[row['feat_idx'], :]
    print("re-ordering ... done")

    X_ordered = ss.csr_matrix(X_ordered)
    print(X_ordered.shape)
    return X_ordered


def diag_primary_physician_role_dv():
    df_base = _load_base()
    df_base['idx'] = list(range(len(df_base)))
    print("load trans")
    df = pd.read_csv(
        g.FILE_DIAG_OUT,
        usecols=['patient_id', 'primary_physician_role'],
        dtype={'primary_physician_role': str},
    )
    df['primary_physician_role'].fillna('NA', inplace=True)
    print("load trans ... done")

    patient_list = []
    df_feat = []
    for patient_id, df_part in df.groupby('patient_id'):
        df_feat.append(Counter(df_part.primary_physician_role))
        patient_list.append(patient_id)
    X = DV(sparse=True).fit_transform(df_feat)
    print(X.shape)

    df_feat_order = pd.DataFrame({
        'patient_id': patient_list,
        'feat_idx': list(range(len(patient_list))),
    })
    df_feat_order = df_feat_order.merge(df_base, how='left', on='patient_id')
    print(df_feat_order.head())

    # Re-ordering
    print("re-ordering")
    sz = len(df_feat_order)
    X_ordered = ss.lil_matrix((len(df_base), X.shape[1]))
    for idx, row in df_feat_order.iterrows():
        if idx % 1000 == 0:
            print(str(datetime.datetime.now()), idx, sz)
        X_ordered[row['idx'], :] = X[row['feat_idx'], :]
    print("re-ordering ... done")

    X_ordered = ss.csr_matrix(X_ordered)
    print(X_ordered.shape)
    return X_ordered


def diag_diagnosis_description_1gram_dv():
    df_base = _load_base()
    df_base['idx'] = list(range(len(df_base)))
    print("load trans")
    df = pd.read_csv(
        g.FILE_DIAG_OUT,
        usecols=['patient_id', 'diagnosis_code'],
        dtype={'diagnosis_code': str},
    )
    df['diagnosis_code'].fillna('NA', inplace=True)
    print("load trans ... done")

    df_diag = pd.read_csv(
        g.FILE_DIAG_CD,
        dtype={'diagnosis_code': str})
    vec = CV(
        min_df=2,
        stop_words='english',
        analyzer='word')
    vec.fit(df_diag.diagnosis_description)

    df = df.merge(df_diag, how='left', on='diagnosis_code')

    patient_list = []
    sz = len(df.groupby('patient_id'))
    X = ss.lil_matrix((sz, len(vec.vocabulary_)))
    for i, (patient_id, df_part) in enumerate(df.groupby('patient_id')):
        if i % 1000 == 0:
            print(str(datetime.datetime.now()), i, sz)
        x_part = vec.transform(df_part.diagnosis_description).sum(axis=0)
        X[i, :] = x_part
        patient_list.append(patient_id)
    X = ss.csr_matrix(X)
    print(X.shape)

    df_feat_order = pd.DataFrame({
        'patient_id': patient_list,
        'feat_idx': list(range(len(patient_list))),
    })
    df_feat_order = df_feat_order.merge(df_base, how='left', on='patient_id')
    print(df_feat_order.head())

    # Re-ordering
    print("re-ordering")
    sz = len(df_feat_order)
    X_ordered = ss.lil_matrix((len(df_base), X.shape[1]))
    for idx, row in df_feat_order.iterrows():
        if idx % 1000 == 0:
            print(str(datetime.datetime.now()), idx, sz)
        X_ordered[row['idx'], :] = X[row['feat_idx'], :]
    print("re-ordering ... done")

    X_ordered = ss.csr_matrix(X_ordered)
    print(X_ordered.shape)
    return X_ordered


def diag_physician_cbsa_dv():
    df_base = _load_base()
    df_base['idx'] = list(range(len(df_base)))
    print("load trans")
    df = pd.read_csv(
        g.FILE_DIAG_OUT,
        usecols=['patient_id', 'primary_practitioner_id'],
        dtype={'primary_practitioner_id': str},
    ).rename(columns={
        'primary_practitioner_id': 'practitioner_id'})

    df_phy = pd.read_csv(
        g.FILE_PHYS,
        usecols=['CBSA', 'practitioner_id'],
        dtype={
            'practitioner_id': str,
            'CBSA': str})

    df['practitioner_id'].fillna('NA', inplace=True)
    df_phy['practitioner_id'].fillna('NA', inplace=True)
    df_phy['CBSA'].fillna('NA', inplace=True)
    df = df.merge(df_phy, how='left', on='practitioner_id')

    print("load trans ... done")

    patient_list = []
    df_feat = []
    for patient_id, df_part in df.groupby('patient_id'):
        df_feat.append(Counter(df_part.CBSA))
        patient_list.append(patient_id)
    X = DV(sparse=True).fit_transform(df_feat)
    print(X.shape)

    df_feat_order = pd.DataFrame({
        'patient_id': patient_list,
        'feat_idx': list(range(len(patient_list))),
    })
    df_feat_order = df_feat_order.merge(df_base, how='left', on='patient_id')
    print(df_feat_order.head())

    # Re-ordering
    print("re-ordering")
    sz = len(df_feat_order)
    X_ordered = ss.lil_matrix((len(df_base), X.shape[1]))
    for idx, row in df_feat_order.iterrows():
        if idx % 1000 == 0:
            print(str(datetime.datetime.now()), idx, sz)
        X_ordered[row['idx'], :] = X[row['feat_idx'], :]
    print("re-ordering ... done")

    X_ordered = ss.csr_matrix(X_ordered)
    print(X_ordered.shape)
    return X_ordered


"""
g.FILE_PRES
"""

# drug_generic_name
# drug_strength
# manufacturer


def pres_drug_manufacturer_dv():
    df_base = _load_base()
    df_base['idx'] = list(range(len(df_base)))
    print("load trans")
    df = pd.read_csv(
        g.FILE_PRES_OUT,
        usecols=['patient_id', 'drug_id'],
        dtype={'drug_id': str})
    print("load trans ... done")

    df['drug_id'].fillna('NA', inplace=True)

    # Drug
    df_drug = pd.read_csv(
        g.FILE_DRUG,
        usecols=['drug_id', 'manufacturer'],
        dtype={'drug_id': str, 'manufacturer': str})
    df_drug['drug_id'].fillna('NA', inplace=True)
    df_drug['manufacturer'].fillna('NA', inplace=True)

    df = df.merge(df_drug, how='left', on='drug_id')

    patient_list = []
    df_feat = []
    for patient_id, df_part in df.groupby('patient_id'):
        df_feat.append(Counter(df_part.manufacturer))
        patient_list.append(patient_id)
    X = DV(sparse=True).fit_transform(df_feat)
    print(X.shape)

    df_feat_order = pd.DataFrame({
        'patient_id': patient_list,
        'feat_idx': list(range(len(patient_list))),
    })
    df_feat_order = df_feat_order.merge(df_base, how='left', on='patient_id')
    print(df_feat_order.head())

    # Re-ordering
    X_ordered = ss.lil_matrix((len(df_base), X.shape[1]))
    for idx, row in df_feat_order.iterrows():
        X_ordered[row['idx'], :] = X[row['feat_idx'], :]

    X_ordered = ss.csr_matrix(X_ordered)
    print(X_ordered.shape)
    return X_ordered


def pres_drug_generic_name_dv():
    df_base = _load_base()
    df_base['idx'] = list(range(len(df_base)))
    print("load trans")
    df = pd.read_csv(
        g.FILE_PRES_OUT,
        usecols=['patient_id', 'drug_id'],
        dtype={'drug_id': str})
    print("load trans ... done")

    df['drug_id'].fillna('NA', inplace=True)

    # Drug
    df_drug = pd.read_csv(
        g.FILE_DRUG,
        usecols=['drug_id', 'drug_generic_name'],
        dtype={'drug_id': str, 'drug_generic_name': str})
    df_drug['drug_id'].fillna('NA', inplace=True)
    df_drug['drug_generic_name'].fillna('NA', inplace=True)

    df = df.merge(df_drug, how='left', on='drug_id')

    patient_list = []
    df_feat = []
    for patient_id, df_part in df.groupby('patient_id'):
        df_feat.append(Counter(df_part.drug_generic_name))
        patient_list.append(patient_id)
    X = DV(sparse=True).fit_transform(df_feat)
    print(X.shape)

    df_feat_order = pd.DataFrame({
        'patient_id': patient_list,
        'feat_idx': list(range(len(patient_list))),
    })
    df_feat_order = df_feat_order.merge(df_base, how='left', on='patient_id')
    print(df_feat_order.head())

    # Re-ordering
    X_ordered = ss.lil_matrix((len(df_base), X.shape[1]))
    for idx, row in df_feat_order.iterrows():
        X_ordered[row['idx'], :] = X[row['feat_idx'], :]

    X_ordered = ss.csr_matrix(X_ordered)
    print(X_ordered.shape)
    return X_ordered


def proc_procedure_description_1gram_dv():
    df_base = _load_base()
    df_base['idx'] = list(range(len(df_base)))
    print("load trans")
    df = pd.read_csv(
        g.FILE_PROC_OUT,
        usecols=['patient_id', 'procedure_code'],
        dtype={'procedure_code': str},
    )
    df['procedure_code'].fillna('NA', inplace=True)
    print("load trans ... done")

    df_proc = pd.read_csv(
        g.FILE_PROC_CD,
        dtype={'procedure_code': str})
    vec = CV(
        min_df=2,
        stop_words='english',
        analyzer='word')
    vec.fit(df_proc.procedure_description)

    df = df.merge(df_proc, how='left', on='procedure_code')

    patient_list = []
    sz = len(df.groupby('patient_id'))
    X = ss.lil_matrix((sz, len(vec.vocabulary_)))
    for i, (patient_id, df_part) in enumerate(df.groupby('patient_id')):
        if i % 1000 == 0:
            print(str(datetime.datetime.now()), i, sz)
        x_part = vec.transform(df_part.procedure_description).sum(axis=0)
        X[i, :] = x_part
        patient_list.append(patient_id)
    X = ss.csr_matrix(X)
    print(X.shape)

    df_feat_order = pd.DataFrame({
        'patient_id': patient_list,
        'feat_idx': list(range(len(patient_list))),
    })
    df_feat_order = df_feat_order.merge(df_base, how='left', on='patient_id')
    print(df_feat_order.head())

    # Re-ordering
    print("re-ordering")
    sz = len(df_feat_order)
    X_ordered = ss.lil_matrix((len(df_base), X.shape[1]))
    for idx, row in df_feat_order.iterrows():
        if idx % 1000 == 0:
            print(str(datetime.datetime.now()), idx, sz)
        X_ordered[row['idx'], :] = X[row['feat_idx'], :]
    print("re-ordering ... done")

    X_ordered = ss.csr_matrix(X_ordered)
    print(X_ordered.shape)
    return X_ordered


def pres_drug_bb_usc_code_dv():
    df_base = _load_base()
    df_base['idx'] = list(range(len(df_base)))
    print("load trans")
    df = pd.read_csv(
        g.FILE_PRES_OUT,
        usecols=['patient_id', 'drug_id'],
        dtype={'drug_id': str})
    print("load trans ... done")

    df['drug_id'].fillna('NA', inplace=True)

    # Drug
    df_drug = pd.read_csv(
        g.FILE_DRUG,
        usecols=['drug_id', 'BB_USC_code'],
        dtype={'drug_id': str, 'BB_USC_code': str})
    df_drug['drug_id'].fillna('NA', inplace=True)
    df_drug['BB_USC_code'].fillna('NA', inplace=True)

    df = df.merge(df_drug, how='left', on='drug_id')

    patient_list = []
    df_feat = []
    for patient_id, df_part in df.groupby('patient_id'):
        df_feat.append(Counter(df_part.BB_USC_code))
        patient_list.append(patient_id)
    X = DV(sparse=True).fit_transform(df_feat)
    print(X.shape)

    df_feat_order = pd.DataFrame({
        'patient_id': patient_list,
        'feat_idx': list(range(len(patient_list))),
    })
    df_feat_order = df_feat_order.merge(df_base, how='left', on='patient_id')
    print(df_feat_order.head())

    # Re-ordering
    X_ordered = ss.lil_matrix((len(df_base), X.shape[1]))
    for idx, row in df_feat_order.iterrows():
        X_ordered[row['idx'], :] = X[row['feat_idx'], :]

    X_ordered = ss.csr_matrix(X_ordered)
    print(X_ordered.shape)
    return X_ordered


def pres_drug_dv():
    """
    2.5 min on 2% dataset
    """
    df_base = _load_base()
    df_base['idx'] = list(range(len(df_base)))
    print("load trans")
    df = pd.read_csv(g.FILE_PRES_OUT, usecols=['patient_id', 'drug_id'])
    print("load trans ... done")

    patient_list = []
    df_feat = []
    for patient_id, df_part in df.groupby('patient_id'):
        df_feat.append(Counter(df_part.drug_id))
        patient_list.append(patient_id)
    X = DV(sparse=True).fit_transform(df_feat)
    print(X.shape)

    df_feat_order = pd.DataFrame({
        'patient_id': patient_list,
        'feat_idx': list(range(len(patient_list))),
    })
    df_feat_order = df_feat_order.merge(df_base, how='left', on='patient_id')
    print(df_feat_order.head())

    # Re-ordering
    X_ordered = ss.lil_matrix((len(df_base), X.shape[1]))
    for idx, row in df_feat_order.iterrows():
        X_ordered[row['idx'], :] = X[row['feat_idx'], :]

    X_ordered = ss.csr_matrix(X_ordered)
    print(X_ordered.shape)
    return X_ordered


def pres_unique_claim_id():
    df_base = _load_base()
    df_base['idx'] = list(range(len(df_base)))

    print("load trans")
    df = pd.read_csv(
        g.FILE_PRES_OUT,
        usecols=['patient_id', 'claim_id'],
        dtype={'claim_id': str})
    df.dropna(subset=['claim_id'], inplace=True)
    print("load trans ... done")

    df = df.groupby('patient_id').agg({
        'claim_id': lambda x: len(x.unique()),
    }).reset_index()
    df.columns = [
        'patient_id',
        'uq_claim_id',
    ]
    df_base = df_base.merge(df, how='left', on='patient_id')
    return np.array(
        df_base.uq_claim_id.fillna(0)
    ).reshape((-1, 1))


def pres_days_supply_stat():
    df_base = _load_base()
    df_base['idx'] = list(range(len(df_base)))

    df = pd.read_csv(
        g.FILE_PRES_OUT,
        usecols=['patient_id', 'days_supply'])
    df.dropna(subset=['days_supply'], inplace=True)

    df = df.groupby('patient_id').agg({
        'days_supply': [np.max, np.min],
    }).reset_index()
    df.columns = [
        'patient_id',
        'days_supply_max',
        'days_supply_min',
    ]
    df_base = df_base.merge(df, how='left', on='patient_id')
    return np.array(df_base.fillna(0)[[
        'days_supply_max',
        'days_supply_min',
    ]])


"""
g.FILE_PROC
"""


def proc_exists_attending_practitioner_id():
    df_base = _load_base()
    df = pd.read_csv(
        g.FILE_PROC_OUT,
        usecols=['patient_id', 'attending_practitioner_id'],
        dtype={'attending_practitioner_id': str})
    df['attending_practitioner_id'].fillna('NA', inplace=True)
    df = df[df['attending_practitioner_id'] != 'NA']
    df['num_record'] = 1
    df = df.groupby('patient_id').agg({'num_record': np.sum}).reset_index()
    df_base = df_base.merge(df, how='left', on='patient_id')
    return np.array(df_base.num_record.fillna(0)).reshape((-1, 1))


def proc_exists_referring_practitioner_id():
    df_base = _load_base()
    df = pd.read_csv(
        g.FILE_PROC_OUT,
        usecols=['patient_id', 'referring_practitioner_id'],
        dtype={'referring_practitioner_id': str})
    df['referring_practitioner_id'].fillna('NA', inplace=True)
    df = df[df['referring_practitioner_id'] != 'NA']
    df['num_record'] = 1
    df = df.groupby('patient_id').agg({'num_record': np.sum}).reset_index()
    df_base = df_base.merge(df, how='left', on='patient_id')
    return np.array(df_base.num_record.fillna(0)).reshape((-1, 1))


def proc_exists_operating_practitioner_id():
    df_base = _load_base()
    df = pd.read_csv(
        g.FILE_PROC_OUT,
        usecols=['patient_id', 'operating_practitioner_id'],
        dtype={'operating_practitioner_id': str})
    df['operating_practitioner_id'].fillna('NA', inplace=True)
    df = df[df['operating_practitioner_id'] != 'NA']
    df['num_record'] = 1
    df = df.groupby('patient_id').agg({'num_record': np.sum}).reset_index()
    df_base = df_base.merge(df, how='left', on='patient_id')
    return np.array(df_base.num_record.fillna(0)).reshape((-1, 1))


def proc_num_record():
    df_base = _load_base()

    df_su = pd.read_csv(g.FILE_PROC_OUT, usecols=['patient_id'])
    df_su['num_record'] = 1
    df_su = df_su.groupby('patient_id').agg({
        'num_record': np.sum,
    }).reset_index()
    df_base = df_base.merge(df_su, how='left', on='patient_id')
    return np.array(df_base.num_record.fillna(0)).reshape((-1, 1))


def proc_uniq_claim_id():
    df_base = _load_base()

    df_su = pd.read_csv(g.FILE_PROC_OUT, usecols=['patient_id', 'claim_id'])
    df_su = df_su.groupby('patient_id').agg({
        'claim_id': lambda x: len(x.unique()),
    }).reset_index()
    df_su.columns = ['patient_id', 'num_uniq']
    df_base = df_base.merge(df_su, how='left', on='patient_id')
    return np.array(df_base.num_uniq.fillna(0)).reshape((-1, 1))


def proc_physician_specialty_description_dv():
    df_base = _load_base()
    df_base['idx'] = list(range(len(df_base)))
    print("load trans")
    df = pd.read_csv(
        g.FILE_PROC_OUT,
        usecols=['patient_id', 'primary_practitioner_id'],
        dtype={'primary_practitioner_id': str},
    ).rename(columns={
        'primary_practitioner_id': 'practitioner_id'})

    df_phy = pd.read_csv(
        g.FILE_PHYS,
        usecols=['specialty_description', 'practitioner_id'],
        dtype={
            'practitioner_id': str,
            'specialty_description': str})

    df['practitioner_id'].fillna('NA', inplace=True)
    df_phy['practitioner_id'].fillna('NA', inplace=True)
    df_phy['specialty_description'].fillna('NA', inplace=True)
    df = df.merge(df_phy, how='left', on='practitioner_id')

    print("load trans ... done")

    patient_list = []
    df_feat = []
    for patient_id, df_part in df.groupby('patient_id'):
        df_feat.append(Counter(df_part.specialty_description))
        patient_list.append(patient_id)
    X = DV(sparse=True).fit_transform(df_feat)
    print(X.shape)

    df_feat_order = pd.DataFrame({
        'patient_id': patient_list,
        'feat_idx': list(range(len(patient_list))),
    })
    df_feat_order = df_feat_order.merge(df_base, how='left', on='patient_id')
    print(df_feat_order.head())

    # Re-ordering
    print("re-ordering")
    sz = len(df_feat_order)
    X_ordered = ss.lil_matrix((len(df_base), X.shape[1]))
    for idx, row in df_feat_order.iterrows():
        if idx % 1000 == 0:
            print(str(datetime.datetime.now()), idx, sz)
        X_ordered[row['idx'], :] = X[row['feat_idx'], :]
    print("re-ordering ... done")

    X_ordered = ss.csr_matrix(X_ordered)
    print(X_ordered.shape)
    return X_ordered


def proc_physician_state_dv():
    df_base = _load_base()
    df_base['idx'] = list(range(len(df_base)))
    print("load trans")
    df = pd.read_csv(
        g.FILE_PROC_OUT,
        usecols=['patient_id', 'primary_practitioner_id'],
        dtype={'primary_practitioner_id': str},
    ).rename(columns={
        'primary_practitioner_id': 'practitioner_id'})

    df_phy = pd.read_csv(
        g.FILE_PHYS,
        usecols=['state', 'practitioner_id'],
        dtype={
            'practitioner_id': str,
            'state': str})

    df['practitioner_id'].fillna('NA', inplace=True)
    df_phy['practitioner_id'].fillna('NA', inplace=True)
    df_phy['state'].fillna('NA', inplace=True)
    df = df.merge(df_phy, how='left', on='practitioner_id')

    print("load trans ... done")

    patient_list = []
    df_feat = []
    for patient_id, df_part in df.groupby('patient_id'):
        df_feat.append(Counter(df_part.state))
        patient_list.append(patient_id)
    X = DV(sparse=True).fit_transform(df_feat)
    print(X.shape)

    df_feat_order = pd.DataFrame({
        'patient_id': patient_list,
        'feat_idx': list(range(len(patient_list))),
    })
    df_feat_order = df_feat_order.merge(df_base, how='left', on='patient_id')
    print(df_feat_order.head())

    # Re-ordering
    print("re-ordering")
    sz = len(df_feat_order)
    X_ordered = ss.lil_matrix((len(df_base), X.shape[1]))
    for idx, row in df_feat_order.iterrows():
        if idx % 1000 == 0:
            print(str(datetime.datetime.now()), idx, sz)
        X_ordered[row['idx'], :] = X[row['feat_idx'], :]
    print("re-ordering ... done")

    X_ordered = ss.csr_matrix(X_ordered)
    print(X_ordered.shape)
    return X_ordered


def proc_place_of_service_dv():
    df_base = _load_base()
    df_base['idx'] = list(range(len(df_base)))
    print("load trans")
    df = pd.read_csv(
        g.FILE_PROC_OUT,
        usecols=['patient_id', 'place_of_service'],
        dtype={'place_of_service': str},
    )

    df['place_of_service'].fillna('NA', inplace=True)
    print("load trans ... done")

    patient_list = []
    df_feat = []
    for patient_id, df_part in df.groupby('patient_id'):
        df_feat.append(Counter(df_part.place_of_service))
        patient_list.append(patient_id)
    X = DV(sparse=True).fit_transform(df_feat)
    print(X.shape)

    df_feat_order = pd.DataFrame({
        'patient_id': patient_list,
        'feat_idx': list(range(len(patient_list))),
    })
    df_feat_order = df_feat_order.merge(df_base, how='left', on='patient_id')
    print(df_feat_order.head())

    # Re-ordering
    print("re-ordering")
    sz = len(df_feat_order)
    X_ordered = ss.lil_matrix((len(df_base), X.shape[1]))
    for idx, row in df_feat_order.iterrows():
        if idx % 1000 == 0:
            print(str(datetime.datetime.now()), idx, sz)
        X_ordered[row['idx'], :] = X[row['feat_idx'], :]
    print("re-ordering ... done")

    X_ordered = ss.csr_matrix(X_ordered)
    print(X_ordered.shape)
    return X_ordered


def proc_plan_type_dv():
    df_base = _load_base()
    df_base['idx'] = list(range(len(df_base)))
    print("load trans")
    df = pd.read_csv(
        g.FILE_PROC_OUT,
        usecols=['patient_id', 'plan_type'],
        dtype={'plan_type': str},
    )

    df['plan_type'].fillna('NA', inplace=True)
    print("load trans ... done")

    patient_list = []
    df_feat = []
    for patient_id, df_part in df.groupby('patient_id'):
        df_feat.append(Counter(df_part.plan_type))
        patient_list.append(patient_id)
    X = DV(sparse=True).fit_transform(df_feat)
    print(X.shape)

    df_feat_order = pd.DataFrame({
        'patient_id': patient_list,
        'feat_idx': list(range(len(patient_list))),
    })
    df_feat_order = df_feat_order.merge(df_base, how='left', on='patient_id')
    print(df_feat_order.head())

    # Re-ordering
    print("re-ordering")
    sz = len(df_feat_order)
    X_ordered = ss.lil_matrix((len(df_base), X.shape[1]))
    for idx, row in df_feat_order.iterrows():
        if idx % 1000 == 0:
            print(str(datetime.datetime.now()), idx, sz)
        X_ordered[row['idx'], :] = X[row['feat_idx'], :]
    print("re-ordering ... done")

    X_ordered = ss.csr_matrix(X_ordered)
    print(X_ordered.shape)
    return X_ordered


def proc_procedure_code_dv():
    df_base = _load_base()
    df_base['idx'] = list(range(len(df_base)))
    print("load trans")
    df = pd.read_csv(
        g.FILE_PROC_OUT,
        usecols=['patient_id', 'procedure_code'],
        dtype={'procedure_code': str},
    )

    df['procedure_code'].fillna('NA', inplace=True)
    print("load trans ... done")

    patient_list = []
    df_feat = []
    for patient_id, df_part in df.groupby('patient_id'):
        df_feat.append(Counter(df_part.procedure_code))
        patient_list.append(patient_id)
    X = DV(sparse=True).fit_transform(df_feat)
    print(X.shape)

    df_feat_order = pd.DataFrame({
        'patient_id': patient_list,
        'feat_idx': list(range(len(patient_list))),
    })
    df_feat_order = df_feat_order.merge(df_base, how='left', on='patient_id')
    print(df_feat_order.head())

    # Re-ordering
    print("re-ordering")
    sz = len(df_feat_order)
    X_ordered = ss.lil_matrix((len(df_base), X.shape[1]))
    for idx, row in df_feat_order.iterrows():
        if idx % 1000 == 0:
            print(str(datetime.datetime.now()), idx, sz)
        X_ordered[row['idx'], :] = X[row['feat_idx'], :]
    print("re-ordering ... done")

    X_ordered = ss.csr_matrix(X_ordered)
    print(X_ordered.shape)
    return X_ordered


"""
g.FILE_SURG
"""


def surg_unuq_claim_id():
    df_base = _load_base()

    df_su = pd.read_csv(g.FILE_SURG_OUT, usecols=['patient_id', 'claim_id'])
    df_su['num_record'] = 1
    df_su = df_su.groupby('patient_id').agg({
        'claim_id': lambda x: len(x.unique()),
    }).reset_index()
    df_su.columns = ['patient_id', 'uniq_claim_id']

    df_base = df_base.merge(df_su, how='left', on='patient_id')
    return np.array(df_base.uniq_claim_id.fillna(0)).reshape((-1, 1))


"""
g.FILE_TRAIN, g.FILE_TEST
"""


def patient_age_group():
    df = pd.concat([
        pd.read_csv(g.FILE_TRAIN_OUT, usecols=['patient_age_group']),
        pd.read_csv(g.FILE_TEST_OUT, usecols=['patient_age_group']),
    ])
    return LE().fit_transform(df.patient_age_group).reshape((-1, 1))


def patient_state():
    df = pd.concat([
        pd.read_csv(g.FILE_TRAIN_OUT, usecols=['patient_state']),
        pd.read_csv(g.FILE_TEST_OUT, usecols=['patient_state']),
    ])
    return LE().fit_transform(df.patient_state).reshape((-1, 1))


def patient_ethinicity():
    df = pd.concat([
        pd.read_csv(g.FILE_TRAIN_OUT, usecols=['ethinicity']),
        pd.read_csv(g.FILE_TEST_OUT, usecols=['ethinicity']),
    ])
    return LE().fit_transform(df.ethinicity).reshape((-1, 1))


def patient_household_income():
    df = pd.concat([
        pd.read_csv(g.FILE_TRAIN_OUT, usecols=['household_income']),
        pd.read_csv(g.FILE_TEST_OUT, usecols=['household_income']),
    ])
    return LE().fit_transform(df.household_income).reshape((-1, 1))


def patient_education_level():
    df = pd.concat([
        pd.read_csv(g.FILE_TRAIN_OUT, usecols=['education_level']),
        pd.read_csv(g.FILE_TEST_OUT, usecols=['education_level']),
    ])
    return LE().fit_transform(df.education_level).reshape((-1, 1))


def y():
    df = pd.read_csv(g.FILE_TRAIN_OUT, usecols=['is_screener'])
    return np.array(df['is_screener'])


if __name__ == '__main__':
    # Generating dense features
    dense_list = [
        "feat.y.blp",
        "feat.patient_age_group.blp",
        "feat.patient_state.blp",
        "feat.patient_ethinicity.blp",
        "feat.patient_household_income.blp",
        "feat.patient_education_level.blp",
        "feat.pres_days_supply_stat.blp",
        "feat.pa_most_recent_year.blp",
        "feat.proc_exists_attending_practitioner_id.blp",
        "feat.proc_exists_referring_practitioner_id.blp",
        "feat.proc_exists_operating_practitioner_id.blp",
        "feat.surg_unuq_claim_id.blp",
        "feat.proc_num_record.blp",
        "feat.proc_uniq_claim_id.blp",
        "feat.pres_unique_claim_id.blp",
        "feat.diag_num_record.blp",
        "feat.diag_uniq_claim_id.blp",
        "feat.diag_diagnosis_date_unique_count.blp",
    ]
    for dense_name in dense_list:
        print("Generating {} ...".format(dense_name))
        function_name = dense_name.split('.')[1]
        caller = getattr(sys.modules[__name__], function_name)
        bp.pack_ndarray_file(caller(), dense_name)

    # Generating sparse features
    sparse_list = [
        "feat.pres_drug_dv.h5",
        "feat.diag_physician_specialty_description_dv.h5",
        "feat.diag_physician_state_dv.h5",
        "feat.diag_physician_cbsa_dv.h5",
        "feat.pres_drug_bb_usc_code_dv.h5",
        "feat.pres_drug_generic_name_dv.h5",
        "feat.pres_drug_manufacturer_dv.h5",
        "feat.proc_physician_specialty_description_dv.h5",
        "feat.diag_diagnosis_code_dv.h5",
        "feat.diag_diagnosis_description_1gram_dv.h5",
        "feat.diag_primary_physician_role_dv.h5",
        "feat.proc_procedure_code_dv.h5",
        "feat.proc_place_of_service_dv.h5",
        "feat.proc_plan_type_dv.h5",
        "feat.proc_physician_state_dv.h5",
        "feat.proc_procedure_description_1gram_dv.h5",
    ]
    for sparse_name in sparse_list:
        print("Generating {} ...".format(sparse_name))
        function_name = sparse_name.split('.')[1]
        caller = getattr(sys.modules[__name__], function_name)
        _savemat(caller(), sparse_name)
