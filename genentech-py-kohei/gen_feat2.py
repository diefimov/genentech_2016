# -*- coding: utf-8 -*-
"""
Feature extraction (missing claims)
"""
import pandas as pd
import bloscpack as bp

import genentech_path as g


def _save_csv(arr, fmt):
    df_train = pd.read_csv(g.FILE_TRAIN_OUT)
    df_test = pd.read_csv(g.FILE_TEST_OUT)

    df_train['num_removed_claim_id'] = arr.ravel()[:len(df_train)].tolist()
    df_test['num_removed_claim_id'] = arr.ravel()[len(df_train):].tolist()

    df_train[['patient_id', 'num_removed_claim_id']].to_csv(
        fmt.format(source="train"), index=False)
    df_test[['patient_id', 'num_removed_claim_id']].to_csv(
        fmt.format(source="test"), index=False)


def num_of_removed_claim_id_from_proc2():
    # Load tables
    df_base = pd.concat([
        pd.read_csv(g.FILE_TRAIN_OUT, usecols=['patient_id']),
        pd.read_csv(g.FILE_TEST_OUT, usecols=['patient_id']),
    ])
    df_proc = pd.read_csv(
        g.FILE_PROC_OUT,
        usecols=['claim_id']).drop_duplicates()
    df_surg = pd.read_csv(
        g.FILE_SURG_OUT,
        usecols=['patient_id', 'claim_id']).drop_duplicates()

    # Case when the claim_id is included on procedure_head
    df_proc['proc_flag'] = 0

    # Fill NaN with 1 when the claim_id isn't included on procedure_head
    df_surg = df_surg.merge(df_proc, how='left', on=['claim_id']).fillna(1)

    # Count the number of claim_id which might be removed from procedure_head
    df_agg = df_surg.groupby('patient_id').agg({
        'proc_flag': 'sum'}).reset_index()

    # Merge with base for ordering patient_id
    df_base = df_base.merge(df_agg, how='left', on='patient_id')

    # Reshape 1-dim array
    return df_base.proc_flag.fillna(0).reshape((-1, 1))


if __name__ == '__main__':
    print("Generate num_of_removed_claim_id_from_proc2 ...")
    _save_csv(num_of_removed_claim_id_from_proc2(), g.FILE_MISSING_CLAIM_PROC2)
