# -*- coding: utf-8 -*-
"""
Kohei's v52 model
"""
import os
import argparse
import datetime
import operator
import json

from sklearn.metrics import roc_auc_score
import scipy.sparse as ss
import scipy.io as sio
import numpy as np
import xgboost
import bloscpack as bp
import tables as tb
import pandas as pd

import genentech_path as g


PARAMS = {
    'objective': 'binary:logistic',
    'learning_rate': 0.04,
    'subsample': 0.9,
    'colsample_bytree': 0.6,
    'max_depth': 14,
    'n_estimators': 12000,
    'seed': 50,
}
FEATURE_LIST = [
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
SPARSE_LIST = [
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


def strfdelta(tdelta, fmt):
    d = {"days": tdelta.days}
    d["hours"], rem = divmod(tdelta.seconds, 3600)
    d["minutes"], d["seconds"] = divmod(rem, 60)
    return fmt.format(**d)


def savemat(X, filepath):
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


def loadmat(filepath):
    with tb.open_file(filepath, 'r') as f:
        l = f.root.shape[1]
        n = f.root.shape[0]
        X = ss.csr_matrix(
            (f.root.data[:],
             f.root.indices[:],
             f.root.indptr[:]),
            shape=(l, n))
        X = ss.csc_matrix(X.T)
    return X


def load_Xy():
    X_np = np.hstack([bp.unpack_ndarray_file(fn) for fn in FEATURE_LIST])
    X_ss = None
    for fn in SPARSE_LIST:
        print("Loading {}".format(fn))
        if X_ss is None:
            X_ss = loadmat(fn)
            X_ss = ss.csr_matrix(X_ss)
        else:
            X_ss = ss.hstack([X_ss, loadmat(fn)])
            X_ss = ss.csr_matrix(X_ss)
    X_np = ss.csr_matrix(X_np)
    print("Concatinate X_ss and X_np")
    X = ss.hstack([X_np, X_ss])
    print("Convert X into CSC matrix")
    X = ss.csc_matrix(X)
    print("done")
    del X_np, X_ss
    y_train = bp.unpack_ndarray_file("feat.y.blp")
    return X, y_train


def main():
    PARAMS['n_estimators'] = 12000

    X, y_train = load_Xy()
    X_train, X_test = X[:len(y_train), :], X[len(y_train):, :]
    clf = xgboost.XGBClassifier(**PARAMS)
    clf.fit(X_train, y_train, eval_set=[
        (X_train, y_train),
    ], eval_metric='auc', early_stopping_rounds=None)
    y_pred = clf.predict_proba(X_test)

    pd.DataFrame({'y': y_pred[:, 1].tolist()})[['y']].to_csv(
        g.TEST_PRED_FILE, index=False, header=False)


def cv(spec_fold=None):
    X, y = load_Xy()
    X_test_full = X[len(y):, :]
    X = X[:len(y), :]

    df_base = pd.read_csv(
        g.FILE_TRAIN_OUT, usecols=['patient_id'])
    df_fold = pd.read_csv(
        g.FILE_DMITRY_CV_IDX, usecols=['patient_id', 'cv_index'])
    df_fold = df_base.merge(df_fold, how='left', on='patient_id').rename(
        columns={'cv_index': 'fold_id'})

    cv_scores = []
    meta_info = {'validation': {}}

    df_pred_array = []
    id_array = np.arange(len(y))
    # Modeling
    for i in range(1, 4):
        if spec_fold is not None and spec_fold != i:
            continue

        idx_train, idx_test = (
            (df_fold.fold_id != i).ravel(),
            (df_fold.fold_id == i).ravel())

        X_train, X_test = X[idx_train, :], X[idx_test, :]
        y_train, y_test = y[idx_train], y[idx_test]

        # training
        start_time = datetime.datetime.now()
        clf = xgboost.XGBClassifier(**PARAMS)
        clf.fit(X_train, y_train, eval_set=[
            (X_train, y_train),
            (X_test, y_test),
        ], eval_metric='auc', early_stopping_rounds=None)
        # ], eval_metric='auc', early_stopping_rounds=300)

        # predict valtest
        y_pred = clf.predict_proba(X_test)
        score = roc_auc_score(y_test, y_pred[:, 1])
        end_time = datetime.datetime.now()
        elapsed_time = strfdelta(
            (end_time - start_time),
            "{days} days {hours}:{minutes}:{seconds}")
        print(elapsed_time)

        meta_info['validation']['fold{}'.format(i)] = {
            # 'clf_best_score': clf.best_score,
            # 'clf_best_iteration': clf.best_iteration,
            'evaluation_result': score,
            'elapsed_time': elapsed_time,
        }

        cv_scores.append(score)
        print("Fold{}: {:.6f}".format(i, score))
        df_pred = pd.DataFrame({
            'id': id_array[idx_test].tolist(),
        })
        df_pred['y'] = y_pred[:, 1].tolist()
        df_pred_array.append(df_pred)

    if spec_fold is None:
        print(cv_scores)
        mean_cv_score = np.mean(cv_scores)

        meta_info['validation']['mean_cv_scores'] = mean_cv_score
        print("CV Fold: {:.6f}".format(mean_cv_score))
        pd.concat(df_pred_array).sort_values('id')[
            'y'
        ].to_csv(g.CV_PRED_FILE, index=False, header=False)

        with open(g.MODEL_META_FILE, 'w') as f:
            json.dump(meta_info, f, indent=4)
    else:
        pd.concat(df_pred_array).sort_values('id')[
            ['id', 'y']
        ].to_csv(g.CV_PRED_FOLD_FILE.format(spec_fold),
                 index=False,
                 header=False)

        with open(g.MODEL_META_FOLD_FILE.format(spec_fold), 'w') as f:
            json.dump(meta_info, f, indent=4)

        # Save pred for test by single fold
        ytest_pred = clf.predict_proba(X_test_full)
        pd.DataFrame({'pred': ytest_pred[:, 1].tolist()})[[
            'pred'
        ]].to_csv(
            g.TEST_PRED_FOLD_FILE.format(spec_fold),
            index=False,
            header=False,
        )


def merge_cv_result():
    meta_files = [g.MODEL_META_FOLD_FILE.format(fold) for fold in range(1, 4)]
    pred_files = [g.CV_PRED_FOLD_FILE.format(fold) for fold in range(1, 4)]
    meta_info = {'validation': {}}
    pred = []

    # filecheck
    for fn in meta_files + pred_files:
        if not os.path.exists(fn):
            raise RuntimeError("No such file: {}".format(fn))

    # merge
    for fn in meta_files:
        with open(fn, 'r') as f:
            json_obj = json.load(f)
            meta_info['validation'].update(json_obj['validation'])
    # dump
    with open(g.MODEL_META_FILE, 'w') as f:
        json.dump(meta_info, f, indent=4)

    # merge
    for fn in pred_files:
        df = pd.read_csv(fn, names=['id', 'y'])
        pred.append(df)
    # dump
    df_train = pd.read_csv(
        g.FILE_TRAIN_OUT,
        usecols=['patient_id'])
    df_train['v52prediction'] = pd.concat(pred).sort_values('id').y.tolist()
    df_train.to_csv(g.CV_PRED_FILE, index=False, header=True)

    # test
    df_test = pd.read_csv(
        g.FILE_TEST_OUT,
        usecols=['patient_id'])
    X_test = np.hstack([
        np.array(pd.read_csv(
            g.TEST_PRED_FOLD_FILE.format(fold_id),
            names=['pred']))
        for fold_id in range(1, 4)]).mean(axis=1)
    pd.DataFrame({
        'v52prediction': X_test.tolist(),
        'patient_id': df_test.patient_id.tolist(),
    }).to_csv(g.TEST_PRED_FILE, index=False, header=True)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--validate', '-v', default=False, action='store_true')
    p.add_argument('--merge', '-m', default=False, action='store_true')
    p.add_argument('--fold', '-f', default=None, type=int)
    p.add_argument('--test', '-t', default=False, action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    arg = parse()
    if arg.validate and arg.fold is None:
        cv()
    elif arg.validate and arg.fold is not None:
        cv(spec_fold=arg.fold)
    elif arg.merge:
        merge_cv_result()
    elif arg.test:
        main()
