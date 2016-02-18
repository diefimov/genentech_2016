from datetime import datetime
import logging
import sys
import pickle
import gzip
import os
import psycopg2

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from multiprocessing import cpu_count
import sklearn.metrics as metrics
import collections
import glob

NA_VAL = -100000
ID_COL = 'patient_id'
TARGET_COL = 'is_screener'
ROUND_PRED = 3


pd.options.display.float_format = '{:.6f}'.format
pd.set_option('max_columns', 100)
pd.set_option('max_rows', 100)

__GLOBAL_VARS = {}
__INPUT_DIR = '../data/input/'
__OUTPUT_DIR = '../data/output-py/'
__OUTPUT_RGF_DIR = '../data/output-rgf/'
__TEAM_DIR = '../data/team/'
__SUBMISSION_DIR = '../data/submission/'
__LOG_DIR = '../data/log/'
__LOG_FORMAT = "[%(asctime)s %(name)s %(levelname)-s] %(message)s"
__LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
__LOG_CONFIGURED = False


class StreamToLogger(object):
    def __init__(self, logger, log_level, stream=None):
        self.logger = logger
        self.log_level = log_level
        self.stream = stream

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())
        if self.stream is not None:
            self.stream.write(buf)
            self.stream.flush()

    def flush(self):
        if self.stream is not None:
            self.stream.flush()

    def close(self):
        pass


def __get_process_log_name():
    return 'LOG_HANDLER:' + str(os.getpid())


def __add_proc_log_to_global_vars(proc_log, replace_sys_streams):
    proc_log_name = __get_process_log_name()
    __GLOBAL_VARS[proc_log_name] = proc_log
    __GLOBAL_VARS[proc_log_name + ':sys.stdout'] = sys.stdout
    __GLOBAL_VARS[proc_log_name + ':sys.stderr'] = sys.stderr
    if replace_sys_streams:
        sys.stdout = StreamToLogger(proc_log, logging.INFO)
        sys.stderr = StreamToLogger(proc_log, logging.ERROR)


def remove_proc_log():
    proc_log_name = __get_process_log_name()
    if proc_log_name in __GLOBAL_VARS:
        del __GLOBAL_VARS[proc_log_name]
        sys.stdout = __GLOBAL_VARS[proc_log_name + ':sys.stdout']
        del __GLOBAL_VARS[proc_log_name + ':sys.stdout']
        sys.stderr = __GLOBAL_VARS[proc_log_name + ':sys.stderr']
        del __GLOBAL_VARS[proc_log_name + ':sys.stderr']
        __add_proc_log_to_global_vars(proc_log=_ROOT_LOGGER, replace_sys_streams=False)


def get_log():
    if __get_process_log_name() in __GLOBAL_VARS:
        log = __GLOBAL_VARS[__get_process_log_name()]
    else:
        log = _ROOT_LOGGER
    assert isinstance(log, logging.Logger)
    return log


def config_file_log(fname, mode='w'):
    fullname = __LOG_DIR + fname
    if not fullname.endswith('.log'):
        fullname += '.log'
    remove_proc_log()

    fdir = os.path.dirname(fullname)
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    proc_log = logging.getLogger(fname)
    proc_log.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(fullname, mode=mode)
    file_handler.setFormatter(fmt=logging.Formatter(fmt=__LOG_FORMAT, datefmt=__LOG_DATE_FORMAT))
    file_handler.setLevel(logging.INFO)
    proc_log.addHandler(file_handler)
    __add_proc_log_to_global_vars(proc_log=proc_log,
                                  replace_sys_streams=True)


if not __LOG_CONFIGURED:
    __LOG_CONFIGURED = True

    logging.basicConfig(
        level=logging.DEBUG,
        format=__LOG_FORMAT,
        datefmt=__LOG_DATE_FORMAT, stream=sys.stdout)
    _ROOT_LOGGER = logging.getLogger()
    __add_proc_log_to_global_vars(proc_log=_ROOT_LOGGER, replace_sys_streams=False)


def reload_module(module):
    import importlib
    importlib.reload(module)


def tic():
    __GLOBAL_VARS['.tic.timer'] = datetime.now()


def toc():
    if '.tic.timer' in __GLOBAL_VARS:
        get_log().info('Elapsed time: %s', str(datetime.now() - __GLOBAL_VARS['.tic.timer']))


def get_input_path(fname):
    return __INPUT_DIR + fname


def get_output_path(fname):
    return __OUTPUT_DIR + fname


def get_team_path(fname):
    return __TEAM_DIR + fname


def get_output_rgf_path(fname):
    return __OUTPUT_RGF_DIR + fname


def save_data(**kwargs):
    for name in kwargs:
        if isinstance(kwargs[name], pd.DataFrame) or isinstance(kwargs[name], pd.Series):
            with pd.HDFStore(get_output_path(name + '.h5'), mode='w', complevel=9, complib='blosc') as store:
                store[name] = kwargs[name]
        else:
            with gzip.open(get_output_path(name + '.pklz'), 'wb') as out_stream:
                pickle.dump(kwargs[name], out_stream)


def copy_saved_data(**kwargs):
    for name in kwargs:
        value = load_data(kwargs[name])
        save_data(**{name: value})


def load_data(name):
    h5_path = get_output_path(name + '.h5')
    if os.path.exists(h5_path):
        with pd.HDFStore(h5_path, mode='r') as store:
            return store[name]
    else:
        with gzip.open(get_output_path(name + '.pklz'), 'rb') as out_stream:
            return pickle.load(out_stream)


def load_df_suffix(name, suffix):
    df = load_data(name + suffix)
    df.columns += suffix
    return df


# noinspection PyUnresolvedReferences
def get_kfold_ids(k_list=None):

    data = load_data('data_all_out')
    kfold = np.sort(data['cv_index'].unique())
    yield_k = lambda x: k_list is None or x in k_list

    # noinspection PyTypeChecker
    for k in kfold[kfold > 0]:
        if yield_k(k):
            # noinspection PyTypeChecker
            # tr_ix = data.index[
            #     np.logical_and(
            #         np.logical_and(data['cv_index'] != k, data['cv_index'] > 0),
            #         ~data['exclude'])
            # ]
            tr_ix = data.index[
                    np.logical_and(data['cv_index'] != k, data['cv_index'] > 0)
            ]
            val_ix = data.index[np.logical_and(data['cv_index'] == k, ~data['exclude'])]
            yield k, tr_ix, val_ix

    if yield_k(0):
        # tr_ix = data.index[np.logical_and(data['cv_index'] > 0, ~data['exclude'])]
        tr_ix = data.index[data['cv_index'] > 0]
        yield 0, tr_ix, data.index[data['cv_index'] == 0]


def auc(actual, pred):
    return metrics.roc_auc_score(y_true=actual, y_score=pred)


def add_inc_pred(data_pred_df, pred, n, pred_col='Pred'):
    if n == 0:
        data_pred_df[pred_col] = 0
    data_pred_df[pred_col] *= n
    data_pred_df[pred_col] += pred
    data_pred_df[pred_col] /= n + 1
    # return data_pred_df


def load_if_str(data):
    if isinstance(data, str):
        data = load_data(name=data)
    return data


def get_prediction_summary(data_pred_df, pred_cols=None, do_print=True, transpose=True, percentiles=None):
    data_pred_df = load_if_str(data_pred_df)

    if pred_cols is None:
        pred_cols = data_pred_df.columns

    if percentiles is None:
        percentiles = []

    pred_summary = data_pred_df.describe(percentiles=percentiles)

    data_all_out = load_data('data_all_out')
    data_all_out = data_all_out[pd.notnull(data_all_out[TARGET_COL])]
    data_pred_df_actual = pd.merge(left=data_all_out, right=data_pred_df, left_index=True, right_index=True)
    if len(data_pred_df_actual) > 0:
        score_ix = len(pred_summary)
        for pred_col in pred_cols:
            try:
                pred_sel = pd.notnull(data_pred_df_actual[pred_col])
                score = auc(actual=data_pred_df_actual.ix[pred_sel, TARGET_COL],
                            pred=data_pred_df_actual.ix[pred_sel, pred_col].round(decimals=ROUND_PRED))
            except ValueError:
                score = np.nan
            pred_summary.loc[score_ix, pred_col] = score
        pred_summary.index = list(pred_summary.index[:-1]) + ['auc']
    if transpose:
        pred_summary = pred_summary.transpose()
    if do_print:
        get_log().info('\nPrediction summary:\n%s' % pred_summary.to_string())
    else:
        return pred_summary


def unlist_dataframe(df_list):
    if isinstance(df_list, pd.DataFrame):
        df_all = df_list.copy()
    else:
        df_all = None
        for df in df_list:
            if df_all is None:
                df_all = df.copy()
            else:
                df_all = df_all.append(df)
    df_all.sort_index(inplace=True)
    return df_all


def write_submission(name, data_pred_df=None, pred_col='Pred', suffix=''):
    if data_pred_df is None:
        data_pred_df = load_data(name)

    if not os.path.exists(__SUBMISSION_DIR):
        os.makedirs(__SUBMISSION_DIR)

    if pred_col not in data_pred_df.columns:
        pred_col = get_default_input_col(data_pred_df)[0]

    data_all_out = load_data('data_all_out')
    data_all_out = data_all_out[pd.isnull(data_all_out[TARGET_COL])]
    data_sub_df = data_pred_df.ix[data_all_out.index][[pred_col]]

    pred_col = 'predict_screener'
    data_sub_df.columns = [pred_col]
    data_sub_df[pred_col] = data_sub_df[pred_col].round(decimals=ROUND_PRED)
    with gzip.open(__SUBMISSION_DIR + name + suffix + '.csv.gz', 'wt') as fout:
        data_sub_df.to_csv(path_or_buf=fout)

    # return data_sub_df


def write_team_csv(name, data_pred_df=None):
    if data_pred_df is None:
        data_pred_df = load_data(name)

    data_pred_df = load_if_str(data_pred_df)

    if not os.path.exists(__TEAM_DIR):
        os.makedirs(__TEAM_DIR)

    # with gzip.open(__TEAM_DIR + name + '.csv.gz', 'wt') as fout:
    #     data_pred_df.to_csv(path_or_buf=fout, index=True, float_format='%1.3f')
    data_pred_df.to_csv(path_or_buf=__TEAM_DIR + name + '.csv.bz2', index=True, float_format='%1.3f', compression='bz2')

    # return data_sub_df


def get_as_list(input_col):
    if isinstance(input_col, str) or not isinstance(input_col, list):
        input_col = [input_col]
    return input_col


def get_last_value_series(x):
    return x.fillna(method='ffill')
    # r = x.copy()
    # last_val = np.nan
    # for iloc, val in enumerate(x):
    #     r.iloc[iloc] = last_val
    #     if not np.isnan(val):
    #         last_val = val
    # return r


def get_next_value_series(x):
    return x.fillna(method='bfill')
    # return get_last_value_series(x.iloc[::-1]).iloc[::-1]


def get_aggr(data, group_col, input_col, fun='size'):
    group_col = get_as_list(group_col)
    input_col = get_as_list(input_col)
    data_group = data[group_col + input_col].groupby(group_col)
    data_counts = data_group.agg(fun)
    if len(group_col) == 1:
        agg_index = pd.Index(data.ix[:, group_col].values.reshape(-1), dtype='object')
    else:
        agg_index = pd.Index(data.ix[:, group_col].values, dtype='object')
    data_counts = data_counts.ix[agg_index]
    data_counts.fillna(0)
    return data_counts


def get_ordinal_recode(data, input_col=None, enc_all=False):
    if input_col is None:
        input_col = get_default_input_col(data)
    for col_nam in input_col:
        # print(col_nam)
        if data[col_nam].dtype == object or enc_all:
            data[col_nam] = preprocessing.LabelEncoder().fit_transform(data[col_nam].fillna('na_val').values)


def get_random_ordinal_recode(data, input_col=None, enc_all=False, random_state=8393478):
    if input_col is None:
        input_col = get_default_input_col(data)
    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    for col_nam in input_col:
        if data[col_nam].dtype == object or enc_all:
            unique_val = np.unique(data[col_nam].values)
            recode_df = pd.DataFrame({'Val': random_state.permutation(len(unique_val))}, index=unique_val)
            data[col_nam] = recode_df.ix[data[col_nam].values, 'Val'].values


def shuff_data(data, input_col=None, random_state=None):
    if input_col is None:
        input_col = get_default_input_col(data)

    if random_state is not None:
        np.random.seed(random_state)

    for col_nam in input_col:
        data_col_unique = data[col_nam].unique()
        data_col_shuff = np.copy(data_col_unique)
        np.random.shuffle(data_col_shuff)
        data_map = pd.DataFrame({'NewVal': data_col_shuff}, index=data_col_unique)
        data[col_nam] = data_map.ix[data[col_nam], 'NewVal'].values
    return data


def get_xy_data_split(data, ids, input_col=None, y_transf=None):
    if input_col is None:
        input_col = get_default_input_col(data)

    elif isinstance(input_col, str):
        input_col = load_data(input_col)

    data_x = data.ix[ids, input_col].values.astype(float)
    data_y = data.ix[ids, TARGET_COL].values.astype(float)
    if y_transf is not None:
        data_y = y_transf(data_y)
    return data_x, data_y


def get_default_input_col(data):
    col_blacklist = []
    return [nam for nam in data.columns if nam not in col_blacklist + [TARGET_COL]]


def get_xgb_data_split(**kwargs):
    data_x, data_y = get_xy_data_split(**kwargs)
    return get_xgb_data_matrix(data_x=data_x, data_y=data_y)


def get_xgb_data_matrix(data_x, data_y, missing=NA_VAL):
    # if isinstance(data_x, pd.DataFrame):
    #     data_x.columns = replace_nonalpha_lst(data_x.columns)
    if data_y is None or np.any(pd.isnull(data_y)):
        xg_data = xgb.DMatrix(data_x, missing=missing)
    else:
        xg_data = xgb.DMatrix(data_x, label=data_y, missing=missing)
    return xg_data


def get_xgb_eval_transf(transf):
    return lambda pred, data_xg: ('auc', auc(actual=transf.pred(data_xg.get_label()), pred=transf.pred(pred)))


def get_xgb_eval(pred, data_xg):
    return 'auc', auc(actual=data_xg.get_label(), pred=pred)


def get_xgb_score(model, input_col=None, do_print=True):
    model_score = model.get_fscore()
    model_score_fnam = [None] * len(model_score)
    model_score_fval = [None] * len(model_score)
    for ix_score, score_item in enumerate(model_score.items()):
        score_fnam, score_val = score_item
        if input_col is None or isinstance(score_fnam, str):
            model_score_fnam[ix_score] = score_fnam
        else:
            ix_col = int(score_fnam[1:])
            model_score_fnam[ix_score] = input_col[ix_col]
        model_score_fval[ix_score] = score_val
    model_score_fval = np.divide(model_score_fval, sum(model_score_fval))
    model_score_df = pd.DataFrame({'Feature': model_score_fnam, 'Score': model_score_fval})
    model_score_df.set_index('Feature', inplace=True)
    model_score_df.sort_values(by=['Score'], ascending=False, inplace=True)
    model_score_df['Order'] = range(1, len(model_score_fval) + 1)
    if do_print:
        get_log().info('\nModel features score:\n%s' % model_score_df.to_string())

    # return model_score


def get_lr_score(model, input_col, do_print=True):
    model_score_df = pd.DataFrame({'Coef': model.coef_}, index=pd.Index(input_col, name='Col'))
    model_score_df['Score'] = np.abs(model_score_df['Coef'])
    model_score_df['Score'] /= model_score_df['Score'].sum()
    model_score_df.sort_values(by=['Score'], ascending=False, inplace=True)
    model_score_df['Order'] = range(1, model_score_df.shape[0] + 1)

    if do_print:
        get_log().info('\nModel features score:\n%s' % model_score_df.to_string())

    # return model_score_df


def get_rf_score(model, input_col, do_print=True):
    model_score_df = pd.DataFrame({'Score': model.feature_importances_}, index=pd.Index(input_col, name='Col'))
    # pd.set_option('max_columns', max([model_score_df.shape[1], 100]))
    # pd.set_option('max_rows',  max([model_score_df.shape[0], 100]))
    model_score_df.sort_values(by=['Score'], ascending=False, inplace=True)
    model_score_df['Order'] = range(1, model_score_df.shape[0] + 1)

    if do_print:
        get_log().info('\nModel features score:\n%s' % model_score_df.to_string())

    return model_score_df


def get_et_score(**kwargs):
    return get_rf_score(**kwargs)


def get_data_pred_df(ids):
    return pd.DataFrame(data={'Pred': 0}, index=pd.Index(ids, name='Id'))


def get_pred_df(data_pred_names, scale=False, preffix='pred_'):
    data_pred_df = load_data('data_all_out')
    for nam in data_pred_names:
        data_pred_cur = load_data(preffix + nam)
        if scale:
            data_pred_cur['Pred'] = preprocessing.StandardScaler().fit_transform(data_pred_cur['Pred'].values)
        new_nam = nam.replace(preffix, '')
        data_pred_df.ix[data_pred_cur.index, new_nam] = data_pred_cur['Pred']

    return data_pred_df


def reescale(data, input_col=None, ignore_zeros=False):
    if isinstance(data, str):
        data = load_data(name=data)

    if input_col is None:
        input_col = get_default_input_col(data)

    for nam in input_col:
        if ignore_zeros:
            not_zero = data[nam] != 0
            data.ix[not_zero, nam] = preprocessing.StandardScaler().fit_transform(
                data.ix[not_zero, [nam]].values.astype(float))[:, 0]
        else:
            data[nam] = preprocessing.StandardScaler().fit_transform(data[[nam]].values.astype(float))[:, 0]

    # return data


def get_unique_count(data, cols=None):
    if isinstance(data, str):
        data = load_data(data)
    if cols is None:
        cols = data.columns
    return [(col_nam, data[col_nam].nunique()) for col_nam in cols]


def get_core_count():
    count = cpu_count()
    if 1 < count <= 12:
        count = int(count / 2)
    return count


def add_cols(data, data_add, cols_add=None, cols_add_suffix=''):
    data = load_if_str(data)
    data_add = load_if_str(data_add)

    if cols_add is None:
        cols_add = data_add.columns

    for col in cols_add:
        data.ix[data_add.index, col + cols_add_suffix] = data_add[col]

    return data


def identity(x):
    return x


def get_identity_transform():
    return TransformY(y=identity, pred=identity)


def get_log_transform():
    return TransformY(
        y=lambda x: np.sign(x) * np.log1p(np.abs(x)),
        pred=lambda x: np.sign(x) * np.expm1(np.abs(x))
    )


# noinspection PyTypeChecker
def get_power_transform(power):
    return TransformY(
        y=lambda x: np.sign(x) * np.power(np.abs(x), power),
        pred=lambda x: np.sign(x) * np.power(np.abs(x), 1. / power)
    )

TransformY = collections.namedtuple('TransformY', ['y', 'pred'])


def get_iloc(data, *kargs):
    ret_val = [None] * len(kargs)
    for ix, ids in enumerate(kargs):
        ret_val[ix] = np.array([data.index.get_loc(oid) for oid in ids], int)
    return tuple(ret_val)


def call_cv_train_sequential(train_func, args_iterator=None):
    if args_iterator is None:
        args_iterator = get_kfold_ids()
    return unlist_dataframe([train_func(*args) for args in args_iterator])


def call_cv_train_parallel(train_func, args_iterator=None):
    if args_iterator is None:
        args_iterator = get_kfold_ids()
    from multiprocessing import Pool
    pool = Pool(get_core_count())
    retval = unlist_dataframe(pool.starmap(train_func, args_iterator))
    pool.terminate()
    return retval


def call_func_parallel(func, args_iterator, workers=-1):
    from multiprocessing import Pool
    if workers == -1:
        workers = get_core_count()
    pool = Pool(workers)
    pool.starmap(func, args_iterator)
    pool.terminate()


def save_rgf_cfg(cfg_params, file):
    with open(file, 'w') as fout:
        for pnam, pval in cfg_params.items():
            if pval is None:
                fout.write('%s\n' % pnam)
            else:
                fout.write('%s=%s\n' % (pnam, str(pval)))


def print_stages(actual, stage_predictions):
    count = 0
    iters = []
    loss = []
    count_factor = 50
    for prediction in stage_predictions:
        count += 1
        if count in [1, 10, 50] or count % count_factor == 0:
            iters.append(count)
            loss.append(auc(actual=actual, pred=prediction))
        if count > 1000:
            count_factor = 500
        elif count > 500:
            count_factor = 200
        elif count > 250:
            count_factor = 100
    loss_df = pd.DataFrame({'Iteration': iters, 'Loss': loss})
    loss_df.rename(columns={'Loss': 'auc'}, inplace=True)
    get_log().info('\nLoss:\n%s' % loss_df.to_string())
    return loss_df


def replace_nonalpha_lst(str_list):
    import re
    str_repl = list(str_list)
    for ix, val in enumerate(str_repl):
        str_repl[ix] = re.sub(r'[\W_]+', '', val)
    return str_repl


def get_lastet_file(search_path):
    return max(glob.iglob(search_path), key=os.path.getmtime)


def to_multi_index(values):
    if values.shape[1] > 1:
        return pd.Index(values, dtype='object')
    else:
        return pd.Index(values[:, 0])


def __db_connect():
	creds = pd.read_csv(get_input_path('credentials.csv'))
	cs = str(creds.loc[creds.field=='REDSHIFT', 'value'].values)
    return psycopg2.connect(cs)


def exec_cmd_db(cmd):
    with __db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(cmd)
            conn.commit()


def load_from_db(query):
    with __db_connect() as con:
        return pd.read_sql_query(sql=query, con=con)


def copy_all_fields(to_data, from_data):
    from_data = load_if_str(from_data)
    for col_nam in from_data.columns:
        to_data[col_nam] = from_data[col_nam]


def copy_all_fields_cv(to_data, from_data, k):
    from_data = drop_other_cv_cols(data=from_data, k=k)
    copy_all_fields(to_data=to_data, from_data=from_data)


def drop_other_cv_cols(data, k):
    data = load_if_str(data)
    cols_drop = [col_nam for col_nam in data.columns if "_cv_ix_" in col_nam and ("_cv_ix_" + str(k)) not in col_nam]
    data.drop(cols_drop, axis='columns', inplace=True)
    return data


def calc_likelihood_db(table, fields, prev_max_vals=0, aggr='max', min_count=1, where1="", where2=""):

    data_all_out = load_data('data_all_out')
    global_avg = data_all_out['is_screener'].mean().round(3)

    fields = get_as_list(fields)
    fields_t4 = ", ".join(['t4.g' + str(ix + 1) for ix, f in enumerate(fields)])
    fields_t3 = ", ".join(['t3.g' + str(ix + 1) for ix, f in enumerate(fields)])
    fields_declare = ", ".join([f + " " + 'g' + str(ix + 1) for ix, f in enumerate(fields)])
    fields_join = " and ".join(['l1.g' + str(ix + 1) + " = " + 'l2.g' + str(ix + 1) for ix, f in enumerate(fields)])

    prev_vals_select = ""
    prev_vals_nth = ""
    if prev_max_vals > 0:
        prev_vals_select = "".join([
            "   {aggr}(cv1_avg_{prev}_nth) as cv1_avg_{prev}_nth, "
            "   {aggr}(cv2_avg_{prev}_nth) as cv2_avg_{prev}_nth, "
            "   {aggr}(cv3_avg_{prev}_nth) as cv3_avg_{prev}_nth, "
            "   {aggr}(cv12_avg_{prev}_nth) as cv12_avg_{prev}_nth, "
            "   {aggr}(cv13_avg_{prev}_nth) as cv13_avg_{prev}_nth, "
            "   {aggr}(cv23_avg_{prev}_nth) as cv23_avg_{prev}_nth, "
            "   {aggr}(cv123_avg_{prev}_nth) as cv123_avg_{prev}_nth, ".format(
                prev=nth + 1, aggr=aggr
            )
            for nth in range(prev_max_vals)])
        prev_sort = ''
        if aggr == 'max':
            prev_sort = 'desc'
        if aggr == 'min':
            prev_sort = 'asc'
        prev_vals_nth = "".join([
            "   nth_value(l1.cv1_avg, {prev_val}) ignore nulls over("
            "       partition by patient_id order by cv1_avg {prev_sort}"
            "       rows between unbounded preceding and unbounded following) as cv1_avg_{prev}_nth, "
            "   nth_value(l1.cv2_avg, {prev_val}) ignore nulls over("
            "       partition by patient_id order by cv2_avg {prev_sort}"
            "       rows between unbounded preceding and unbounded following) as cv2_avg_{prev}_nth, "
            "   nth_value(l1.cv3_avg, {prev_val}) ignore nulls over("
            "       partition by patient_id order by cv3_avg {prev_sort}"
            "       rows between unbounded preceding and unbounded following) as cv3_avg_{prev}_nth, "
            "   nth_value(l1.cv12_avg, {prev_val}) ignore nulls over("
            "       partition by patient_id order by cv12_avg {prev_sort}"
            "       rows between unbounded preceding and unbounded following) as cv12_avg_{prev}_nth, "
            "   nth_value(l1.cv13_avg, {prev_val}) ignore nulls over("
            "       partition by patient_id order by cv13_avg {prev_sort}"
            "       rows between unbounded preceding and unbounded following) as cv13_avg_{prev}_nth, "
            "   nth_value(l1.cv23_avg, {prev_val}) ignore nulls over("
            "       partition by patient_id order by cv23_avg {prev_sort}"
            "       rows between unbounded preceding and unbounded following) as cv23_avg_{prev}_nth, "
            "   nth_value(l1.cv123_avg, {prev_val}) ignore nulls over("
            "       partition by patient_id order by cv123_avg {prev_sort}"
            "       rows between unbounded preceding and unbounded following) as cv123_avg_{prev}_nth, ".format(
                prev_val=nth + 2, prev=nth + 1, prev_sort=prev_sort
            )
            for nth in range(prev_max_vals)])

    min_count_where = "" if min_count <= 1 else (" where (t4.cv1_cnt + t4.cv2_cnt + t4.cv3_cnt) >= %d " % min_count)
    sql = "select " \
          "  patient_id, " \
          "  {prev_max_vals_select} " \
          "  {aggr}(cv1_avg) as cv1_avg, " \
          "  {aggr}(cv2_avg) as cv2_avg, " \
          "  {aggr}(cv3_avg) as cv3_avg, " \
          "  {aggr}(cv12_avg) as cv12_avg, " \
          "  {aggr}(cv13_avg) as cv13_avg, " \
          "  {aggr}(cv23_avg) as cv23_avg, " \
          "  {aggr}(cv123_avg) as cv123_avg " \
          "from " \
          "(select " \
          "  {prev_max_vals_nth} " \
          "* " \
          "from  " \
          "	(select  " \
          "	  {fields_t4}, " \
          "	  (t4.cv1_pos + t4.g_smooth*t4.g_avg)/(t4.cv1_cnt + t4.g_smooth) as cv1_avg, " \
          "	  (t4.cv2_pos + t4.g_smooth*t4.g_avg)/(t4.cv2_cnt + t4.g_smooth) as cv2_avg, " \
          "	  (t4.cv3_pos + t4.g_smooth*t4.g_avg)/(t4.cv3_cnt + t4.g_smooth) as cv3_avg, " \
          "	  (t4.cv1_pos + t4.cv2_pos + t4.g_smooth*t4.g_avg)/(t4.cv1_cnt + t4.cv2_cnt + t4.g_smooth) as cv12_avg, " \
          "	  (t4.cv1_pos + t4.cv3_pos + t4.g_smooth*t4.g_avg)/(t4.cv1_cnt + t4.cv3_cnt + t4.g_smooth) as cv13_avg, " \
          "	  (t4.cv2_pos + t4.cv3_pos + t4.g_smooth*t4.g_avg)/(t4.cv2_cnt + t4.cv3_cnt + t4.g_smooth) as cv23_avg, " \
          "	  (t4.cv1_pos + t4.cv2_pos + t4.cv3_pos + t4.g_smooth*t4.g_avg) / " \
          "     (t4.cv1_cnt + t4.cv2_cnt + t4.cv3_cnt + t4.g_smooth) as cv123_avg " \
          "	from ( " \
          "	  select  " \
          "  		{fields_t3},  " \
          "  		sum(cast(t3.cv_index = 1 as integer)) cv1_cnt,  " \
          "  		sum(cast(t3.cv_index = 1 as integer)*cast(is_screener as float)) as cv1_pos, " \
          "  		sum(cast(t3.cv_index = 2 as integer)) cv2_cnt,  " \
          "  		sum(cast(t3.cv_index = 2 as integer)*cast(is_screener as float)) as cv2_pos, " \
          "  		sum(cast(t3.cv_index = 3 as integer)) cv3_cnt,  " \
          "  		sum(cast(t3.cv_index = 3 as integer)*cast(is_screener as float)) as cv3_pos, " \
          "  		{global_avg} as g_avg, " \
          "  		30 as g_smooth " \
          "	  from " \
          "		((select patient_id, {fields_declare} from {table} {where1}) t1 " \
          "		 inner join " \
          "		 (select patient_id, is_screener, cv_index from train_cv_indices where not is_screener is null) t2 " \
          "		 on t1.patient_id = t2.patient_id) t3 " \
          "	  group by {fields_t3}) t4 {min_count_where}) l1 " \
          "	  inner join (select distinct patient_id, {fields_declare} from {table} {where2}) l2 on {fields_join} ) " \
          "group by patient_id ".format(table=table, fields_t4=fields_t4, fields_t3=fields_t3,
                                        fields_declare=fields_declare, fields_join=fields_join,
                                        prev_max_vals_select=prev_vals_select,
                                        prev_max_vals_nth=prev_vals_nth,
                                        global_avg=str(global_avg),
                                        aggr=aggr, min_count_where=min_count_where,
                                        where1=where1, where2=where2)

    # get_log().info('\n\n' + sql + '\n\n')

    data_likelihood = load_from_db(sql)
    data_likelihood.set_index('patient_id', inplace=True)
    data_likelihood = data_likelihood.ix[data_all_out.index, :].copy()
    data_likelihood.fillna(0.0, inplace=True)

    # save_data(data_likelihood_last=data_likelihood)

    cv_ix_all = np.sort(data_all_out['cv_index'].unique())
    cv_ix_tr = cv_ix_all[cv_ix_all != 0]

    data_likelihood_ret = data_likelihood[[]].copy()

    for nth in range(prev_max_vals + 1):
        nth_suffix = '' if nth == 0 else '_%d_nth' % nth
        for k in cv_ix_all:
            col_nam = aggr + "_avg" + nth_suffix + "_" + "_".join(fields) + "_cv_ix_" + str(k)
            data_likelihood_ret[col_nam] = np.nan
            cv_ix_tr_cur = cv_ix_tr[cv_ix_tr != k]

            for k_val in cv_ix_all:
                data_likelihood_ret.ix[data_all_out['cv_index'] == k_val, col_nam] = \
                    data_likelihood['cv%s_avg' % ''.join(list(cv_ix_tr_cur[cv_ix_tr_cur != k_val].astype(str))) +
                                    nth_suffix]

    return data_likelihood_ret


def calc_max_val_value_db(table, fields, prev_max_vals=1):

    data_all_out = load_data('data_all_out')
    global_avg = data_all_out['is_screener'].mean().round(3)

    fields = get_as_list(fields)
    fields_t4 = ", ".join(['t4.g' + str(ix + 1) for ix, f in enumerate(fields)])
    fields_t3 = ", ".join(['t3.g' + str(ix + 1) for ix, f in enumerate(fields)])
    fields_declare = ", ".join([f + " " + 'g' + str(ix + 1) for ix, f in enumerate(fields)])
    fields_join = " and ".join(['l1.g' + str(ix + 1) + " = " + 'l2.g' + str(ix + 1) for ix, f in enumerate(fields)])
    fields_vals = " || ".join(['l1.g' + str(ix + 1) for ix, f in enumerate(fields)])

    prev_max_vals_select = "".join([
        "   max(cv1_avg_{prev}_nth) as cv1_avg_{prev}_nth, "
        "   max(cv2_avg_{prev}_nth) as cv2_avg_{prev}_nth, "
        "   max(cv3_avg_{prev}_nth) as cv3_avg_{prev}_nth, "
        "   max(cv12_avg_{prev}_nth) as cv12_avg_{prev}_nth, "
        "   max(cv13_avg_{prev}_nth) as cv13_avg_{prev}_nth, "
        "   max(cv23_avg_{prev}_nth) as cv23_avg_{prev}_nth, "
        "   max(cv123_avg_{prev}_nth) as cv123_avg_{prev}_nth, ".format(
            prev=nth + 1
        )
        for nth in range(prev_max_vals)])
    prev_max_vals_nth = "".join([
        "   nth_value({fields_vals}, {prev}) ignore nulls over(partition by patient_id order by cv1_avg desc"
        "       rows between unbounded preceding and unbounded following) as cv1_avg_{prev}_nth, "
        "   nth_value({fields_vals}, {prev}) ignore nulls over(partition by patient_id order by cv2_avg desc"
        "       rows between unbounded preceding and unbounded following) as cv2_avg_{prev}_nth, "
        "   nth_value({fields_vals}, {prev}) ignore nulls over(partition by patient_id order by cv3_avg desc"
        "       rows between unbounded preceding and unbounded following) as cv3_avg_{prev}_nth, "
        "   nth_value({fields_vals}, {prev}) ignore nulls over(partition by patient_id order by cv12_avg desc"
        "       rows between unbounded preceding and unbounded following) as cv12_avg_{prev}_nth, "
        "   nth_value({fields_vals}, {prev}) ignore nulls over(partition by patient_id order by cv13_avg desc"
        "       rows between unbounded preceding and unbounded following) as cv13_avg_{prev}_nth, "
        "   nth_value({fields_vals}, {prev}) ignore nulls over(partition by patient_id order by cv23_avg desc"
        "       rows between unbounded preceding and unbounded following) as cv23_avg_{prev}_nth, "
        "   nth_value({fields_vals}, {prev}) ignore nulls over(partition by patient_id order by cv123_avg desc"
        "       rows between unbounded preceding and unbounded following) as cv123_avg_{prev}_nth, ".format(
            prev=nth + 1, fields_vals=fields_vals
        )
        for nth in range(prev_max_vals)])

    sql = "select " \
          "  {prev_max_vals_select} " \
          "  patient_id " \
          "from " \
          "(select " \
          "  {prev_max_vals_nth} " \
          "* " \
          "from  " \
          "	(select  " \
          "	  {fields_t4}, " \
          "	  (t4.cv1_pos + t4.g_smooth*t4.g_avg)/(t4.cv1_cnt + t4.g_smooth) as cv1_avg, " \
          "	  (t4.cv2_pos + t4.g_smooth*t4.g_avg)/(t4.cv2_cnt + t4.g_smooth) as cv2_avg, " \
          "	  (t4.cv3_pos + t4.g_smooth*t4.g_avg)/(t4.cv3_cnt + t4.g_smooth) as cv3_avg, " \
          "	  (t4.cv1_pos + t4.cv2_pos + t4.g_smooth*t4.g_avg)/(t4.cv1_cnt + t4.cv2_cnt + t4.g_smooth) as cv12_avg, " \
          "	  (t4.cv1_pos + t4.cv3_pos + t4.g_smooth*t4.g_avg)/(t4.cv1_cnt + t4.cv3_cnt + t4.g_smooth) as cv13_avg, " \
          "	  (t4.cv2_pos + t4.cv3_pos + t4.g_smooth*t4.g_avg)/(t4.cv2_cnt + t4.cv3_cnt + t4.g_smooth) as cv23_avg, " \
          "	  (t4.cv1_pos + t4.cv2_pos + t4.cv3_pos + t4.g_smooth*t4.g_avg) / " \
          "     (t4.cv1_cnt + t4.cv2_cnt + t4.cv3_cnt + t4.g_smooth) as cv123_avg " \
          "	from ( " \
          "	  select  " \
          "  		{fields_t3},  " \
          "  		sum(cast(t3.cv_index = 1 as integer)) cv1_cnt,  " \
          "  		sum(cast(t3.cv_index = 1 as integer)*cast(is_screener as float)) as cv1_pos, " \
          "  		sum(cast(t3.cv_index = 2 as integer)) cv2_cnt,  " \
          "  		sum(cast(t3.cv_index = 2 as integer)*cast(is_screener as float)) as cv2_pos, " \
          "  		sum(cast(t3.cv_index = 3 as integer)) cv3_cnt,  " \
          "  		sum(cast(t3.cv_index = 3 as integer)*cast(is_screener as float)) as cv3_pos, " \
          "  		{global_avg} as g_avg, " \
          "  		30 as g_smooth " \
          "	  from " \
          "		((select patient_id, {fields_declare} from {table}) t1 " \
          "		 inner join " \
          "		 (select patient_id, is_screener, cv_index from train_cv_indices where not is_screener is null) t2 " \
          "		 on t1.patient_id = t2.patient_id) t3 " \
          "	  group by {fields_t3}) t4) l1 " \
          "	  inner join (select distinct patient_id, {fields_declare} from {table}) l2 on {fields_join} ) " \
          "group by patient_id ".format(table=table, fields_t4=fields_t4, fields_t3=fields_t3,
                                        fields_declare=fields_declare, fields_join=fields_join,
                                        prev_max_vals_select=prev_max_vals_select,
                                        prev_max_vals_nth=prev_max_vals_nth,
                                        global_avg=str(global_avg))

    # get_log().info('\n\n' + sql + '\n\n')

    data_max_val = load_from_db(sql)
    data_max_val.set_index('patient_id', inplace=True)
    data_max_val = data_max_val.ix[data_all_out.index, :].copy()
    data_max_val.fillna('NA_VAL', inplace=True)

    # save_data(data_max_val_last=data_max_val)

    cv_ix_all = np.sort(data_all_out['cv_index'].unique())
    cv_ix_tr = cv_ix_all[cv_ix_all != 0]

    data_max_val_ret = data_max_val[[]].copy()

    for nth in range(prev_max_vals):
        nth_suffix = '_%d_nth' % (nth + 1)
        for k in cv_ix_all:
            col_nam = "val" + nth_suffix + "_" + "_".join(fields) + "_cv_ix_" + str(k)
            data_max_val_ret[col_nam] = np.nan
            cv_ix_tr_cur = cv_ix_tr[cv_ix_tr != k]

            for k_val in cv_ix_all:
                data_max_val_ret.ix[data_all_out['cv_index'] == k_val, col_nam] = \
                    data_max_val['cv%s_avg' % ''.join(list(cv_ix_tr_cur[cv_ix_tr_cur != k_val].astype(str))) +
                                 nth_suffix]

    return data_max_val_ret


def calc_likelihood_count_db(table, fields, intervals, return_query=False):

    # table = 'diagnosis_feats'
    # fields = ['diagnosis_code']
    # intervals = np.array([0.0] + list(np.linspace(0.0, 1.0, num=21)[2:]))

    data_all_out = load_data('data_all_out')
    global_avg = data_all_out['is_screener'].mean().round(3)

    fields = get_as_list(fields)
    fields_t4 = ", ".join(['t4.g' + str(ix + 1) for ix, f in enumerate(fields)])
    fields_t3 = ", ".join(['t3.g' + str(ix + 1) for ix, f in enumerate(fields)])
    fields_declare = ", ".join([f + " " + 'g' + str(ix + 1) for ix, f in enumerate(fields)])
    fields_join = " and ".join(['l1.g' + str(ix + 1) + " = " + 'l2.g' + str(ix + 1) for ix, f in enumerate(fields)])

    fields_cnt = ""
    for ix in range(1, len(intervals)):
        fields_mask = '  sum(cast((cv{cv}_avg >{eq_low} {rng_low} and ' \
            'cv{cv}_avg <= {rng_hi}) as integer)) as cv{cv}_bin_cnt_{ix}'. \
            format(eq_low='' if ix > 1 else '=',
                   rng_low="{:1.2f}".format(intervals[ix - 1]),
                   rng_hi="{:1.2f}".format(intervals[ix]),
                   ix=ix, cv='{cv}')
        if len(fields_cnt) > 0:
            fields_cnt += ', \n'
        fields_cnt += ", \n".join([fields_mask.format(cv=cv) for cv in ['1', '2', '3', '12', '13', '23', '123']])
    # print(fields_cnt)

    sql = "select \n" \
          "  patient_id, \n" \
          "{fields_cnt} \n" \
          "from \n" \
          "(select \n" \
          "* \n" \
          "from  \n" \
          "	(select  \n" \
          "	  {fields_t4}, \n" \
          "	  (t4.cv1_pos + t4.g_smooth*t4.g_avg)/(t4.cv1_cnt + t4.g_smooth) as cv1_avg, \n" \
          "	  (t4.cv2_pos + t4.g_smooth*t4.g_avg)/(t4.cv2_cnt + t4.g_smooth) as cv2_avg, \n" \
          "	  (t4.cv3_pos + t4.g_smooth*t4.g_avg)/(t4.cv3_cnt + t4.g_smooth) as cv3_avg, \n" \
          "	  (t4.cv1_pos + t4.cv2_pos + t4.g_smooth*t4.g_avg)/(t4.cv1_cnt + t4.cv2_cnt + t4.g_smooth) \n" \
          "     as cv12_avg, \n" \
          "	  (t4.cv1_pos + t4.cv3_pos + t4.g_smooth*t4.g_avg)/(t4.cv1_cnt + t4.cv3_cnt + t4.g_smooth) \n" \
          "     as cv13_avg, \n" \
          "	  (t4.cv2_pos + t4.cv3_pos + t4.g_smooth*t4.g_avg)/(t4.cv2_cnt + t4.cv3_cnt + t4.g_smooth) \n" \
          "     as cv23_avg, \n" \
          "	  (t4.cv1_pos + t4.cv2_pos + t4.cv3_pos + t4.g_smooth*t4.g_avg) / \n" \
          "     (t4.cv1_cnt + t4.cv2_cnt + t4.cv3_cnt + t4.g_smooth) as cv123_avg \n" \
          "	from ( \n" \
          "	  select  \n" \
          "  		{fields_t3},  \n" \
          "  		sum(cast(t3.cv_index = 1 as integer)) cv1_cnt,  \n" \
          "  		sum(cast(t3.cv_index = 1 as integer)*cast(is_screener as float)) as cv1_pos, \n" \
          "  		sum(cast(t3.cv_index = 2 as integer)) cv2_cnt,  \n" \
          "  		sum(cast(t3.cv_index = 2 as integer)*cast(is_screener as float)) as cv2_pos, \n" \
          "  		sum(cast(t3.cv_index = 3 as integer)) cv3_cnt,  \n" \
          "  		sum(cast(t3.cv_index = 3 as integer)*cast(is_screener as float)) as cv3_pos, \n" \
          "  		{global_avg} as g_avg, \n" \
          "  		100 as g_smooth \n" \
          "	  from \n" \
          "		((select patient_id, {fields_declare} from {table}) t1 \n" \
          "		 inner join \n" \
          "		 (select patient_id, is_screener, cv_index from train_cv_indices where not is_screener is null) t2 \n" \
          "		 on t1.patient_id = t2.patient_id) t3 \n" \
          "	  group by {fields_t3}) t4) l1 \n" \
          "	  inner join (select distinct patient_id, {fields_declare} from {table}) l2 on {fields_join} ) \n" \
          "group by patient_id \n".format(table=table, fields_t4=fields_t4, fields_t3=fields_t3,
                                          fields_declare=fields_declare, fields_join=fields_join,
                                          global_avg=str(global_avg), fields_cnt=fields_cnt)

    if return_query:
        return sql

    data_llh_cnt = load_from_db(sql)
    data_llh_cnt.set_index('patient_id', inplace=True)
    data_llh_cnt = data_llh_cnt.ix[data_all_out.index, :].copy()
    data_llh_cnt.fillna(0.0, inplace=True)

    # save_data(data_llh_cnt_last=data_llh_cnt)

    cv_ix_all = np.sort(data_all_out['cv_index'].unique())
    cv_ix_tr = cv_ix_all[cv_ix_all != 0]

    data_llh_cnt_ret = data_llh_cnt[[]].copy()

    for ix in range(1, len(intervals)):
        for k in cv_ix_all:
            col_nam = 'bin_cnt_{ix}_{fields}_cv_ix_{k}'.format(ix=ix, fields="_".join(fields), k=k)
            data_llh_cnt_ret[col_nam] = np.nan
            cv_ix_tr_cur = cv_ix_tr[cv_ix_tr != k]

            for k_val in cv_ix_all:
                cnt_col = 'cv{cv}_bin_cnt_{ix}'.format(
                    cv=''.join(list(cv_ix_tr_cur[cv_ix_tr_cur != k_val].astype(str))),
                    ix=ix)
                data_llh_cnt_ret.ix[data_all_out['cv_index'] == k_val, col_nam] = \
                    data_llh_cnt[cnt_col]

    return data_llh_cnt_ret


def round_df(data, decimals=3):
    for col_nam in data.columns:
        if data[col_nam].dtype == float:
            data[col_nam] = data[col_nam].round(decimals=decimals)
