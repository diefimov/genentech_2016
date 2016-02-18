import psycopg2
import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedKFold
import xgboost as xgb
from ml_metrics import auc
from sklearn.preprocessing import StandardScaler
import random
from scipy.stats.stats import pearsonr
import os
import glob
import itertools
import utils

def read_patients_to_exclude():
    with open('../data/input/train_patients_to_exclude.csv', 'rb') as f:
        reader = csv.reader(f)
        train_patients_to_exclude = [int(x[0]) for x in reader]

    with open('../data/input/test_patients_to_exclude.csv', 'rb') as f:
        reader = csv.reader(f)
        test_patients_to_exclude = [int(x[0]) for x in reader]

    patients_to_exclude = train_patients_to_exclude + test_patients_to_exclude
    return patients_to_exclude

def calculate_basic(df):
    df['patient_age_group'] = df['patient_age_group'].str.split("-").map(lambda x: x[0]).values
    df.loc[df['household_income']=='UNKNOWN', 'household_income'] = -999
    df.loc[df['household_income']=='<=$49K', 'household_income'] = 1
    df.loc[df['household_income']=='<$50-99K', 'household_income'] = 2
    df.loc[df['household_income']=='$100K+', 'household_income'] = 3
    df['household_income'] = df['household_income'].astype(int)
    df.loc[df['education_level']=='UNKNOWN', 'education_level'] = -999
    df.loc[df['education_level']=='HIGH SCHOOL OR LESS', 'education_level'] = 1
    df.loc[df['education_level']=='SOME COLLEGE', 'education_level'] = 2
    df.loc[df['education_level']=='ASSOCIATE DEGREE AND ABOVE', 'education_level'] = 3
    df['education_level'] = df['education_level'].astype(int)
    df['activity_type_ratio'] = df['activity_type_r_count_all'].divide(df['activity_type_a_count_all'])
    return df

def build_train_test():
    conn = utils.connect_to_database()
    target = ['is_screener']
    flist_basic = ['patient_age_group', 'patient_state', 'ethinicity',
                   'household_income', 'education_level']
    flist_pah = ['activity_type_r_count_all', 'activity_type_a_count_all', 'activity_type_count_all']

    sql_query = "SELECT t1.patient_id,t1.is_screener," + ",".join(['t1.'+x for x in flist_basic]) +\
                "," + ",".join(['t2.'+x for x in flist_pah]) +\
                " FROM patients_train t1\
                 LEFT JOIN patient_activity_feats t2\
                 ON t1.patient_id=t2.patient_id;"
    train = pd.read_sql_query(sql_query, conn)
    train.reset_index(drop=True, inplace=True)

    sql_query = "SELECT t1.patient_id," + ",".join(['t1.'+x for x in flist_basic]) +\
                "," + ",".join(['t2.'+x for x in flist_pah]) +\
                " FROM patients_test2 t1\
                 LEFT JOIN patient_activity_feats t2\
                 ON t1.patient_id=t2.patient_id;"
    test = pd.read_sql_query(sql_query, conn)
    test.reset_index(drop=True, inplace=True)

    cv_indices = pd.read_sql_query('SELECT patient_id, cv_index FROM train_cv_indices;', conn)
    train = pd.merge(train, cv_indices, on='patient_id', how='left')

    train = calculate_basic(train)
    test = calculate_basic(test)
    train, test = encode_onehot(train, test, ['patient_state', 'ethinicity'])

    print "Writing to HDF5 store..."
    store = pd.HDFStore('../data/output-py/train_test.h5')
    store.append('train', train)
    store.append('test', test)
    store.close()
    conn.close()
    return train, test


def save_ftrl_data(data_type, fnames, ftablename, test_folds, train_folds, ftrl_type, optional_date_ftrl3, optional_condition_ftrl4):
    conn = utils.connect_to_database()
    cur = conn.cursor()
    
    path = '../data/output-py/ftrl/'
    if (data_type=='val') or (data_type=='train'):
        path_part = 'train'
    else:
        path_part = 'test'

    temp_path = '../data/output-py/ftrl/temp/'
    file_name = data_type + '_ftrl_folds.csv'

    sql_query = open('genentech-sql/pattern_ftrl_' + path_part + ftrl_type + '.sql').read()
    if data_type == 'train':
        sql_query = sql_query.replace('OPTIONAL_CV_EXPRESSION', 'WHERE ' + ' OR '.join(['cv_index='+str(x) for x in train_folds]))
    if data_type == 'val':
        sql_query = sql_query.replace('OPTIONAL_CV_EXPRESSION', 'WHERE ' + ' OR '.join(['cv_index='+str(x) for x in test_folds]))
    sql_query = sql_query.replace('FEATURE_TABLE_NAME', ftablename)
    sql_query = sql_query.replace('FEATURES_LIST_COMMA_SEPARATED', ','.join(fnames))
    sql_query = sql_query.replace('T1_FEATURES_COMMA_SEPARATED', ','.join(['t1.'+x for x in fnames]))
    sql_query = sql_query.replace('OPTIONAL_DATE_FTRL3', optional_date_ftrl3)
    sql_query = sql_query.replace('OPTIONAL_CONDITION_FTRL4', optional_condition_ftrl4)

    copy_string = "unload ('" + sql_query + "') to 's3://genentech-2016/ftrl/" + file_name + "' " +\
                  "credentials " + utils.S3_CONNECTION_STRING +\
                  "delimiter ',' gzip allowoverwrite;"

    cur.execute(copy_string)
    conn.commit()

    cur.close()
    conn.close()

    os.system('aws s3 cp s3://genentech-2016/ftrl/ ' + temp_path + ' --recursive')
    os.system('aws s3 rm s3://genentech-2016/ftrl/ --recursive')
    os.system('find ' + temp_path + ' -name \*.gz -exec gunzip {} \;')
    data_parts = ' '.join(sorted(glob.glob(temp_path + '*')))
    if data_type == 'test':
        header = 'patient_id,' + ','.join(fnames) + '\n'
    else:
        header = 'patient_id,' + ','.join(fnames) + ',is_screener\n'
    with open(temp_path + "header.csv", "w") as text_file:
        text_file.write("%s" % header)
    os.system('cat ' + temp_path + 'header.csv ' + data_parts + ' > ' + path + file_name)
    os.system('rm -R ' + temp_path + '/*')

    return path + file_name

def calculate_ftrl_features(train, test, fnames, ftablename, ftrl_type='', optional_date_ftrl3='', optional_condition_ftrl4=''):
    folds = [x for x in range(1, nfold+1)]
    global_mean = np.mean(train.is_screener)
    pred_file = '../data/output-py/ftrl/pred_ftrl.csv'

    ftrl_all = pd.DataFrame()
    count = 0
    for L in range(1, len(folds)+1):
        for train_folds in itertools.combinations(folds, L):
            count = count + 1
            print train_folds
            test_folds = [x for x in folds if not x in list(train_folds)]
            if len(test_folds) == 0:
                test_folds = [0]
            print test_folds

            if False:
                store = pd.HDFStore('../data/output-py/ftrl/ftrl_feats' + str(count) + '.h5')
                ftrl_feats = store.get('ftrl_feats')
                store.close()
            else:
                train_file = save_ftrl_data('train', fnames, ftablename, test_folds, list(train_folds), ftrl_type, optional_date_ftrl3, optional_condition_ftrl4)
                if 0 in test_folds:
                    test_file = save_ftrl_data('test', fnames, ftablename, test_folds, list(train_folds), ftrl_type, optional_date_ftrl3, optional_condition_ftrl4)
                else:
                    test_file = save_ftrl_data('val', fnames, ftablename, test_folds, list(train_folds), ftrl_type, optional_date_ftrl3, optional_condition_ftrl4)

                non_factor_cols = "''"
                non_feature_cols = "''"
                text_cols = "'diagnosis_description'"

                os.system('pypy ftrl' + ftrl_type + '.py' +
                          ' --alpha ' + str(0.07) +
                          ' --beta ' + str(1.0) +
                          ' --L1 ' + str(0.01) +
                          ' --L2 ' + str(1.0) +
                          ' --epoch ' + str(1) +
                          ' --train ' + train_file +
                          ' --test ' + test_file +
                          ' --submission ' + pred_file +
                          ' --non_feature_cols ' + non_feature_cols +
                          ' --non_factor_cols ' + non_factor_cols + 
                          ' --text_cols ' + text_cols)

                ftrl_feats = pd.read_csv(pred_file)
                ftrl_feats = ftrl_feats.groupby('patient_id')['is_screener_pred'].max().reset_index()

                for x in folds:
                    if x in list(train_folds):
                        ftrl_feats['fold'+str(x)] = 1
                    else:
                        ftrl_feats['fold'+str(x)] = 0
                store = pd.HDFStore('../data/output-py/ftrl/ftrl_feats' + str(count) + '.h5')
                store.append('ftrl_feats', ftrl_feats)
                store.close()
                os.system('rm -R ' + train_file)
                os.system('rm -R ' + test_file)
                os.system('rm -R ' + pred_file)

            ftrl_all = ftrl_all.append(ftrl_feats, ignore_index=True)

            ftrl_feats = pd.merge(ftrl_feats, train[['patient_id', 'is_screener']], on='patient_id', how='inner')
            if len(ftrl_feats)>0:
                print "Pearson correlation: " + str(pearsonr(ftrl_feats.is_screener, ftrl_feats.is_screener_pred))
                print "AUC: " + str(auc(ftrl_feats.is_screener, ftrl_feats.is_screener_pred))
            del ftrl_feats

    feats_all = train[['patient_id']].append(test[['patient_id']], ignore_index=True)
    for test_fold in ([0] + folds):
        train_folds = [x for x in folds if (x != test_fold) and (x != 0)]

        if len(train_folds) == len(folds):
            pd_query = ' and '.join(['fold'+str(x)+'==1' for x in train_folds])
        else:
            pd_query = ' and '.join(['fold'+str(x)+'==1' for x in train_folds]) + ' and ' + ' and '.join(['fold'+str(x)+'==0' for x in folds if not x in train_folds])
        print pd_query

        ftrl_feats = ftrl_all.query(pd_query).copy().reset_index(drop=True)
        for x in folds:
            ftrl_feats.drop('fold'+str(x), axis=1, inplace=True)

        if test_fold == 0:
            feats_fold = test[['patient_id']].copy()
        else:
            feats_fold = train.query('cv_index==@test_fold')[['patient_id']].copy()
        feats_fold = pd.merge(feats_fold, ftrl_feats, on='patient_id', how='left')
        del ftrl_feats

        for val_fold in [x for x in folds if (x != test_fold) and (x != 0)]:
            train_folds = [x for x in folds if (x != test_fold) and (x != val_fold) and (x != 0)]
            pd_query = ' and '.join(['fold'+str(x)+'==1' for x in train_folds]) + ' and ' + ' and '.join(['fold'+str(x)+'==0' for x in folds if not x in train_folds])
            
            ftrl_feats = ftrl_all.query(pd_query).copy().reset_index(drop=True)
            for x in folds:
                ftrl_feats.drop('fold'+str(x), axis=1, inplace=True)

            feats_val_fold = train.query('cv_index==@val_fold')[['patient_id']].copy()
            feats_val_fold = pd.merge(feats_val_fold, ftrl_feats, on='patient_id', how='left')
            del ftrl_feats
            feats_fold = feats_fold.append(feats_val_fold, ignore_index=True)

        feats_fold = feats_fold.reset_index(drop=True)
        feats_fold['is_screener_pred'].fillna(global_mean, inplace=True)
        feats_fold = feats_fold.rename(columns={'is_screener_pred' : '_'.join(fnames) + '_' + ftablename + '_ftrl' + ftrl_type + '_fold_'+str(test_fold)})
        feats_all = pd.merge(feats_all, feats_fold, on='patient_id', how='left')

    print "Writing to HDF5 store..."
    store = pd.HDFStore('../data/output-py/' + '_'.join(fnames) + '_' + ftablename + '_ftrl' + ftrl_type + '.h5')
    store.append('feats_all', feats_all)
    print 'Feature ' + '_'.join(fnames) + '_' + ftablename + '_ftrl' + ftrl_type + ' is saved in file.'
    store.close()
    return '_'.join(fnames) + '_' + ftablename + '_ftrl' + ftrl_type

def calculate_likelihoods(train, test, fnames, ftablename, function_type='max', query_type='', optional_filter_feature_likeli6='', optional_filter_value_likeli6=''):
    global_mean = np.mean(train.is_screener)
    folds = [x for x in range(1, nfold+1)]

    likeli_all = pd.DataFrame()
    for L in range(1, len(folds)+1):
        for train_folds in itertools.combinations(folds, L):
            print train_folds
            sql_query = open('genentech-sql/pattern_likeli_multiple' + query_type + '.sql').read()
            sql_query = sql_query.replace('FEATURE_TABLE_NAME', ftablename)
            sql_query = sql_query.replace('GENERIC_FEATURE_NAME', '_'.join(fnames))
            sql_query = sql_query.replace('FEATURE_NAMES_COMMA_SEPARATED', ','.join(fnames))
            sql_query = sql_query.replace('T1_COMMA_SEPARATED', ','.join(['t1.'+x for x in fnames]))
            sql_query = sql_query.replace('T3_T4_CONDITION', ' AND '.join(['t3.'+x+'=t4.'+x for x in fnames]))
            sql_query = sql_query.replace('OPTIONAL_CV_EXPRESSION', 'WHERE ' + ' OR '.join(['cv_index='+str(x) for x in list(train_folds)]))
            sql_query = sql_query.replace('GROUP_FUNCTION', function_type)
            sql_query = sql_query.replace('OPTIONAL_CONDITION_LIKELI6', 'WHERE ' + optional_filter_feature_likeli6 + "='" + optional_filter_value_likeli6 + "'")
            #sql_query = sql_query.replace('OPTIONAL_CONDITION_LIKELI6', 'WHERE ' + optional_filter_feature_likeli6 + ">=" + optional_filter_value_likeli6)
            if len(list(train_folds)) == len(folds):
                choosing_patients_expression = 'patients_test2'
            else:
                choosing_patients_expression = 'train_cv_indices ' + 'WHERE ' + ' OR '.join(['cv_index='+str(x) for x in folds if not x in list(train_folds)])
            sql_query = sql_query.replace('CHOOSING_PATIENTS_EXPRESSION', choosing_patients_expression)

            conn = utils.connect_to_database()
            cur = conn.cursor()
            cur.execute(sql_query)
            if (query_type == '3') or (query_type == '4') or (query_type == '5'):
                conn.commit()
                sql_query = open('genentech-sql/pattern_likeli_multiple' + query_type + '_2.sql').read()
                sql_query = sql_query.replace('GENERIC_FEATURE_NAME', '_'.join(fnames))
                sql_query = sql_query.replace('FEATURE_TABLE_NAME', ftablename)
                cur.execute(sql_query)
                likeli = pd.DataFrame(cur.fetchall())
                likeli.columns = [x.name for x in cur.description]
                cur.execute('DROP TABLE patient_likeli_table;')
                conn.commit()
            else:
                likeli = pd.DataFrame(cur.fetchall())
                likeli.columns = [x.name for x in cur.description]

            for x in folds:
                if x in list(train_folds):
                    likeli['fold'+str(x)] = 1
                else:
                    likeli['fold'+str(x)] = 0
            cur.close()
            conn.close()
            
            likeli_all = likeli_all.append(likeli, ignore_index=True)
            col = likeli.columns[1]
            likeli = pd.merge(likeli, train[['patient_id', 'is_screener']], on='patient_id', how='inner')
            if len(likeli)>0:
                print "Pearson correlation: " + str(pearsonr(likeli.is_screener, likeli[col]))
                print "AUC: " + str(auc(likeli.is_screener, likeli[col]))
            del likeli

    feats_all = train[['patient_id']].append(test[['patient_id']], ignore_index=True)
    for test_fold in ([0] + folds):
        train_folds = [x for x in folds if (x != test_fold) and (x != 0)]

        if len(train_folds) == len(folds):
            pd_query = ' and '.join(['fold'+str(x)+'==1' for x in train_folds])
        else:
            pd_query = ' and '.join(['fold'+str(x)+'==1' for x in train_folds]) + ' and ' + ' and '.join(['fold'+str(x)+'==0' for x in folds if not x in train_folds])
        print pd_query

        likeli = likeli_all.query(pd_query).copy().reset_index(drop=True)
        for x in folds:
            likeli.drop('fold'+str(x), axis=1, inplace=True)

        if test_fold == 0:
            feats_fold = test[['patient_id']].copy()
        else:
            feats_fold = train.query('cv_index==@test_fold')[['patient_id']].copy()
        feats_fold = pd.merge(feats_fold, likeli, on='patient_id', how='left')
        del likeli

        for val_fold in [x for x in folds if (x != test_fold) and (x != 0)]:
            train_folds = [x for x in folds if (x != test_fold) and (x != val_fold) and (x != 0)]
            pd_query = ' and '.join(['fold'+str(x)+'==1' for x in train_folds]) + ' and ' + ' and '.join(['fold'+str(x)+'==0' for x in folds if not x in train_folds])
            
            likeli = likeli_all.query(pd_query).copy().reset_index(drop=True)
            for x in folds:
                likeli.drop('fold'+str(x), axis=1, inplace=True)

            feats_val_fold = train.query('cv_index==@val_fold')[['patient_id']].copy()
            feats_val_fold = pd.merge(feats_val_fold, likeli, on='patient_id', how='left')
            del likeli
            feats_fold = feats_fold.append(feats_val_fold, ignore_index=True)

        col = feats_fold.columns[1]
        feats_fold = feats_fold.reset_index(drop=True)
        feats_fold[col].fillna(global_mean, inplace=True)
        #feats_fold[fname_w_likeli].fillna(global_mean, inplace=True)
        feats_fold = feats_fold.rename(columns={col : col+'_fold_'+str(test_fold)})
        #feats_fold = feats_fold.rename(columns={fname_w_likeli : fname_w_likeli+'_fold_'+str(test_fold)})
        feats_all = pd.merge(feats_all, feats_fold, on='patient_id', how='left')

    print "Writing to HDF5 store..."
    store = pd.HDFStore('../data/output-py/' + col + '.h5')
    store.append('feats_all', feats_all)
    store.close()
    conn.close()
    print "Feature " + col + " is saved in file."
    return col

def generate_likelihood_table(likeli_table_name, fnames, ftablename, train_folds):
    sql_query = open('genentech-sql/pattern_likeli_table.sql').read()
    sql_query = sql_query.replace('LIKELI_TABLE_NAME', likeli_table_name)
    sql_query = sql_query.replace('T1_COMMA_SEPARATED', ','.join(['t1.'+x for x in fnames]))
    sql_query = sql_query.replace('FEATURE_NAMES_COMMA_SEPARATED', ','.join(fnames))
    sql_query = sql_query.replace('FEATURE_TABLE_NAME', ftablename)
    sql_query = sql_query.replace('OPTIONAL_CV_EXPRESSION', 'WHERE ' + ' OR '.join(['cv_index='+str(x) for x in list(train_folds)]))
    conn = utils.connect_to_database()
    cur = conn.cursor()
    cur.execute(sql_query)
    conn.commit()
    cur.close()
    conn.close()
    return None

def merge_likelihood_tables(fnames_list, ftablename, train_folds):
    folds = [x for x in range(1, nfold+1)]

    sql_query = open('genentech-sql/pattern_merge_likeli.sql').read()
    sql_query = sql_query.replace('FEATURE_TABLE_NAME', ftablename)
    sql_query = sql_query.replace('FEATURE_NAMES_COMMA_SEPARATED', ','.join([','.join(x) for x in fnames_list]))

    likeli_tables_for_join = ''
    
    count = 2
    for fnames in fnames_list:
         likeli_tables_for_join = likeli_tables_for_join + ' INNER JOIN ' + '_'.join(fnames) + '_likeli_table t' +\
                                  str(count) + ' ON ' + ' AND '.join(['t1.' + x + '=t'+str(count)+'.'+x for x in fnames])
         count = count + 1
    sql_query = sql_query.replace('LIKELI_TABLES_FOR_JOIN', likeli_tables_for_join)
    sql_query = sql_query.replace('GENERIC_FEATURE_NAME', '_'.join(['_'.join(x) for x in fnames_list]))
    likeli_function = 'MAX(' + '+'.join(['t'+str(x)+'.feature_avg' for x in range(2, len(fnames_list)+2)]) + ')'
    #likeli_function = 'MAX(1.0-' + '*'.join(['(1.0-t'+str(x)+'.feature_avg)' for x in range(2, len(fnames_list)+2)]) + ')'
    sql_query = sql_query.replace('LIKELI_FUNCTION', likeli_function)

    if len(train_folds) == len(folds):
        choosing_patients_expression = 'patients_test2'
    else:
        choosing_patients_expression = 'train_cv_indices ' + 'WHERE ' + ' OR '.join(['cv_index='+str(x) for x in folds if not x in list(train_folds)])

    sql_query = sql_query.replace('CHOOSING_PATIENTS_EXPRESSION', choosing_patients_expression)

    conn = utils.connect_to_database()
    cur = conn.cursor()
    cur.execute(sql_query)
    likeli = pd.DataFrame(cur.fetchall())
    likeli.columns = [x.name for x in cur.description]
    cur.close()
    conn.close()
    return likeli   

def drop_likelihood_table(likeli_table_name):
    conn = utils.connect_to_database()
    cur = conn.cursor()
    cur.execute('DROP TABLE ' + likeli_table_name + ';')
    conn.commit()
    cur.close()
    conn.close()
    return None

def calculate_likelihoods2(train, test, fnames_list, ftablename):
    global_mean = np.mean(train.is_screener)
    folds = [x for x in range(1, nfold+1)]

    likeli_all = pd.DataFrame()
    for L in range(1, len(folds)+1):
        for train_folds in itertools.combinations(folds, L):
            print train_folds
            test_folds = [x for x in folds if not x in list(train_folds)]
            if len(test_folds) == 0:
                test_folds = [0]
            print test_folds

            for fnames in fnames_list:
                likeli_table_name = '_'.join(fnames) + '_likeli_table'
                generate_likelihood_table(likeli_table_name, fnames, ftablename, train_folds)
            likeli = merge_likelihood_tables(fnames_list, ftablename, train_folds)
            for fnames in fnames_list:
                likeli_table_name = '_'.join(fnames) + '_likeli_table'
                drop_likelihood_table(likeli_table_name)

            for x in folds:
                if x in list(train_folds):
                    likeli['fold'+str(x)] = 1
                else:
                    likeli['fold'+str(x)] = 0
            likeli_all = likeli_all.append(likeli, ignore_index=True)
            
            col = likeli.columns[1]
            likeli = pd.merge(likeli, train[['patient_id', 'is_screener']], on='patient_id', how='inner')
            if len(likeli)>0:
                print "Pearson correlation: " + str(pearsonr(likeli.is_screener, likeli[col]))
                print "AUC: " + str(auc(likeli.is_screener, likeli[col]))
            del likeli

    file_name = likeli_all.columns[1]
    feats_all = train[['patient_id']].append(test[['patient_id']], ignore_index=True)
    for test_fold in ([0] + folds):
        train_folds = [x for x in folds if (x != test_fold) and (x != 0)]

        if len(train_folds) == len(folds):
            pd_query = ' and '.join(['fold'+str(x)+'==1' for x in train_folds])
        else:
            pd_query = ' and '.join(['fold'+str(x)+'==1' for x in train_folds]) + ' and ' + ' and '.join(['fold'+str(x)+'==0' for x in folds if not x in train_folds])
        print pd_query

        likeli = likeli_all.query(pd_query).copy().reset_index(drop=True)
        for x in folds:
            likeli.drop('fold'+str(x), axis=1, inplace=True)

        if test_fold == 0:
            feats_fold = test[['patient_id']].copy()
        else:
            feats_fold = train.query('cv_index==@test_fold')[['patient_id']].copy()
        feats_fold = pd.merge(feats_fold, likeli, on='patient_id', how='left')
        del likeli

        for val_fold in [x for x in folds if (x != test_fold) and (x != 0)]:
            train_folds = [x for x in folds if (x != test_fold) and (x != val_fold) and (x != 0)]
            pd_query = ' and '.join(['fold'+str(x)+'==1' for x in train_folds]) + ' and ' + ' and '.join(['fold'+str(x)+'==0' for x in folds if not x in train_folds])
            
            likeli = likeli_all.query(pd_query).copy().reset_index(drop=True)
            for x in folds:
                likeli.drop('fold'+str(x), axis=1, inplace=True)

            feats_val_fold = train.query('cv_index==@val_fold')[['patient_id']].copy()
            feats_val_fold = pd.merge(feats_val_fold, likeli, on='patient_id', how='left')
            del likeli
            feats_fold = feats_fold.append(feats_val_fold, ignore_index=True)

        feats_fold = feats_fold.reset_index(drop=True)
        for cols in [x for x in feats_fold.columns if x != 'patient_id']:
            feats_fold[cols].fillna(global_mean*len(fnames), inplace=True)            
            feats_fold = feats_fold.rename(columns={cols : cols+'_fold_'+str(test_fold)})
        feats_all = pd.merge(feats_all, feats_fold, on='patient_id', how='left')

    print "Writing to HDF5 store..."
    store = pd.HDFStore('../data/output-py/' + file_name + '.h5')
    store.append('feats_all', feats_all)
    print "Feature " + file_name + " is saved in file."
    store.close()
    return file_name

##########################################################################
##########################   BUILD DATASET   #############################
##########################################################################

train, test = build_train_test()
nfold = max(train.cv_index)

flist0 = ['patient_age_group', 'household_income', 'education_level'] +\
         [x for x in train.columns if 'patient_state' in x] +\
         [x for x in train.columns if 'ethinicity' in x] +\
         ['activity_type_count_all', 'activity_type_ratio']

#store = pd.HDFStore('../data/output-py/train_test.h5')
#train = store.get('train')
#test = store.get('test')
#store.close()
#nfold = max(train.cv_index)

##########################################################################
##########################   EXTERNAL FEATURES   #########################
##########################################################################

conn = utils.connect_to_database()
cols_add = pd.read_sql_query('SELECT patient_id, patient_state FROM patients_train;', conn)
train = pd.merge(train, cols_add, on='patient_id', how='left')
cols_add = pd.read_sql_query('SELECT patient_id, patient_state FROM patients_test;', conn)
test = pd.merge(test, cols_add, on='patient_id', how='left')
conn.close()
state_lat_lang = pd.read_csv('../data/input/external/state_latlon.csv')
train = pd.merge(train, state_lat_lang, on='patient_state', how='left')
test = pd.merge(test, state_lat_lang, on='patient_state', how='left')
flist0 = flist0 + ['patient_state_latitude', 'patient_state_longitude']
train['patient_state_latitude'].fillna(-999, inplace=True)
train['patient_state_longitude'].fillna(-999, inplace=True)
test['patient_state_latitude'].fillna(-999, inplace=True)
test['patient_state_longitude'].fillna(-999, inplace=True)

##########################################################################
##########################   COUNT FEATURES   ############################
##########################################################################

for f in range(1, 17):
    conn = utils.connect_to_database()
    sql_query = open('genentech-sql/patient_info' + str(f) + '.sql').read()
    feats = pd.read_sql_query(sql_query, conn)
    feats.fillna(0, inplace=True)
    store = pd.HDFStore('../data/output-py/patient_info' + str(f) + '.h5')
    store.append('feats', feats)
    store.close()
    conn.close()
    for fname in [x for x in feats.columns if x != 'patient_id']:
        train[fname].fillna(0, inplace=True)
        test[fname].fillna(0, inplace=True)
        flist0 = flist0 + [fname]

##########################################################################
##########################   FTRL FEATURES   #############################
##########################################################################
flist_likeli = []

fname = calculate_ftrl_features(train, test, 
                                fnames = ['diagnosis_practitioner_id', 'diagnosis_code', 'patient_age_group', 'patient_state', 'ethinicity', 'household_income', 'education_level'],
                                ftablename = 'diagnosis_feats')
flist_likeli = flist_likeli + [fname]

fname = calculate_ftrl_features(train, test, 
                                fnames = ['diagnosis_code', 'procedure_code'], 
                                ftablename = 'diagnosis_procedure_link')
flist_likeli = flist_likeli + [fname]

fname = calculate_ftrl_features(train, test, 
                                fnames = ['diagnosis_code', 'procedure_code'], 
                                ftablename = 'diagnosis_procedure_link', 
                                ftrl_type = '2')
flist_likeli = flist_likeli + [fname]

fname = calculate_ftrl_features(train, test, 
                                fnames = ['diagnosis_practitioner_id', 'diagnosis_code', 'patient_age_group', 'patient_state', 'ethinicity', 'household_income', 'education_level'],
                                ftablename = 'diagnosis_feats',
                                ftrl_type = '3', 
                                optional_date_ftrl3 = 'diagnosis_date')
flist_likeli = flist_likeli + [fname]

fname = calculate_ftrl_features(train, test, 
                                fnames = ['claim_id', 'procedure_code', 'procedure_primary_practitioner_id'],
                                ftablename = 'procedure_head2',
                                ftrl_type = '4')
flist_likeli = flist_likeli + [fname]

fname = calculate_ftrl_features(train, test, 
                                fnames = ['claim_id', 'procedure_code'],
                                ftablename = 'procedure_head2',
                                ftrl_type = '4')
flist_likeli = flist_likeli + [fname]

fname = calculate_ftrl_features(train, test, 
                                fnames = ['claim_id', 'diagnosis_practitioner_id', 'diagnosis_code', 'primary_physician_role'],
                                ftablename = 'diagnosis_feats2',
                                ftrl_type = '4')
flist_likeli = flist_likeli + [fname]

fname = calculate_ftrl_features(train, test, 
                                fnames = ['claim_id', 'diagnosis_code', 'primary_physician_role'],
                                ftablename = 'diagnosis_feats2',
                                ftrl_type = '4')
flist_likeli = flist_likeli + [fname]

fname = calculate_ftrl_features(train, test, 
                                fnames = ['claim_id', 'diagnosis_code'],
                                ftablename = 'diagnosis_feats2',
                                ftrl_type = '4')
flist_likeli = flist_likeli + [fname]

##########################################################################
####################   PRESCRIPTION_FEATS FEATURES   #####################
##########################################################################

fname = calculate_likelihoods(train, test, ['bb_usc_name', 'prescription_practitioner_id'], 'prescription_feats', 'max')
flist_likeli = flist_likeli + [fname]
fname = calculate_likelihoods(train, test, ['bb_usc_name'], 'prescription_feats', 'max')
flist_likeli = flist_likeli + [fname]
fname = calculate_likelihoods(train, test, ['payment_type'], 'prescription_feats', 'max')
flist_likeli = flist_likeli + [fname]
fname = calculate_likelihoods(train, test, ['prescription_practitioner_id'], 'prescription_feats', 'max')
flist_likeli = flist_likeli + [fname]

##########################################################################
#####################   DIAGNOSIS_FEATS FEATURES   #######################
##########################################################################

fname = calculate_likelihoods(train, test, ['diagnosis_code'], 'diagnosis_feats', 'max')
flist_likeli = flist_likeli + [fname]
fname = calculate_likelihoods(train, test, ['diagnosis_code'], 'diagnosis_feats', 'max', '2')
flist_likeli = flist_likeli + [fname]
fname = calculate_likelihoods(train, test, ['diagnosis_code'], 'diagnosis_feats', 'max', '3')
flist_likeli = flist_likeli + [fname]
fname = calculate_likelihoods(train, test, ['diagnosis_code'], 'diagnosis_feats', 'max', '4')
flist_likeli = flist_likeli + [fname]
fname = calculate_likelihoods(train, test, ['diagnosis_code'], 'diagnosis_feats', 'max', '5')
flist_likeli = flist_likeli + [fname]
fname = calculate_likelihoods(train, test, ['diagnosis_practitioner_id'], 'diagnosis_feats', 'max')
flist_likeli = flist_likeli + [fname]
fname = calculate_likelihoods(train, test, ['diagnosis_practitioner_id', 'diagnosis_code'], 'diagnosis_feats', 'max')
flist_likeli = flist_likeli + [fname]
fname = calculate_likelihoods(train, test, ['diagnosis_code', 'diagnosis_practitioner_state'], 'diagnosis_feats', 'max')
flist_likeli = flist_likeli + [fname]
fname = calculate_likelihoods(train, test, ['diagnosis_code', 'diagnosis_practitioner_state', 'diagnosis_practitioner_specialty_code'], 'diagnosis_feats', 'max')
flist_likeli = flist_likeli + [fname]
fname = calculate_likelihoods(train, test, ['diagnosis_practitioner_specialty_code'], 'diagnosis_feats', 'max')
flist_likeli = flist_likeli + [fname]
fname = calculate_likelihoods(train, test, ['diagnosis_code', 'patient_age_group'], 'diagnosis_feats', 'max')
flist_likeli = flist_likeli + [fname]
fname = calculate_likelihoods(train, test, ['diagnosis_code', 'patient_age_group'], 'diagnosis_feats', 'max', '2')
flist_likeli = flist_likeli + [fname]
fname = calculate_likelihoods(train, test, ['diagnosis_practitioner_id', 'patient_age_group'], 'diagnosis_feats', 'max')
flist_likeli = flist_likeli + [fname]
fname = calculate_likelihoods(train, test, ['diagnosis_practitioner_specialty_code', 'diagnosis_code'], 'diagnosis_feats', 'max')
flist_likeli = flist_likeli + [fname]
fname = calculate_likelihoods(train, test, ['diagnosis_practitioner_specialty_code', 'diagnosis_code'], 'diagnosis_feats', 'max', '4')
flist_likeli = flist_likeli + [fname]
fname = calculate_likelihoods(train, test, ['diagnosis_code', 'primary_physician_role'], 'diagnosis_feats', 'max')
flist_likeli = flist_likeli + [fname]
fname = calculate_likelihoods(train, test, ['diagnosis_code', 'primary_physician_role'], 'diagnosis_feats', 'max', '4')
flist_likeli = flist_likeli + [fname]
fname = calculate_likelihoods(train, test, ['diagnosis_code', 'primary_physician_role'], 'diagnosis_feats', 'max', '2')
flist_likeli = flist_likeli + [fname]
fname = calculate_likelihoods2(train, test, [['diagnosis_code'], ['diagnosis_practitioner_id']], 'diagnosis_feats')
flist_likeli = flist_likeli + [fname]
fname = calculate_likelihoods2(train, test, [['diagnosis_code'], ['diagnosis_practitioner_id'], ['diagnosis_code_prefix'], ['diagnosis_practitioner_specialty_code']], 'diagnosis_feats')
flist_likeli = flist_likeli + [fname]
fname = calculate_likelihoods(train, test, ['diagnosis_code', 'diagnosis_practitioner_cbsa'], 'diagnosis_feats', 'max')
flist_likeli = flist_likeli + [fname]
fname = calculate_likelihoods(train, test, ['diagnosis_code', 'diagnosis_practitioner_specialty_code', 'primary_physician_role'], 'diagnosis_feats', 'max')
flist_likeli = flist_likeli + [fname]
fname = calculate_likelihoods(train, test, ['diagnosis_code', 'diagnosis_practitioner_cbsa', 'primary_physician_role'], 'diagnosis_feats', 'max')
flist_likeli = flist_likeli + [fname]
fname = calculate_likelihoods(train, test, 
                              fnames = ['diagnosis_code'], 
                              ftablename = 'diagnosis_feats',
                              function_type = 'max',
                              query_type = '6',
                              optional_filter_feature_likeli6 =  'primary_physician_role',
                              optional_filter_value_likeli6 = 'ATG')
flist_likeli = flist_likeli + [fname]

##########################################################################
######################   PROCEDURE_HEAD FEATURES   #######################
##########################################################################

fname = calculate_likelihoods(train, test, ['procedure_primary_practitioner_id'], 'procedure_head', 'avg')
flist_likeli = flist_likeli + [fname]
fname = calculate_likelihoods(train, test, ['procedure_code'], 'procedure_head', 'max')
flist_likeli = flist_likeli + [fname]
fname = calculate_likelihoods(train, test, ['procedure_code', 'place_of_service'], 'procedure_head', 'max')
flist_likeli = flist_likeli + [fname]
fname = calculate_likelihoods(train, test, ['procedure_rendering_practitioner_id'], 'procedure_head', 'max')
flist_likeli = flist_likeli + [fname]

##########################################################################
#################   DIAGNOSIS_PROCEDURE_LINK FEATURES   ##################
##########################################################################

fname = calculate_likelihoods(train, test, ['procedure_code', 'diagnosis_code'], 'diagnosis_procedure_link', 'max')
flist_likeli = flist_likeli + [fname]
fname = calculate_likelihoods(train, test, ['procedure_code', 'diagnosis_code'], 'diagnosis_procedure_link', 'max', '2')
flist_likeli = flist_likeli + [fname]
fname = calculate_likelihoods(train, test, ['procedure_code', 'diagnosis_code'], 'diagnosis_procedure_link', 'max', '3')
flist_likeli = flist_likeli + [fname]
fname = calculate_likelihoods(train, test, ['procedure_code', 'diagnosis_code'], 'diagnosis_procedure_link', 'max', '4')
flist_likeli = flist_likeli + [fname]
fname = calculate_likelihoods(train, test, ['procedure_code', 'diagnosis_code'], 'diagnosis_procedure_link', 'max', '5')
flist_likeli = flist_likeli + [fname]
fname = calculate_likelihoods(train, test, ['diagnosis_primary_practitioner_id', 'procedure_code', 'diagnosis_code'], 'diagnosis_procedure_link', 'max')
flist_likeli = flist_likeli + [fname]
fname = calculate_likelihoods(train, test, ['diagnosis_code', 'procedure_code', 'primary_physician_role'], 'diagnosis_procedure_link', 'max')
flist_likeli = flist_likeli + [fname]
fname = calculate_likelihoods(train, test, ['diagnosis_code', 'place_of_service'], 'diagnosis_procedure_link', 'max')
flist_likeli = flist_likeli + [fname]
fname = calculate_likelihoods(train, test, ['diagnosis_code', 'procedure_code', 'place_of_service'], 'diagnosis_procedure_link', 'max')
flist_likeli = flist_likeli + [fname]
fname = calculate_likelihoods2(train, test, [['procedure_code'], ['diagnosis_code']], 'diagnosis_procedure_link')
flist_likeli = flist_likeli + [fname]

##########################################################################
###########################   OTHER FEATURES   ###########################
##########################################################################

fname = calculate_likelihoods(train, test, 
                              fnames = ['diagnosis_code', 'plan_type'], 
                              ftablename = 'diagnosis_procedure_link2',
                              function_type = 'max')
flist_likeli = flist_likeli + [fname]

##########################################################################
####################   ADD FTRL AND LIKELI FEATURES   ####################
##########################################################################

patients_to_exclude = read_patients_to_exclude()

for fold in range(nfold+1):
    flist = flist0
    Xtr = train.loc[train['cv_index']!=fold].copy().reset_index(drop=True)
    global_mean = np.mean(Xtr.is_screener)
    if fold !=0 :
        Xtest = train.loc[train['cv_index']==fold].copy().reset_index(drop=True)
    else:
        Xtest = test.copy()
    Xtest = Xtest.loc[~Xtest['patient_id'].isin(patients_to_exclude)].reset_index(drop=True)

    for f in flist_likeli:
        store = pd.HDFStore('../data/output-py/' + f + '.h5')
        feats = store.get('feats_all')
        store.close()
        feats = feats.rename(columns={f+'_fold_'+str(fold) : f})
        feats = feats.groupby('patient_id')[f].max().reset_index()
        Xtr = pd.merge(Xtr, feats[['patient_id', f]], on='patient_id', how='left')
        Xtr[f].fillna(global_mean, inplace=True)
        Xtest = pd.merge(Xtest, feats[['patient_id', f]], on='patient_id', how='left')
        Xtest[f].fillna(global_mean, inplace=True)
        flist = flist + [f]
        flist_nonfactor = flist_nonfactor + [f]
        #print 'Correlations for ' + f + ':'
        #print pearsonr(Xtr[f], Xtr.is_screener)
        #print pearsonr(Xtest[f], Xtest.is_screener)

    Xtr[['patient_id'] + flist].to_csv('../data/team/dmitry/train_feats_dmitry_fold' + str(fold) + '.csv', index=False)
    Xtest[['patient_id'] + flist].to_csv('../data/team/dmitry/test_feats_dmitry_fold' + str(fold) + '.csv', index=False)
