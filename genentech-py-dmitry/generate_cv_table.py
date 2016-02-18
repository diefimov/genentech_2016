import psycopg2
import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedKFold
import random
import os
import sys
import getopt

import utils

opts, args = getopt.getopt(sys.argv[1:], "n:", ["nfold="])
opts = {x[0]:x[1] for x in opts}
nfold = int(opts['--nfold'])

conn = utils.connect_to_database()
cur = conn.cursor()

sql_query = "SELECT patient_id, is_screener \
             FROM patients_train;"
train = pd.read_sql_query(sql_query, conn)
train.reset_index(drop=True, inplace=True)
train['cv_index'] = 0

skf = StratifiedKFold(train.is_screener.values, nfold, random_state=2016)
count_cv = 0
for train_index, valid_index in skf:
    count_cv = count_cv + 1
    train.loc[valid_index, 'cv_index'] = count_cv
train.to_csv('../data/output-py/train_cv_indices.csv', index=False)

os.system('aws s3 cp ../data/output-py/train_cv_indices.csv s3://genentech-2016')

cur.execute("DROP TABLE IF EXISTS train_cv_indices;")
cur.execute("CREATE TABLE train_cv_indices (patient_id integer, is_screener integer, cv_index integer)\
                                           interleaved sortkey(patient_id, cv_index);")
conn.commit()
copy_string = "copy train_cv_indices from 's3://genentech-2016/train_cv_indices.csv'\
               credentials " + utils.S3_CONNECTION_STRING + " ignoreheader 1 csv;" 
cur.execute(copy_string)
conn.commit()

cur.close()
conn.close()
