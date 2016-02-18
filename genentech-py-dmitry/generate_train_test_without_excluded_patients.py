import psycopg2
import pandas as pd
import numpy as np
import csv
import os

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


conn = utils.connect_to_database()
cur = conn.cursor()

patients_to_exclude = read_patients_to_exclude()

sql_query = "SELECT * FROM patients_train;"
train = pd.read_sql_query(sql_query, conn)
train.reset_index(drop=True, inplace=True)
train = train.loc[~train['patient_id'].isin(patients_to_exclude)].reset_index(drop=True)
train.to_csv('../data/input/train2.csv', index=False)
os.system('aws s3 cp ../data/input/train2.csv s3://genentech-2016')

sql_query = "SELECT * FROM patients_test;"
test = pd.read_sql_query(sql_query, conn)
test.reset_index(drop=True, inplace=True)
test = test.loc[~test['patient_id'].isin(patients_to_exclude)].reset_index(drop=True)
test.to_csv('../data/input/test2.csv', index=False)
os.system('aws s3 cp ../data/input/test2.csv s3://genentech-2016')

cur.execute("DROP TABLE IF EXISTS patients_train2;")
cur.execute("CREATE TABLE patients_train2 (patient_id integer, patient_age_group varchar, patient_gender varchar, patient_state varchar,\
                                           ethinicity varchar, household_income varchar, education_level varchar, is_screener integer)\
                                          compound sortkey(patient_id);")
conn.commit()
copy_string = "copy patients_train2 from 's3://genentech-2016/train2.csv'\
               credentials " + utils.S3_CONNECTION_STRING + " ignoreheader 1 csv;"
cur.execute(copy_string)
conn.commit()

cur.execute("DROP TABLE IF EXISTS patients_test2;")
cur.execute("CREATE TABLE patients_test2 (patient_id integer, patient_age_group varchar, patient_gender varchar, patient_state varchar,\
                                          ethinicity varchar, household_income varchar, education_level varchar)\
                                         compound sortkey(patient_id);")
conn.commit()
copy_string = "copy patients_test2 from 's3://genentech-2016/test2.csv'\
               credentials " + utils.S3_CONNECTION_STRING + " ignoreheader 1 csv;"
cur.execute(copy_string)
conn.commit()

cur.close()
conn.close()
