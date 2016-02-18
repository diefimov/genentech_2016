import psycopg2
import pandas as pd
import numpy as np
import os
from nltk.corpus import stopwords

import utils

conn = utils.connect_to_database()
df = pd.read_sql_query('SELECT * FROM diagnosis_code_table;', conn)
df['diagnosis_description'] = df['diagnosis_description'].str.strip()
df['diagnosis_description'] = df['diagnosis_description'].str.lower()
df['diagnosis_description'] = df['diagnosis_description'].str.replace('[^\w\s]','')
for word in stopwords.words('english'):
    df['diagnosis_description'] = df['diagnosis_description'].str.replace(' ' + word + ' ', ' ')
    df['diagnosis_description'] = df['diagnosis_description'].str.replace('^' + word + ' ', 'EF')
    df['diagnosis_description'] = df['diagnosis_description'].str.replace(' ' + word + '$', 'EF')
    df['diagnosis_description'] = df['diagnosis_description'].str.replace('EF', '')
df.to_csv('../data/output-py/diagnosis_code_table_cleaned.csv', index=False)
os.system('aws s3 cp ../data/output-py/diagnosis_code_table_cleaned.csv s3://genentech-2016')

cur = conn.cursor()
cur.execute("DROP TABLE IF EXISTS diagnosis_code_table_cleaned;")
cur.execute("CREATE TABLE diagnosis_code_table_cleaned (diagnosis_code varchar, diagnosis_description varchar)\
                                                        compound sortkey(diagnosis_code);")
conn.commit()
copy_string = "copy diagnosis_code_table_cleaned from 's3://genentech-2016/diagnosis_code_table_cleaned.csv'\
               credentials " + utils.S3_CONNECTION_STRING + " ignoreheader 1 csv;"
cur.execute(copy_string)
conn.commit()

cur.close()
conn.close()
