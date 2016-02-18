import psycopg2
import pandas as pd
from scipy.stats.stats import pearsonr

import utils

conn = utils.connect_to_database()
cur = conn.cursor()

cur.execute(open('genentech-sql/patient_activity_feats.sql').read())
conn.commit()

cur.execute(open('genentech-sql/prescription_feats.sql').read())
conn.commit()

cur.execute(open('genentech-sql/diagnosis_feats.sql').read())
conn.commit()

cur.execute(open('genentech-sql/diagnosis_procedure_link.sql').read())
conn.commit()

cur.execute(open('genentech-sql/diagnosis_pairs.sql').read())
conn.commit()

cur.execute(open('genentech-sql/diagnosis_feats2.sql').read())
conn.commit()

cur.execute(open('genentech-sql/procedure_head2.sql').read())
conn.commit()

cur.execute(open('genentech-sql/diagnosis_procedure_link2.sql').read())
conn.commit()

cur.close()
conn.close()

