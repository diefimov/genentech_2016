import psycopg2
import pandas as pd

import utils

conn = utils.connect_to_database()
cur = conn.cursor()

##### diagnosis_code table ######
cur.execute("CREATE TABLE diagnosis_code_table (diagnosis_code varchar, diagnosis_description varchar)\
                                               compound sortkey(diagnosis_code);")
conn.commit()
copy_string = "copy diagnosis_code_table from 's3://genentech-2016/diagnosis_code.csv.gz'\
               credentials " + utils.S3_CONNECTION_STRING + " ignoreheader 1 csv gzip;"               
cur.execute(copy_string)
conn.commit()
#########

##### diagnosis_head table #####
cur.execute("CREATE TABLE diagnosis_head (patient_id integer, claim_id varchar, claim_type varchar, diagnosis_date varchar,\
                                          diagnosis_code varchar, primary_practitioner_id integer, primary_physician_role varchar)\
                                         interleaved sortkey(claim_id, patient_id, primary_practitioner_id, diagnosis_code);")
conn.commit()
copy_string = "copy diagnosis_head from 's3://genentech-2016/diagnosis_head.csv.gz'\
               credentials " + utils.S3_CONNECTION_STRING + " ignoreheader 1 csv gzip;"               
cur.execute(copy_string)
conn.commit()
##########

##### prescription_head table ######
cur.execute("CREATE TABLE prescription_head (claim_id varchar, patient_id integer, drug_id varchar, prescription_practitioner_id integer,\
                                             refill_code varchar, days_supply integer, rx_fill_date date,\
                                             rx_number varchar, payment_type varchar)\
                                            interleaved sortkey(patient_id, prescription_practitioner_id, drug_id, rx_number, rx_fill_date);")
conn.commit()
copy_string = "copy prescription_head from 's3://genentech-2016/prescription_head.csv.gz'\
               credentials " + utils.S3_CONNECTION_STRING + " ignoreheader 1 csv gzip;"               
cur.execute(copy_string)
conn.commit()
sql_query = 'ALTER TABLE prescription_head ADD drug_date varchar;'
cur.execute(sql_query)
conn.commit()
RIGHT('000' + CAST(@number1 AS NCHAR(3)), 3 )
sql_query = 'UPDATE prescription_head SET drug_date = (CAST(DATE_PART(yr, rx_fill_date) AS varchar) + RIGHT('00'+CAST(DATE_PART(mon, rx_fill_date) AS varchar),2));'
cur.execute(sql_query)
conn.commit()

############

##### drugs table ######
cur.execute("CREATE TABLE drugs (drug_id varchar, NDC11 varchar, drug_name varchar, BGI varchar, BB_USC_code varchar, BB_USC_name varchar,\
                                 drug_generic_name varchar, drug_strength varchar, drug_form varchar, package_size numeric,\
                                 package_description varchar, manufacturer varchar, NDC_start_date date)\
                                compound sortkey(drug_id);")
conn.commit()
copy_string = "copy drugs from 's3://genentech-2016/drugs.csv.gz'\
               credentials " + utils.S3_CONNECTION_STRING + " ignoreheader 1 csv gzip;"               
cur.execute(copy_string)
conn.commit()
############

##### patient_activity_head table ######
cur.execute("CREATE TABLE patient_activity_head (patient_id integer, activity_type varchar, activity_year varchar, activity_month varchar)\
                                                compound sortkey(patient_id, activity_year, activity_month);")
conn.commit()
copy_string = "copy patient_activity_head from 's3://genentech-2016/patient_activity_head.csv.gz'\
               credentials " + utils.S3_CONNECTION_STRING + " ignoreheader 1 csv gzip;"               
cur.execute(copy_string)
conn.commit()
############

##### patients_train table ######
cur.execute("CREATE TABLE patients_train (patient_id integer, patient_age_group varchar, patient_gender varchar, patient_state varchar,\
                                          ethinicity varchar, household_income varchar, education_level varchar, is_screener integer)\
                                         compound sortkey(patient_id);")
conn.commit()
copy_string = "copy patients_train from 's3://genentech-2016/patients_train.csv.gz'\
               credentials " + utils.S3_CONNECTION_STRING + " ignoreheader 1 csv gzip;"               
cur.execute(copy_string)
conn.commit()
############

##### patients_test table ######
cur.execute("CREATE TABLE patients_test (patient_id integer, patient_age_group varchar, patient_gender varchar, patient_state varchar,\
                                         ethinicity varchar, household_income varchar, education_level varchar)\
                                        compound sortkey(patient_id);")
conn.commit()
copy_string = "copy patients_test from 's3://genentech-2016/patients_test.csv.gz'\
               credentials " + utils.S3_CONNECTION_STRING + " ignoreheader 1 csv gzip;"               
cur.execute(copy_string)
conn.commit()
############

##### physicians table ######
cur.execute("CREATE TABLE physicians (physician_id integer, practitioner_id integer, state varchar,\
                                      specialty_code varchar, specialty_description varchar, CBSA varchar)\
                                     compound sortkey(practitioner_id);")
conn.commit()
copy_string = "copy physicians from 's3://genentech-2016/physicians.csv.gz'\
               credentials " + utils.S3_CONNECTION_STRING + " ignoreheader 1 csv gzip;"               
cur.execute(copy_string)
conn.commit()
############

##### procedure_head table #####
cur.execute("CREATE TABLE procedure_head (patient_id integer, claim_id varchar, claim_line_item integer, claim_type varchar,\
                                          procedure_code varchar, procedure_date varchar, place_of_service varchar, plan_type varchar,\
                                          procedure_primary_practitioner_id integer, units_administered bigint, charge_amount integer,\
                                          procedure_primary_physician_role varchar, procedure_attending_practitioner_id integer, procedure_referring_practitioner_id integer,\
                                          procedure_rendering_practitioner_id integer, procedure_ordering_practitioner_id integer, procedure_operating_practitioner_id integer)\
                                         interleaved sortkey(claim_id, patient_id, procedure_code, procedure_primary_practitioner_id);")
conn.commit()
copy_string = "copy procedure_head from 's3://genentech-2016/procedure_head.csv.gz'\
               credentials " + utils.S3_CONNECTION_STRING + " ignoreheader 1 csv gzip;"               
cur.execute(copy_string)
conn.commit()
##########

##### procedure_code table ######
cur.execute("CREATE TABLE procedure_code_table (procedure_code varchar, procedure_description varchar)\
                                               compound sortkey(procedure_code);")
conn.commit()
copy_string = "copy procedure_code_table from 's3://genentech-2016/procedure_code.csv.gz'\
               credentials " + utils.S3_CONNECTION_STRING + " ignoreheader 1 csv gzip;"               
cur.execute(copy_string)
conn.commit()
############

##### surgical_head table ######
cur.execute("CREATE TABLE surgical_head (patient_id integer, claim_id varchar, procedure_type_code varchar, surgical_claim_type varchar,\
                                         surgical_code varchar, surgical_procedure_date varchar, surgical_place_of_service varchar,\
                                         surgical_plan_type varchar, surgical_practitioner_id integer, surgical_primary_physician_role varchar)\
                                        interleaved sortkey (patient_id, surgical_code, surgical_practitioner_id);")
conn.commit()
copy_string = "copy surgical_head from 's3://genentech-2016/surgical_head.csv.gz'\
               credentials " + utils.S3_CONNECTION_STRING + " ignoreheader 1 csv gzip;"               
cur.execute(copy_string)
conn.commit()
############

##### surgical_code table ######
cur.execute("CREATE TABLE surgical_code_table (surgical_code varchar, surgical_description varchar)\
                                              compound sortkey (surgical_code);")
conn.commit()
copy_string = "copy surgical_code_table from 's3://genentech-2016/surgical_code.csv.gz'\
               credentials " + utils.S3_CONNECTION_STRING + " ignoreheader 1 csv gzip;"               
cur.execute(copy_string)
conn.commit()
############

#rows = pd.read_sql_query("SELECT TOP 100 * FROM diagnosis_head;", conn)
#cur.execute("SELECT * FROM diagnosis_code;")
#cur.fetchall()

cur.close()
conn.close()


