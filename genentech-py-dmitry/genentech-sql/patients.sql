CREATE TABLE patients
interleaved sortkey(patient_id)
AS 
SELECT * FROM
(SELECT patient_id, patient_age_group, patient_state, ethinicity, household_income, education_level FROM patients_train
UNION ALL
SELECT patient_id, patient_age_group, patient_state, ethinicity, household_income, education_level FROM patients_test2);