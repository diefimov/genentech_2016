CREATE TABLE diagnosis_feats
interleaved sortkey(patient_id, diagnosis_practitioner_id, diagnosis_code, diagnosis_date, diagnosis_code_prefix)
AS 
SELECT t1.patient_id, t1.primary_practitioner_id AS diagnosis_practitioner_id, t1.primary_physician_role,
t1.claim_type, t1.diagnosis_date, t1.diagnosis_code, t1.diagnosis_code_prefix, t2.diagnosis_description,
t3.state AS diagnosis_practitioner_state, t3.specialty_code AS diagnosis_practitioner_specialty_code,
t3.specialty_description AS diagnosis_practitioner_specialty_description, t3.cbsa as diagnosis_practitioner_cbsa,
t4.patient_age_group, t4.patient_state, t4.ethinicity, t4.household_income, t4.education_level
FROM
((SELECT patient_id, claim_type, diagnosis_date, diagnosis_code, primary_practitioner_id, primary_physician_role,
CASE WHEN diagnosis_code LIKE '%.%' THEN LEFT(diagnosis_code,CHARINDEX('.',diagnosis_code)-1) ELSE diagnosis_code END AS diagnosis_code_prefix
FROM diagnosis_head) t1
LEFT JOIN
diagnosis_code_table_cleaned t2
ON t1.diagnosis_code=t2.diagnosis_code
LEFT JOIN
physicians t3
ON t1.primary_practitioner_id=t3.practitioner_id
LEFT JOIN
patients t4
ON t1.patient_id=t4.patient_id);
