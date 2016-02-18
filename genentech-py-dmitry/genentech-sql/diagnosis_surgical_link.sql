CREATE TABLE diagnosis_surgical_link
interleaved sortkey(patient_id, diagnosis_primary_practitioner_id, surgical_practitioner_id, diagnosis_code, diagnosis_date, surgical_code, surgical_procedure_date)
AS 
SELECT t1.patient_id, t1.diagnosis_date, t1.diagnosis_code, t1.primary_practitioner_id AS diagnosis_primary_practitioner_id,
t2.surgical_code, t2.surgical_procedure_date, t2.surgical_practitioner_id
FROM
(diagnosis_head t1
INNER JOIN
surgical_head t2
ON t1.claim_id=t2.claim_id);
