CREATE TABLE diagnosis_procedure_link2
interleaved sortkey(patient_id, diagnosis_primary_practitioner_id, procedure_primary_practitioner_id, diagnosis_code, diagnosis_date, procedure_code)
AS 
SELECT t1.patient_id, t1.diagnosis_date, t1.diagnosis_code, t1.primary_practitioner_id AS diagnosis_primary_practitioner_id,
t2.procedure_code, t2.procedure_date, t2.procedure_primary_practitioner_id, t2.plan_type, t2.charge_amount, t1.primary_physician_role
FROM
(diagnosis_head t1
LEFT OUTER JOIN
procedure_head t2
ON t1.claim_id=t2.claim_id);