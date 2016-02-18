CREATE TABLE diagnosis_procedure_link
interleaved sortkey(patient_id, diagnosis_primary_practitioner_id, procedure_primary_practitioner_id, diagnosis_code, diagnosis_date, procedure_code)
AS 
SELECT t1.patient_id, t1.diagnosis_date, t1.diagnosis_code, t1.primary_practitioner_id AS diagnosis_primary_practitioner_id,
t2.procedure_code, t2.procedure_date, t2.procedure_primary_practitioner_id, t2.place_of_service, t1.primary_physician_role
FROM
(diagnosis_head t1
INNER JOIN
procedure_head t2
ON t1.claim_id=t2.claim_id);
