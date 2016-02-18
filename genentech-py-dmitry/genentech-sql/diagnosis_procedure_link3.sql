CREATE VIEW diagnosis_procedure_link3
AS 
SELECT t1.patient_id, t1.claim_id, t1.diagnosis_practitioner_id, t1.diagnosis_code, t1.primary_physician_role,
t2.procedure_code, t2.procedure_primary_practitioner_id
FROM
(diagnosis_feats2 t1
LEFT OUTER JOIN
procedure_head2 t2
ON t1.claim_id=t2.claim_id);