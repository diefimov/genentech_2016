CREATE TABLE diagnosis_feats2
interleaved sortkey(patient_id, diagnosis_practitioner_id, diagnosis_code, primary_physician_role)
AS
SELECT t1.patient_id, t1.claim_id, t1.diagnosis_practitioner_id, t1.diagnosis_code, t1.primary_physician_role FROM
((SELECT patient_id, claim_id, primary_practitioner_id AS diagnosis_practitioner_id, diagnosis_code, primary_physician_role FROM diagnosis_head) t1
INNER JOIN
(SELECT claim_id FROM procedure_head2) t2
ON t1.claim_id = t2.claim_id);