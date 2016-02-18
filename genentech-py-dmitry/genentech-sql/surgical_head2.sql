CREATE TABLE surgical_head2
interleaved sortkey(patient_id, claim_id)
AS
SELECT t1.patient_id, t1.claim_id, t1.surgical_practitioner_id, t1.surgical_code, t1.surgical_primary_physician_role, t1.surgical_plan_type FROM
((SELECT patient_id, claim_id, surgical_practitioner_id, surgical_code, surgical_primary_physician_role, surgical_plan_type FROM surgical_head) t1
INNER JOIN
(SELECT claim_id FROM procedure_head2) t2
ON t1.claim_id = t2.claim_id);