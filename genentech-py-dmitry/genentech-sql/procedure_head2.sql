CREATE TABLE procedure_head2
interleaved sortkey(patient_id, claim_id, procedure_code, procedure_primary_practitioner_id)
AS 
SELECT t1.patient_id, t1.claim_id, t1.procedure_code, t1.procedure_primary_practitioner_id FROM
((SELECT * FROM procedure_head) t1
INNER JOIN
(SELECT * FROM
(SELECT claim_id, MAX(claim_line_item)-COUNT(claim_line_item) AS claim_line_item_diff
FROM procedure_head
GROUP BY claim_id)
WHERE claim_line_item_diff > 0) t2
ON t1.claim_id=t2.claim_id);