CREATE VIEW procedure_head3
AS 
SELECT t1.patient_id, t1.claim_id, t1.procedure_code, t1.procedure_primary_practitioner_id, t2.proc_claim_line_diff,
t2.proc_max_claim_line, t2.proc_count_claim_line
FROM
(procedure_head t1
INNER JOIN
missed_claims_disc t2
ON t1.claim_id=t2.claim_id);