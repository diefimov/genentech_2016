CREATE TABLE missed_claims
interleaved sortkey(claim_id)
AS 
SELECT 
 claim_id,
 max(claim_line_item) - count(claim_line_item) AS proc_claim_line_diff,
 max(claim_line_item) AS proc_max_claim_line,
 count(claim_line_item) AS proc_count_claim_line
FROM procedure_head
GROUP BY claim_id

UNION

SELECT
t1.claim_id,
-1 AS proc_claim_line_diff,
-1 AS proc_max_claim_line,
-1 AS proc_count_claim_line
FROM 
diagnosis_head t1
LEFT JOIN procedure_head t2 ON t2.claim_id = t1.claim_id 
WHERE t2.procedure_code is null
GROUP BY t1.claim_id;
