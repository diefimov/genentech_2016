CREATE VIEW missed_claims_disc AS
SELECT 
claim_id,
CASE
WHEN proc_claim_line_diff = -1 THEN '-1'
WHEN proc_claim_line_diff = 0 THEN '0'
WHEN proc_claim_line_diff = 1 THEN '1'
WHEN proc_claim_line_diff = 2 THEN '2'
WHEN proc_claim_line_diff = 3 THEN '3'
WHEN proc_claim_line_diff = 4 THEN '4'
WHEN proc_claim_line_diff = 5 THEN '5'
WHEN proc_claim_line_diff <= 10 THEN '10'
ELSE 'Inf'
END AS proc_claim_line_diff,
CASE
WHEN proc_max_claim_line = -1 THEN '-1'
WHEN proc_max_claim_line = 0 THEN '0'
WHEN proc_max_claim_line = 1 THEN '1'
WHEN proc_max_claim_line = 2 THEN '2'
WHEN proc_max_claim_line = 3 THEN '3'
WHEN proc_max_claim_line = 4 THEN '4'
WHEN proc_max_claim_line = 5 THEN '5'
WHEN proc_max_claim_line <= 10 THEN '10'
WHEN proc_max_claim_line <= 30 THEN '50'
WHEN proc_max_claim_line <= 50 THEN '100'
ELSE 'Inf'
END AS proc_max_claim_line,
CASE
WHEN proc_count_claim_line = -1 THEN '-1'
WHEN proc_count_claim_line = 0 THEN '0'
WHEN proc_count_claim_line = 1 THEN '1'
WHEN proc_count_claim_line = 2 THEN '2'
WHEN proc_count_claim_line = 3 THEN '3'
WHEN proc_count_claim_line = 4 THEN '4'
WHEN proc_count_claim_line = 5 THEN '5'
WHEN proc_count_claim_line <= 10 THEN '10'
WHEN proc_count_claim_line <= 30 THEN '50'
WHEN proc_max_claim_line <= 50 THEN '100'
ELSE 'Inf'
END AS proc_count_claim_line
FROM missed_claims;