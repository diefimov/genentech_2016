CREATE TABLE patient_activity_feats
interleaved sortkey(patient_id)
AS
SELECT patient_id,
COUNT(*) AS activity_type_count_all,
COUNT(CASE WHEN activity_type='R' THEN 1 ELSE null END) AS activity_type_r_count_all,
COUNT(CASE WHEN activity_type='A' THEN 1 ELSE null END) AS activity_type_a_count_all
FROM patient_activity_head
GROUP BY patient_id;
