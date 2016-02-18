SELECT patient_id, AVG(feature_avg) AS GENERIC_FEATURE_NAME_FEATURE_TABLE_NAME_likeli5
FROM
(SELECT t1.patient_id, t1.feature_avg FROM
((SELECT patient_id, feature_avg
FROM patient_likeli_table) t1
INNER JOIN
(SELECT patient_id, feature_avg,
RANK() OVER(PARTITION BY patient_id ORDER BY feature_avg DESC) feature_rank
FROM patient_likeli_table) t2
ON (t1.patient_id=t2.patient_id) and (t1.feature_avg=t2.feature_avg))
WHERE t2.feature_rank=2)
GROUP BY patient_id;