SELECT t3.patient_id, GROUP_FUNCTION(t4.feature_avg) AS GENERIC_FEATURE_NAME_FEATURE_TABLE_NAME_likeli2_GROUP_FUNCTION FROM
((SELECT patient_id, FEATURE_NAMES_COMMA_SEPARATED FROM FEATURE_TABLE_NAME) t3
INNER JOIN
(SELECT * FROM (SELECT FEATURE_NAMES_COMMA_SEPARATED,
                       AVG(CAST(is_screener AS float)) AS feature_avg,
                       COUNT(is_screener) AS feature_count
FROM
	(SELECT FEATURE_NAMES_COMMA_SEPARATED, t1.patient_id, MAX(t2.is_screener) AS is_screener FROM
    ((SELECT patient_id, FEATURE_NAMES_COMMA_SEPARATED FROM FEATURE_TABLE_NAME) t1
     INNER JOIN
     (SELECT patient_id, is_screener FROM train_cv_indices OPTIONAL_CV_EXPRESSION) t2
     ON t1.patient_id = t2.patient_id)
    GROUP BY FEATURE_NAMES_COMMA_SEPARATED, t1.patient_id)
GROUP BY FEATURE_NAMES_COMMA_SEPARATED)
WHERE feature_count>50) t4
ON T3_T4_CONDITION)
GROUP BY t3.patient_id;
