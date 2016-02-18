SELECT * INTO LIKELI_TABLE_NAME
FROM (SELECT T1_COMMA_SEPARATED,
             AVG(CAST(t2.is_screener AS float)) AS feature_avg,
             COUNT(t2.is_screener) AS feature_count
FROM
(SELECT patient_id, FEATURE_NAMES_COMMA_SEPARATED FROM FEATURE_TABLE_NAME) t1
INNER JOIN
(SELECT patient_id, is_screener FROM train_cv_indices OPTIONAL_CV_EXPRESSION) t2
ON t1.patient_id = t2.patient_id
GROUP BY T1_COMMA_SEPARATED)
WHERE feature_count>50;
