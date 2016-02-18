SELECT patient_id, FEATURES_LIST_COMMA_SEPARATED, MAX(is_screener) AS is_screener
FROM
(SELECT t1.patient_id, T1_FEATURES_COMMA_SEPARATED, t2.is_screener FROM
((SELECT patient_id, FEATURES_LIST_COMMA_SEPARATED FROM FEATURE_TABLE_NAME) t1
INNER JOIN
(SELECT patient_id, is_screener FROM train_cv_indices OPTIONAL_CV_EXPRESSION) t2
ON t1.patient_id = t2.patient_id))
GROUP BY patient_id, FEATURES_LIST_COMMA_SEPARATED;
