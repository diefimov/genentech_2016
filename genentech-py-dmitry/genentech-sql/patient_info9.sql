SELECT patient_id, COUNT(DISTINCT diagnosis_code) AS patient_id_diff_diagnosis_code_count
FROM diagnosis_feats
GROUP BY patient_id;