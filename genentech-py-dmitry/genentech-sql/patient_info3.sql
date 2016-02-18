SELECT patient_id, COUNT(DISTINCT CAST(diagnosis_practitioner_id AS varchar) + '*' + diagnosis_date) AS patient_id_diagnosis_visit_count
FROM diagnosis_feats
GROUP BY patient_id;