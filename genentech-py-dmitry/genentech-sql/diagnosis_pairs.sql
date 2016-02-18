SELECT *, ROW_NUMBER() OVER() AS id 
INTO diagnosis_pairs_temp FROM 
(SELECT patient_id, diagnosis_code, diagnosis_date FROM diagnosis_feats ORDER BY patient_id, diagnosis_date);

SELECT t1.patient_id, t1.diagnosis_code AS diagnosis_code1, t2.diagnosis_code AS diagnosis_code2
INTO diagnosis_pairs
FROM
diagnosis_pairs_temp t1
JOIN diagnosis_pairs_temp t2
ON t2.id=t1.id+1
WHERE t1.patient_id=t2.patient_id;

DROP TABLE diagnosis_pairs_temp;
