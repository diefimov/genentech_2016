SELECT patient_id, COUNT(DISTINCT drug_id) AS patient_id_diff_drug_count
FROM prescription_feats
GROUP BY patient_id;