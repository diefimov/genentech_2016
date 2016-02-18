SELECT patient_id, COUNT(DISTINCT rx_number) AS patient_id_diff_rx_number_count
FROM prescription_feats
GROUP BY patient_id;