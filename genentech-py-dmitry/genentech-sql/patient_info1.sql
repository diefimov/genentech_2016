SELECT patient_id, COUNT(DISTINCT CAST(prescription_practitioner_id AS varchar) + '*' + rx_number) AS patient_id_prescription_visit_count
FROM prescription_feats
GROUP BY patient_id;