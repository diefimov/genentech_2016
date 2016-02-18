SELECT patient_id, COUNT(DISTINCT CAST(surgical_practitioner_id AS varchar) + '*' + surgical_procedure_date) AS patient_id_surgical_visit_count
FROM surgical_head
GROUP BY patient_id;