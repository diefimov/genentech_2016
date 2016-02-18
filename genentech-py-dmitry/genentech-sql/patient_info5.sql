SELECT patient_id, COUNT(DISTINCT CAST(procedure_primary_practitioner_id AS varchar) + '*' + procedure_date) AS patient_id_procedure_visit_count
FROM procedure_head
GROUP BY patient_id;