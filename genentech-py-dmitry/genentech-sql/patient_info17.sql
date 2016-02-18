SELECT patient_id FROM
(SELECT patient_id, procedure_code 
FROM procedure_head
WHERE procedure_code='REPLACE_PROCEDURE_CODE')
GROUP BY patient_id;
