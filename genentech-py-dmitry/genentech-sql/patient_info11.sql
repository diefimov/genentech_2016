SELECT patient_id, 
MAX(charge_amount) as procedure_charge_amount_max
FROM procedure_head
GROUP BY patient_id;