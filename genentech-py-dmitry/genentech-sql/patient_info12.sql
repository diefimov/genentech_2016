SELECT patient_id, 
AVG(charge_amount) as procedure_charge_amount_mean
FROM procedure_head
GROUP BY patient_id;