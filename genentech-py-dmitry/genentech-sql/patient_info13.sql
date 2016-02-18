SELECT patient_id, 
SUM(charge_amount) as procedure_charge_amount_sum
FROM procedure_head
GROUP BY patient_id;