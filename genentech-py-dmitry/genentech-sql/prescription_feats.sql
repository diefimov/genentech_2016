CREATE TABLE prescription_feats
interleaved sortkey(patient_id, prescription_practitioner_id, drug_id, rx_number, rx_fill_year)
AS 
SELECT t1.patient_id, t1.prescription_practitioner_id, 
t1.drug_id, t1.rx_fill_date, t1.refill_code, t1.days_supply,
t1.payment_type, t2.drug_name, t2.bgi, t2.bb_usc_code, t2.bb_usc_name,
t2.drug_generic_name, t2.drug_strength, t2.drug_form, t2.package_size, t2.manufacturer, t2.ndc_start_date,
t1.rx_fill_day, t1.rx_fill_month, t1.rx_fill_year, t2.ndc_start_year, t1.rx_number,
t3.state AS prescription_practitioner_state, t3.specialty_code AS prescription_practitioner_specialty_code,
t3.specialty_description AS prescription_practitioner_specialty_description, t3.cbsa as prescription_practitioner_cbsa 
FROM
((SELECT patient_id, prescription_practitioner_id, drug_id, rx_fill_date, refill_code, days_supply, payment_type, rx_number,
DATE_PART(yr, rx_fill_date) AS rx_fill_year,
DATE_PART(mon, rx_fill_date) AS rx_fill_month,
DATE_PART(d, rx_fill_date) AS rx_fill_day
FROM prescription_head) t1
LEFT JOIN
(SELECT *,
DATE_PART(yr, ndc_start_date) AS ndc_start_year
FROM drugs) t2
ON t1.drug_id=t2.drug_id
LEFT JOIN
physicians t3
ON t1.prescription_practitioner_id=t3.practitioner_id);