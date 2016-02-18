DROP TABLE IF EXISTS diagnosis_claim_count CASCADE;

CREATE TABLE diagnosis_claim_count
(
   claim_id               varchar(256),
   diag_count_claim   bigint
);

COMMIT;

insert into diagnosis_claim_count
select 
  claim_id,
  count(*) as diag_count_claim
from diagnosis_head
group by claim_id

union

select
	t1.claim_id,
	-1 as diag_count_claim
from 
	procedure_head t1
	left join diagnosis_head t2 on t2.claim_id = t1.claim_id
where t2.diagnosis_code is null
group by t1.claim_id;


COMMIT;

create or replace view diagnosis_claim_count_disc as
select 
	claim_id,
	CASE
		WHEN diag_count_claim = -1 THEN '-1'
		WHEN diag_count_claim = 0 THEN '0'
		WHEN diag_count_claim = 1 THEN '1'
		WHEN diag_count_claim = 2 THEN '2'
		WHEN diag_count_claim = 3 THEN '3'
		WHEN diag_count_claim = 4 THEN '4'
		WHEN diag_count_claim = 5 THEN '5'
		WHEN diag_count_claim <= 10 THEN '10'
		ELSE 'Inf'
	END as diag_count_claim
from diagnosis_claim_count

COMMIT;

CREATE OR REPLACE VIEW procedure_head3
AS 
 SELECT t1.patient_id, t1.claim_id, t1.procedure_code, t1.procedure_primary_practitioner_id, t2.proc_claim_line_diff, t2.proc_max_claim_line, t2.proc_count_claim_line, t3.diag_count_claim
   FROM procedure_head t1
   JOIN missed_claims_disc t2 ON t1.claim_id = t2.claim_id
   join diagnosis_claim_count_disc t3 on t1.claim_id = t3.claim_id;

COMMIT;

CREATE OR REPLACE VIEW diagnosis_head2
AS 
 SELECT t1.*, t2.proc_claim_line_diff, t2.proc_max_claim_line, t2.proc_count_claim_line, t3.diag_count_claim
   FROM diagnosis_head t1
   JOIN missed_claims_disc t2 ON t1.claim_id = t2.claim_id
   join diagnosis_claim_count_disc t3 on t1.claim_id = t3.claim_id;

COMMIT;