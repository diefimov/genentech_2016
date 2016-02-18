import pandas as pd
import base_util as util
import numpy as np


################################################################
# CV fold indices
################################################################
def load_indices():
    util.get_log().info('Loading CV fold indices...')
    util.tic()

    data_cv = util.load_from_db("select * from train_cv_indices")
    data_cv.set_index('patient_id', inplace=True)
    data_cv = data_cv['cv_index']

    util.save_data(data_cv=data_cv)
    util.toc()


################################################################
# patient data
################################################################
def load_patient_data():
    util.get_log().info('Loading Patient data...')
    util.tic()

    data_tr = util.load_from_db("select * from patients_train")
    data_tst = util.load_from_db("select * from patients_test")
    data_patient_all = data_tr.append(data_tst)

    data_patient_all.sort_values(by=['patient_id'], inplace=True)
    data_patient_all.set_index('patient_id', inplace=True)

    data_all_out = data_patient_all[['is_screener']].copy()

    data_cv = util.load_data('data_cv')
    data_all_out['cv_index'] = data_cv.ix[data_all_out.index].values
    data_all_out.ix[pd.isnull(data_all_out['is_screener']), 'cv_index'] = 0.0
    data_all_out['cv_index'] = data_all_out['cv_index'].astype(int)

    data_all_out['exclude'] = True
    data_keep = util.load_from_db("select patient_id from patients_train2 union select patient_id from patients_test2")
    data_all_out.ix[data_keep['patient_id'].values, 'exclude'] = False

    data_patient_all.drop(['is_screener'], axis='columns', inplace=True)

    util.save_data(data_patient_all=data_patient_all, data_all_out=data_all_out)
    util.toc()


################################################################
# activity data
################################################################
def load_activity_data():
    util.get_log().info('Loading activity data...')
    util.tic()

    data_actv = util.load_from_db(
        "select patient_id, "
        "sum(cast(activity_type = 'R' as integer)) as act_r_count, "
        "sum(cast(activity_type = 'A' as integer)) as act_a_count, "
        "min(activity_year || '-' || activity_month || '-01') as act_start, "
        "max(activity_year || '-' || activity_month || '-01') as act_end "
        "from patient_activity_head group by patient_id")

    data_actv['act_start'] = pd.to_datetime(data_actv['act_start'], format="%Y-%m-%d").astype(int).astype(float)
    data_actv['act_end'] = pd.to_datetime(data_actv['act_end'], format="%Y-%m-%d").astype(int).astype(float)
    data_actv['act_period'] = data_actv['act_end'] - data_actv['act_start']
    data_actv.drop(['act_start', 'act_end'], axis='columns', inplace=True)

    data_actv.set_index('patient_id', inplace=True)
    data_actv = data_actv.ix[util.load_data('data_all_out').index, :].copy()

    util.save_data(data_actv=data_actv)
    util.toc()


################################################################
# diagnostics data
################################################################
def load_diag_data():

    util.get_log().info('Loading diagnosis data...')
    util.tic()

    util.get_log().info('   diagnosis base data...')
    data_diag_base = util.load_from_db(
        "select "
        "  patient_id,  "
        "  count(*) diag_count  "
        " group by patient_id ")
    data_diag_base.set_index('patient_id', inplace=True)
    data_diag_base = data_diag_base.ix[util.load_data('data_all_out').index, :].copy()
    data_diag_base.fillna(0, inplace=True)
    util.save_data(data_diag_base=data_diag_base)
    del data_diag_base

    util.get_log().info('   diagnosis base description count data...')
    data_diag_base_desc = util.load_from_db(
        "select "
        "  patient_id,  "
        "  sum(cast(diagnosis_description like '%infection%' as integer)) diag_desc_infect,   "
        "  sum(cast(diagnosis_description like '%malignant%'as integer)) diag_desc_malig,   "
        "  sum(cast(diagnosis_description like '%carcinoma%' as integer)) diag_desc_carcin,   "
        "  sum(cast(diagnosis_description like '%oma %' as integer)) diag_desc_oma,   "
        "  sum(cast(diagnosis_description like '%hodgkins%' as integer)) diag_desc_hodk,   "
        "  sum(cast(diagnosis_description like '%leukemia%' as integer)) diag_desc_leuk, "
        "  sum(cast(diagnosis_description like '%benign%' as integer)) diag_desc_benig, "
        "  sum(cast(diagnosis_description like '%tuberculosis%' as integer)) diag_desc_tuberc, "
        "  sum(cast(diagnosis_description like '%virus%' as integer)) diag_desc_virus, "
        "  sum(cast(diagnosis_description like '%diabetes%' as integer)) diag_desc_diab, "
        "  sum(cast(diagnosis_description like '%vagina%' as integer)) diag_desc_vag, "
        "  sum(cast(diagnosis_description like '%smear%' as integer)) diag_desc_smr, "
        "  sum(cast(diagnosis_description like '%cervical%' as integer)) diag_desc_cvc, "
        "  sum(cast(diagnosis_description like '%cervix%' as integer)) diag_desc_cvx, "
        "  sum(cast(diagnosis_description like '%papiloma%' as integer)) diag_desc_ppl, "
        "  sum(cast(diagnosis_description like '%chlamy%' as integer)) diag_desc_chm, "
        "  sum(cast(diagnosis_description like '%gyneco%' as integer)) diag_desc_gyn, "
        "  sum(cast(diagnosis_description like '%hiv%' as integer)) diag_desc_hiv "
        " from diagnosis_feats  "
        " group by patient_id ")
    data_diag_base_desc.set_index('patient_id', inplace=True)
    data_diag_base_desc = data_diag_base_desc.ix[util.load_data('data_all_out').index, :].copy()
    data_diag_base_desc.fillna(0, inplace=True)
    util.save_data(data_diag_base_desc=data_diag_base_desc)
    del data_diag_base_desc

    util.get_log().info('   diagnosis_code likelihood discretized...')
    data_diag_code_cnt = util.calc_likelihood_count_db(
        table='diagnosis_feats', fields=['diagnosis_code'],
        intervals=np.array([0.0] + list(np.linspace(0.0, 1.0, num=21)[2:])))
    util.save_data(data_diag_code_cnt=data_diag_code_cnt)
    del data_diag_code_cnt

    util.get_log().info('   diagnosis_code_x_primary_physician_role likelihood discretized...')
    data_diag_code_phys_cnt = util.calc_likelihood_count_db(
        table='diagnosis_feats', fields=['diagnosis_code', 'primary_physician_role'],
        intervals=np.array([0.0] + list(np.linspace(0.0, 1.0, num=21)[2:])))
    util.save_data(data_diag_code_phys_cnt=data_diag_code_phys_cnt)
    del data_diag_code_phys_cnt

    util.get_log().info('   diagnosis specialty data...')
    data_diag_spec = util.load_from_db(
        "select "
        "  patient_id,  "
        "  sum(cast(diagnosis_practitioner_specialty_description = 'NURSE MIDWIFE' as integer)) "
        "     as diag_phys_nrs_mw,   "
        "  sum(cast(diagnosis_practitioner_specialty_description = 'MATERNAL AND FETAL MEDICINE' as integer)) "
        "     as diag_phys_mfm,   "
        "  sum(cast(diagnosis_practitioner_specialty_description = 'PATHOLOGY, CYTOPATHOLOGY' as integer)) "
        "     as diag_phys_pc,   "
        "  sum(cast(diagnosis_practitioner_specialty_description = 'GYNECOLOGY' as integer)) "
        "     as diag_phys_gy,   "
        "  sum(cast(diagnosis_practitioner_specialty_description = 'ANATOMIC PATHOLOGY' as integer)) "
        "     as diag_phys_ap,   "
        "  sum(cast(diagnosis_practitioner_specialty_description = 'OBSTETRICS AND GYNECOLOGY' as integer)) "
        "     as diag_phys_og,   "
        "  sum(cast(diagnosis_practitioner_specialty_description = 'PATHOLOGY, ANATOMIC/CLINICAL' as integer)) "
        "     as diag_phys_pac,   "
        "  sum(cast(diagnosis_practitioner_specialty_description = 'DERMATOPATHOLOGY' as integer)) "
        "     as diag_phys_dp,   "
        "  sum(cast(diagnosis_practitioner_specialty_description = 'DERMATOLOGY' as integer)) "
        "     as diag_phys_dt,   "
        "  sum(cast(diagnosis_practitioner_specialty_description = 'PULMONARY CRITICAL CARE MEDICINE' as integer)) "
        "     as diag_phys_pccm,   "
        "  sum(cast(diagnosis_practitioner_specialty_description = 'NEPHROLOGY' as integer)) "
        "     as diag_phys_nph,   "
        "  sum(cast(diagnosis_practitioner_specialty_description = 'PEDIATRIC RADIOLOGY' as integer)) "
        "     as diag_phys_pr,   "
        "  sum(cast(diagnosis_practitioner_specialty_description = 'PEDIATRICS' as integer)) "
        "     as diag_phys_pd   "
        " from diagnosis_feats  "
        " group by patient_id ")
    data_diag_spec.set_index('patient_id', inplace=True)
    data_diag_spec = data_diag_spec.ix[util.load_data('data_all_out').index, :].copy()
    data_diag_spec.fillna(0, inplace=True)
    util.save_data(data_diag_spec=data_diag_spec)

    util.get_log().info('   diagnosis_code_prefix likelihood...')
    data_diag_code_pref_avg = util.calc_likelihood_db(table='diagnosis_feats', fields=['diagnosis_code_prefix'])
    util.save_data(data_diag_code_pref_avg=data_diag_code_pref_avg)

    util.get_log().info('   diagnosis_code_prefix_x_primary_physician_role  likelihood...')
    data_diag_cod_pref_phys_avg = util.calc_likelihood_db(
        table='diagnosis_feats', fields=['diagnosis_code_prefix', 'primary_physician_role'])
    util.save_data(data_diag_cod_pref_phys_avg=data_diag_cod_pref_phys_avg)

    util.get_log().info('   diagnosis_code_prefix_x_diagnosis_practitioner_specialty_code  likelihood...')
    data_diag_cod_pref_diag_spec = util.calc_likelihood_db(
        table='diagnosis_feats', fields=['diagnosis_code_prefix', 'diagnosis_practitioner_specialty_code'])
    util.save_data(data_diag_cod_pref_diag_spec=data_diag_cod_pref_diag_spec)

    util.get_log().info('   diagnosis_code likelihood...')
    data_diag_code_avg = util.calc_likelihood_db(table='diagnosis_feats', fields=['diagnosis_code'],
                                                 prev_max_vals=3)
    util.save_data(data_diag_code_avg=data_diag_code_avg)

    util.get_log().info('   diagnosis_code_x_primary_physician_role last 5y likelihood...')
    data_diag_cod_phys_avg_l5 = util.calc_likelihood_db(table='diagnosis_feats',
                                                        fields=['diagnosis_code', 'primary_physician_role'],
                                                        where1="where diagnosis_date >= '201001'",
                                                        where2="where diagnosis_date >= '201001'")
    data_diag_cod_phys_avg_l5.columns = 'last5_' + data_diag_cod_phys_avg_l5.columns
    util.save_data(data_diag_cod_phys_avg_l5=data_diag_cod_phys_avg_l5)

    util.get_log().info('   diagnosis_code_x_primary_physician_role last 3y likelihood...')
    data_diag_cod_phys_avg_l3 = util.calc_likelihood_db(table='diagnosis_feats',
                                                        fields=['diagnosis_code', 'primary_physician_role'],
                                                        where1="where diagnosis_date >= '201201'",
                                                        where2="where diagnosis_date >= '201201'")
    data_diag_cod_phys_avg_l3.columns = 'last3_' + data_diag_cod_phys_avg_l3.columns
    util.save_data(data_diag_cod_phys_avg_l3=data_diag_cod_phys_avg_l3)


    util.get_log().info('   diagnosis_code_x_primary_physician_role  likelihood...')
    data_diag_cod_phys_avg = util.calc_likelihood_db(
        table='diagnosis_feats', fields=['diagnosis_code', 'primary_physician_role'], prev_max_vals=3)
    util.save_data(data_diag_cod_phys_avg=data_diag_cod_phys_avg)

    util.get_log().info('   diagnosis_code_x_diagnosis_practitioner_specialty_code  likelihood...')
    data_diag_cod_diag_spec = util.calc_likelihood_db(
        table='diagnosis_feats', fields=['diagnosis_code', 'diagnosis_practitioner_specialty_code'],
        prev_max_vals=3)
    util.save_data(data_diag_cod_diag_spec=data_diag_cod_diag_spec)

    util.get_log().info('   diagnosis_code_x_ethinicity  likelihood...')
    data_diag_cod_eth = util.calc_likelihood_db(
        table='diagnosis_feats', fields=['diagnosis_code', 'ethinicity'])
    util.save_data(data_diag_cod_eth=data_diag_cod_eth)

    util.get_log().info('   diagnosis_practitioner_id likelihood...')
    data_diag_prat_id_avg = util.calc_likelihood_db(table='diagnosis_feats', fields=['diagnosis_practitioner_id'],
                                                    prev_max_vals=3)
    util.save_data(data_diag_prat_id_avg=data_diag_prat_id_avg)

    util.get_log().info('   diagnosis_code max val...')
    data_diag_cod_val = util.calc_max_val_value_db(table='diagnosis_feats', fields=['diagnosis_code'],
                                                   prev_max_vals=5)
    util.save_data(data_diag_cod_val=data_diag_cod_val)

    util.get_log().info('   diagnosis_code feats2 likelihood...')
    data_diag2_code_avg = util.calc_likelihood_db(table='diagnosis_feats2', fields=['diagnosis_code'],
                                                  prev_max_vals=3)
    data_diag2_code_avg.columns = 'diag2_' + data_diag2_code_avg.columns
    util.save_data(data_diag2_code_avg=data_diag2_code_avg)

    util.get_log().info('   diagnosis_code_x_primary_physician_role  feats2 likelihood...')
    data_diag2_cod_phys_avg = util.calc_likelihood_db(
        table='diagnosis_feats2', fields=['diagnosis_code', 'primary_physician_role'], prev_max_vals=3)
    data_diag2_cod_phys_avg.columns = 'diag2_' + data_diag2_cod_phys_avg.columns
    util.save_data(data_diag2_cod_phys_avg=data_diag2_cod_phys_avg)

    util.get_log().info('   diagnosis_code_x_primary_physician_role_x_proc_claim_line_diff likelihood...')
    data_diag3_cod_phys_cd_avg = util.calc_likelihood_db(
        table='diagnosis_head2', fields=['diagnosis_code', 'primary_physician_role', 'proc_claim_line_diff'],
        prev_max_vals=1)
    data_diag3_cod_phys_cd_avg.columns = 'diag3_' + data_diag3_cod_phys_cd_avg.columns
    util.save_data(data_diag3_cod_phys_cd_avg=data_diag3_cod_phys_cd_avg)

    util.get_log().info('   diagnosis_code_x_primary_physician_role_x_diag_count_claim likelihood...')
    data_diag3_cod_phys_dc_avg = util.calc_likelihood_db(
        table='diagnosis_head2', fields=['diagnosis_code', 'primary_physician_role', 'diag_count_claim'])
    data_diag3_cod_phys_dc_avg.columns = 'diag3_' + data_diag3_cod_phys_dc_avg.columns
    util.save_data(data_diag3_cod_phys_dc_avg=data_diag3_cod_phys_dc_avg)

    util.get_log().info('   diagnosis_code_x_primary_physician_role_x_diag_count_claim_x_proc_claim_line_diff '
                        'likelihood...')
    data_diag3_cod_phys_dc_cd_avg = util.calc_likelihood_db(
        table='diagnosis_head2', fields=['diagnosis_code', 'primary_physician_role', 'diag_count_claim',
                                         'proc_claim_line_diff'])
    data_diag3_cod_phys_dc_cd_avg.columns = 'diag3_' + data_diag3_cod_phys_dc_cd_avg.columns
    util.save_data(data_diag3_cod_phys_dc_cd_avg=data_diag3_cod_phys_dc_cd_avg)

    util.get_log().info('   diag_count_claim_x_proc_claim_line_diff likelihood...')
    data_diag3_cd_dc_avg = util.calc_likelihood_db(
        table='diagnosis_head2', fields=['diag_count_claim', 'proc_claim_line_diff', 'primary_physician_role'],
        prev_max_vals=1)
    data_diag3_cd_dc_avg.columns = 'diag3_' + data_diag3_cd_dc_avg.columns
    util.save_data(data_diag3_cd_dc_avg=data_diag3_cd_dc_avg)

    util.get_log().info('   diagnosis without procedures...')
    data_diag3_no_proc = util.load_from_db(
        "select "
        "  patient_id,  "
        "  count(distinct claim_id) diag_no_proc   "
        " from diagnosis_head2  "
        " where proc_count_claim_line = '-1'  "
        " group by patient_id ")
    data_diag3_no_proc.set_index('patient_id', inplace=True)
    data_diag3_no_proc = data_diag3_no_proc.ix[util.load_data('data_all_out').index, :].copy()
    data_diag3_no_proc.fillna(0, inplace=True)
    util.save_data(data_diag3_no_proc=data_diag3_no_proc)
    del data_diag3_no_proc

    util.get_log().info('   diag_count_claim_x_primary_physician_role likelihood...')
    data_diag3_dc_phys_avg = util.calc_likelihood_db(
        table='diagnosis_head2', fields=['diag_count_claim', 'primary_physician_role'])
    data_diag3_dc_phys_avg.columns = 'diag3_' + data_diag3_dc_phys_avg.columns
    util.save_data(data_diag3_dc_phys_avg=data_diag3_dc_phys_avg)

    util.get_log().info('   proc_claim_line_diff_x_primary_physician_role likelihood...')
    data_diag3_cd_phys_avg = util.calc_likelihood_db(
        table='diagnosis_head2', fields=['proc_claim_line_diff', 'primary_physician_role'])
    data_diag3_cd_phys_avg.columns = 'diag3_' + data_diag3_cd_phys_avg.columns
    util.save_data(data_diag3_cd_phys_avg=data_diag3_cd_phys_avg)

    util.get_log().info('   code x physician role same state likelihood...')
    data_diag4_cod_phys_avg = util.calc_likelihood_db(
        table='diagnosis_feats_same_state', fields=['diagnosis_code', 'primary_physician_role'])
    data_diag4_cod_phys_avg.columns = 'diag4_' + data_diag4_cod_phys_avg.columns
    util.save_data(data_diag4_cod_phys_avg=data_diag4_cod_phys_avg)

    util.toc()


################################################################
# procedures data
################################################################
def load_procedure_data():

    util.get_log().info('Loading procedure data...')
    util.tic()

    util.get_log().info('   procedure base data...')
    data_proc_base = util.load_from_db(
        "select "
        "  patient_id,  "
        "  count(*) proc_count  "
        " from procedure_head  "
        " group by patient_id ")
    data_proc_base.set_index('patient_id', inplace=True)
    data_proc_base = data_proc_base.ix[util.load_data('data_all_out').index, :].copy()
    data_proc_base.fillna(0, inplace=True)
    util.save_data(data_proc_base=data_proc_base)

    util.get_log().info('   procedure_code_x_procedure_primary_physician_rol discretized likelihood...')
    data_proc_cod_phys_cnt = util.calc_likelihood_count_db(
        table='procedure_head', fields=['procedure_code', 'procedure_primary_physician_role'],
        intervals=np.array([0.0] + list(np.linspace(0.0, 1.0, num=21)[2:])))
    util.save_data(data_proc_cod_phys_cnt=data_proc_cod_phys_cnt)

    util.get_log().info('   procedure_code likelihood...')
    data_proc_cod_avg = util.calc_likelihood_db(
        table='procedure_head', fields=['procedure_code'], prev_max_vals=2)
    util.save_data(data_proc_cod_avg=data_proc_cod_avg)

    util.get_log().info('   procedure_code_x_procedure_primary_physician_role likelihood...')
    data_proc_cod_phys_avg = util.calc_likelihood_db(
        table='procedure_head', fields=['procedure_code', 'procedure_primary_physician_role'], prev_max_vals=2)
    util.save_data(data_proc_cod_phys_avg=data_proc_cod_phys_avg)

    util.get_log().info('   procedure_primary_practitioner_id likelihood...')
    data_proc_pract_avg = util.calc_likelihood_db(
        table='procedure_head', fields=['procedure_primary_practitioner_id'], prev_max_vals=2)
    util.save_data(data_proc_pract_avg=data_proc_pract_avg)

    util.get_log().info('   procedure_attending_practitioner_id likelihood...')
    data_proc_pract_att_avg = util.calc_likelihood_db(
        table='procedure_head', fields=['procedure_attending_practitioner_id'])
    util.save_data(data_proc_pract_att_avg=data_proc_pract_att_avg)

    util.get_log().info('   procedure_referring_practitioner_id likelihood...')
    data_proc_pract_ref_avg = util.calc_likelihood_db(
        table='procedure_head', fields=['procedure_referring_practitioner_id'])
    util.save_data(data_proc_pract_ref_avg=data_proc_pract_ref_avg)

    util.get_log().info('   procedure_rendering_practitioner_id likelihood...')
    data_proc_pract_rend_avg = util.calc_likelihood_db(
        table='procedure_head', fields=['procedure_rendering_practitioner_id'])
    util.save_data(data_proc_pract_rend_avg=data_proc_pract_rend_avg)

    util.get_log().info('   procedure_operating_practitioner_id likelihood...')
    data_proc_pract_op_avg = util.calc_likelihood_db(
        table='procedure_head', fields=['procedure_operating_practitioner_id'])
    util.save_data(data_proc_pract_op_avg=data_proc_pract_op_avg)

    util.get_log().info('   proc2 procedure_code likelihood...')
    data_proc2_cod_avg = util.calc_likelihood_db(
        table='procedure_head2', fields=['procedure_code'], prev_max_vals=2)
    data_proc2_cod_avg.columns = 'proc2_' + data_proc2_cod_avg.columns
    util.save_data(data_proc2_cod_avg=data_proc2_cod_avg)

    util.get_log().info('   proc2 procedure_primary_practitioner_id likelihood...')
    data_proc2_pract_avg = util.calc_likelihood_db(
        table='procedure_head2', fields=['procedure_primary_practitioner_id'])
    data_proc2_pract_avg.columns = 'proc2_' + data_proc2_pract_avg.columns
    util.save_data(data_proc2_pract_avg=data_proc2_pract_avg)

    util.get_log().info('   procedure_code_x_proc_claim_line_diff likelihood...')
    data_proc3_cod_cd_avg = util.calc_likelihood_db(
        table='procedure_head3', fields=['procedure_code', 'proc_claim_line_diff'],
        prev_max_vals=1)
    data_proc3_cod_cd_avg.columns = 'proc3_' + data_proc3_cod_cd_avg.columns
    util.save_data(data_proc3_cod_cd_avg=data_proc3_cod_cd_avg)

    util.get_log().info('   procedure_code_x_diag_count_claim likelihood...')
    data_proc3_cod_dc_avg = util.calc_likelihood_db(
        table='procedure_head3', fields=['procedure_code', 'diag_count_claim'])
    data_proc3_cod_dc_avg.columns = 'proc3_' + data_proc3_cod_dc_avg.columns
    util.save_data(data_proc3_cod_dc_avg=data_proc3_cod_dc_avg)

    util.get_log().info('   procedure_code_x_diag_count_claim_x_proc_claim_line_diff likelihood...')
    data_proc3_cod_dc_cd_avg = util.calc_likelihood_db(
        table='procedure_head3', fields=['procedure_code', 'diag_count_claim', 'proc_claim_line_diff'])
    data_proc3_cod_dc_cd_avg.columns = 'proc3_' + data_proc3_cod_dc_cd_avg.columns
    util.save_data(data_proc3_cod_dc_cd_avg=data_proc3_cod_dc_cd_avg)

    util.get_log().info('   diag_count_claim_x_proc_claim_line_diff_x_primary_practitioner_id likelihood...')
    data_proc3_cd_dc_pract_avg = util.calc_likelihood_db(
        table='procedure_head3', fields=['diag_count_claim', 'proc_claim_line_diff',
                                         'procedure_primary_practitioner_id'],
        prev_max_vals=2)
    data_proc3_cd_dc_pract_avg.columns = 'proc3_' + data_proc3_cd_dc_pract_avg.columns
    util.save_data(data_proc3_cd_dc_pract_avg=data_proc3_cd_dc_pract_avg)

    util.get_log().info('   procedures without diagnosis...')
    data_proc3_no_proc = util.load_from_db(
        "select "
        "  patient_id,  "
        "  count(distinct claim_id) proc_no_diag   "
        " from procedure_head3  "
        " where diag_count_claim = '-1'  "
        " group by patient_id ")
    data_proc3_no_proc.set_index('patient_id', inplace=True)
    data_proc3_no_proc = data_proc3_no_proc.ix[util.load_data('data_all_out').index, :].copy()
    data_proc3_no_proc.fillna(0, inplace=True)
    util.save_data(data_proc3_no_proc=data_proc3_no_proc)
    del data_proc3_no_proc

    util.get_log().info('   diag_count_claim_x_procedure_primary_practitioner_id likelihood...')
    data_proc3_dc_pract_avg = util.calc_likelihood_db(
        table='procedure_head3', fields=['diag_count_claim', 'procedure_primary_practitioner_id'])
    data_proc3_dc_pract_avg.columns = 'proc3_' + data_proc3_dc_pract_avg.columns
    util.save_data(data_proc3_dc_pract_avg=data_proc3_dc_pract_avg)

    util.get_log().info('   proc_claim_line_diff_x_procedure_primary_practitioner_id likelihood...')
    data_proc3_cd_pract_avg = util.calc_likelihood_db(
        table='procedure_head3', fields=['proc_claim_line_diff', 'procedure_primary_practitioner_id'])
    data_proc3_cd_pract_avg.columns = 'proc3_' + data_proc3_cd_pract_avg.columns
    util.save_data(data_proc3_cd_pract_avg=data_proc3_cd_pract_avg)

    util.toc()


################################################################
# prescription data
################################################################
def load_prescription_data():

    util.get_log().info('Loading prescription data...')
    util.tic()

    util.get_log().info('   prescription base data...')
    data_presc_base = util.load_from_db(
        "select "
        "  patient_id,  "
        "  count(*) presc_count,  "
        "  avg(days_supply) presc_day_supply "
        " from prescription_feats  "
        " group by patient_id ")
    data_presc_base.set_index('patient_id', inplace=True)
    data_presc_base = data_presc_base.ix[util.load_data('data_all_out').index, :].copy()
    data_presc_base.fillna(0, inplace=True)
    util.save_data(data_presc_base=data_presc_base)

    util.get_log().info('   drug_name likelihood...')
    data_presc_drug_nam_avg = util.calc_likelihood_db(
        table='prescription_feats', fields=['drug_name'], prev_max_vals=3)
    util.save_data(data_presc_drug_nam_avg=data_presc_drug_nam_avg)

    util.get_log().info('   drug_generic_name likelihood...')
    data_presc_drug_gnam_avg = util.calc_likelihood_db(
        table='prescription_feats', fields=['drug_generic_name'], prev_max_vals=3)
    util.save_data(data_presc_drug_gnam_avg=data_presc_drug_gnam_avg)

    util.get_log().info('   drug_name_x_prescription_practitioner_specialty_code likelihood...')
    data_presc_drug_phys_avg = util.calc_likelihood_db(
        table='prescription_feats', fields=['drug_name', 'prescription_practitioner_specialty_code'], prev_max_vals=3)
    util.save_data(data_presc_drug_phys_avg=data_presc_drug_phys_avg)

    util.get_log().info('   bb_usc_code likelihood...')
    data_presc_cod_avg = util.calc_likelihood_db(
        table='prescription_feats', fields=['bb_usc_code'], prev_max_vals=3)
    util.save_data(data_presc_cod_avg=data_presc_cod_avg)

    util.get_log().info('   bb_usc_code_x_prescription_practitioner_specialty_code likelihood...')
    data_presc_cod_phys_avg = util.calc_likelihood_db(
        table='prescription_feats', fields=['bb_usc_code', 'prescription_practitioner_specialty_code'],
        prev_max_vals=3)
    util.save_data(data_presc_cod_phys_avg=data_presc_cod_phys_avg)

    util.get_log().info('   prescription_practitioner_id likelihood...')
    data_presc_pract_avg = util.calc_likelihood_db(
        table='prescription_feats', fields=['prescription_practitioner_id'], prev_max_vals=3)
    util.save_data(data_presc_pract_avg=data_presc_pract_avg)

    util.get_log().info('   payment_type likelihood...')
    data_presc_pay_avg = util.calc_likelihood_db(
        table='prescription_feats', fields=['payment_type'])
    util.save_data(data_presc_pay_avg=data_presc_pay_avg)

    util.get_log().info('   drug_strength likelihood...')
    data_presc_drug_str_avg = util.calc_likelihood_db(
        table='prescription_feats', fields=['drug_strength'], aggr='avg')
    util.save_data(data_presc_drug_str_avg=data_presc_drug_str_avg)

    util.get_log().info('   drug_id likelihood...')
    data_presc_drug_id_avg = util.calc_likelihood_db(
        table='prescription_feats', fields=['drug_id'], aggr='avg')
    util.save_data(data_presc_drug_id_avg=data_presc_drug_id_avg)

    util.get_log().info('   manufacturer likelihood...')
    data_presc_manufact_avg = util.calc_likelihood_db(
        table='prescription_feats', fields=['manufacturer'])
    util.save_data(data_presc_manufact_avg=data_presc_manufact_avg)

    util.toc()


################################################################
# surgery data
################################################################
def load_surgery_data():

    util.get_log().info('Loading surgery data...')
    util.tic()

    util.get_log().info('   surgery base data...')
    data_surg_base = util.load_from_db(
        "select "
        "  patient_id,  "
        "  count(*) surg_count  "
        " from surgical_head  "
        " group by patient_id ")
    data_surg_base.set_index('patient_id', inplace=True)
    data_surg_base = data_surg_base.ix[util.load_data('data_all_out').index, :].copy()
    data_surg_base.fillna(0, inplace=True)
    util.save_data(data_surg_base=data_surg_base)

    util.get_log().info('   procedure_type_code likelihood...')
    data_surg_typ_avg = util.calc_likelihood_db(
        table='surgical_head', fields=['procedure_type_code'], prev_max_vals=3)
    util.save_data(data_surg_typ_avg=data_surg_typ_avg)

    util.get_log().info('   surgical_code likelihood...')
    data_surg_cod_avg = util.calc_likelihood_db(
        table='surgical_head', fields=['surgical_code'], prev_max_vals=3)
    util.save_data(data_surg_cod_avg=data_surg_cod_avg)

    util.get_log().info('   surgical_practitioner_id likelihood...')
    data_surg_pract_avg = util.calc_likelihood_db(
        table='surgical_head', fields=['surgical_practitioner_id'], prev_max_vals=3)
    util.save_data(data_surg_pract_avg=data_surg_pract_avg)

    util.get_log().info('   surgical_primary_physician_role likelihood...')
    data_surg_phys_role_avg = util.calc_likelihood_db(
        table='surgical_head', fields=['surgical_primary_physician_role'], prev_max_vals=3)
    util.save_data(data_surg_phys_role_avg=data_surg_phys_role_avg)

    util.toc()


################################################################
# all pratictioner data
################################################################
def load_pratictioner_data():

    util.get_log().info('Loading pratictioner data...')
    util.tic()

    util.get_log().info('   creating pratictioner_head view')
    util.exec_cmd_db(
        " CREATE table pratictioner_head "
        " AS  "
        "  SELECT t1.patient_id, t1.all_practitioner_id, t1.proc_claim_line_diff, t1.diag_count_claim, "
        "         t2.all_pract_state, t2.all_pract_specialty_code, "
        "         t2.all_pract_cbsa "
        "    FROM ((((((( SELECT t1.patient_id, t1.primary_practitioner_id AS all_practitioner_id, "
        " 					t2.proc_claim_line_diff, "
        " 					t3.diag_count_claim "
        "            FROM diagnosis_head t1 "
        " 		   	inner join missed_claims_disc t2 on t2.claim_id = t1.claim_id "
        " 			inner join diagnosis_claim_count_disc t3 on t3.claim_id = t1.claim_id "
        "           WHERE NOT t1.primary_practitioner_id IS NULL "
        " UNION  "
        "          SELECT t1.patient_id, t1.procedure_primary_practitioner_id AS all_practitioner_id, "
        " 					t2.proc_claim_line_diff, "
        " 					t3.diag_count_claim "
        "            FROM procedure_head t1 "
        " 		   	inner join missed_claims_disc t2 on t2.claim_id = t1.claim_id "
        " 			inner join diagnosis_claim_count_disc t3 on t3.claim_id = t1.claim_id "
        "           WHERE NOT t1.procedure_primary_practitioner_id IS NULL) "
        " UNION  "
        "          SELECT t1.patient_id, t1.procedure_attending_practitioner_id AS all_practitioner_id, "
        " 					t2.proc_claim_line_diff, "
        " 					t3.diag_count_claim "
        "            FROM procedure_head t1 "
        " 		   	inner join missed_claims_disc t2 on t2.claim_id = t1.claim_id "
        " 			inner join diagnosis_claim_count_disc t3 on t3.claim_id = t1.claim_id "
        "           WHERE NOT t1.procedure_attending_practitioner_id IS NULL) "
        " UNION  "
        "          SELECT t1.patient_id, t1.procedure_referring_practitioner_id AS all_practitioner_id, "
        " 					t2.proc_claim_line_diff, "
        " 					t3.diag_count_claim "
        "            FROM procedure_head t1 "
        " 		   	inner join missed_claims_disc t2 on t2.claim_id = t1.claim_id "
        " 			inner join diagnosis_claim_count_disc t3 on t3.claim_id = t1.claim_id "
        "           WHERE NOT t1.procedure_referring_practitioner_id IS NULL) "
        " UNION  "
        "          SELECT t1.patient_id, t1.procedure_ordering_practitioner_id AS all_practitioner_id, "
        " 					t2.proc_claim_line_diff, "
        " 					t3.diag_count_claim "
        "            FROM procedure_head t1 "
        " 		   	inner join missed_claims_disc t2 on t2.claim_id = t1.claim_id "
        " 			inner join diagnosis_claim_count_disc t3 on t3.claim_id = t1.claim_id "
        "           WHERE NOT t1.procedure_ordering_practitioner_id IS NULL) "
        " UNION  "
        "          SELECT t1.patient_id, t1.procedure_operating_practitioner_id AS all_practitioner_id, "
        " 					t2.proc_claim_line_diff, "
        " 					t3.diag_count_claim "
        "            FROM procedure_head t1 "
        " 		   	inner join missed_claims_disc t2 on t2.claim_id = t1.claim_id "
        " 			inner join diagnosis_claim_count_disc t3 on t3.claim_id = t1.claim_id "
        "           WHERE NOT t1.procedure_operating_practitioner_id IS NULL) "
        " UNION  "
        "          SELECT t1.patient_id, t1.prescription_practitioner_id AS all_practitioner_id, "
        " 					nvl(t2.proc_claim_line_diff, '-2') as proc_claim_line_diff, "
        " 					nvl(t3.diag_count_claim, '-2') as diag_count_claim "
        "            FROM prescription_head t1 "
        " 		   	left join missed_claims_disc t2 on t2.claim_id = t1.claim_id "
        " 			left join diagnosis_claim_count_disc t3 on t3.claim_id = t1.claim_id "
        "           WHERE NOT t1.prescription_practitioner_id IS NULL) "
        " UNION  "
        "          SELECT t1.patient_id, t1.surgical_practitioner_id AS all_practitioner_id, "
        " 					nvl(t2.proc_claim_line_diff, '-2') as proc_claim_line_diff, "
        " 					nvl(t3.diag_count_claim, '-2') as diag_count_claim "
        "            FROM surgical_head t1 "
        " 		   	left join missed_claims_disc t2 on t2.claim_id = t1.claim_id "
        " 			left join diagnosis_claim_count_disc t3 on t3.claim_id = t1.claim_id "
        "           WHERE NOT t1.surgical_practitioner_id IS NULL) t1 "
        "    JOIN ( "
        "           SELECT physicians.practitioner_id AS all_practitioner_id, physicians.state AS all_pract_state, "
        "                  physicians.specialty_code AS all_pract_specialty_code, "
        "                  physicians.specialty_description AS all_pract_specialty_description, "
        "                  physicians.cbsa AS all_pract_cbsa "
        "            FROM physicians) t2 ON t1.all_practitioner_id = t2.all_practitioner_id "
    )

    util.get_log().info('   all_pract_specialty_code likelihood...')
    data_pratic_spect_avg = util.calc_likelihood_db(
        table='pratictioner_head', fields=['all_pract_specialty_code'])
    util.save_data(data_pratic_spect_avg=data_pratic_spect_avg)

    util.get_log().info('   all_pract_state likelihood...')
    data_pratic_state_avg = util.calc_likelihood_db(
        table='pratictioner_head', fields=['all_pract_state'])
    util.save_data(data_pratic_state_avg=data_pratic_state_avg)

    util.get_log().info('   "specialty_code x claim diff x diag count x state" likelihood...')
    data_pratic_spect_st_cd_dc_avg = util.calc_likelihood_db(
        table='pratictioner_head', fields=['all_pract_specialty_code', 'proc_claim_line_diff',
                                           'diag_count_claim', 'all_pract_state'])
    util.save_data(data_pratic_spect_st_cd_dc_avg=data_pratic_spect_st_cd_dc_avg)

    util.get_log().info('   "practitioner" likelihood...')
    data_pratic_cod_avg = util.calc_likelihood_db(
        table='pratictioner_head', fields=['all_practitioner_id'], prev_max_vals=2)
    util.save_data(data_pratic_cod_avg=data_pratic_cod_avg)

    util.get_log().info('   "practitioner x claim diff" likelihood...')
    data_pratic_cod_cd_avg = util.calc_likelihood_db(
        table='pratictioner_head', fields=['all_practitioner_id', 'proc_claim_line_diff'], prev_max_vals=2)
    util.save_data(data_pratic_cod_cd_avg=data_pratic_cod_cd_avg)

    util.toc()


################################################################
# combined data
################################################################
def load_combined_data():

    util.get_log().info('Loading combined data...')
    util.tic()

    util.get_log().info('   creating diagnosis_pairs_unique view...')
    util.exec_cmd_db("create or replace view diagnosis_pairs_unique as "
                     "select * from diagnosis_pairs where diagnosis_code1 >= diagnosis_code2")

    util.get_log().info('   diagnosis_pairs likelihood...')
    data_diag_pair = util.calc_likelihood_db(
        table='diagnosis_pairs_unique', fields=['diagnosis_code1', 'diagnosis_code2'])
    util.save_data(data_diag_pair=data_diag_pair)

    util.get_log().info('   diagnosis_prescription_link likelihood...')
    data_diag_presc_dd = util.calc_likelihood_db(
        table='diagnosis_prescription_link', fields=['diagnosis_code', 'drug_generic_name'])
    util.save_data(data_diag_presc_dd=data_diag_presc_dd)

    util.get_log().info('   diagnosis_surgical_link likelihood...')
    data_diag_surg_ds = util.calc_likelihood_db(
        table='diagnosis_surgical_link', fields=['diagnosis_code', 'surgical_code'])
    util.save_data(data_diag_surg_ds=data_diag_surg_ds)

    util.get_log().info('   procedure_surgical_link likelihood...')
    data_proc_surg_ps = util.calc_likelihood_db(
        table='procedure_surgical_link', fields=['procedure_code', 'surgical_code'])
    util.save_data(data_proc_surg_ps=data_proc_surg_ps)

    util.get_log().info('   diagnosis_procedure_surgical_link likelihood...')
    data_diag_proc_surg_dps = util.calc_likelihood_db(
        table='diagnosis_procedure_surgical_link', fields=['diagnosis_code', 'procedure_code', 'surgical_code'])
    util.save_data(data_diag_proc_surg_dps=data_diag_proc_surg_dps)

    util.get_log().info('   diagnosis_procedure_link likelihood...')
    data_diag_proc_dp = util.calc_likelihood_db(
        table='diagnosis_procedure_link', fields=['diagnosis_code', 'procedure_code'])
    util.save_data(data_diag_proc_dp=data_diag_proc_dp)

    util.get_log().info('   diagnosis_procedure_link2 charge data...')
    data_diag_proc_lnk2_charge = util.load_from_db(
        "select patient_id, max(charge_amount) diag_proc_lnk_charge from "
        "(select "
        "  patient_id,  "
        "  sum(charge_amount) charge_amount  "
        " from diagnosis_procedure_link2  "
        " group by patient_id, diagnosis_code)"
        " group by patient_id")
    data_diag_proc_lnk2_charge.set_index('patient_id', inplace=True)
    data_diag_proc_lnk2_charge = data_diag_proc_lnk2_charge.ix[util.load_data('data_all_out').index, :].copy()
    data_diag_proc_lnk2_charge.fillna(0, inplace=True)
    util.save_data(data_diag_proc_lnk2_charge=data_diag_proc_lnk2_charge)

    util.get_log().info('   diagnosis_procedure_link2 likelihood...')
    data_diag_proc_typ_dp = util.calc_likelihood_db(
        table='diagnosis_procedure_link2', fields=['diagnosis_code', 'procedure_code', 'plan_type'])
    util.save_data(data_diag_proc_typ_dp=data_diag_proc_typ_dp)

    util.get_log().info('   diagnosis_procedure_link4 "code x code" likelihood...')
    data_diag_proc2_cod_cod = util.calc_likelihood_db(
        table='diagnosis_procedure_link4', fields=['diagnosis_code', 'procedure_code',
                                                   'proc_claim_line_diff', 'diag_count_claim'])
    util.save_data(data_diag_proc2_cod_cod=data_diag_proc2_cod_cod)

    util.get_log().info('   diagnosis_procedure_link4 "plan x code" likelihood...')
    data_diag_proc2_plan_cod = util.calc_likelihood_db(
        table='diagnosis_procedure_link4', fields=['diagnosis_code', 'plan_type',
                                                   'proc_claim_line_diff', 'diag_count_claim'])
    util.save_data(data_diag_proc2_plan_cod=data_diag_proc2_plan_cod)

    util.get_log().info('   diagnosis_procedure_link4 "code x code x phys_role" likelihood...')
    data_diag_proc2_cod_cod_phys = util.calc_likelihood_db(
        table='diagnosis_procedure_link4', fields=['diagnosis_code', 'procedure_code', 'primary_physician_role',
                                                   'proc_claim_line_diff', 'diag_count_claim'])
    util.save_data(data_diag_proc2_cod_cod_phys=data_diag_proc2_cod_cod_phys)

    util.get_log().info('   diagnosis_procedure_link4 "plan x code x phys_role" likelihood...')
    data_diag_proc2_plan_cod_phys = util.calc_likelihood_db(
        table='diagnosis_procedure_link4', fields=['diagnosis_code', 'plan_type', 'primary_physician_role',
                                                   'proc_claim_line_diff', 'diag_count_claim'])
    util.save_data(data_diag_proc2_plan_cod_phys=data_diag_proc2_plan_cod_phys)

    util.get_log().info('   create claim_type_head table...')
    util.exec_cmd_db(
        " create table claim_type_head as "
        " select  "
        "   t1.claim_id,  "
        "    CASE "
        "         WHEN t3.patient_id is null and t1.proc_claim_line_diff <> '-1' "
        "              and t2.diag_count_claim <> '-1' THEN 'dp' "
        "         WHEN t3.patient_id is null and t2.diag_count_claim <> '-1' THEN 'd' "
        "         WHEN t3.patient_id is null and t1.proc_claim_line_diff <> '-1' THEN 'p' "
        "          "
        "         WHEN not t3.patient_id is null and nvl(t1.proc_claim_line_diff, '-1') <> '-1' "
        "              and nvl(t2.diag_count_claim, '-1') <> '-1' THEN 'dps' "
        "         WHEN not t3.patient_id is null and nvl(t1.proc_claim_line_diff, '-1')  = '-1' "
        "              and nvl(t2.diag_count_claim, '-1')  = '-1' THEN 's' "
        "         WHEN not t3.patient_id is null and nvl(t1.proc_claim_line_diff, '-1') <> '-1' THEN 'ps' "
        "         WHEN not t3.patient_id is null and nvl(t2.diag_count_claim, '-1') <> '-1' THEN 'ds' "
        "         ELSE 'other' "
        "     END AS claim_type "
        " from  "
        "   missed_claims t1 "
        "   inner join diagnosis_claim_count t2 on t1.claim_id = t2.claim_id "
        "   left outer join ( "
        "     select patient_id, claim_id "
        "     from surgical_head "
        "     group by patient_id, claim_id) t3 on t1.claim_id = t3.claim_id  "
    )

    util.get_log().info('   create patient_claim table...')
    util.exec_cmd_db(
        " create table patient_claim as "
        " select t1.*, t2.claim_type from ("
        " select patient_id, claim_id "
        " from ( "
        " 	select patient_id, claim_id  "
        " 	from diagnosis_head "
        " 	group by patient_id, claim_id "
        " 	union "
        " 	select patient_id, claim_id  "
        " 	from procedure_head "
        " 	group by patient_id, claim_id "
        " 	union "
        " 	select patient_id, claim_id  "
        " 	from surgical_head "
        " 	group by patient_id, claim_id "
        " ) "
        " group by patient_id, claim_id "
        ") t1 join claim_type_head t2 on t1.claim_id = t2.claim_id"
    )

    util.get_log().info('   patient_claim likelihood...')
    data_pat_claim = util.calc_likelihood_db(
        table='patient_claim', fields=['claim_type'])
    util.save_data(data_pat_claim=data_pat_claim)

    util.get_log().info('   patient_claim count...')
    data_pat_claim_cnt = util.load_from_db(
        "select "
        "  patient_id,  "
        "  sum(cast(claim_type = 'ds' as integer)) as pat_ct_ds  , "
        "  sum(cast(claim_type = 'd' as integer)) as pat_ct_d  , "
        "  sum(cast(claim_type = 'ps' as integer)) as pat_ct_ps  , "
        "  sum(cast(claim_type = 'dps' as integer)) as pat_ct_dps  , "
        "  sum(cast(claim_type = 'dp' as integer)) as pat_ct_dp  , "
        "  sum(cast(claim_type = 'p' as integer)) as pat_ct_p,  "
        "  count(*) as pat_ct_cnt  "
        " from patient_claim  "
        " group by patient_id ")
    data_pat_claim_cnt.set_index('patient_id', inplace=True)
    data_pat_claim_cnt = data_pat_claim_cnt.ix[util.load_data('data_all_out').index, :].copy()
    data_pat_claim_cnt.fillna(0, inplace=True)
    util.save_data(data_pat_claim_cnt=data_pat_claim_cnt)
    del data_pat_claim_cnt

    util.get_log().info('   diagnosis_head2 "code x role x type" likelihood...')
    data_diag_cod_role_typ = util.calc_likelihood_db(
        table='diagnosis_head2', fields=['diagnosis_code', 'primary_physician_role', 'pat_claim_type'])
    util.save_data(data_diag_cod_role_typ=data_diag_cod_role_typ)

    util.get_log().info('   procedure_head3 "code x type" likelihood...')
    data_proc_cod_role_typ = util.calc_likelihood_db(
        table='procedure_head3', fields=['procedure_code', 'pat_claim_type'])
    util.save_data(data_proc_cod_role_typ=data_proc_cod_role_typ)

    util.get_log().info('   surgical_head3 "code x role x type" likelihood...')
    data_surg_cod_role_typ = util.calc_likelihood_db(
        table='surgical_head3', fields=['surgical_code', 'surgical_primary_physician_role', 'pat_claim_type'])
    util.save_data(data_surg_cod_role_typ=data_surg_cod_role_typ)

    util.get_log().info('   surgical_head3 "plan type x role x type" likelihood...')
    data_surg_ptyp_role_typ = util.calc_likelihood_db(
        table='surgical_head3', fields=['surgical_plan_type', 'surgical_primary_physician_role', 'pat_claim_type'])
    util.save_data(data_surg_ptyp_role_typ=data_surg_ptyp_role_typ)

    util.toc()


################################################################
# Tree data 01
################################################################
def build_tree_01_data():
    util.get_log().info('Building tree data 01 data...')
    util.tic()

    cv_ix_all = np.sort(util.load_data('data_all_out')['cv_index'].unique())

    for k in cv_ix_all:
        util.get_log().info('   fold %d...' % k)

        data_in_tree_k_01 = util.load_data('data_patient_all')

        # activities data
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_actv', k=k)

        # dignostics data
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_diag_base', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_diag_base_desc', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_diag_spec', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_diag_code_pref_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_diag_cod_pref_phys_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_diag_cod_pref_diag_spec', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_diag_cod_eth', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_diag_code_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_diag_cod_phys_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_diag_cod_diag_spec', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_diag_prat_id_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_diag_cod_val', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_diag_code_phys_cnt', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_diag_cod_phys_avg_l3', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_diag_cod_phys_avg_l5', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_diag2_code_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_diag2_cod_phys_avg', k=k)

        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_diag3_cod_phys_cd_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_diag3_cod_phys_dc_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_diag3_cod_phys_dc_cd_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_diag3_cd_dc_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_diag3_no_proc', k=k)

        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_diag3_dc_phys_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_diag3_cd_phys_avg', k=k)

        # procedure data
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_proc_base', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_proc_cod_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_proc_cod_phys_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_proc_pract_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_proc_pract_att_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_proc_pract_ref_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_proc_pract_rend_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_proc_pract_op_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_proc_cod_phys_cnt', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_proc2_cod_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_proc2_pract_avg', k=k)

        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_proc3_cod_cd_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_proc3_cod_dc_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_proc3_cod_dc_cd_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_proc3_cd_dc_pract_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_proc3_no_proc', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_proc3_dc_pract_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_proc3_cd_pract_avg', k=k)

        # prescription data
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_presc_base', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_presc_drug_nam_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_presc_drug_gnam_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_presc_drug_phys_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_presc_cod_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_presc_cod_phys_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_presc_pract_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_presc_pay_avg', k=k)

        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_presc_drug_str_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_presc_drug_id_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_presc_manufact_avg', k=k)

        # surgical data
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_surg_base', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_surg_typ_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_surg_cod_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_surg_pract_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_surg_phys_role_avg', k=k)

        # pratictioner
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_pratic_spect_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_pratic_state_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_pratic_spect_st_cd_dc_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_pratic_cod_avg', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_pratic_cod_cd_avg', k=k)

        # combined
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_diag_pair', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_diag_presc_dd', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_diag_proc_dp', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_diag_proc_surg_dps', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_diag_surg_ds', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_proc_surg_ps', k=k)

        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_diag_proc_lnk2_charge', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_diag_proc_typ_dp', k=k)

        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_diag_proc2_cod_cod', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_diag_proc2_plan_cod', k=k)
        util.copy_all_fields_cv(to_data=data_in_tree_k_01, from_data='data_diag_proc2_plan_cod_phys', k=k)

        util.get_ordinal_recode(data_in_tree_k_01)

        data_name = ('data_in_tree_cv%d_01' % k)
        util.save_data(**{data_name: data_in_tree_k_01})

    util.toc()


def load_team_data_csv():
    util.get_log().info('Loading team data csv...')
    util.tic()

    data_all_out = util.load_data('data_all_out')
    cv_ix_all = np.sort(data_all_out['cv_index'].unique())
    patient_ix = data_all_out.index

    for k in cv_ix_all:
        util.get_log().info('   fold %d...' % k)

        util.get_log().info('       giba')
        data_in_giba = pd.read_csv(
            filepath_or_buffer=util.get_team_path('giba/train4.cv%d.csv.gz' % k), index_col='patient_id')
        data_in_giba = data_in_giba.ix[patient_ix, :].copy()
        data_in_giba.columns = 'gb_' + data_in_giba.columns
        data_name = ('data_in_giba_01_cv%d' % k)
        util.save_data(**{data_name: data_in_giba})

        util.get_log().info('       dmitry')

        dtry_suffix = "" if k == 0 else ("_fold%d" % k)
        data_in_dtry = pd.read_csv(
            filepath_or_buffer=util.get_team_path('dmitry/train_feats_dmitry%s.csv.gz' % dtry_suffix),
            index_col='patient_id')
        data_in_dtry = data_in_dtry.append(pd.read_csv(
            filepath_or_buffer=util.get_team_path('dmitry/test_feats_dmitry%s.csv.gz' % dtry_suffix),
            index_col='patient_id'))
        data_in_dtry.drop(['is_screener'], axis='columns', inplace=True)

        data_in_dtry = data_in_dtry.ix[patient_ix, :].copy()

        data_in_dtry.fillna(data_in_dtry.mean(), inplace=True)
        data_in_dtry.columns = 'dtry_' + data_in_dtry.columns

        if k > 0:
            data_in_dtry_base = util.load_data('data_in_dtry_01_cv0')
            util.copy_all_fields(to_data=data_in_dtry_base, from_data=data_in_dtry)
            data_in_dtry = data_in_dtry_base
            del data_in_dtry_base

        data_name = ('data_in_dtry_01_cv%d' % k)
        util.save_data(**{data_name: data_in_dtry})

    util.get_log().info('   kohei data...')
    data_in_kohei = pd.read_csv(
        filepath_or_buffer=util.get_team_path('kohei/kohei_v52_prediction_train.csv.gz'), index_col='patient_id')
    data_in_kohei.columns = ['predict_screener']

    random_state = np.random.RandomState(2345825)
    data_in_kohei['predict_screener'] += \
        random_state.normal(
            loc=0.0, scale=data_in_kohei['predict_screener'].std() * 0.04, size=data_in_kohei.shape[0])
    data_in_kohei['predict_screener'][data_in_kohei['predict_screener'] > 1.0] = 1.0
    data_in_kohei['predict_screener'][data_in_kohei['predict_screener'] < 0.0] = 0.0

    data_in_kohei = data_in_kohei.append(
        pd.read_csv(filepath_or_buffer=util.get_team_path('kohei/kohei_v52_prediction_test_full.csv.gz'),
                    index_col='patient_id'))
    data_in_kohei = data_in_kohei.ix[patient_ix, :].copy()
    data_in_kohei.columns = ['kh_predict_screener']

    for kh_extra in ['_num_removed_claim_proc_via_surg.csv']:
        data_in_kh_extra = pd.read_csv(filepath_or_buffer=util.get_team_path('kohei/train%s' % kh_extra),
                                       index_col='patient_id')
        data_in_kh_extra = data_in_kh_extra.append(
            pd.read_csv(filepath_or_buffer=util.get_team_path('kohei/test%s' % kh_extra),
                        index_col='patient_id'))
        data_in_kh_extra = data_in_kh_extra.ix[patient_ix, :].copy()
        util.copy_all_fields(to_data=data_in_kohei, from_data=data_in_kh_extra)

    util.round_df(data=data_in_kohei)
    util.save_data(data_in_kohei=data_in_kohei)

    util.toc()

################################################################
# execution flow
################################################################

load_indices()
load_patient_data()
load_activity_data()
load_diag_data()
load_procedure_data()
load_prescription_data()
load_surgery_data()
load_pratictioner_data()
load_combined_data()
build_tree_01_data()
load_team_data_csv()
