import xgboost as xgb
import pandas as pd

import base_util as util
import gc


def train(k, tr_ids, tst_ids):
    util.config_file_log('s01_xgb_02/m_01_k_%02d' % k)

    data_in = pd.concat([
        util.load_data('data_in_tree_cv%d_01' % k),
        util.load_data('data_in_giba_01_cv%d' % k),
        util.load_data('data_in_dtry_01_cv%d' % k),
        util.load_data('data_in_kohei')
    ], axis=1)
    data_out = util.load_data('data_all_out')

    xg_params = {
        "objective": "binary:logistic",
        "eta": 0.03,
        "max_depth": 10,
        "colsample_bytree": 0.6667,
        "min_child_weight": 2,
        "silent": 1,
        'nthread': util.get_core_count(),
        "eval_metric": 'auc',
    }

    xg_num_rounds = 700
    xg_reps = 1 if k > 0 else 3

    data_pred_tst = pd.DataFrame(data={}, index=tst_ids)

    transforms = [
        ('xgb_02', util.get_identity_transform(), 43592 + 67894 + 6678),
    ]

    for pred_ix, (pred_name, transform, seed) in enumerate(transforms):

        xg_params['seed'] = seed

        xg_data_tr = util.get_xgb_data_matrix(
            data_x=data_in.ix[tr_ids, :], data_y=transform.y(data_out.ix[tr_ids, util.TARGET_COL]))
        xg_data_tst = util.get_xgb_data_matrix(
            data_x=data_in.ix[tst_ids, :], data_y=transform.y(data_out.ix[tst_ids, util.TARGET_COL]))

        xg_feval = None  # util.get_xgb_eval_transf(transform)
        if k > 0:
            xg_evals = [(xg_data_tr, 'tr'), (xg_data_tst, 'val')]
        else:
            xg_evals = [(xg_data_tr, 'tr')]

        for ix in range(xg_reps):

            util.get_log().info('Iteration %d of %d (transformation %s, shape %s)' %
                                (1 + ix + pred_ix * len(transforms), xg_reps * len(transforms), pred_name,
                                 repr(data_in.shape)))

            xg_params['seed'] += ix
            xg_model = xgb.train(
                params=list(xg_params.items()),
                dtrain=xg_data_tr,
                num_boost_round=xg_num_rounds,
                evals=xg_evals,
                feval=xg_feval,
            )
            cur_pred = transform.pred(xg_model.predict(xg_data_tst))
            util.add_inc_pred(data_pred_df=data_pred_tst, pred=cur_pred, pred_col=pred_name, n=ix)

            util.get_xgb_score(model=xg_model)

            util.get_prediction_summary(data_pred_df=data_pred_tst)

    return data_pred_tst


util.tic()
util.get_log().info('Starting training')

pred_s01_xgb_02_tmp = None
# pred_s01_xgb_02_tmp = util.load_data('pred_s01_xgb_02_tmp')
for k_it, tr_ids_it, tst_ids_it in util.get_kfold_ids(k_list=[0, 1]):
    pred_s01_xgb_02_tmp_cur = train(k=k_it, tr_ids=tr_ids_it, tst_ids=tst_ids_it)
    if pred_s01_xgb_02_tmp is None:
        pred_s01_xgb_02_tmp = pred_s01_xgb_02_tmp_cur
    else:
        pred_s01_xgb_02_tmp = pred_s01_xgb_02_tmp.append(pred_s01_xgb_02_tmp_cur)
    util.save_data(pred_s01_xgb_02_tmp=pred_s01_xgb_02_tmp)
    del pred_s01_xgb_02_tmp_cur
    gc.collect()

util.save_data(pred_s01_xgb_02_tmp=pred_s01_xgb_02_tmp)

util.get_prediction_summary(data_pred_df='pred_s01_xgb_02_tmp')

# Prediction summary:
#                count     mean      std      min      50%      max      auc
# xgb_02 412093.000000 0.540319 0.426931 0.000251 0.526055 1.000000 0.963468

# lb: 0.96726

util.copy_saved_data(pred_s01_xgb_02='pred_s01_xgb_02_tmp')
util.get_prediction_summary(data_pred_df='pred_s01_xgb_02')
util.write_submission(name='pred_s01_xgb_02')

util.toc()
