import datetime
import gc
import json
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import time
import warnings
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK
from contextlib import contextmanager
from pandas.core.common import SettingWithCopyWarning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

from features import train_test, historical_transactions, new_merchant_transactions, additional_features


warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

FEATS_EXCLUDED = ['first_active_month', 'target', 'card_id', 'outliers',
                  'hist_purchase_date_max', 'hist_purchase_date_min', 'hist_card_id_size',
                  'new_purchase_date_max', 'new_purchase_date_min', 'new_card_id_size',
                  'OOF_PRED', 'month_0']


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')


def opt(train_df, test_df, num_folds, stratified=False, debug=False):
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=326)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=326)

    parameter_space = {
        'learning_rate': hp.quniform("learning_rate", 0.01, 0.02, 0.001),
        'subsample': hp.quniform("subsample", 0.1, 1.0, 0.01),
        'max_depth': hp.quniform("max_depth", 4, 12, 1),
        # 'top_rate': hp.quniform("top_rate", 0.01, 0.99, 0.01),
        'num_leaves': hp.quniform("num_leaves", 16, 256, 1),
        'min_child_weight': hp.quniform("min_child_weight", 1.0, 100.0, 0.1),
        "reg_alpha": hp.quniform("reg_alpha", 0, 100, 0.1),
        "reg_lambda": hp.quniform("reg_lambda", 0, 100, 0.1),
        "colsample_bytree": hp.quniform("colsample_bytree", 0.01, 1.0, 0.01),
        'min_split_gain': hp.quniform("min_split_gain", 0.01, 5.0, 0.1),
        'min_data_in_leaf': hp.quniform("min_data_in_leaf", 1, 100, 1)
    }

    opt_idx = 0
    out_dir_base = "../data/hyperopt_output_gbdt"

    def objective(opt_params):
        nonlocal opt_idx
        opt_idx += 1
        # Create arrays and dataframes to store results
        oof_preds = np.zeros(train_df.shape[0])
        sub_preds = np.zeros(test_df.shape[0])
        feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]

        # k-fold
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outliers'])):
            train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
            valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]

            # set data structure
            lgb_train = lgb.Dataset(train_x,
                                    label=train_y,
                                    free_raw_data=False)
            lgb_val = lgb.Dataset(valid_x,
                                   label=valid_y,
                                   free_raw_data=False)

            params = {
                'task': 'train',
                # 'boosting': 'goss',
                'boosting': 'gbdt',
                'objective': 'regression',
                'metric': 'rmse',
                'verbose': -1,
                'seed': int(2**n_fold),
                # 'bagging_seed': int(2**n_fold),
                # 'drop_seed': int(2**n_fold),
                'learning_rate': float(opt_params["learning_rate"]),
                'subsample': float(opt_params["subsample"]),
                'max_depth': int(opt_params["max_depth"]),
                # 'top_rate': float(opt_params["top_rate"]),
                'num_leaves': int(opt_params["num_leaves"]),
                'min_child_weight': float(opt_params["min_child_weight"]),
                # 'other_rate': 1.0 - float(opt_params["top_rate"]),
                "reg_alpha": float(opt_params["reg_alpha"]),
                "reg_lambda": float(opt_params["reg_lambda"]),
                "colsample_bytree": float(opt_params["colsample_bytree"]),
                'min_split_gain': float(opt_params["min_split_gain"]),
                'min_data_in_leaf': int(opt_params["min_data_in_leaf"])
            }

            reg = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_train, lgb_val],
                valid_names=['train', 'val'],
                num_boost_round=10000,
                early_stopping_rounds=200,
                verbose_eval=False
            )

            oof_preds[valid_idx] = reg.predict(valid_x, num_iteration=reg.best_iteration)
            sub_preds += reg.predict(test_df[feats], num_iteration=reg.best_iteration) / folds.n_splits

            del reg, train_x, train_y, valid_x, valid_y
            gc.collect()

        score = rmse(train_df['target'], oof_preds)

        print(f"==== [{opt_idx:0>4}] RMSE: {score} ====")
        out_dir = os.path.join(out_dir_base, f"{opt_idx:0>4}_{score}")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        oof_df = train_df.copy().reset_index()
        oof_df['target'] = oof_preds
        oof_df[['card_id', 'target']].to_csv(os.path.join(out_dir, f"oof.csv"), index=False)

        submission = test_df.copy().reset_index()
        submission['target'] = sub_preds
        submission[['card_id', 'target']].to_csv(os.path.join(out_dir, f"submission.csv"), index=False)

        with open(os.path.join(out_dir, "params.json"), "w") as fp:
            json.dump(params, fp, indent=2)
        return {'loss': score, 'status': STATUS_OK}

    print("====== optimize lgbm parameters ======")
    trials = hyperopt.Trials()
    best = fmin(objective, parameter_space, algo=tpe.suggest,
                max_evals=200, trials=trials, verbose=1)
    print("====== best estimate parameters ======")
    for key, val in best.items():
        print(f"    {key}: {val}")
    print("============= best score =============")
    best_score = trials.best_trial['result']['loss']
    print(best_score)
    pickle.dump(trials.trials, open(os.path.join(out_dir_base, f'trials_{best_score}.pkl', 'wb')))


def main(debug=False):
    num_rows = 10000 if debug else None

    with timer("train & test"):
        df = train_test(num_rows)
    with timer("historical transactions"):
        df = pd.merge(df, historical_transactions(num_rows), on='card_id', how='outer')
    with timer("new merchants"):
        df = pd.merge(df, new_merchant_transactions(num_rows), on='card_id', how='outer')
    with timer("additional features"):
        df = additional_features(df)
    with timer("split train & test"):
        train_df = df[df['target'].notnull()]
        test_df = df[df['target'].isnull()]
        del df
        gc.collect()
    with timer("Run LightGBM with kfold"):
        opt(train_df, test_df, num_folds=11, stratified=False, debug=debug)


if __name__ == "__main__":
    submission_file_name = "submission.csv"
    with timer("Full model run"):
        main(debug=False)
