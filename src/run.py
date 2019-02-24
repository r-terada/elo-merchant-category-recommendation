import datetime
import gc
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import time
import warnings

from contextlib import contextmanager
from pandas.core.common import SettingWithCopyWarning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

from features import train_test, historical_transactions, new_merchant_transactions, additional_features


warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

with open("../data/misc/split_corr_under_threshold_99.txt", "r") as fp:
    line = fp.read()
    features_to_be_excluded = eval(line)

FEATS_EXCLUDED = ['first_active_month', 'target', 'card_id', 'outliers',
                  'hist_purchase_date_max', 'hist_purchase_date_min', 'hist_card_id_size',
                  'new_purchase_date_max', 'new_purchase_date_min', 'new_card_id_size',
                  'OOF_PRED', 'month_0'] #  + features_to_be_excluded


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def plot_importance(feature_importance_df_, out_dir):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'lgbm_importances.png'))


def kfold_lightgbm(train_df, test_df, num_folds, out_dir_name, stratified=False, debug=False):
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=326)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=326)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]
    print(f"feats: {len(feats)}\n{feats}")
    print(f"Starting LightGBM.\nTrain shape: {train_df[feats].shape}, {train_df['target'].shape}\nTest shape: {test_df[feats].shape}, {test_df['target'].shape}")
    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outliers'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]


        # set data structure
        lgb_train = lgb.Dataset(train_x,
                                label=train_y,
                                free_raw_data=False)
        lgb_test = lgb.Dataset(valid_x,
                               label=valid_y,
                               free_raw_data=False)

        # params optimized by optuna
        params = {
            'task': 'train',
            # 'boosting': 'goss',
            'boosting': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.01,
            'subsample': 0.9855232997390695,
            'max_depth': 7,
            'num_leaves': 63,
            'min_child_weight': 41.9612869171337,
            # 'top_rate': 0.9064148448434349,
            # 'other_rate': 0.0721768246018207,
            'reg_alpha': 9.677537745007898,
            'colsample_bytree': 0.5665320670155495,
            'min_split_gain': 9.820197773625843,
            'reg_lambda': 8.2532317400459,
            'min_data_in_leaf': 21,
            'verbose': -1,
            'seed': int(2**n_fold),
            # 'bagging_seed': int(2**n_fold),
            # 'drop_seed': int(2**n_fold)
        }

        reg = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_test],
            valid_names=['train', 'test'],
            num_boost_round=10000,
            early_stopping_rounds=200,
            verbose_eval=100
        )

        oof_preds[valid_idx] = reg.predict(valid_x, num_iteration=reg.best_iteration)
        sub_preds += reg.predict(test_df[feats], num_iteration=reg.best_iteration) / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = np.log1p(reg.feature_importance(importance_type='gain', iteration=reg.best_iteration))
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d RMSE : %.6f' % (n_fold + 1, rmse(valid_y, oof_preds[valid_idx])))
        del reg, train_x, train_y, valid_x, valid_y
        gc.collect()

    score = rmse(train_df['target'], oof_preds)
    print(f"==== ALL RMSE: {score} ====")
    out_dir_base = "../data/output"
    out_dir = os.path.join(out_dir_base, f"{out_dir_name}_{score}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plot_importance(feature_importance_df, out_dir)
    feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False).to_csv(os.path.join(out_dir, f"faeture_importance.csv"), index=False)

    oof_df = train_df.copy().reset_index()
    oof_df['target'] = oof_preds
    oof_df[['card_id', 'target']].to_csv(os.path.join(out_dir, f"oof.csv"), index=False)

    submission = test_df.copy().reset_index()
    submission['target'] = sub_preds
    submission[['card_id', 'target']].to_csv(os.path.join(out_dir, f"submission.csv"), index=False)


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
        kfold_lightgbm(train_df, test_df, out_dir_name="20190223_0017_add_category_agg", num_folds=11, stratified=False, debug=debug)


if __name__ == "__main__":
    with timer("Full model run"):
        main(debug=False)
