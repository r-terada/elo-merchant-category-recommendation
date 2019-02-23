import os
import gc
import sys
import time
import click
import random
import sklearn
import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm
from pprint import pprint
from functools import reduce
from lightgbm import LGBMClassifier
from contextlib import contextmanager
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

from features import train_test, historical_transactions, new_merchant_transactions, additional_features

import warnings
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


def get_train_test():
    with timer("train & test"):
        df = train_test()
    with timer("historical transactions"):
        df = pd.merge(df, historical_transactions(), on='card_id', how='outer')
    with timer("new merchants"):
        df = pd.merge(df, new_merchant_transactions(), on='card_id', how='outer')
    with timer("additional features"):
        df = additional_features(df)
    with timer("split train & test"):
        train_df = df[df['target'].notnull()]
        test_df = df[df['target'].isnull()]
        del df
        gc.collect()

    return train_df, test_df


def get_feature_importances(data, shuffle, seed=None):
    # Gather real features
    train_features = [f for f in data.columns if f not in FEATS_EXCLUDED]
    # Go over fold and keep track of CV score (train and valid) and feature importances

    # Shuffle target if required
    y = data['target'].copy()
    if shuffle:
        # Here you could as well use a binomial distribution
        y = data['target'].copy().sample(frac=1.0)

    # Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest
    dtrain = lgb.Dataset(data[train_features], y, free_raw_data=False, silent=True)
    lgb_params = {
        'objective': 'regression',
        'boosting_type': 'rf',
        'subsample': 0.623,
        'colsample_bytree': 0.7,
        'num_leaves': 127,
        'max_depth': 8,
        'seed': seed,
        'bagging_freq': 1,
        'num_threads': 4,
        'verbose': -1
    }

    # Fit the model
    clf = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=600)

    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(train_features)
    imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
    imp_df["importance_split"] = clf.feature_importance(importance_type='split')
    imp_df['trn_score'] = rmse(y, clf.predict(data[train_features]))

    return imp_df




def lgb_cv(train_df, feats, target):
    # Cross validation model
    scores = []
    folds = KFold(n_splits=11, shuffle=True, random_state=326)
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outliers'])):
        train_x, train_y = train_df[feats].iloc[train_idx], target.iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], target.iloc[valid_idx]

        # set data structure
        lgb_train = lgb.Dataset(train_x,
                                label=train_y,
                                free_raw_data=False)
        lgb_test = lgb.Dataset(valid_x,
                               label=valid_y,
                               free_raw_data=False)

        params = {
            'metric': 'rmse',
            'objective': 'regression',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'learning_rate': 0.01,
            'subsample': 0.9855232997390695,
            'max_depth': 7,
            'num_leaves': 63,
            'min_child_weight': 41.9612869171337,
            'reg_alpha': 9.677537745007898,
            'reg_lambda': 8.2532317400459,
            'colsample_bytree': 0.5665320670155495,
            'min_split_gain': 9.820197773625843,
            'min_data_in_leaf': 21
            # 'learning_rate': .1,
            # 'subsample': 0.8,
            # 'colsample_bytree': 0.8,
            # 'num_leaves': 31,
            # 'max_depth': -1,
            # 'seed': 13,
            # 'num_threads': 4,
            # 'min_split_gain': .00001,
            # 'reg_alpha': .00001,
            # 'reg_lambda': .00001,
        }

        reg = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_test],
            valid_names=['train', 'test'],
            num_boost_round=10000,
            early_stopping_rounds=200,
            verbose_eval=False
        )

        preds = reg.predict(valid_x, num_iteration=reg.best_iteration)
        scores.append(rmse(valid_y, preds))
        del reg, train_x, train_y, valid_x, valid_y
        gc.collect()

    score_mean = np.mean(scores)
    score_std = np.std(scores)
    return score_mean, score_std


def score_feature_selection(df=None, train_features=None, target=None):
    # Fit the model
    mean, std = lgb_cv(df, train_features, target)
    # Return the last mean / std values
    return mean, std


def main():
    np.random.seed(47)

    data, _ = get_train_test()

    if not os.path.exists("../data/misc"):
        os.makedirs("../data/misc")

    with timer("calc actual importance"):
        if os.path.exists("../data/misc/actual_imp_df.pkl"):
            actual_imp_df = pd.read_pickle("../data/misc/actual_imp_df.pkl")
        else:
            actual_imp_df = get_feature_importances(data=data, shuffle=False)
            actual_imp_df.to_pickle("../data/misc/actual_imp_df.pkl")

    print(actual_imp_df.head())

    with timer("calc null importance"):
        nb_runs = 100

        if os.path.exists(f"../data/misc/null_imp_df_run{nb_runs}time.pkl"):
            null_imp_df = pd.read_pickle(f"../data/misc/null_imp_df_run{nb_runs}time.pkl")
        else:
            null_imp_df = pd.DataFrame()
            for i in range(nb_runs):
                start = time.time()
                # Get current run importances
                imp_df = get_feature_importances(data=data, shuffle=True)
                imp_df['run'] = i + 1
                # Concat the latest importances with the old ones
                null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
                # Display current run and time used
                spent = (time.time() - start) / 60
                dsp = '\rDone with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)
                print(dsp, end='', flush=True)
            null_imp_df.to_pickle(f"../data/misc/null_imp_df_run{nb_runs}time.pkl")

    print(null_imp_df.head())

    with timer('score features'):
        if os.path.exists("../data/misc/feature_scores_df.pkl"):
            scores_df = pd.read_pickle("../data/misc/feature_scores_df.pkl")
        else:
            feature_scores = []
            for _f in actual_imp_df['feature'].unique():
                f_null_imps_gain = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
                f_act_imps_gain = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].mean()
                gain_score = np.log(1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))  # Avoid didvide by zero
                f_null_imps_split = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
                f_act_imps_split = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].mean()
                split_score = np.log(1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75)))  # Avoid didvide by zero
                feature_scores.append((_f, split_score, gain_score))

            scores_df = pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score'])
            scores_df.to_pickle("../data/misc/feature_scores_df.pkl")

    with timer('calc correlation'):
        if os.path.exists("../data/misc/corr_scores_df.pkl"):
            corr_scores_df = pd.read_pickle("../data/misc/corr_scores_df.pkl")
        else:
            correlation_scores = []
            for _f in actual_imp_df['feature'].unique():
                f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
                f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].values
                gain_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
                f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
                f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].values
                split_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
                correlation_scores.append((_f, split_score, gain_score))

            corr_scores_df = pd.DataFrame(correlation_scores, columns=['feature', 'split_score', 'gain_score'])
            corr_scores_df.to_pickle("../data/misc/corr_scores_df.pkl")

    with timer('score feature removal by corr_scores'):
        # for threshold in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99][::-1]:
        #     with open(f"../data/misc/split_corr_under_threshold_{threshold}.txt", "w") as fp:
        #         print([_f for _f, _score, _ in corr_scores_df.values if _score < threshold], file=fp)
        #     with open(f"../data/misc/gain_corr_under_threshold_{threshold}.txt", "w") as fp:
        #         print([_f for _f, _, _score in corr_scores_df.values if _score < threshold], file=fp)

        for threshold in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99][::-1]:
            split_feats = [_f for _f, _score, _ in corr_scores_df.values if _score >= threshold]
            gain_feats = [_f for _f, _, _score in corr_scores_df.values if _score >= threshold]

            print('Results for threshold %3d' % threshold)
            print(f'split: use {len(split_feats)} features')
            split_results = score_feature_selection(df=data, train_features=split_feats, target=data['target'])
            print('\t SPLIT : %.6f +/- %.6f' % (split_results[0], split_results[1]))
            print(f'gain: use {len(gain_feats)} features')
            gain_results = score_feature_selection(df=data, train_features=gain_feats, target=data['target'])
            print('\t GAIN  : %.6f +/- %.6f' % (gain_results[0], gain_results[1]))


if __name__ == '__main__':
    main()
