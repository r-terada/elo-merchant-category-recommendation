import gc
import os
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb
from functools import reduce
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def stacking(train_df, test_df, save=True, verbose_fold=True):
    folds = KFold(n_splits=11, shuffle=True, random_state=326)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['target', 'card_id']]

    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['target'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]

        clf = LinearRegression(n_jobs=-1)
        clf.fit(train_x.values, train_y.values)

        oof_preds[valid_idx] = clf.predict(valid_x.values)
        sub_preds += clf.predict(test_df[feats].values) / folds.n_splits

        if verbose_fold:
            print('Fold %2d RMSE : %.6f' % (n_fold + 1, rmse(valid_y, oof_preds[valid_idx])))

    score = rmse(train_df['target'], oof_preds)
    print(f'ALL RMSE: {score}')

    if save:
        out_dir = ("../data/output/stacking")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        with open(os.path.join(out_dir, f"params_{score:.5f}.txt"), "w") as fp:
            print(",".join(feats), file=fp)
        # oof_df = train_df.copy().reset_index()
        # oof_df['target'] = oof_preds
        # oof_df[['card_id', 'target']].to_csv(os.path.join(out_dir, f"oof.csv"), index=False)
        submission = test_df.copy().reset_index()
        submission['target'] = sub_preds
        submission[['card_id', 'target']].to_csv(os.path.join(out_dir, f"stacking_{score:.5f}.csv"), index=False)

    return score


def stack_all():
    train_df = pd.read_csv("../data/input/train.csv.zip")[["card_id", "target"]]
    for fname in glob.glob("../data/hyperopt_output/*/oof.csv"):
        df = pd.read_csv(fname).rename(columns={"target": os.path.dirname(fname).split("/")[-1].split("_")[0]})
        train_df = train_df.merge(df, on="card_id")
        del df
        gc.collect()

    print(train_df.head())

    test_df = pd.read_csv("../data/input/test.csv.zip")[["card_id"]]
    for fname in glob.glob("../data/hyperopt_output/*/submission.csv"):
        df = pd.read_csv(fname).rename(columns={"target": os.path.dirname(fname).split("/")[-1].split("_")[0]})
        test_df = test_df.merge(df, on="card_id")
        del df
        gc.collect()

    print(test_df.head())

    stacking(train_df, test_df)


def stack_bruteforce():
    train_base = pd.read_csv("../data/input/train.csv.zip")[["card_id", "target"]]
    train_dfs = {}
    for fname in glob.glob("../data/hyperopt_output/*/oof.csv"):
        model_idx = os.path.dirname(fname).split("/")[-1].split("_")[0]
        train_dfs[model_idx] = pd.read_csv(fname).rename(columns={"target": model_idx})

    test_base = pd.read_csv("../data/input/test.csv.zip")[["card_id"]]
    test_dfs = {}
    for fname in glob.glob("../data/hyperopt_output/*/submission.csv"):
        model_idx = os.path.dirname(fname).split("/")[-1].split("_")[0]
        test_dfs[model_idx] = pd.read_csv(fname).rename(columns={"target": model_idx})

    best_score = 100000
    best_features = []
    feature_names = [f for f in train_dfs.keys() if f not in ['card_id', 'target']]
    while True:
        best_feature = None
        print(f"current best_score: {best_score}")
        print(f"current features: {best_features}")
        for n in feature_names:
            print(f"test {n}")
            train_f_list = [train_base] + [train_dfs[f] for f in best_features] + [train_dfs[n]]
            train_df = reduce(lambda l, r: l.merge(r, on="card_id"), train_f_list)
            test_f_list = [test_base] + [test_dfs[f] for f in best_features] + [test_dfs[n]]
            test_df = reduce(lambda l, r: l.merge(r, on="card_id"), test_f_list)
            score = stacking(train_df, test_df, False, False)
            if score < best_score:
                best_score = score
                best_feature = n

        if best_feature is None:
            print("score does not improve")
            break
        else:
            best_features.append(best_feature)
            feature_names.remove(best_feature)

    print(f"final best_score: {best_score}")
    print(f"final features: {best_features}")
    train_f_list = [train_base] + [train_dfs[f] for f in best_features]
    train_df = reduce(lambda l, r: l.merge(r, on="card_id"), train_f_list)
    test_f_list = [test_base] + [test_dfs[f] for f in best_features]
    test_df = reduce(lambda l, r: l.merge(r, on="card_id"), test_f_list)
    score = stacking(train_df, test_df, True, True)


def main():
    # stack_all()
    stack_bruteforce()


if __name__ == '__main__':
    main()
