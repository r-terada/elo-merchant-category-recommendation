import gc
import os
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm
from functools import reduce
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, BayesianRidge, Ridge, Lasso, ElasticNet, SGDRegressor, HuberRegressor


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def stacking(train_df, test_df, save=True, verbose=True):
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

        # clf = LinearRegression(n_jobs=-1)
        clf = BayesianRidge()
        # clf = Ridge()
        # clf = Lasso()
        # clf = ElasticNet()
        # clf = SGDRegressor()
        # clf = HuberRegressor()
        clf.fit(train_x.values, train_y.values)

        oof_preds[valid_idx] = clf.predict(valid_x.values)
        sub_preds += clf.predict(test_df[feats].values) / folds.n_splits

        if verbose:
            print('Fold %2d RMSE : %.6f' % (n_fold + 1, rmse(valid_y, oof_preds[valid_idx])))

    score = rmse(train_df['target'], oof_preds)
    if verbose:
        print(f'ALL RMSE: {score}')

    if save:
        out_dir = ("../data/output/stacking")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        with open(os.path.join(out_dir, f"params_{score:.5f}.txt"), "w") as fp:
            print(",".join(feats), file=fp)
        oof_df = train_df.copy().reset_index()
        oof_df['target'] = oof_preds
        oof_df[['card_id', 'target']].to_csv(os.path.join(out_dir, f"oof_{score:.5f}.csv"), index=False)
        submission = test_df.copy().reset_index()
        submission['target'] = sub_preds
        submission[['card_id', 'target']].to_csv(os.path.join(out_dir, f"stacking_{score:.5f}.csv"), index=False)

    return score


def stack_all():
    train_df = pd.read_csv("../data/input/train.csv.zip")[["card_id", "target"]]
    for fname in glob.glob("../data/output/**/oof.csv", recursive=True):
        df = pd.read_csv(fname).rename(columns={"target": os.path.dirname(fname).split("output")[-1]})
        train_df = train_df.merge(df, on="card_id")
        del df
        gc.collect()

    print(train_df.head())

    test_df = pd.read_csv("../data/input/test.csv.zip")[["card_id"]]
    for fname in glob.glob("../data/output/**/oof.csv", recursive=True):
        fname = fname.replace("oof", "submission")
        df = pd.read_csv(fname).rename(columns={"target": os.path.dirname(fname).split("output")[-1]})
        test_df = test_df.merge(df, on="card_id")
        del df
        gc.collect()

    print(test_df.head())

    stacking(train_df, test_df)


def stack_bruteforce():
    train_base = pd.read_csv("../data/input/train.csv.zip")[["card_id", "target"]]
    train_dfs = {}
    for fname in glob.glob("../data/output/**/oof.csv", recursive=True):
        model_idx = os.path.dirname(fname).split("output")[-1]
        train_dfs[model_idx] = pd.read_csv(fname).rename(columns={"target": model_idx})

    test_base = pd.read_csv("../data/input/test.csv.zip")[["card_id"]]
    test_dfs = {}
    for fname in glob.glob("../data/output/**/oof.csv", recursive=True):
        fname = fname.replace("oof", "submission")
        model_idx = os.path.dirname(fname).split("output")[-1]
        test_dfs[model_idx] = pd.read_csv(fname).rename(columns={"target": model_idx})

    best_score = 100000
    best_features = []
    feature_names = [f for f in train_dfs.keys() if f not in ['card_id', 'target']]
    while True:
        best_feature = None
        print(f"current best_score: {best_score}")
        print(f"current features: {best_features}")
        for n in tqdm(feature_names, desc="test stacking", ncols=100):
            # print(f"test {n}")
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


def stack_specified():
    dir_names = [
        "20190221_0004_features_sum_product_3.6478211935730918",
        "hyperopt_goss/0018_3.6500527247789143",
        "20190221_0010_stratified_3.6494040531741043",
        "hyperopt_gbdt/0036_3.650217284097732",
        "hyperopt_gbdt/0001_3.6533637171589413",
        "hyperopt_goss/0025_3.65060454884021",
        "hyperopt_goss/0019_3.6494077365567765",
        "20190224_0020_xgboost_3.6503550670356386",
        "20190223_0015_add_features_3.6605727017505107",
        "hyperopt_gbdt/0022_3.6547204358846415",
        "hyperopt_gbdt/0009_3.6556283525618425",
        "hyperopt_gbdt/0016_3.651468530751637",
        "hyperopt_goss/0008_3.653070140341593",
        "hyperopt_goss/0029_3.6497186322103863",
        "hyperopt_gbdt/0014_3.658375006600629",
        "hyperopt_gbdt/0032_3.6525504112273146",
        "hyperopt_gbdt/0015_3.6543947094545657",
        "hyperopt_gbdt/0029_3.651514948215108",
        "hyperopt_gbdt/0004_3.6537836461048685",
        "20190226_0025_outlier_prediction_as_feature_3.654405778061392"
    ]

    train_base = pd.read_csv("../data/input/train.csv.zip")[["card_id", "target"]]
    train_dfs = [train_base]
    for dir_name in tqdm(dir_names, desc="read oofs", ncols=100):
        fpath = os.path.join("../data/output/", dir_name, "oof.csv")
        train_dfs.append(pd.read_csv(fpath).rename(columns={"target": dir_name}))

    train_df = reduce(lambda l, r: l.merge(r, on="card_id"), train_dfs)

    test_base = pd.read_csv("../data/input/test.csv.zip")[["card_id"]]
    test_dfs = [test_base]
    for dir_name in tqdm(dir_names, desc="read submissions", ncols=100):
        fpath = os.path.join("../data/output/", dir_name, "submission.csv")
        test_dfs.append(pd.read_csv(fpath).rename(columns={"target": dir_name}))

    test_df = reduce(lambda l, r: l.merge(r, on="card_id"), test_dfs)

    score = stacking(train_df, test_df, True, True)


def main():
    # stack_all()
    # stack_bruteforce()
    stack_specified()


if __name__ == '__main__':
    main()
