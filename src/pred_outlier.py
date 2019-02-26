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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend
from contextlib import contextmanager
from pandas.core.common import SettingWithCopyWarning
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

from features import train_test, historical_transactions, new_merchant_transactions, additional_features, FEATS_EXCLUDED


warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


def plot_importance(feature_importance_df_, out_dir):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'lgbm_importances.png'))


def kfold_training(train_func, train_df, test_df, out_dir_name, num_folds, stratified=False, debug=False):
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=326)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=326)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    # feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]
    feats = [f for f in train_df.columns if (f not in FEATS_EXCLUDED) and not (f.startswith("feature_"))]
    # train_df = train_df[train_df["outliers"] == 0]
    print(f"feats: {len(feats)}\n{feats}")
    print(f"Start Training.\nTrain shape: {train_df[feats].shape}, {train_df['outliers'].shape}\nTest shape: {test_df[feats].shape}, {test_df['outliers'].shape}")

    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outliers'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['outliers'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['outliers'].iloc[valid_idx]

        result = train_func(train_x, train_y, valid_x, valid_y, test_df[feats], feats, n_fold)
        oof_preds[valid_idx] = result["oof_preds"]
        sub_preds += result["sub_preds"] / folds.n_splits
        print('Fold %2d LOGLOSS : %.6f' % (n_fold + 1, log_loss(valid_y, oof_preds[valid_idx])))

        if "feature_importance_df" in result:
            feature_importance_df = pd.concat(
                [feature_importance_df, result["feature_importance_df"]],
                axis=0
            )

    score = log_loss(train_df['outliers'], oof_preds)
    print(f"==== ALL LOGLOSS: {score} ====")
    out_dir_base = "../data/output"
    out_dir = os.path.join(out_dir_base, f"{out_dir_name}_{score}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if len(feature_importance_df) > 0:
        plot_importance(feature_importance_df, out_dir)
        feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False).to_csv(os.path.join(out_dir, f"faeture_importance.csv"), index=False)

    oof_df = train_df.copy().reset_index()
    oof_df['target'] = oof_preds
    oof_df[['card_id', 'target']].to_csv(os.path.join(out_dir, f"oof.csv"), index=False)

    submission = test_df.copy().reset_index()
    submission['target'] = sub_preds
    submission[['card_id', 'target']].to_csv(os.path.join(out_dir, f"submission.csv"), index=False)


def train_xgboost(train_x, train_y, valid_x, valid_y, test_x, feats, n_fold, stratified=False, debug=False):
    params = {
        "n_jobs": -1,
        "n_estimators": 10000,
        "learning_rate": 0.01,
        "max_depth": 7,
        "num_leaves": 63,
        'subsample': 0.9855232997390695,
        "colsample_bytree": 0.5665320670155495,
        "reg_alpha": 9.677537745007898,
        "reg_lambda": 8.2532317400459,
        "gamma": 9.820197773625843,
        "min_child_weight": 40,
        "seed": 131,
        "silent": True,
        "tree_method": "approx"
    }

    fit_params = {
        "eval_metric": "rmse",
        "verbose": 100,
        "early_stopping_rounds": 200
    }

    clf = XGBRegressor(**params)
    clf.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], **fit_params)
    result = {}

    result["oof_preds"] = clf.predict(valid_x)
    result["sub_preds"] = clf.predict(test_x)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = feats
    fold_importance_df["importance"] = np.log1p(clf.feature_importances_)
    fold_importance_df["fold"] = n_fold + 1
    result["feature_importance_df"] = fold_importance_df

    return result


def train_nn(train_x, train_y, valid_x, valid_y, test_x, feats, n_fold, stratified=False, debug=False):
    sc = MinMaxScaler()
    train_x = sc.fit_transform(train_x.clip(lower=np.finfo(np.float32).min, upper=np.finfo(np.float32).max), train_y)
    valid_x = sc.transform(valid_x.clip(lower=np.finfo(np.float32).min, upper=np.finfo(np.float32).max))
    test_x = sc.transform(test_x.clip(lower=np.finfo(np.float32).min, upper=np.finfo(np.float32).max))

    def keras_rmse(y_true, y_pred):
        return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

    def build_model():
        model = keras.Sequential([
            layers.Dense(64, activation=tf.nn.relu, input_shape=[len(feats)]),
            layers.Dense(64, activation=tf.nn.relu),
            layers.Dense(1)
        ])
        optimizer = tf.keras.optimizers.RMSprop(0.001)
        # optimizer = tf.keras.optimizers.SGD(lr=0.0001, clipnorm=5)
        model.compile(
            loss=keras_rmse,
            optimizer=optimizer,
            metrics=[keras_rmse]
        )
        return model

    def plot_history(history, n_fold):
        from matplotlib import pyplot as plt
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Square Error [$MPG^2$]')
        plt.plot(hist['epoch'], hist['keras_rmse'],
                 label='Train Error')
        plt.plot(hist['epoch'], hist['val_keras_rmse'],
                 label='Val Error')
        # plt.ylim([0, 20])
        plt.legend()
        plt.savefig(f"nn_loss_{n_fold}.png")

    model = build_model()
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    history = model.fit(
        train_x, train_y,
        epochs=1000, validation_data=(valid_x, valid_y),
        callbacks=[early_stop],
        batch_size=256,
    )

    plot_history(history, n_fold)

    result = {}
    result["oof_preds"] = model.predict(valid_x).flatten()
    result["sub_preds"] = model.predict(test_x).flatten()

    print(result)

    return result


def train_lightgbm(train_x, train_y, valid_x, valid_y, test_x, feats, n_fold, stratified=False, debug=False):
        # set data structure
    lgb_train = lgb.Dataset(train_x,
                            label=train_y,
                            free_raw_data=False)
    lgb_valid = lgb.Dataset(valid_x,
                            label=valid_y,
                            free_raw_data=False)

    # params optimized by optuna
    params = {
        'task': 'train',
        # 'boosting': 'goss',
        'boosting': 'gbdt',
        'objective': 'binary',
        'metrics': ['binary_logloss', 'auc'],
        # 'objective': 'None',
        # 'metric': 'None',
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
        valid_sets=[lgb_train, lgb_valid],
        valid_names=['train', 'valid'],
        num_boost_round=10000,
        early_stopping_rounds=200,
        verbose_eval=100,
        # fobj=custom_asymmetric_train,
        # feval=custom_asymmetric_valid,
    )

    result = {}

    result["oof_preds"] = reg.predict(valid_x, num_iteration=reg.best_iteration)
    result["sub_preds"] = reg.predict(test_x, num_iteration=reg.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = feats
    fold_importance_df["importance"] = np.log1p(reg.feature_importance(importance_type='gain', iteration=reg.best_iteration))
    fold_importance_df["fold"] = n_fold + 1
    result["feature_importance_df"] = fold_importance_df

    return result


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

    _out_dir = "20190226_0023_pred_outlier"
    with timer("Run LightGBM with kfold"):
        kfold_training(train_lightgbm, train_df, test_df, out_dir_name=_out_dir, num_folds=5, stratified=True, debug=debug)
    # with timer("Run XGBoost with kfold"):
        # kfold_training(train_xgboost, train_df, test_df, out_dir_name=_out_dir, num_folds=11, stratified=False, debug=debug)
    # with timer("Run NN with kfold"):
        # kfold_training(train_nn, train_df, test_df, out_dir_name=_out_dir, num_folds=11, stratified=False, debug=debug)


if __name__ == "__main__":
    with timer("Full model run"):
        # main(debug=True)
        main(debug=False)
