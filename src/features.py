import gc
import os
import datetime
import numpy as np
import pandas as pd
from sklearn.externals.joblib import memory


cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/cache')
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
mem = memory.Memory(location=cache_dir, verbose=1)


def one_hot_encoder(df, nan_as_category=True, not_ohe_cols=None):
    original_columns = list(df.columns)
    if not_ohe_cols is None:
        not_ohe_cols = []
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object' and col not in not_ohe_cols]
    print(f"[one_hot_encoder]: encode these columns {categorical_columns}")
    # for c in categorical_columns:
        # print(f"{c} has {df[c].nunique()} unique values")
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

# preprocessing train & test


@mem.cache
def train_test(num_rows=None):

    # load csv
    train_df = pd.read_csv('../data/input/train.csv.zip', index_col=['card_id'], nrows=num_rows)
    test_df = pd.read_csv('../data/input/test.csv.zip', index_col=['card_id'], nrows=num_rows)

    print("Train samples: {}, test samples: {}".format(len(train_df), len(test_df)))

    # outlier
    train_df['outliers'] = 0
    train_df.loc[train_df['target'] < -30, 'outliers'] = 1

    # set target as nan
    test_df['target'] = np.nan

    # merge
    df = train_df.append(test_df)

    del train_df, test_df
    gc.collect()

    # to datetime
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])

    # datetime features
    df['quarter'] = df['first_active_month'].dt.quarter
    df['elapsed_time'] = (datetime.datetime(2019, 1, 20, 0, 0) - df['first_active_month']).dt.days
    # df['first_month'] = df['first_active_month'].dt.month
    # df['first_day'] = df['first_active_month'].dt.day
    # df['first_hour'] = df['first_active_month'].dt.hour
    # df['first_weekofyear'] = df['first_active_month'].dt.weekofyear
    # df['first_weekday'] = df['first_active_month'].dt.weekday
    # df['first_weekend'] = (df['first_active_month'].dt.weekday >= 5).astype(int)
    df['feature_1'] = df['feature_1'].astype("object")
    df['feature_2'] = df['feature_2'].astype("object")
    df['feature_3'] = df['feature_3'].astype("object")
    for f in ['feature_1', 'feature_2', 'feature_3']:
        order_label = df.groupby([f])['outliers'].mean()
        df[f"{f}_outlier_prob"] = df[f].map(order_label)

    # one hot encoding
    df, cols = one_hot_encoder(df, nan_as_category=False)

    # df['days_feature1'] = df['elapsed_time'] * df['feature_1']
    # df['days_feature2'] = df['elapsed_time'] * df['feature_2']
    # df['days_feature3'] = df['elapsed_time'] * df['feature_3']

    # df['days_feature1_ratio'] = df['feature_1'] / df['elapsed_time']
    # df['days_feature2_ratio'] = df['feature_2'] / df['elapsed_time']
    # df['days_feature3_ratio'] = df['feature_3'] / df['elapsed_time']

    df['feature_outlier_prob_sum'] = df['feature_1_outlier_prob'] + df['feature_2_outlier_prob'] + df['feature_3_outlier_prob']
    # df['feature1_plus_feature2'] = df['feature_1'] + df['feature_2']
    # df['feature1_plus_feature3'] = df['feature_1'] + df['feature_3']
    # df['feature2_plus_feature3'] = df['feature_2'] + df['feature_3']
    df['feature_outlier_prob_product'] = df['feature_1_outlier_prob'] * df['feature_2_outlier_prob'] * df['feature_3_outlier_prob']
    # df['feature1_mul_feature2'] = df['feature_1_outlier_prob'] * df['feature_2_outlier_prob']
    # df['feature1_mul_feature3'] = df['feature_1_outlier_prob'] * df['feature_3_outlier_prob']
    # df['feature2_mul_feature3'] = df['feature_2_outlier_prob'] * df['feature_3_outlier_prob']
    df['feature_outlier_prob_mean'] = df['feature_outlier_prob_sum'] / 3
    df['feature_outlier_prob_max'] = df[['feature_1_outlier_prob', 'feature_2_outlier_prob', 'feature_3_outlier_prob']].max(axis=1)
    df['feature_outlier_prob_min'] = df[['feature_1_outlier_prob', 'feature_2_outlier_prob', 'feature_3_outlier_prob']].min(axis=1)
    df['feature_outlier_prob_var'] = df[['feature_1_outlier_prob', 'feature_2_outlier_prob', 'feature_3_outlier_prob']].std(axis=1)

    return df

# preprocessing historical transactions
@mem.cache
def read_historical_transactions(num_rows):
    return pd.read_csv('../data/input/historical_transactions.csv.zip', nrows=num_rows)


@mem.cache
def historical_transactions(num_rows=None):
    # load csv
    hist_df = read_historical_transactions(num_rows)

    # fillna
    hist_df['category_2'].fillna(1.0, inplace=True)
    hist_df['category_3'].fillna('A', inplace=True)
    hist_df['merchant_id'].fillna('M_ID_00a6ca8a8a', inplace=True)
    hist_df['installments'].replace(-1, np.nan, inplace=True)
    hist_df['installments'].replace(999, np.nan, inplace=True)

    # trim
    hist_df['purchase_amount'] = hist_df['purchase_amount'].apply(lambda x: min(x, 0.8))

    # Y/N to 1/0
    hist_df['authorized_flag'] = hist_df['authorized_flag'].map({'Y': 1, 'N': 0}).astype(int)
    hist_df['category_1'] = hist_df['category_1'].map({'Y': 1, 'N': 0}).astype(int)
    hist_df['category_3'] = hist_df['category_3'].map({'A': 0, 'B': 1, 'C': 2})

    # datetime features
    hist_df['purchase_date'] = pd.to_datetime(hist_df['purchase_date'])
    hist_df['month'] = hist_df['purchase_date'].dt.month
    hist_df['day'] = hist_df['purchase_date'].dt.day
    hist_df['hour'] = hist_df['purchase_date'].dt.hour
    hist_df['weekofyear'] = hist_df['purchase_date'].dt.weekofyear
    hist_df['weekday'] = hist_df['purchase_date'].dt.weekday
    hist_df['weekend'] = (hist_df['purchase_date'].dt.weekday >= 5).astype(int)

    # additional features
    hist_df['price'] = hist_df['purchase_amount'] / hist_df['installments']

    # Christmas : December 25 2017
    hist_df['Christmas_Day_2017'] = (pd.to_datetime('2017-12-25')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    # Mothers Day: May 14 2017
    hist_df['Mothers_Day_2017'] = (pd.to_datetime('2017-06-04')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    # fathers day: August 13 2017
    hist_df['fathers_day_2017'] = (pd.to_datetime('2017-08-13')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    # Childrens day: October 12 2017
    hist_df['Children_day_2017'] = (pd.to_datetime('2017-10-12')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    # Valentine's Day : 12th June, 2017
    hist_df['Valentine_Day_2017'] = (pd.to_datetime('2017-06-12')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    # Black Friday : 24th November 2017
    hist_df['Black_Friday_2017'] = (pd.to_datetime('2017-11-24') - hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

    # 2018
    # Mothers Day: May 13 2018
    hist_df['Mothers_Day_2018'] = (pd.to_datetime('2018-05-13')-hist_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

    hist_df['month_diff'] = ((datetime.datetime(2019, 1, 20, 0, 0) - hist_df['purchase_date']).dt.days)//30
    hist_df['month_diff'] += hist_df['month_lag']

    # additional features
    hist_df['duration'] = hist_df['purchase_amount']*hist_df['month_diff']
    hist_df['amount_month_ratio'] = hist_df['purchase_amount']/hist_df['month_diff']

    # reduce memory usage
    hist_df = reduce_mem_usage(hist_df)

    col_unique = ['subsector_id', 'merchant_id', 'merchant_category_id']
    col_seas = ['month', 'hour', 'weekofyear', 'weekday', 'day']

    aggs = {}
    for col in col_unique:
        aggs[col] = ['nunique']

    for col in col_seas:
        aggs[col] = ['nunique', 'mean', 'min', 'max']

    aggs['purchase_amount'] = ['sum', 'max', 'min', 'mean', 'var', 'skew']
    aggs['installments'] = ['sum', 'max', 'mean', 'var', 'skew']
    aggs['purchase_date'] = ['max', 'min', 'nunique', 'count']
    aggs['month_lag'] = ['max', 'min', 'mean', 'var', 'skew']
    aggs['month_diff'] = ['max', 'min', 'mean', 'var', 'skew']
    aggs['authorized_flag'] = ['mean']
    aggs['weekend'] = ['mean', 'count']  # overwrite
    aggs['weekday'] = ['mean', 'count']  # overwrite
    aggs['day'] = ['nunique', 'mean', 'max', 'min', 'count']  # overwrite
    # aggs['category_1'] = ['mean']
    # aggs['category_2'] = ['mean']
    # aggs['category_3'] = ['mean']
    aggs['category_1'] = ['count']
    aggs['category_2'] = ['count']
    aggs['category_3'] = ['count']
    aggs['card_id'] = ['size', 'count']
    aggs['price'] = ['sum', 'mean', 'max', 'min', 'var']
    aggs['Christmas_Day_2017'] = ['mean', 'count']
    aggs['Mothers_Day_2017'] = ['mean', 'count']
    aggs['fathers_day_2017'] = ['mean', 'count']
    aggs['Children_day_2017'] = ['mean', 'count']
    aggs['Valentine_Day_2017'] = ['mean', 'count']
    aggs['Black_Friday_2017'] = ['mean', 'count']
    aggs['Mothers_Day_2018'] = ['mean', 'count']
    aggs['duration'] = ['mean', 'min', 'max', 'var', 'skew']
    aggs['amount_month_ratio'] = ['mean', 'min', 'max', 'var', 'skew']

    for col in ['category_2', 'category_3']:
        hist_df[col+'_mean'] = hist_df.groupby([col])['purchase_amount'].transform('mean')
        hist_df[col+'_min'] = hist_df.groupby([col])['purchase_amount'].transform('min')
        hist_df[col+'_max'] = hist_df.groupby([col])['purchase_amount'].transform('max')
        hist_df[col+'_sum'] = hist_df.groupby([col])['purchase_amount'].transform('sum')
        aggs[col+'_mean'] = ['mean']

    hist_df = hist_df.reset_index().groupby('card_id').agg(aggs)

    # change column name
    hist_df.columns = pd.Index([e[0] + "_" + e[1] for e in hist_df.columns.tolist()])
    hist_df.columns = ['hist_' + c for c in hist_df.columns]

    hist_df['hist_purchase_date_diff'] = (hist_df['hist_purchase_date_max']-hist_df['hist_purchase_date_min']).dt.days
    hist_df['hist_purchase_date_average'] = hist_df['hist_purchase_date_diff']/hist_df['hist_card_id_size']
    hist_df['hist_purchase_date_uptonow'] = (datetime.datetime(2019, 1, 20, 0, 0)-hist_df['hist_purchase_date_max']).dt.days
    hist_df['hist_purchase_date_uptomin'] = (datetime.datetime(2019, 1, 20, 0, 0)-hist_df['hist_purchase_date_min']).dt.days
    hist_df['week_end_ratio'] = hist_df['week_end_count'] / hist_df['day_count']

    # reduce memory usage
    hist_df = reduce_mem_usage(hist_df)

    return hist_df

# preprocessing new_merchant_transactions
@mem.cache
def read_new_merchant_transactions(num_rows):
    return pd.read_csv('../data/input/new_merchant_transactions.csv.zip', nrows=num_rows)


@mem.cache
def new_merchant_transactions(num_rows=None):
    # load csv
    new_merchant_df = read_new_merchant_transactions(num_rows)
    # fillna
    new_merchant_df['category_2'].fillna(1.0, inplace=True)
    new_merchant_df['category_3'].fillna('A', inplace=True)
    new_merchant_df['merchant_id'].fillna('M_ID_00a6ca8a8a', inplace=True)
    new_merchant_df['installments'].replace(-1, np.nan, inplace=True)
    new_merchant_df['installments'].replace(999, np.nan, inplace=True)

    # trim
    new_merchant_df['purchase_amount'] = new_merchant_df['purchase_amount'].apply(lambda x: min(x, 0.8))

    # Y/N to 1/0
    new_merchant_df['authorized_flag'] = new_merchant_df['authorized_flag'].map({'Y': 1, 'N': 0}).astype(int)
    new_merchant_df['category_1'] = new_merchant_df['category_1'].map({'Y': 1, 'N': 0}).astype(int)
    new_merchant_df['category_3'] = new_merchant_df['category_3'].map({'A': 0, 'B': 1, 'C': 2}).astype(int)

    # datetime features
    new_merchant_df['purchase_date'] = pd.to_datetime(new_merchant_df['purchase_date'])
    new_merchant_df['month'] = new_merchant_df['purchase_date'].dt.month
    new_merchant_df['day'] = new_merchant_df['purchase_date'].dt.day
    new_merchant_df['hour'] = new_merchant_df['purchase_date'].dt.hour
    new_merchant_df['weekofyear'] = new_merchant_df['purchase_date'].dt.weekofyear
    new_merchant_df['weekday'] = new_merchant_df['purchase_date'].dt.weekday
    new_merchant_df['weekend'] = (new_merchant_df['purchase_date'].dt.weekday >= 5).astype(int)

    # additional features
    new_merchant_df['price'] = new_merchant_df['purchase_amount'] / new_merchant_df['installments']

    # Christmas : December 25 2017
    new_merchant_df['Christmas_Day_2017'] = (pd.to_datetime('2017-12-25')-new_merchant_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    # Childrens day: October 12 2017
    new_merchant_df['Children_day_2017'] = (pd.to_datetime('2017-10-12')-new_merchant_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    # Black Friday : 24th November 2017
    new_merchant_df['Black_Friday_2017'] = (pd.to_datetime('2017-11-24') - new_merchant_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

    # Mothers Day: May 13 2018
    new_merchant_df['Mothers_Day_2018'] = (pd.to_datetime('2018-05-13')-new_merchant_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

    new_merchant_df['month_diff'] = ((datetime.datetime(2019, 1, 20, 0, 0) - new_merchant_df['purchase_date']).dt.days)//30
    new_merchant_df['month_diff'] += new_merchant_df['month_lag']

    # additional features
    new_merchant_df['duration'] = new_merchant_df['purchase_amount']*new_merchant_df['month_diff']
    new_merchant_df['amount_month_ratio'] = new_merchant_df['purchase_amount']/new_merchant_df['month_diff']

    # reduce memory usage
    new_merchant_df = reduce_mem_usage(new_merchant_df)

    col_unique = ['subsector_id', 'merchant_id', 'merchant_category_id']
    col_seas = ['month', 'hour', 'weekofyear', 'weekday', 'day']

    aggs = {}
    for col in col_unique:
        aggs[col] = ['nunique']

    for col in col_seas:
        aggs[col] = ['nunique', 'mean', 'min', 'max']

    aggs['purchase_amount'] = ['sum', 'max', 'min', 'mean', 'var', 'skew']
    aggs['installments'] = ['sum', 'max', 'mean', 'var', 'skew']
    aggs['purchase_date'] = ['max', 'min']
    aggs['month_lag'] = ['max', 'min', 'mean', 'var', 'skew']
    aggs['month_diff'] = ['mean', 'var', 'skew']
    aggs['weekend'] = ['mean']
    aggs['month'] = ['mean', 'min', 'max']
    aggs['weekday'] = ['mean', 'min', 'max']
    aggs['category_1'] = ['mean']
    aggs['category_2'] = ['mean']
    aggs['category_3'] = ['mean']
    aggs['card_id'] = ['size', 'count']
    aggs['price'] = ['mean', 'max', 'min', 'var']
    aggs['Christmas_Day_2017'] = ['mean']
    aggs['Children_day_2017'] = ['mean']
    aggs['Black_Friday_2017'] = ['mean']
    aggs['Mothers_Day_2018'] = ['mean']
    aggs['duration'] = ['mean', 'min', 'max', 'var', 'skew']
    aggs['amount_month_ratio'] = ['mean', 'min', 'max', 'var', 'skew']

    for col in ['category_2', 'category_3']:
        new_merchant_df[col+'_mean'] = new_merchant_df.groupby([col])['purchase_amount'].transform('mean')
        new_merchant_df[col+'_min'] = new_merchant_df.groupby([col])['purchase_amount'].transform('min')
        new_merchant_df[col+'_max'] = new_merchant_df.groupby([col])['purchase_amount'].transform('max')
        new_merchant_df[col+'_sum'] = new_merchant_df.groupby([col])['purchase_amount'].transform('sum')
        aggs[col+'_mean'] = ['mean']

    new_merchant_df = new_merchant_df.reset_index().groupby('card_id').agg(aggs)

    # change column name
    new_merchant_df.columns = pd.Index([e[0] + "_" + e[1] for e in new_merchant_df.columns.tolist()])
    new_merchant_df.columns = ['new_' + c for c in new_merchant_df.columns]

    new_merchant_df['new_purchase_date_diff'] = (new_merchant_df['new_purchase_date_max']-new_merchant_df['new_purchase_date_min']).dt.days
    new_merchant_df['new_purchase_date_average'] = new_merchant_df['new_purchase_date_diff']/new_merchant_df['new_card_id_size']
    new_merchant_df['new_purchase_date_uptonow'] = (datetime.datetime(2019, 1, 20, 0, 0)-new_merchant_df['new_purchase_date_max']).dt.days
    new_merchant_df['new_purchase_date_uptomin'] = (datetime.datetime(2019, 1, 20, 0, 0)-new_merchant_df['new_purchase_date_min']).dt.days

    # reduce memory usage
    new_merchant_df = reduce_mem_usage(new_merchant_df)

    return new_merchant_df


# additional features
def additional_features(df):
    df['hist_first_buy'] = (df['hist_purchase_date_min'] - df['first_active_month']).dt.days
    df['hist_last_buy'] = (df['hist_purchase_date_max'] - df['first_active_month']).dt.days
    df['new_first_buy'] = (df['new_purchase_date_min'] - df['first_active_month']).dt.days
    df['new_last_buy'] = (df['new_purchase_date_max'] - df['first_active_month']).dt.days

    date_features = ['hist_purchase_date_max', 'hist_purchase_date_min',
                     'new_purchase_date_max', 'new_purchase_date_min']

    for f in date_features:
        df[f] = df[f].astype(np.int64) * 1e-9

    df['card_id_total'] = df['new_card_id_size']+df['hist_card_id_size']
    df['card_id_cnt_total'] = df['new_card_id_count']+df['hist_card_id_count']
    df['card_id_cnt_ratio'] = df['new_card_id_count']/df['hist_card_id_count']
    df['purchase_amount_total'] = df['new_purchase_amount_sum']+df['hist_purchase_amount_sum']
    df['purchase_amount_mean'] = df['new_purchase_amount_mean']+df['hist_purchase_amount_mean']
    df['purchase_amount_max'] = df['new_purchase_amount_max']+df['hist_purchase_amount_max']
    df['purchase_amount_min'] = df['new_purchase_amount_min']+df['hist_purchase_amount_min']
    df['purchase_amount_ratio'] = df['new_purchase_amount_sum']/df['hist_purchase_amount_sum']
    df['month_diff_mean'] = df['new_month_diff_mean']+df['hist_month_diff_mean']
    df['month_diff_ratio'] = df['new_month_diff_mean']/df['hist_month_diff_mean']
    df['month_lag_mean'] = df['new_month_lag_mean']+df['hist_month_lag_mean']
    df['month_lag_max'] = df['new_month_lag_max']+df['hist_month_lag_max']
    df['month_lag_min'] = df['new_month_lag_min']+df['hist_month_lag_min']
    df['category_1_mean'] = df['new_category_1_mean']+df['hist_category_1_mean']
    df['installments_total'] = df['new_installments_sum']+df['hist_installments_sum']
    df['installments_mean'] = df['new_installments_mean']+df['hist_installments_mean']
    df['installments_max'] = df['new_installments_max']+df['hist_installments_max']
    df['installments_ratio'] = df['new_installments_sum']/df['hist_installments_sum']
    df['price_total'] = df['purchase_amount_total'] / df['installments_total']
    df['price_mean'] = df['purchase_amount_mean'] / df['installments_mean']
    df['price_max'] = df['purchase_amount_max'] / df['installments_max']
    df['duration_mean'] = df['new_duration_mean']+df['hist_duration_mean']
    df['duration_min'] = df['new_duration_min']+df['hist_duration_min']
    df['duration_max'] = df['new_duration_max']+df['hist_duration_max']
    df['amount_month_ratio_mean'] = df['new_amount_month_ratio_mean']+df['hist_amount_month_ratio_mean']
    df['amount_month_ratio_min'] = df['new_amount_month_ratio_min']+df['hist_amount_month_ratio_min']
    df['amount_month_ratio_max'] = df['new_amount_month_ratio_max']+df['hist_amount_month_ratio_max']
    df['new_CLV'] = df['new_card_id_count'] * df['new_purchase_amount_sum'] / df['new_month_diff_mean']
    df['hist_CLV'] = df['hist_card_id_count'] * df['hist_purchase_amount_sum'] / df['hist_month_diff_mean']
    df['CLV_ratio'] = df['new_CLV'] / df['hist_CLV']

    return df
