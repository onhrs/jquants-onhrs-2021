# -*- coding: utf-8 -*-
import io
import os
import sys
import multiprocessing
import pickle
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from tqdm.auto import tqdm

from pyti.relative_strength_index import relative_strength_index as rsi
from pyti.average_true_range import average_true_range as atr
from pyti.moving_average_convergence_divergence import moving_average_convergence_divergence as macd
from pyti.simple_moving_average import simple_moving_average as sma


import dask.dataframe as dd

import predict.src.config as config


formatter = '%(levelname)s : %(asctime)s : %(message)s'
logging.basicConfig(level=logging.INFO, format=formatter)


class ScoringServiceFundamental(object):
    # 訓練期間終了日
    TRAIN_END = "2018-12-31"
    # 評価期間開始日
    VAL_START = "2019-02-01"
    # 評価期間終了日
    VAL_END = "2019-12-01"
    # テスト期間開始日
    TEST_START = "2020-01-01"

    # 予測に必要な対象期間
    TARGET_DATE = str((pd.Timestamp(TEST_START) - pd.offsets.BDay(90)).date())

    # 目的変数
    TARGET_LABELS = ["label_high_20", "label_low_20"]

    # データをこの変数に読み込む
    dfs = None
    # モデルをこの変数に読み込む
    models = None
    # 対象の銘柄コードをこの変数に読み込む
    codes = None

    # 特徴量エンジニアリング
    feats = None

    # stock_priceのmulti index
    price_multi_index = None

    df_learning = None

    df_predict = None

    @classmethod
    def get_inputs(cls, dataset_dir):
        """
        Args:
            dataset_dir (str)  : path to dataset directory
        Returns:
            dict[str]: path to dataset files
        """
        inputs = {
            "stock_list": f"{dataset_dir}/stock_list.csv.gz",
            "stock_price": f"{dataset_dir}/stock_price.csv.gz",
            "stock_fin": f"{dataset_dir}/stock_fin.csv.gz",
            # "stock_fin_price": f"{dataset_dir}/stock_fin_price.csv.gz",
            "stock_labels": f"{dataset_dir}/stock_labels.csv.gz",
        }
        return inputs

    @classmethod
    def get_dataset(cls, inputs):
        """
        Args:
            inputs (list[str]): path to dataset files
        Returns:
            dict[pd.DataFrame]: loaded data
        """
        if cls.dfs is None:
            cls.dfs = {}
        for k, v in inputs.items():
            cls.dfs[k] = pd.read_csv(v)
            # DataFrameのindexを設定します。
            if k == "stock_price":
                cls.dfs[k].loc[:, "datetime"] = pd.to_datetime(
                    cls.dfs[k].loc[:, "EndOfDayQuote Date"]
                )
                cls.dfs[k].set_index("datetime", inplace=True)
            elif k in ["stock_fin", "stock_fin_price", "stock_labels"]:
                cls.dfs[k].loc[:, "datetime"] = pd.to_datetime(
                    cls.dfs[k].loc[:, "base_date"]
                )
                cls.dfs[k].set_index("datetime", inplace=True)
            elif k == "stock_list":
                cls.dfs[k]['17 Sector(Code)'] = cls.dfs[k]['17 Sector(Code)'].astype(int)
                cls.dfs[k]['33 Sector(Code)'] = cls.dfs[k]['33 Sector(Code)'].astype(int)

            cls.dfs[k]['Local Code'] = cls.dfs[k]['Local Code'].astype(int)

        return cls.dfs

    @classmethod
    def specified_data(cls):
        if cls.dfs is None:
            raise Exception('dfs is none, please call get_dataset method')
        if cls.codes is None:
            cls.get_codes(cls.dfs)
        for k in cls.dfs.keys():
            if k == "stock_list":
                continue
            cls.dfs[k] = cls.dfs[k].loc[cls.TARGET_DATE:]

        return cls.dfs

    @classmethod
    def get_codes(cls, dfs):
        """
        Args:
            dfs (dict[pd.DataFrame]): loaded data
        Returns:
            array: list of stock codes
        """
        stock_list = dfs["stock_list"].copy()
        # 予測対象の銘柄コードを取得
        cls.codes = stock_list[stock_list["prediction_target"] == True][
            "Local Code"
        ].values
        return cls.codes

    @classmethod
    def get_features_and_label(cls, dfs, codes, feature, label):
        """
        Args:
            dfs (dict[pd.DataFrame]): loaded data
            codes  (array) : target codes
            feature (pd.DataFrame): features
            label (str) : label column name
        Returns:
            train_X (pd.DataFrame): training data
            train_y (pd.DataFrame): label for train_X
            val_X (pd.DataFrame): validation data
            val_y (pd.DataFrame): label for val_X
            test_X (pd.DataFrame): test data
            test_y (pd.DataFrame): label for test_X
        """
        # 分割データ用の変数を定義
        trains_X, vals_X, tests_X = [], [], []
        trains_y, vals_y, tests_y = [], [], []

        # 銘柄コード毎に特徴量を作成
        for code in tqdm(codes):
            # 特徴量取得
            feats = feature[feature["code"] == code]

            # stock_labelデータを読み込み
            stock_labels = dfs["stock_labels"]
            # 特定の銘柄コードのデータに絞る
            stock_labels = stock_labels[stock_labels["Local Code"] == code]

            # 特定の目的変数に絞る
            labels = stock_labels[label].copy()
            # nanを削除
            labels.dropna(inplace=True)

            if feats.shape[0] > 0 and labels.shape[0] > 0:
                # 特徴量と目的変数のインデックスを合わせる
                labels = labels.loc[labels.index.isin(feats.index)]
                feats = feats.loc[feats.index.isin(labels.index)]
                labels.index = feats.index

                # データを分割
                _train_X = feats[: cls.TRAIN_END]
                _val_X = feats[cls.VAL_START : cls.VAL_END]
                _test_X = feats[cls.TEST_START :]

                _train_y = labels[: cls.TRAIN_END]
                _val_y = labels[cls.VAL_START : cls.VAL_END]
                _test_y = labels[cls.TEST_START :]

                # データを配列に格納 (後ほど結合するため)
                trains_X.append(_train_X)
                vals_X.append(_val_X)
                tests_X.append(_test_X)

                trains_y.append(_train_y)
                vals_y.append(_val_y)
                tests_y.append(_test_y)
        # 銘柄毎に作成した説明変数データを結合します。
        train_X = pd.concat(trains_X)
        val_X = pd.concat(vals_X)
        test_X = pd.concat(tests_X)
        # 銘柄毎に作成した目的変数データを結合します。
        train_y = pd.concat(trains_y)
        val_y = pd.concat(vals_y)
        test_y = pd.concat(tests_y)

        return train_X, train_y, val_X, val_y, test_X, test_y

    @classmethod
    def get_features_for_predict(cls, codes, is_dask=False):
        # 特徴量を作成
        price = cls.dfs["stock_price"]

        price_prediction_target = price[price['Local Code'].isin(codes)]
        if is_dask:
            logging.info('exec FE by dask')
            price_dd = dd.from_pandas(price_prediction_target, npartitions=multiprocessing.cpu_count())
            price_indicator = price_dd.groupby('Local Code').apply(cls.get_features_indicator,
                                                              meta=config.meta_feature)
            feats = price_indicator.compute(scheduler='processes')
        else:
            logging.info('exec FE by pandas')
            feats = price_prediction_target.groupby('Local Code').apply(cls.get_features_indicator)

        logging.info('price_indicator FE end')

        logging.info('start lag FE')
        feats['dollar_vol'] = feats.loc[:, 'EndOfDayQuote ExchangeOfficialClose'].mul(
            feats.loc[:, 'EndOfDayQuote Volume'], axis=0)

        q = 0.0001
        lags = [1, 5, 10, 21, 42, 63]
        for lag in lags:
            feats[f'return_{lag}d'] = (
                feats.groupby(level='Local Code')["EndOfDayQuote ExchangeOfficialClose"].pct_change(lag)
                .pipe(lambda x: x.clip(lower=x.quantile(q), upper=x.quantile(1 - q)))
                .add(1)
                .pow(1 / lag)
                .sub(1)
                )

        ### Shift lagged returns
        for t in [1, 2, 3, 4, 5]:
            for lag in [1, 5, 10, 21]:
                feats[f'return_{lag}d_lag{t}'] = (feats.groupby(level='Local Code')[f'return_{lag}d'].shift(t * lag))

        # 終値の20営業日リターン
        feats["return_1month"] = feats.groupby(level='Local Code')["EndOfDayQuote ExchangeOfficialClose"].pct_change(20)
        # 終値の40営業日リターン
        feats["return_2month"] = feats.groupby(level='Local Code')["EndOfDayQuote ExchangeOfficialClose"].pct_change(40)
        # 終値の60営業日リターン
        feats["return_3month"] = feats.groupby(level='Local Code')["EndOfDayQuote ExchangeOfficialClose"].pct_change(60)

        # ボラティリティ
        lags_vol = [2, 5, 10, 20, 40, 60]
        for lags in lags_vol:
            feats[f"volatility_{lags}days"] = feats.groupby(level='Local Code')[
                "EndOfDayQuote ExchangeOfficialClose"].diff().rolling(lags).std()

        feats['year'] = feats.index.get_level_values('datetime').year
        feats['month'] = feats.index.get_level_values('datetime').month

        logging.info('end lag FE')

        return feats

    @classmethod
    def get_fundamental_feature(cls, feats, codes, start_dt="2016-01-01"):

        stock_fin = cls.dfs["stock_fin"]
        fin_data = stock_fin[stock_fin["Local Code"].isin(codes)]
        stock_labels = cls.dfs["stock_labels"]

        # 特定の銘柄コードのデータに絞る
        stock_labels = stock_labels[stock_labels["Local Code"].isin(codes)]

        # 結合
        stock_fin_multi_index = fin_data.reset_index().set_index(['Local Code', 'datetime'])
        stock_labels_multi_index = stock_labels.reset_index().set_index(['Local Code', 'datetime'])
        cls.price_multi_index = cls.dfs['stock_price'].reset_index().set_index(['Local Code', 'datetime']).drop(
            ["EndOfDayQuote ExchangeOfficialClose", "EndOfDayQuote Volume"], axis=1)

        df_learning = pd.merge(stock_fin_multi_index, feats, left_index=True, right_index=True, how='left')
        del stock_labels_multi_index['base_date']

        df_learning = pd.merge(df_learning, stock_labels_multi_index, left_index=True, right_index=True, how='left')
        df_learning = pd.merge(df_learning, cls.price_multi_index, left_index=True, right_index=True,
                               how='left').reset_index().set_index('datetime')

        # 欠損値処理
        df_learning = df_learning.replace([np.inf, -np.inf], 0).rename(columns={'Local Code': 'code'})#.fillna(0)

        return df_learning

    @classmethod
    def calc_fundamental_feature(cls, df_learning, is_predict):
        stock_list = cls.dfs['stock_list'].rename(columns={'Local Code': 'code'})
        df_learning_stock_list_raw = pd.merge(df_learning.reset_index(), stock_list, on=['code'], how='left').set_index(
            'datetime')

        # 計算式は https://github.com/UKI000/JQuants-Forum/blob/main/210204_forum01.ipynb

        df_learning_stock_list = df_learning_stock_list_raw.copy()

        df_learning_stock_list["market_cap"] =\
            df_learning_stock_list["EndOfDayQuote Close"] * df_learning_stock_list["IssuedShareEquityQuote IssuedShare"]

        df_learning_stock_list["per"] = df_learning_stock_list["EndOfDayQuote Close"] / (
                    df_learning_stock_list["Result_FinancialStatement NetIncome"] * 1000000 / df_learning_stock_list[
                "IssuedShareEquityQuote IssuedShare"])

        df_learning_stock_list["per"].loc[
            df_learning_stock_list["Result_FinancialStatement CashFlowsFromOperatingActivities"] == 0] = np.nan

        df_learning_stock_list["pbr"] = df_learning_stock_list["EndOfDayQuote Close"] / (
                    df_learning_stock_list["Result_FinancialStatement NetAssets"] * 1000000 / df_learning_stock_list[
                "IssuedShareEquityQuote IssuedShare"])

        df_learning_stock_list["roe"] = df_learning_stock_list["pbr"] / df_learning_stock_list["per"]

        df_learning_stock_list["profit_margin"] =\
            df_learning_stock_list["Result_FinancialStatement NetIncome"] / df_learning_stock_list["Result_FinancialStatement NetSales"]

        df_learning_stock_list["profit_margin"].loc[
            df_learning_stock_list["Result_FinancialStatement CashFlowsFromOperatingActivities"] == 0] = np.nan

        df_learning_stock_list["equity_ratio"] \
            = df_learning_stock_list["Result_FinancialStatement NetAssets"] / df_learning_stock_list["Result_FinancialStatement TotalAssets"]

        df_learning_stock_list[config.fundamental_cols_category] = \
            df_learning_stock_list[config.fundamental_cols_category].fillna('NaN').astype('str')

        df_learning_stock_list = df_learning_stock_list.replace([np.inf, -np.inf], 0)

        # trainの時
        if not is_predict:
            df_learning_stock_list = df_learning_stock_list.dropna(subset=cls.TARGET_LABELS)

        return df_learning_stock_list

    @classmethod
    def get_feature_columns(cls, dfs, feats, column_group="fundamental+technical"):
        technical_cols = cls.price_multi_index.columns
        technical_cols = technical_cols[technical_cols != 'EndOfDayQuote Date']
        technical_cols = technical_cols[technical_cols != 'EndOfDayQuote PreviousCloseDate']
        technical_cols = technical_cols[technical_cols != 'EndOfDayQuote PreviousExchangeOfficialCloseDate']
        technical_cols = technical_cols.tolist() + feats.columns.tolist()

        fundamental_cols = dfs["stock_fin"].select_dtypes("float64").columns
        fundamental_cols = fundamental_cols[fundamental_cols != "Result_Dividend DividendPayableDate"]
        fundamental_cols = fundamental_cols[fundamental_cols != "Local Code"].tolist()

        fundamental_cols += ['IssuedShareEquityQuote IssuedShare', 'market_cap', 'per', 'pbr', 'roe', 'profit_margin',
                             'equity_ratio']

        columns = {
            "fundamental_only": fundamental_cols + ['code'],
            "technical_only": technical_cols + ['code'],
            "fundamental+technical": list(fundamental_cols) + list(technical_cols) + config.fundamental_cols_category + ['code'],
        }

        return columns[column_group]

    @classmethod
    def create_model(cls, dfs, codes, label):
        """
        Args:
            dfs (dict)  : dict of pd.DataFrame include stock_fin, stock_price
            codes (list[int]): A local code for a listed company
            label (str): prediction target label
        Returns:
            RandomForestRegressor
        """
        # 特徴量を取得
        buff = []
        for code in codes:
            buff.append(cls.get_features_for_predict(cls.dfs, code))
        feature = pd.concat(buff)
        # 特徴量と目的変数を一致させて、データを分割
        train_X, train_y, _, _, _, _ = cls.get_features_and_label(
            dfs, codes, feature, label
        )
        # 特徴量カラムを指定
        feature_columns = cls.get_feature_columns(dfs, train_X)
        # モデル作成
        model = RandomForestRegressor(random_state=0)
        model.fit(train_X[feature_columns], train_y)

        return model

    @classmethod
    def save_model(cls, model, label, model_path="../model"):
        """
        Args:
            model (RandomForestRegressor): trained model
            label (str): prediction target label
            model_path (str): path to save model
        Returns:
            -
        """
        # tag::save_model_partial[]
        # モデル保存先ディレクトリを作成
        os.makedirs(model_path, exist_ok=True)
        with open(os.path.join(model_path, f"my_model_{label}.pkl"), "wb") as f:
            # モデルをpickle形式で保存
            pickle.dump(model, f)
        # end::save_model_partial[]

    @classmethod
    def get_features_indicator(cls, price_data):
        price_data = price_data.select_dtypes(include=["int64", "float64"])
        # 欠損値処理
        price_data = price_data.fillna(0)

        target_col_list = ["EndOfDayQuote ExchangeOfficialClose", "EndOfDayQuote Volume"]
        feats = price_data[target_col_list]
        feats = feats.copy()
        feats['rsi'] = rsi(feats['EndOfDayQuote ExchangeOfficialClose'], 14)
        feats['atr'] = atr(feats['EndOfDayQuote ExchangeOfficialClose'], 14)
        macd_period = {'long': 26, 'short': 12}
        sma_period = 9
        feats['macd'] = macd(feats['EndOfDayQuote ExchangeOfficialClose'], macd_period['short'], macd_period['long'])
        feats['signal'] = sma(feats['EndOfDayQuote ExchangeOfficialClose'], sma_period)

        return feats

    @classmethod
    def get_model(cls, model_path="../model", labels=None):
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.
            labels (arrayt): list of prediction target labels

        Returns:
            bool: The return value. True for success, False otherwise.

        """
        if cls.models is None:
            cls.models = {}
        if labels is None:
            labels = cls.TARGET_LABELS
        for label in labels:
            m = os.path.join(model_path, f"my_model_{label}.pkl")
            with open(m, "rb") as f:
                # pickle形式で保存されているモデルを読み込み
                cls.models[label] = pickle.load(f)

        return True

    @classmethod
    def train_and_save_model(
        cls, inputs, labels=None, codes=None, model_path="../model"
    ):
        """Predict method

        Args:
            inputs (str)   : paths to the dataset files
            labels (array) : labels which is used in prediction model
            codes  (array) : target codes
            model_path (str): Path to the trained model directory.
        Returns:
            Dict[pd.DataFrame]: Inference for the given input.
        """
        if cls.dfs is None:
            cls.get_dataset(inputs)
            cls.get_codes(cls.dfs)
        if codes is None:
            codes = cls.codes
        if labels is None:
            labels = cls.TARGET_LABELS
        for label in labels:
            logging.info(label)
            model = cls.create_model(cls.dfs, codes=codes, label=label)
            cls.save_model(model, label, model_path=model_path)

    @classmethod
    def predict(cls, inputs, labels=None, codes=None, is_dask=False):
        """Predict method

        Args:
            inputs (dict[str]): paths to the dataset files
            labels (list[str]): target label names
            codes (list[int]): traget codes
            is_dask(bool) : use dask

        Returns:
            str: Inference for the given input.
        """

        logging.info('read data')
        # データ読み込み
        if cls.dfs is None:
            cls.get_dataset(inputs)
            cls.get_codes(cls.dfs)

        # 予測対象の銘柄コードと目的変数を設定
        if codes is None:
            codes = cls.codes[:]
        if labels is None:
            labels = cls.TARGET_LABELS

        logging.info('specified　data')
        cls.specified_data()

        logging.info('feature engineering dask or pandas')
        cls.feats = cls.get_features_for_predict(codes, is_dask=is_dask)

        logging.info('create learning_data')
        cls.df_learning = cls.get_fundamental_feature(cls.feats, codes)

        logging.info('calc　fundamental feature')
        cls.df_predict = cls.calc_fundamental_feature(cls.df_learning)
        # １列目:datetimeとcodeをつなげたもの(Ex 2016-05-09-1301)
        # ２列目:label_high_20　終値→最高値への変化率
        # ３列目:label_low_20　終値→最安値への変化率
        # headerはなし、B列C列はfloat64

        # 日付と銘柄コードに絞り込み
        df = cls.df_predict.loc[:, ["code"]].copy()
        # codeを出力形式の１列目と一致させる
        df.loc[:, "code"] = df.index.strftime("%Y-%m-%d-") + df.loc[:, "code"].astype(
            str
        )

        # 出力対象列を定義
        output_columns = ["code"]

        # 特徴量カラムを指定
        feature_columns = cls.get_feature_columns(cls.dfs, cls.feats)

        logging.info('start predict')
        # 目的変数毎に予測
        for label in labels:
            # 予測実施
            df[label] = cls.models[label].predict(cls.df_predict[feature_columns])
            # 出力対象列に追加
            output_columns.append(label)

        out = io.StringIO()
        df.to_csv(out, header=False, index=False, columns=output_columns)

        return out.getvalue()