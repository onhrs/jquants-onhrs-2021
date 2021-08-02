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

#import predict.src.config as config
from module import SentimentGenerator
import config

formatter = '%(levelname)s : %(asctime)s : %(message)s'
logging.basicConfig(level=logging.INFO, format=formatter)


class ScoringService(object):
    # テスト期間開始日
    TEST_START = "2021-02-01"


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
            "stock_price": f"{dataset_dir}/1st/stock_price.csv.gz",
            "stock_fin": f"{dataset_dir}/1st/stock_fin.csv.gz",
            "stock_fin_price": f"{dataset_dir}stock_fin_price.csv.gz",
            # ニュースデータ
            "tdnet": f"{dataset_dir}/tdnet.csv.gz",
            "disclosureItems": f"{dataset_dir}/disclosureItems.csv.gz",
            "nikkei_article": f"{dataset_dir}/nikkei_article.csv.gz",
            "article": f"{dataset_dir}/article.csv.gz",
            "industry": f"{dataset_dir}/industry.csv.gz",
            "industry2": f"{dataset_dir}/industry2.csv.gz",
            "region": f"{dataset_dir}/region.csv.gz",
            "theme": f"{dataset_dir}/theme.csv.gz",
            # 目的変数データ
            "stock_labels": f"{dataset_dir}/1st/stock_labels.csv.gz",
            "purchase_date": f"{dataset_dir}/purchase_date.csv",

        }
        return inputs

    @classmethod
    def get_dataset(cls, inputs, load_data):
        """
        Args:
            inputs (list[str]): path to dataset files
            load_data (list[str]): specify loading data
        Returns:
            dict[pd.DataFrame]: loaded data
        """

        if cls.dfs is None:
            cls.dfs = {}
        for k, v in inputs.items():
            # 必要なデータのみ読み込みます
            if k not in load_data:
                continue
            logging.info(f'read {k}')
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

            elif k == "tdnet":
                cls.dfs[k]['code'] = cls.dfs[k]['code'].astype(int)

            elif k in ["purchase_date", "nikkei_article"]:
                continue

            else:
                cls.dfs[k]['Local Code'] = cls.dfs[k]['Local Code'].astype(int)

        return cls.dfs

    @classmethod
    def specified_data(cls, target_date=TEST_START):
        # 予測に必要な対象期間
        TARGET_DATE = str((pd.Timestamp(target_date) - pd.offsets.BDay(90)).date())

        if cls.dfs is None:
            raise Exception('dfs is none, please call get_dataset method')
        if cls.codes is None:
            cls.get_codes(cls.dfs)
        for k in cls.dfs.keys():
            if k == "stock_list":
                continue
            cls.dfs[k] = cls.dfs[k].loc[TARGET_DATE:]

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
        cls.codes = stock_list[stock_list["universe_comp2"] == True][
            "Local Code"
        ].values
        return cls.codes


    @classmethod
    def get_features_for_predict_portfolio(cls, codes, is_dask=False):
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

        # 2stのsubmit用に変更（how='left'から how='right'）
        df_learning = pd.merge(stock_fin_multi_index, feats, left_index=True, right_index=True, how='right')
        del stock_labels_multi_index['base_date']

        df_learning = pd.merge(df_learning, stock_labels_multi_index, left_index=True, right_index=True, how='left')
        df_learning = pd.merge(df_learning, cls.price_multi_index, left_index=True, right_index=True,
                               how='left').reset_index().set_index('datetime')

        # 欠損値処理
        df_learning = df_learning.replace([np.inf, -np.inf], 0).rename(columns={'Local Code': 'code'})#.fillna(0)

        # 2stのsubmit用に追記
        df_learning['Local Code'] = df_learning['code']
        df_learning_ffill = df_learning.groupby('Local Code').ffill()

        return df_learning_ffill

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
    def transform_yearweek_to_monday(cls, year, week):
        """
        ニュースから抽出した特徴量データのindexは (year, week) なので、
        (year, week) => YYYY-MM-DD 形式(月曜日) に変換します。
        """
        for s in pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D"):
            if s.week == week:
                # to return Monday of the first week of the year
                # e.g. "2020-01-01" => "2019-12-30"
                return s - pd.Timedelta(f"{s.dayofweek}D")

    @classmethod
    def load_sentiments(cls, path=None):
        DIST_END_DT = "2020-09-25"

        print(f"[+] load prepared sentiment: {path}")

        # 事前に出力したセンチメントの分布を読み込み
        df_sentiments = pd.read_pickle(path)

        # indexを日付型に変換します変換します。
        df_sentiments.loc[:, "index"] = df_sentiments.index.map(
            lambda x: cls.transform_yearweek_to_monday(x[0], x[1])
        )
        # indexを設定します
        df_sentiments.set_index("index", inplace=True)
        # カラム名を変更します
        df_sentiments.rename(columns={0: "headline_m2_sentiment_0"}, inplace=True)
        # 分布として使用するデータの範囲に絞り込みます
        df_sentiments = df_sentiments.loc[:DIST_END_DT]

        # 金曜日日付に変更します
        df_sentiments.index = df_sentiments.index + pd.Timedelta("4D")

        return df_sentiments

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
            buff.append(cls.get_features_for_predict_portfolio(cls.dfs, code))
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
        try:
            feats['rsi'] = rsi(feats['EndOfDayQuote ExchangeOfficialClose'], 14)
            feats['atr'] = atr(feats['EndOfDayQuote ExchangeOfficialClose'], 14)
            macd_period = {'long': 26, 'short': 12}
            sma_period = 9
            feats['macd'] = macd(feats['EndOfDayQuote ExchangeOfficialClose'], macd_period['short'], macd_period['long'])
            feats['signal'] = sma(feats['EndOfDayQuote ExchangeOfficialClose'], sma_period)
        except:
            for col_nan in ['rsi', 'atr', 'macd', 'signal']:
                feats[col_nan] = np.nan
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

        # SentimentGeneratorクラスの初期設定を実施
        SentimentGenerator.initialize(model_path)

        return True

    @classmethod
    def get_exclude(
            cls,
            df_tdnet,  # tdnetのデータ
            start_dt=None,  # データ取得対象の開始日、Noneの場合は制限なし
            end_dt=None,  # データ取得対象の終了日、Noneの場合は制限なし
            lookback=7,  # 除外考慮期間 (days)
            target_day_of_week=4,  # 起点となる曜日
    ):
        # 特別損失のレコードを取得
        special_loss = df_tdnet[df_tdnet["disclosureItems"].str.contains('201"')].copy()
        # 日付型を調整
        special_loss["date"] = pd.to_datetime(special_loss["disclosedDate"])
        # 処理対象開始日が設定されていない場合はデータの最初の日付を取得
        if start_dt is None:
            start_dt = special_loss["date"].iloc[0]
        # 処理対象終了日が設定されていない場合はデータの最後の日付を取得
        if end_dt is None:
            end_dt = special_loss["date"].iloc[-1]
        #  処理対象日で絞り込み
        special_loss = special_loss[
            (start_dt <= special_loss["date"]) & (special_loss["date"] <= end_dt)
            ]
        # 出力用にカラムを調整
        res = special_loss[["code", "disclosedDate", "date"]].copy()
        # 銘柄コードを4桁にする
        res["code"] = res["code"].astype(str).str[:-1]
        # 予測の基準となる金曜日の日付にするために調整
        res["remain"] = (target_day_of_week - res["date"].dt.dayofweek) % 7
        res["start_dt"] = res["date"] + pd.to_timedelta(res["remain"], unit="d")
        res["end_dt"] = res["start_dt"] + pd.Timedelta(days=lookback)
        columns = ["code", "date", "start_dt", "end_dt"]
        return res[columns].reset_index(drop=True)

    @classmethod
    def strategy(cls, strategy_id, df, df_tdnet):
        df = df.copy()
        # 銘柄選択方法選択
        if strategy_id in [1, 4]:
            # 最高値モデル +　最安値モデル
            df.loc[:, "pred"] = df.loc[:, "label_high_20"] + df.loc[:, "label_low_20"]
        elif strategy_id in [2, 5]:
            # 最高値モデル
            df.loc[:, "pred"] = df.loc[:, "label_high_20"]
        elif strategy_id in [3, 6]:
            # 最高値モデル
            df.loc[:, "pred"] = df.loc[:, "label_low_20"]
        else:
            raise ValueError("no strategy_id selected")

        # 特別損失を除外する場合
        if strategy_id in [4, 5, 6]:
            # 特別損失が発生した銘柄一覧を取得
            df_exclude = cls.get_exclude(df_tdnet)
            # 除外用にユニークな列を作成します。
            df_exclude.loc[:, "date-code_lastweek"] = df_exclude.loc[:, "start_dt"].dt.strftime(
                "%Y-%m-%d-") + df_exclude.loc[:, "code"]
            df_exclude.loc[:, "date-code_thisweek"] = df_exclude.loc[:, "end_dt"].dt.strftime(
                "%Y-%m-%d-") + df_exclude.loc[:, "code"]
            df.loc[:, "date-code_lastweek"] = (df.index - pd.Timedelta("7D")).strftime("%Y-%m-%d-") + df.loc[:,
                                                                                                      "code"].astype(
                str)
            df.loc[:, "date-code_thisweek"] = df.index.strftime("%Y-%m-%d-") + df.loc[:, "code"].astype(str)
            # 特別損失銘柄を除外
            df = df.loc[~df.loc[:, "date-code_lastweek"].isin(df_exclude.loc[:, "date-code_lastweek"])]
            df = df.loc[~df.loc[:, "date-code_thisweek"].isin(df_exclude.loc[:, "date-code_thisweek"])]

        # 予測出力を降順に並び替え
        df = df.sort_values("pred", ascending=False)
        # 予測出力の大きいものを取得
        df = df.groupby("datetime").head(50)

        return df

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
    def filter_friday(cls):
        if cls.df_predict is None:
            raise Exception('df_predict is none, please create FE method')
        df_predict_resample_business = cls.df_predict.groupby('code').resample("B").ffill().drop(['code'], axis=1)
        df_business_set_datetime = df_predict_resample_business.reset_index().set_index('datetime')
        FRIDAY = 4
        df_friday = df_business_set_datetime.loc[df_business_set_datetime.index.dayofweek == FRIDAY]
        cls.df_predict = df_friday

    @classmethod
    def predict(
            cls,
            inputs,
            labels=None,
            codes=None,
            start_dt=TEST_START,
            load_data=["stock_list", "stock_fin", "stock_labels", "stock_price", "tdnet", "purchase_date"],
            is_dask=False,
            strategy_id=5
    ):
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
            cls.get_dataset(inputs, load_data)
            cls.get_codes(cls.dfs)
        # 予測対象の銘柄コードと目的変数を設定
        if codes is None:
            codes = cls.codes[:]
        if labels is None:
            labels = cls.TARGET_LABELS

        if "purchase_date" in inputs.keys():
            # ランタイム環境では指定された投資対象日付を使用します
            # purchase_dateを読み込み
            df_purchase_date = pd.read_csv(inputs["purchase_date"])
            # purchase_dateの最も古い日付を設定
            start_dt = pd.Timestamp(
            df_purchase_date.sort_values("Purchase Date").iloc[0, 0]
            )
            logging.info(f'purchase_date is {start_dt}')

        # 予測対象日を調整
        # start_dtにはポートフォリオの購入日を指定しているため、
        # 予測に使用する前週の金曜日を指定します。
        start_dt_friday = pd.Timestamp(start_dt) - pd.Timedelta("3D")

        logging.info('specified　data')
        cls.specified_data(target_date=start_dt_friday)

        logging.info('feature engineering dask or pandas')
        cls.feats = cls.get_features_for_predict_portfolio(codes, is_dask=is_dask)

        logging.info('create learning_data')
        cls.df_learning = cls.get_fundamental_feature(cls.feats, codes)

        logging.info('calc　fundamental feature')
        cls.df_predict = cls.calc_fundamental_feature(cls.df_learning, is_predict=True)

        logging.info('filter friday')
        cls.filter_friday()

        ###################
        # センチメント情報取得
        ###################
        # ニュース見出しデータへのパスを指定
        start_dt_news = start_dt - pd.Timedelta("7D")
        start_dt_news = start_dt_news.strftime("%Y-%m-%d")
        df_sentiments = cls.get_sentiment(inputs, start_dt=start_dt_news)
        #
        # 金曜日日付に変更
        df_sentiments.index = df_sentiments.index + pd.Timedelta("4D")

        df_cash = df_sentiments.copy()
        df_cash['ratio'] = df_cash['headline_m2_sentiment_0'].values * 100 / 50

        # １列目:datetimeとcodeをつなげたもの(Ex 2016-05-09-1301)
        # ２列目:label_high_20　終値→最高値への変化率
        # ３列目:label_low_20　終値→最安値への変化率
        # headerはなし、B列C列はfloat64

        # 日付と銘柄コードに絞り込み
        df = cls.df_predict.loc[:, ["code"]].copy()

        # 購入金額を設定 (ここでは一律30000とする)
        df.loc[:, "budget"] = 30000

        df = pd.merge(df,
                      df_cash,
                      how='left',
                      left_index=True,
                      right_index=True
                      )
        df['headline_m2_sentiment_0'] = df['headline_m2_sentiment_0'].fillna(0.5)
        df['ratio'] = df['ratio'].fillna(1)

        df['budget'] = (df['budget'] * df['ratio']).round(-3).astype(int)
        df = df.reset_index().rename(columns={'index': 'datetime'}).set_index('datetime')

        # 特徴量カラムを指定
        feature_columns = cls.get_feature_columns(cls.dfs, cls.feats)

        logging.info('start predict')
        # 目的変数毎に予測
        for label in labels:
            # 予測実施
            df[label] = cls.models[label].predict(cls.df_predict[feature_columns])
            # 出力対象列に追加

        # 銘柄選択方法選択
        df = cls.strategy(strategy_id, df, cls.dfs["tdnet"])

        # 日付順に並び替え
        df.sort_index(kind="mergesort", inplace=True)
        # 月曜日日付に変更
        df.index = df.index + pd.Timedelta("3D")
        # 出力用に調整
        df.index.name = "date"
        df.rename(columns={"code": "Local Code"}, inplace=True)
        df.reset_index(inplace=True)

        # 出力対象列を定義
        output_columns = ["date", "Local Code", "budget"]

        out = io.StringIO()
        df_submit = df[df['date'] == start_dt]
        df_submit.to_csv(out, header=True, index=False, columns=output_columns)

        return out.getvalue()

    @classmethod
    def get_sentiment(cls, inputs, start_dt="2020-12-31"):
        # ニュース見出しデータへのパスを指定
        article_path = inputs["nikkei_article"]
        target_feature_types = ["headline"]
        df_sentiments = SentimentGenerator.generate_lstm_features(
            article_path,
            start_dt=start_dt,
            target_feature_types=target_feature_types,
        )["headline_features"]

        df_sentiments.loc[:, "index"] = df_sentiments.index.map(
            lambda x: cls.transform_yearweek_to_monday(x[0], x[1])
        )
        df_sentiments.set_index("index", inplace=True)
        df_sentiments.rename(columns={0: "headline_m2_sentiment_0"}, inplace=True)
        return df_sentiments