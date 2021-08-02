import datetime

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from IPython.display import display
from scipy.stats import spearmanr

from predict.src.pipeline import ml_pipeline
from predict.src import config

import typing as t


class ExperimentManagement(object):
    def __init__(self, df_learning, feature_columns):
        self.df_learning = df_learning
        self.feature_columns = feature_columns
        self.result = {}
        self.result_list = []
        self.model_pipeline = None
        self.model = None
        self.label = None

    def set_model_label(self, model, label) -> None:
        self.model = model
        self.label = label

    def create_model(self, model, train):
        self.set_model_label(model, self.label)
        self.model_pipeline = ml_pipeline(model=model, features=self.feature_columns,
                                      category_variable=config.fundamental_cols_category)
        self.model_pipeline.fit(train[self.feature_columns], train[self.label])

        return self.model_pipeline

    def get_result(self, pred_model, validation:pd.DataFrame, is_plot=False) -> dict:
        try:
            self.result[self.label] = pd.DataFrame(pred_model.predict(validation[self.feature_columns]), columns=["predict"])
        except ValueError:
            self.result[self.label] = pd.DataFrame(pred_model.predict(validation[self.feature_columns].fillna(-1)), columns=["predict"])


        # 予測結果に日付と銘柄コードを追加
        self.result[self.label]["datetime"] = validation[self.feature_columns].index
        self.result[self.label]["code"] = validation["code"].values

        # 予測の符号を取得
        self.result[self.label]["predict_dir"] = np.sign(self.result[self.label]["predict"])

        # 実際の値を追加
        self.result[self.label]["actual"] = validation[self.label].values
        if is_plot:
            display(self.result[self.label].loc[:, ["predict", "actual"]].corr())
            sns.jointplot(data=self.result[self.label], x="predict", y="actual")
            plt.show()

        self.result[self.label].loc[:, ["predict", "actual"]].corr()

        return self.result

    def get_spearman_corr(self, init: t.Union[datetime.date, str],
                                end: t.Union[datetime.date, str],
                                train: pd.DataFrame
                          ) -> float:

        val = self.df_learning.loc[init: end]

        self.model_pipeline = self.create_model(self.model, train)

        # 予測
        self.result = self.get_result(self.model_pipeline, val, is_plot=True)

        result = self.result.copy()

        self.result_list.append(result)



        spearman_corr = spearmanr(self.result[self.label]["actual"], self.result[self.label]["predict"])[0]
        print(f'spearman_corr : {spearman_corr}')
        print('-----------------------------------------')

        return spearman_corr

    def evaluate_time_split_corr(self, init_date_list:list, end_date_list:list, rolling=True) -> list:
        """
        :param init_date_list:
        :param end_date_list:
        :param model:
        :param label:
        :param rolling:
        :return:
        """
        self.result_list = []

        spearman_corr_list =[]
        print(f'rolling : {rolling}')
        for init, end in zip(init_date_list, end_date_list):
            if isinstance(init, datetime.date):
                init = str(init)
                end = str(end)
                init_first = str(init_date_list[0])
            else:
                init_first = init_date_list[0]

            if not rolling:
                train = self.df_learning.loc[: init_first]
            else:
                train = self.df_learning.loc[: init]

            print(f'学習期間 : 　2016-01-01 ~ {init}')
            print(f'検証期間 : {init} ~ {end}')
            spearman_corr = self.get_spearman_corr(init, end, train)

            spearman_corr_list.append(spearman_corr)

        return spearman_corr_list

