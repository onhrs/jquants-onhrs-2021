import json
import requests


import pandas as pd
import numpy as np
from tqdm import tqdm


def match_type_df(df_target, df_matched):
    """
    :param df_target: 合わせたいデータフレーム
    :param df_matched: マッチさせたい型の入ったデータフレーム
    :return:
    """
    df_target = df_target[df_matched.columns]

    ser_matched_dtypes = df_matched.dtypes

    for col, dtype in ser_matched_dtypes.iteritems():
        if dtype == 'float':
            df_target[col] = df_target[col].replace('', np.nan).astype(float)
        elif dtype == 'int':
            df_target[col] = df_target[col].replace('', np.nan).astype(int)

    return df_target


def fetch_jquant_data(*, codes: list, paramdict: dict, target: str , idtk: str) -> pd.DataFrame:
    """
    :param codes:
    :param paramdict:
    :param target:
    :param idtk:
    :return:
    """
    df_stock_data_list = []

    for code in tqdm(codes):
        stock_data_dict = call_jquants_api(paramdict, idtk,target, f"{code}")
        key = list(stock_data_dict.keys())[0]
        stock_data = pd.DataFrame(stock_data_dict[key])
        df_stock_data_list.append(stock_data)

    df_stock_data = pd.concat(df_stock_data_list)

    return df_stock_data


def call_refresh_api(refreshtoken: str):
    """
    idTokenをリフレッシュするメソッド。

    Parameters
    ----------
    refreshtoken : str
        refreshtoken。ログイン後の画面からご確認いただけます。

    Returns
    -------
    resjson : dict
        新しいidtokenが格納されたAPIレスポンス(json形式)
    """
    headers = {"accept": "application/json"}
    data = {"refresh-token": refreshtoken}

    response = requests.post(
        "https://api.jpx-jquants.com/refresh", headers=headers, data=json.dumps(data)
    )

    resjson = json.loads(response.text)
    return resjson


def call_jquants_api(params: dict, idtoken: str, apitype: str, code: str = None):
    """
    J-QuantsのAPIを試すメソッド。

    Parameters
    ----------
    params : dict
        リクエストパラメータ。
    idtoken : str
        idTokenはログイン後の画面からご確認いただけます。
    apitype: str
        APIの種類。"news", "prices", "lists"などがあります。
    code: str
        銘柄を指定するAPIの場合に設定します。

    Returns
    -------
    resjson : dict
        APIレスポンス(json形式)
    """
    datefrom = params.get("datefrom", None)
    dateto = params.get("dateto", None)
    date = params.get("date", None)
    includedetails = params.get("includedetails", "false")
    keyword = params.get("keyword", None)
    headline = params.get("headline", None)
    paramcode = params.get("code", None)
    nexttoken = params.get("nextToken", None)
    scrollid = params.get("scrollId", None)
    headers = {"accept": "application/json", "Authorization": idtoken}
    data = {
        "from": datefrom,
        "to": dateto,
        "includeDetails": includedetails,
        "nextToken": nexttoken,
        "date": date,
        "keyword": keyword,
        "headline": headline,
        "code": paramcode,
        "scrollId": scrollid
    }

    if code:
        code = "/" + code
        r = requests.get(
            "https://api.jpx-jquants.com/" + apitype + code,
            params=data,
            headers=headers,
        )
    else:
        r = requests.get(
            "https://api.jpx-jquants.com/" + apitype, params=data, headers=headers
        )
    resjson = json.loads(r.text)
    return resjson
