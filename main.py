# 起動用コマンド
# streamlit run C:\Users\inosh\OneDrive\デスクトップ\Streamlit\KABU-Low\main.py
# pip install 

# 必要なライブラリをインポート
import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from prophet import Prophet
from prophet.plot import add_changepoints_to_plot

from pandas_datareader.yahoo.daily import YahooDailyReader
import pandas_datareader.data as pdr
import yfinance as yf
from datetime import datetime,timedelta

# YahooDailyReader のインポート
yf.pdr_override()


# タイトルとテキストを記入
st.title('KABU-Low')
st.subheader('～カブロウ～')
st.text('指定した株価の押し目を予測するアプリです。')

# セレクトボックス
option = st.sidebar.selectbox(
    '銘柄を選んでください。',
    ['', '【2432】DeNA', '【7201】日産自動車']
)



# date_st = st.sidebar.date_input('株価取得の始期',
#     min_value=datetime(2022, 4, 1),
#     max_value=datetime.today() - timedelta(days=1),
#     value=datetime.today() - timedelta(days=90),
# )
# st.write('株価取得の始期: ', date_st)

date_fn = st.sidebar.date_input('いつまでの株価を予測の算出に使いますか？\n(初期値:今日)',
    min_value=datetime(2022, 4, 1),
    max_value=datetime.today(),
    value=datetime.today(),
)

# 数値入力
days = st.sidebar.number_input('過去何日分にさかのぼって株価を取得しますか？\n(初期値:90)',
    value=90,
)

date_st = date_fn - timedelta(days)

st.write('予測に反映する株価の期間: ', date_st, '～', date_fn)


# def main():
#     date_st = st.sidebar.date_input('取得開始',
#         min_value=datetime(2022, 4, 1),
#         max_value=datetime.today(),
#         value=datetime(2023, 1, 1),
#     )
#     st.write('date: ', date_st)

# if __name__ == '__main__':
#     main()

# def main():
#     date_fn = st.sidebar.date_input('取得終了',
#         min_value=datetime(2022, 4, 1),
#         max_value=datetime.today(),
#         value=datetime(2023, 1, 1),
#     )
#     st.write('date: ', date_fn)

# if __name__ == '__main__':
#     main()


st.write('株価の押し目を予測します。')

# データ取得期間の設定
# date_st = datetime(2022, 3, 1)
# date_fn = datetime(2023, 3, 1)



# def main():
#     birthday = st.date_input('When is your birthday?',
#         min_value=datetime(2000, 1, 1),
#         max_value=datetime.today(),
#         value=datetime(2023, 1, 1),
#     )
#     st.write('Birthday: ', birthday)

# if __name__ == '__main__':
#     main()

if option == '【2432】DeNA':

    # データ取得
    df_dena =pdr.get_data_yahoo('2432.T', date_st, date_fn).reset_index() # 日産自動車

    # prophet の仕様に合わせ、カラム名を変更
    df_dena['ds'] = df_dena['Date'] # 日付データを「ds」に変更
    df_dena['y'] = df_dena['Close'] # 予測したいデータを「y」に変更

    # Prophet のモデルをインスタンス化＆学習
    model = Prophet()
    model.fit(df_dena)

    # 1週間後までの株価を予測
    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)

    # 予測結果を可視化
    fig1 = model.plot(forecast)

    # 押し目株価を抽出
    df_dena_p = forecast['yhat'] # 通常の押し目株価
    df_dena_pl = forecast['yhat_lower'] # 最小の押し目株価

    #予測結果を表示
    st.dataframe(df_dena_p)
    st.dataframe(df_dena_pl)
   
  

elif option == '【7201】日産自動車':

    # データ取得
    df_nissan =pdr.get_data_yahoo('7201.T', date_st, date_fn).reset_index() # 日産自動車

    # prophet の仕様に合わせ、カラム名を変更
    df_nissan['ds'] = df_nissan['Date'] # 日付データを「ds」に変更
    df_nissan['y'] = df_nissan['Close'] # 予測したいデータを「y」に変更

    # Prophet のモデルをインスタンス化＆学習
    model = Prophet()
    model.fit(df_nissan)

    # 1週間後までの株価を予測
    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)

    # 予測結果を可視化
    fig1 = model.plot(forecast)

    # 押し目株価を抽出
    df_nissan_p = forecast['yhat'] # 通常の押し目株価
    df_nissan_pl = forecast['yhat_lower'] # 最小の押し目株価

    #予測結果を表示
    st.dataframe(df_nissan_p)
    st.dataframe(df_nissan_pl)
