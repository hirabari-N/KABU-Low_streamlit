
# 起動用コマンド
# streamlit run C:\Users\inosh\OneDrive\デスクトップ\Streamlit\KABU-Low\main.py
# pip install

#GitHub に push するためのコマンド
# git add .
# git commit -m "20230520 commmit" ※日付を変える
# git push origin main


# 必要なライブラリをインポート
import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from prophet import Prophet
from prophet.plot import add_changepoints_to_plot

from pandas_datareader.yahoo.daily import YahooDailyReader
import pandas_datareader.data as pdr
import yfinance as yf
from datetime import datetime,timedelta

# YahooDailyReader のインポート
yf.pdr_override()


# 証券コード一覧を取得　(!!!注意!!!　コード調整時は「stockllist」以外をOFFにする)
# import requests
# url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
# r = requests.get(url)
# with open('data_j.xls', 'wb') as output:
#     output.write(r.content)
stocklist = pd.read_excel("./data_j.xls")
stocklist["コード"] = stocklist["コード"].astype(str)

st.sidebar.title('☆上場企業リストの絞り込み☆')

# 市場区分の絞り込み
st.sidebar.text('◎市場区分を選んでください')
col = st.sidebar.columns(3)
prime = col[0].checkbox(label='プライム', value=True)
standard = col[1].checkbox(label='スタンダード')
growth = col[2].checkbox(label='グロース')
 
markets = []
if prime:
    markets.append('プライム（内国株式）')
if standard:
    markets.append('スタンダード（内国株式）')
if growth:
    markets.append('グロース（内国株式）')

# 業種の絞り込み
industries = st.sidebar.multiselect(label="◎業種を選んでください",
             options=['水産・農林業', '鉱業', '建設業',
                      '食料品', '繊維製品', 'パルプ・紙', '化学', '医薬品', '石油・石炭製品', 'ゴム製品', 'ガラス・土石製品', '鉄鋼', '非鉄金属', '金属製品', '機械', '電気機器', '輸送用機器', '精密機器', 'その他製品',
                      '電気・ガス業', '陸運業', '海運業', '空運業', '倉庫・運輸関連業', '情報・通信業', '卸売業', '小売業', '銀行業', '証券、商品先物取引業', '保険業', 'その他金融業', '不動産業', 'サービス業'],
             default=['食料品', '医薬品', '電気機器', '精密機器', '情報・通信業', '小売業', 'サービス業']
)

# 会社規模の絞り込み(規模コード)
min_value, max_value = st.sidebar.slider(label='◎会社規模の範囲を指定してください(1:大規模 ～ 7:小規模)',
                                     min_value=1,
                                     max_value=7,
                                     value=(1, 4),
                                     )
scale = list(range(min_value, max_value + 1))

col = st.sidebar.columns(1)
scale_none = col[0].checkbox(label='※規模コード「-」の銘柄も抽出する場合はチェック')

if scale_none:
     scale.append('-')


st.sidebar.title('☆特定銘柄の株価予測☆')

# 証券コードの数値入力 (+必要処理)
code = st.sidebar.text_input('◎4桁の証券コードを入力してください(半角英数字)', '2432')
code_t = code + ".T"
codes = stocklist["コード"].values


# 株価の予測算出における対象期間
date_fn = st.sidebar.date_input('◎いつまでの株価を予測の算出に使いますか？\n(初期値:今日)',
    min_value=datetime(2022, 1, 1),
    max_value=datetime.today(),
    value=datetime.today(),
)

days = st.sidebar.number_input('◎過去何日分にさかのぼって株価を取得しますか？\n(初期値:90)',
    value=90,
)

date_st = date_fn - timedelta(days)


# 日付範囲を設定
start_date = pd.to_datetime('2022-01-01')
end_date = pd.to_datetime('2023-12-31')

# 日付範囲内の日付を生成
date_range = pd.date_range(start=start_date, end=end_date)


# タイトルとテキストを記入
st.title('KABU-Low')
st.subheader('～カブロウ～')

st.write('予測に反映する株価の期間: ', date_st, '～', date_fn)

st.title('上場企業リスト')


# 上場企業リストの呼び出し
stocklist_l = stocklist.loc[stocklist["市場・商品区分"].isin(markets)
              & stocklist["33業種区分"].isin(industries)
              & stocklist["規模コード"].isin(scale),
            ["コード","銘柄名","市場・商品区分","33業種コード","33業種区分","規模コード","規模区分"]
            ]
stocklist_l_c = stocklist_l.set_index('コード')
stocklist_l_c

# # 証券コードのみを抽出
# mg_codes = stocklist_l[['コード','銘柄名']]

# # 複数の銘柄の株価グラフを表示
# if len(mg_codes) <= 20:

#     for 

#     #銘柄名の表示
#     stock_name = stocklist.loc[stocklist["コード"]==code, ["銘柄名"]].iat[0, 0]
#     st.title(stock_name)

#     # データ取得
#     df_stock =pdr.get_data_yahoo(code_t, date_st, date_fn).reset_index()

#     # prophet の仕様に合わせ、カラム名を変更
#     df_stock['ds'] = df_stock['Date'] # 日付データを「ds」に変更
#     df_stock['y'] = df_stock['Close'] # 予測したいデータを「y」に変更

#     # Prophet のモデルをインスタンス化＆学習
#     model = Prophet()
#     model.fit(df_stock)

#     # 1週間後までの株価を予測
#     future = model.make_future_dataframe(periods=7)
#     future_no_weekend = future[future['ds'].dt.weekday < 5]
#     forecast = model.predict(future_no_weekend)
 
#     #インデックスとして使用する列名を指定
#     forecast = forecast.set_index('ds')

#     # 株価の実測値を抽出(グラフ用)
#     df_stock_n = forecast['yhat']
#     df_stock_nl = forecast['yhat_lower']
  
#     # 株価の予測値を抽出(グラフ用)
#     df_stock_p = forecast['yhat'].tail(7) #.reset_index(drop=True) # 通常の押し目株価
#     df_stock_pl = forecast['yhat_lower'].tail(7) #.reset_index(drop=True) # 最小の押し目株価

#     # チャートグラフの描画
#     fig, ax = plt.subplots()
#     ax.plot(df_stock_n, color='y')
#     ax.plot(df_stock_nl, color='c')
#     ax.plot(df_stock_p, color='r')
#     ax.plot(df_stock_pl, color='#984ea3')

#     # グラフのタイトルと軸ラベルの設定
#     ax.set_title('Stock Price Prediction')

#     # メモリの非表示化
#     plt.xticks([])
#     plt.yticks([])

#     # Streamlit上でグラフを表示
#     st.pyplot(fig)

# else:
#     st.error(f'対象銘柄が多すぎるため、グラフ一覧を表示できません。')







# 個別銘柄の抽出
if code in codes:

    #銘柄名の表示
    stock_name = stocklist.loc[stocklist["コード"]==code, ["銘柄名"]].iat[0, 0]
    st.title(stock_name + 'の株価予測')

    # データ取得
    df_stock =pdr.get_data_yahoo(code_t, date_st, date_fn).reset_index()

    # prophet の仕様に合わせ、カラム名を変更
    df_stock['ds'] = df_stock['Date'] # 日付データを「ds」に変更
    df_stock['y'] = df_stock['Close'] # 予測したいデータを「y」に変更

    # Prophet のモデルをインスタンス化＆学習
    model = Prophet()
    model.fit(df_stock)

    # 1週間後までの株価を予測
    future = model.make_future_dataframe(periods=7)
    future_no_weekend = future[future['ds'].dt.weekday < 5]
    forecast = model.predict(future_no_weekend)
 
    #インデックスとして使用する列名を指定
    forecast = forecast.set_index('ds')

    # 営業日以外を「None」にしたパターン(表用)
    forecast_f = forecast.asfreq('D')
    forecast_f.index = forecast_f.reset_index()['ds'].dt.strftime('%Y-%m-%d')

    # # prophet の仕様に合わせ、カラム名を変更
    forecast_f['通常株価'] = forecast_f['yhat'] # 日付データを「ds」に変更
    forecast_f['最低株価'] = forecast_f['yhat_lower'] # 予測したいデータを「y」に変更

    # 株価の実測値を抽出(表用)
    df_stock_n_f = forecast_f['通常株価'].round()
    df_stock_nl_f = forecast_f['最低株価'].round()
 
    # 株価の予測値を抽出(表用)
    df_stock_p_f = forecast_f['通常株価'].round().tail(7) #.reset_index(drop=True) # 通常の押し目株価
    df_stock_pl_f = forecast_f['最低株価'].round().tail(7) #.reset_index(drop=True) # 最小の押し目株価

    #予測結果を表示
    df = st.dataframe([df_stock_p_f, df_stock_pl_f])


    # 株価の実測値を抽出(グラフ用)
    df_stock_n = forecast['yhat']
    df_stock_nl = forecast['yhat_lower']
  
    # 株価の予測値を抽出(グラフ用)
    df_stock_p = forecast['yhat'].tail(7) #.reset_index(drop=True) # 通常の押し目株価
    df_stock_pl = forecast['yhat_lower'].tail(7) #.reset_index(drop=True) # 最小の押し目株価

    # チャートグラフの描画
    fig, ax = plt.subplots()
    ax.plot(df_stock_n, color='y')
    ax.plot(df_stock_nl, color='c')
    ax.plot(df_stock_p, color='r')
    ax.plot(df_stock_pl, color='#984ea3')

    # グラフのタイトルと軸ラベルの設定
    ax.set_title('Stock Price Prediction')
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price')

    # X軸ラベルを縦書きに変更
    plt.xticks(rotation=90)

    # Streamlit上でグラフを表示
    st.pyplot(fig)

else:
    st.error(f'証券コード「{code}」は存在しません。')