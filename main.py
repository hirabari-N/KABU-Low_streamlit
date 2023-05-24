
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

import mplfinance as mpf
from PIL import Image

# YahooDailyReader のインポート
yf.pdr_override()

# Streamlit の表示調整
st.set_page_config(layout="wide")

# 証券コード一覧を取得　(!!!注意!!!　コード調整時は「stockllist」以外をOFFにする)
# import requests
# url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
# r = requests.get(url)
# with open('data_j.xls', 'wb') as output:
#     output.write(r.content)
stocklist = pd.read_excel("./data_j.xls")
stocklist["コード"] = stocklist["コード"].astype(str)





# ↓↓↓↓↓サイドバー↓↓↓↓↓


# リロード注意
st.sidebar.warning('アップロード時はWebスクレイピングを必ずOFFに', icon="⚠️")


# ↓↓↓「上場企業リスト」調整用↓↓↓
st.sidebar.header(':classical_building:上場企業リストの絞り込み:classical_building:')

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
industries = st.sidebar.multiselect(label="◎業種を選んでください :green[(複数選択可)]",
             options=['水産・農林業', '鉱業', '建設業',
                      '食料品', '繊維製品', 'パルプ・紙', '化学', '医薬品', '石油・石炭製品', 'ゴム製品', 'ガラス・土石製品', '鉄鋼', '非鉄金属', '金属製品', '機械', '電気機器', '輸送用機器', '精密機器', 'その他製品',
                      '電気・ガス業', '陸運業', '海運業', '空運業', '倉庫・運輸関連業', '情報・通信業', '卸売業', '小売業', '銀行業', '証券、商品先物取引業', '保険業', 'その他金融業', '不動産業', 'サービス業'],
             default=['サービス業']
)

# 会社規模の絞り込み(規模コード)
min_value, max_value = st.sidebar.slider(label='◎会社規模の範囲を指定してください :green[(1:大規模 ～ 7:小規模)]', min_value=1, max_value=7, value=(1, 4)) # value は初期値
scale = list(range(min_value, max_value + 1))

col = st.sidebar.columns(1)
scale_none = col[0].checkbox(label='← 規模コード「-」の銘柄も抽出する場合はチェック')

if scale_none:
     scale.append('-')



# ↓↓↓「特定銘柄の株価」調整用↓↓↓
st.sidebar.header(':office:特定銘柄の株価チェック:office:')

# 証券コードの数値入力 (+必要処理)
code = st.sidebar.text_input('◎4桁の証券コードを入力してください :green[(半角英数字、初期値:7203)]', '7203')
code_t = code + ".T"
codes = stocklist["コード"].values


# 株価の予測算出における対象期間
date_fn = st.sidebar.date_input('◎いつまでの株価を予測の算出に使いますか？ :green[(初期値:今日)]',
    min_value=datetime(2010, 1, 1),
    max_value=datetime.today(),
    value=datetime.today(),
)

days = st.sidebar.number_input('◎過去何日分にさかのぼって株価を取得しますか？ :green[(初期値:365)]', value=365,)
date_st = date_fn - timedelta(days)

# チャートにおける表示範囲の調整
days_2 = st.sidebar.slider('◎チャートに表示する期間を決めてください :green[(単位:日、初期値:90)]', value=90, min_value=0, max_value=days)

# チャートにおける時間足の調整
time_frame = st.sidebar.radio('◎チャートの時間足を決めてください :green[(初期値:1日)]',
                              options=('1日','5日','1週間','1ヶ月','3ヶ月'),  # 時間足の選択
                              index=0, horizontal=True,) # index はラジオボタンの初期値

# チャート生成時に備えて時間足をコードに変換
if time_frame == '1日':
    time_frame_e = '1d'
if time_frame == '5日':
    time_frame_e = '5d'
if time_frame == '1週間':
    time_frame_e = '1wk'
if time_frame == '1ヶ月':
    time_frame_e = '1mo'
if time_frame == '3ヶ月':
    time_frame_e = '3mo'



# ↓↓↓↓↓メインフィールド↓↓↓↓↓

# アプリ名
st.title(':chart: :green[_≪≪≪_] :red[_KABU-Low_] :green[_≫≫≫_] :japan:')



# ↓↓↓上場企業リスト↓↓↓
st.header(':classical_building: 上場企業リスト :classical_building:')

# 上場企業リストの呼び出し
stocklist_l = stocklist.loc[stocklist["市場・商品区分"].isin(markets)
              & stocklist["33業種区分"].isin(industries)
              & stocklist["規模コード"].isin(scale),
            ["コード","銘柄名","市場・商品区分","33業種コード","33業種区分","規模コード","規模区分"]
            ]
stocklist_l_c = stocklist_l.set_index('コード')
stocklist_l_c

# # 証券コードのみを抽出
mg_codes = stocklist_l[['コード','銘柄名']]

# 複数の銘柄の株価グラフを表示
if len(mg_codes) == 0:
    st.error(f'対象銘柄がありません。')

elif len(mg_codes) <= 40:

    if st.button(label='各銘柄のチャートを表示'):

        # 4列のグリッドを作成
        col0, col1, col2, col3= st.columns(4) 

        i = 0

        for index, data in mg_codes.iterrows():
                
            # データ取得
            code_t_m = data['コード'] + ".T"
            df_stock_m =pdr.get_data_yahoo(code_t_m, date_st, date_fn)

            # チャートグラフの描画
            fig, ax = plt.subplots()
            ax.plot(df_stock_m['Close'], color='#000000')

            # メモリの非表示化
            plt.xticks([])
            plt.yticks([])

            # Streamlit上でチャートを4列になるように表示
            if i == 0:
                with col0:
                    # 銘柄名の表示
                    st.write('証券コード：' + data['コード'])
                    st.caption(data['銘柄名'])
                    st.pyplot(fig)

            elif i == 1:
                with col1:
                    # 銘柄名の表示
                    st.write('証券コード：' + data['コード'])
                    st.caption(data['銘柄名'])
                    st.pyplot(fig)
            
            elif i == 2:
                with col2:
                    # 銘柄名の表示
                    st.write('証券コード：' + data['コード'])
                    st.caption(data['銘柄名'])
                    st.pyplot(fig)
                    
            else:
                with col3:
                    # 銘柄名の表示
                    st.write('証券コード：' + data['コード'])
                    st.caption(data['銘柄名'])
                    st.pyplot(fig)
                
            i = (i + 1) % 4

else:
    st.error(f'※対象銘柄が40個以下であれば、チャート一覧を表示できます')



# ↓↓↓個別銘柄の実測値＆予測値↓↓↓

# 個別銘柄の抽出
if code in codes:

    #銘柄名の表示
    stock_name = stocklist.loc[stocklist["コード"]==code, ["銘柄名"]].iat[0, 0]
    st.header(':office: :green[_≪' + stock_name + '≫_] の株価チェック :office:')


    # ↓実測値↓
    st.subheader('◎実測値のチャートと数値一覧')

    #引数『y_stock』にyfinance.downloadで取得した価格データを入れるコード
    df_stock = yf.download(code_t, start=date_st, end=date_fn,
                           interval=time_frame_e, # interval：時間軸
                           auto_adjust=True) # auto_adjust：『始値 / 高値 / 安値 / 終値』の四本値を自動的に調整

    # チャートを生成    
    def main():

        # 図として保存＆呼び出し
        fname='./mpf_candle.png' # 保存先のファイル名を指定
        mpf.plot(df_stock[:days_2], type='candle', # ローソク足描画
                figscale=2.0, # 図の大きさの倍率を指定、デフォルトは1
                tight_layout=True, # 図の端の余白を狭くして最適化するかどうかを指定
                volume=True, savefig=fname)
        img=Image.open(fname) # PILのイメージオブジェクトとして呼び出し
        st.image(img) # 図を表示

    if __name__ == '__main__':
        main()

    # 実測値の表示
    df_stock = df_stock.reset_index()
    df_stock_i = df_stock.rename(columns={'Open': '始値', 'High': '高値', 'Low': '安値', 'Close': '終値', 'Volume': '出来高'})
    df_stock_i['Date'] = df_stock_i['Date'].dt.strftime('%Y-%m-%d')
    df_stock_i = df_stock_i.set_index('Date')
    df_stock_i = df_stock_i.round()
    df_stock_i.T


    # ↓予測値↓
    st.subheader('◎予測値のチャートと数値一覧')
    st.write('予測に反映する株価の期間: ', date_st, '～', date_fn)

     # prophet の仕様に合わせ、カラム名を変更
    df_stock['ds'] = df_stock['Date'] # 日付データを「ds」に変更
    df_stock['y'] = df_stock['Close'] # 予測したいデータを「y」に変更

    # Prophet のモデルをインスタンス化＆学習
    model = Prophet()
    model.fit(df_stock)

    # 1週間後までの株価を予測
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    forecast = forecast[forecast['ds'].dt.weekday < 5]

    #インデックスとして使用する列名を指定
    forecast = forecast.set_index('ds')

    # 株価の実測値を抽出(グラフ用)
    actual = df_stock.set_index('ds') #インデックスとして使用する列名を指定
    df_stock_nh = actual['High']
    df_stock_n = actual['Close']
    df_stock_nl = actual['Low']

    # 株価の予測値を抽出(グラフ用)
    df_stock_ph = forecast['yhat_upper'].tail(22) #.reset_index(drop=True) # 通常の押し目株価
    df_stock_p = forecast['yhat'].tail(22) #.reset_index(drop=True) # 通常の押し目株価
    df_stock_pl = forecast['yhat_lower'].tail(22) #.reset_index(drop=True) # 最小の押し目株価

    # チャートグラフの描画
    fig, ax = plt.subplots()
    ax.plot(df_stock_nh, color='#ff0000')
    ax.plot(df_stock_nl, color='#00a1e9')
    ax.plot(df_stock_n, color='#000000')
    ax.plot(df_stock_ph, color='#f5b2b2')
    ax.plot(df_stock_pl, color='#bbe2f1')
    ax.plot(df_stock_p, color='#c0c0c0')

    # グラフのタイトルと軸ラベルの設定
    ax.set_title('Stock Price Prediction')
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price')

    # X軸ラベルを縦書きに変更
    plt.xticks(rotation=90)

    # Streamlit上でグラフを表示
    st.pyplot(fig)

    # 営業日以外を「None」にしたパターン(表用)
    forecast_f = forecast.asfreq('D')
    forecast_f.index = forecast_f.reset_index()['ds'].dt.strftime('%Y-%m-%d')

    # 表のインデックス名を変更
    forecast_f['高値'] = forecast_f['yhat_upper']
    forecast_f['通常'] = forecast_f['yhat']
    forecast_f['安値'] = forecast_f['yhat_lower']

    # 株価の予測値を抽出(表用)
    df_stock_nh_f = forecast_f['高値'].round().tail(30)
    df_stock_p_f = forecast_f['通常'].round().tail(30)
    df_stock_pl_f = forecast_f['安値'].round().tail(30)

    #予測結果を表示
    st.dataframe([df_stock_nh_f, df_stock_p_f, df_stock_pl_f])

else:
    st.error(f'証券コード「{code}」は存在しません。')