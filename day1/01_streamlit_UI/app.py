import streamlit as st
import pandas as pd
import numpy as np
import time

# ============================================
# ページ設定
# ============================================
# st.set_page_config(
#     page_title="Streamlit デモ",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# ============================================
# タイトルと説明
# ============================================
st.title("Streamlit 初心者向けデモ")
st.markdown("### コメントを解除しながらStreamlitの機能を学びましょう")
st.markdown("このデモコードでは、コメントアウトされた部分を順番に解除しながらUIの変化を確認できます。")

# ============================================
# サイドバー 
# ============================================
st.sidebar.header("デモのガイド")
st.sidebar.info("コードのコメントを解除して、Streamlitのさまざまな機能を確認しましょう。")

# ============================================
# 基本的なUI要素
# ============================================
st.header("基本的なUI要素")

# テキスト入力
st.subheader("テキスト入力")
name = st.text_input("あなたの名前", "ゲスト")
st.write(f"こんにちは、{name}さん！")

# ボタン
# st.subheader("ボタン")
# if st.button("クリックしてください"):
#     st.success("ボタンがクリックされました！")

# チェックボックス
# st.subheader("チェックボックス")
# if st.checkbox("チェックを入れると追加コンテンツが表示されます"):
#     st.info("これは隠れたコンテンツです！")

# スライダー
# st.subheader("スライダー")
# age = st.slider("年齢", 0, 100, 25)
# st.write(f"あなたの年齢: {age}")

# セレクトボックス
# st.subheader("セレクトボックス")
# option = st.selectbox(
#     "好きなプログラミング言語は?",
#     ["Python", "JavaScript", "Java", "C++", "Go", "Rust"]
# )
# st.write(f"あなたは{option}を選びました")

# ============================================
# レイアウト
# ============================================
# st.header("レイアウト")

# カラム
# st.subheader("カラムレイアウト")
# col1, col2 = st.columns(2)
# with col1:
#     st.write("これは左カラムです")
#     st.number_input("数値を入力", value=10)
# with col2:
#     st.write("これは右カラムです")
#     st.metric("メトリクス", "42", "2%")

# タブ
# st.subheader("タブ")
# tab1, tab2 = st.tabs(["第1タブ", "第2タブ"])
# with tab1:
#     st.write("これは第1タブの内容です")
# with tab2:
#     st.write("これは第2タブの内容です")

# エクスパンダー
# st.subheader("エクスパンダー")
# with st.expander("詳細を表示"):
#     st.write("これはエクスパンダー内の隠れたコンテンツです")
#     st.code("print('Hello, Streamlit！')")

# ============================================
# データ表示
# ============================================
# st.header("データの表示")

# サンプルデータフレームを作成
# df = pd.DataFrame({
#     '名前': ['田中', '鈴木', '佐藤', '高橋', '伊藤'],
#     '年齢': [25, 30, 22, 28, 33],
#     '都市': ['東京', '大阪', '福岡', '札幌', '名古屋']
# })

# データフレーム表示
# st.subheader("データフレーム")
# st.dataframe(df, use_container_width=True)

# テーブル表示
# st.subheader("テーブル")
# st.table(df)

# メトリクス表示
# st.subheader("メトリクス")
# col1, col2, col3 = st.columns(3)
# col1.metric("温度", "23°C", "1.5°C")
# col2.metric("湿度", "45%", "-5%")
# col3.metric("気圧", "1013hPa", "0.1hPa")

# ============================================
# グラフ表示
# ============================================
# st.header("グラフの表示")

# ラインチャート
# st.subheader("ラインチャート")
# chart_data = pd.DataFrame(
#     np.random.randn(20, 3),
#     columns=['A', 'B', 'C'])
# st.line_chart(chart_data)

# バーチャート
# st.subheader("バーチャート")
# chart_data = pd.DataFrame({
#     'カテゴリ': ['A', 'B', 'C', 'D'],
#     '値': [10, 25, 15, 30]
# }).set_index('カテゴリ')
# st.bar_chart(chart_data)

# ============================================
# インタラクティブ機能
# ============================================
# st.header("インタラクティブ機能")

# プログレスバー
# st.subheader("プログレスバー")
# progress = st.progress(0)
# if st.button("進捗をシミュレート"):
#     for i in range(101):
#         time.sleep(0.01)
#         progress.progress(i / 100)
#     st.balloons()

# ファイルアップロード
# st.subheader("ファイルアップロード")
# uploaded_file = st.file_uploader("ファイルをアップロード", type=["csv", "txt"])
# if uploaded_file is not None:
#     # ファイルのデータを表示
#     bytes_data = uploaded_file.getvalue()
#     st.write(f"ファイルサイズ: {len(bytes_data)} bytes")
#     
#     # CSVの場合はデータフレームとして読み込む
#     if uploaded_file.name.endswith('.csv'):
#         df = pd.read_csv(uploaded_file)
#         st.write("CSVデータのプレビュー:")
#         st.dataframe(df.head())

# ============================================
# カスタマイズ
# ============================================
# st.header("スタイルのカスタマイズ")

# カスタムCSS
# st.markdown("""
# <style>
# .big-font {
#     font-size:20px ！important;
#     font-weight: bold;
#     color: #0066cc;
# }
# </style>
# """, unsafe_allow_html=True)
# 
# st.markdown('<p class="big-font">これはカスタムCSSでスタイリングされたテキストです！</p>', unsafe_allow_html=True)

# ============================================
# デモの使用方法
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.markdown("""
<style>
/* メイン運勢 */
.main-fortune {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    color: #8e44ad;
    margin: 40px 0 30px 0;
}

/* 小運勢（恋愛・仕事・金運） */
.sub-fortune-container {
    display: flex;
    justify-content: center;
    gap: 40px;
    margin-bottom: 30px;
}
.sub-fortune {
    background-color: #f4ecf7;
    padding: 15px 25px;
    border-radius: 10px;
    text-align: center;
    font-size: 18px;
    font-weight: 500;
    color: #333;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
}

/* ラッキー情報 */
.lucky-info-container {
    display: flex;
    justify-content: space-around;
    margin-top: 20px;
    flex-wrap: wrap;
}
.lucky-card {
    background-color: #fef9e7;
    padding: 15px 20px;
    border-radius: 10px;
    text-align: center;
    width: 30%;
    min-width: 150px;
    margin: 10px;
    box-shadow: 1px 1px 6px rgba(0,0,0,0.05);
}
.lucky-title {
    font-weight: bold;
    margin-bottom: 8px;
    font-size: 16px;
    color: #6e2c00;
}
.lucky-value {
    font-size: 20px;
    color: #117a65;
}
</style>
""", unsafe_allow_html=True)

df_fortune = pd.DataFrame({
    "運勢": ["大吉", "中吉", "小吉", "凶"],
    "恋愛運": ["絶好調", "良好", "普通", "注意"],
    "仕事運": ["絶好調", "マイペースで", "停滞気味", "注意が必要"],
    "金運": ["大金運", "そこそこ", "散財注意", "出費多め"]
})

colors = ["赤", "青", "緑", "金", "紫"]
items = ["ハンカチ", "水筒", "スマホスタンド", "イヤホン", "アロマ"]
keywords = ["挑戦", "癒し", "笑顔", "冷静さ", "冒険"]
advices = [
    "一歩踏み出す勇気が運を引き寄せます。",
    "小さな優しさが大きなチャンスに変わるかも。",
    "いつもより丁寧に行動してみましょう。",
    "自分を信じて、直感を大切にして。"
]

st.title("占いの館")

name = st.text_input("お名前を入力してください")

if name and st.button("運勢を占う"):
    today = datetime.now().strftime("%Y年%m月%d日")
    result = df_fortune.sample(1).iloc[0]
    color = np.random.choice(colors)
    item = np.random.choice(items)
    keyword = np.random.choice(keywords)
    advice = np.random.choice(advices)

    st.markdown(f"<div class='main-fortune'>{name}さんの今日の運勢：{result['運勢']}</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='sub-fortune-container'>
        <div class='sub-fortune'>恋愛運<br>{}</div>
        <div class='sub-fortune'>仕事運<br>{}</div>
        <div class='sub-fortune'>金銭運<br>{}</div>
    </div>
    """.format(result['恋愛運'], result['仕事運'], result['金運']), unsafe_allow_html=True)

    st.markdown("""
    <div class='lucky-info-container'>
        <div class='lucky-card'>
            <div class='lucky-title'>ラッキーカラー</div>
            <div class='lucky-value'>{}</div>
        </div>
        <div class='lucky-card'>
            <div class='lucky-title'>ラッキーアイテム</div>
            <div class='lucky-value'>{}</div>
        </div>
        <div class='lucky-card'>
            <div class='lucky-title'>ラッキーワード</div>
            <div class='lucky-value'>{}</div>
        </div>
    </div>
    """.format(color, item, keyword), unsafe_allow_html=True)

    with st.expander("今日のアドバイス"):
        st.write(advice)

    st.balloons()