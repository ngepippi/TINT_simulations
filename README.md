# 不定自然変換理論に基づく比喩理解モデル：<br>シミュレーションプログラム
## コードの概要
### データ
- three_metaphor_data/three_metaphor_images.csv
    - 「蝶は踊り子のようだ」、「粉雪は羽毛のようだ」、「笑顔は花のようだ」の喩辞、被喩辞から連想したイメージ(それぞれ8個)のデータ
- three_metaphor_data/three_metaphor_assoc_data.csv
    - 「蝶は踊り子のようだ」、「粉雪は羽毛のようだ」、「笑顔は花のようだ」の全てのイメージ間の連想データ
- human_correspondence/human_correspondence.csv
    - 「蝶は踊り子のようだ」に関して実験を行った結果の人間の比喩の解釈となるデータ

### プログラム
- common. py
    - グラフの操作、恒等射の追加、関手をなしているか、自然変換をなしているかなどの関数の定義
- data_load.py
    - 人間の連想確率のデータを読み出すための関数を定義
- object_established_coslice_simulator.py
    - 対象同士の対応づけのシミュレーションを行うためのコード
- graph_show.py
    - グラフを表示するための関数をまとめたもの
- TINT_condig.py
    - シミュレーションを行う際のパラメータや設定を扱うためのクラス
- TINT_recoder.py
    - シミュレーション結果である対応づけを記録するためのクラス
- TINT_rule.py
    - TINTで用いる射の励起緩和のルールを関数で定義
- tri_established_coslice_simulator.py
    - 三角構造同士のシミュレーションを行うためのコード
- analysis. py
    - 対象同士、三角構造同士のシミュレーション結果のヒートマップの作成
    - 人間の比喩の解釈となるヒートマップの作成
    - 対象同士、三角構造同士のシミュレーション結果と人間の比喩解釈データから相関係数を計算
    - 対象同士の人間、三角構造同士と人間の相関係数の比較

## 論文再現の手順
1. ### シミュレーションの実行
    - 対象同士のシミュレーションを行うためにobject_established_coslice_simulator.pyを実行する。(この結果はobject_edge_correspondenceに格納される)
    - 三角構造同士のシミュレーションを行うためにtri_established_coslice_simulator.pyを実行する。(この結果はtri_edge_correspondenceに格納される)

2. ### 連想確率、シミュレーション結果、人間の比喩解釈のヒートマップ出力
    - analysis.pyのmain関数の上段のブロックを実行する(結果は/heatmapに格納される)
    - 連想確率のヒートマップを作成する(論文中図6)：<br>adj_matrix
    - 人間の対応づけのヒートマップを作成する(論文中図8)：<br>human_correspondence_heatmap
    - 対象同士のTINTの対応づけのヒートマップを作成する(論文中図9)：<br>object_TINT_edge_correspondence_heatmap
    - 三角構造同士の対応づけのヒートマップを作成する(論文中図10)：<br>tri_TINT_edge_correspondence_heatmap

3. ### 相関係数の計算
    - analysis.pyのmain関数の中段のブロックを実行する(結果は/correfに格納される)
    - 人間と対象同士の対応づけを行うTINTの相関係数を計算する：<br>human_tri_data_correlation_to_csv
    - 人間と三角構造同士の対応づけを行うTINTの相関係数を計算する<br>    human_object_data_correlation_to_csv

4. ### 対象同士と三角構造同士の相関係数の比較
    - analysis.pyのmain関数の下段のブロックを実行する(標準出力に表示される)：<br> object_correlation_analysis_over_th
    - 対象同士と人間の相関係数で閾値(0.4)を超えているものを表示：<br>tri_correlation_analysis_over_th
    - 三角構造同士と人間の相関係数で閾値(0.4)を超えているものを表示(論文中表3)：<br>
    - 対象同士の相関係数を上回っている三角構造、下回っている三角構造をそれぞれ表示(論文中表4,5)：<br>compare_tri_and_object_correlation
    - 上回っている、下回っている三角構造でその射の連想確率が0.75以上のものを表示(論文中表6,7)：<br>over_less_tri_and_object_assoc_prob_th