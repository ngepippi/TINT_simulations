from common import *
from TINT_rule import *
from graph_show import *
from data_load import *
from TINT_config import *
from TINT_recoder import *

import networkx as nx
import matplotlib.pyplot as plt
import itertools as iter
import random
import pandas as pd
import numpy as np
from tqdm import tqdm,trange

#得られた関手を解釈しやすいAにとってのCはBにとってのDという形で標準出力に表示する
def show_metaphar(F_edge_dict,target_A,target_B):
    for base,target in F_edge_dict.items():
        b_dom,b_cod = base
        t_dom,t_cod = target
        # prob = est_A[t_dom][t_cod]["weight"]
        # print(b_dom,"にとっての",b_cod,"は",t_dom,"にとっての",t_cod,"：",prob)
        print("{0:<5} にとっての {1:<5}\t {2:<5} にとっての {3:<5}".format(b_dom,b_cod,t_dom,t_cod))

#得られた関手を解釈しやすいAにとってのCはBにとってのDという形でファイルに書き出す
def save_metaphar(fname,F_edge_dict,target_A,target_B):
    with open(save_dir+fname,'w') as f:
        for base,target in F_edge_dict.items():
            b_dom,b_cod = base
            t_dom,t_cod = target
            if b_dom == b_cod and t_dom==t_cod:
                continue
            f.write("{0:<5} にとっての {1:<5}\t {2:<5} にとっての {3:<5}\n".format(b_dom,b_cod,t_dom,t_cod))

#喩辞と被喩辞の意味を表す圏の間での射の対応を探索する
def categories_nt_search(g, T, S, est_T, est_S, cutoff):
    if len(list(est_T.nodes())) == 0:
        print("コスライス圏Aのノード数が０")
    if len(list(est_S.nodes())) == 0:
        print("コスライス圏Bのノード数が０")

    #B->Bはfork ruleのやらない、B->Aの射は想定してない
    S_succs = np.array([cod for cod in est_S.successors(S) if cod != S and cod != T])#Bから出ている対象で恒等射とAは省く
    T_succs = np.array([cod for cod in est_T.successors(T) if cod != T])#Aから出ている対象で恒等射は省く

    #いま現状は痩せた圏なので集合型でいい
    fork_edges = set()  #自然変換
    T_rem_edges = set() #Aで残る射
    S_rem_edges = set() #Bで残る射
    BMF_node_dict = {S:T}
    F_node_dict = {S:T}

    #ネストを深くしたくなかったので、numpy配列のブーリアンインデックスで
    #全ての自然変換の候補が励起するかどうかを全ての喩辞の対象について一気に判断する
    #TODO：np.ix_をうまく使えれば最後のforもいらなくなりそう

    #自然変換の探索で使う全ての重みを取得  #確率を取る部分を変えれば合成射まで探すことも可能
    nt_weight_mat = np.array([[g[s_node][t_node]["weight"] for t_node in T_succs] for s_node in S_succs])
    #重みと同じ形状のランダム値が入った配列を生成
    rnd_mat = np.random.rand(len(S_succs),len(T_succs))

    for i, (rnd_list, nt_weight_list) in enumerate(zip(rnd_mat, nt_weight_mat)):
        nt_cand_list = T_succs[rnd_list < nt_weight_list]                   #自然変換の候補の取得
        if nt_cand_list.size == 0:#候補が一つも励起されなかった場合飛ばす
            continue
        nt_cand_weights = nt_weight_list[rnd_list < nt_weight_list]         #候補の部分の重みを取得
        nt_idx_list = np.where(nt_cand_weights == np.max(nt_cand_weights))  #最大の重みのリストを取得
        nt_idx = random.choice(nt_idx_list[0])                              #最大のものが複数あったときにはランダムで
        S_node, T_node = S_succs[i] , nt_cand_list[nt_idx]
        BMF_node_dict[S_node] = S_node  #BMFの記録
        F_node_dict[S_node] = T_node    #Fの記録
        fork_edges.add((S_node, T_node))    #自然変換の要素を記録
        T_rem_edges.add((T,T_node))         #Fでうつされる射を記録
        S_rem_edges.add((S,S_node))         #BMF,Fのうつり元となる射を記録
    return fork_edges, T_rem_edges, S_rem_edges, BMF_node_dict, F_node_dict

#喩辞・被喩辞のコスライス圏の両方が確立している場合のシミュレーション関数
def TINT_simu_est(g, A, B, est_A, est_B, config, recoder):
    sim_times = config.sim_times
    anti_time = config.anti_time
    anti_type = config.anti_type
    seed = config.seed
    fork_cutoff = 1
    nt_cutoff = config.nt_cutoff
    is_show = config.is_show
    is_save = config.is_save
    neigh_limit = config.neigh_limit
    A_name_dict = config.A_name_dict
    B_name_dict = config.B_name_dict

    BMF_is_functor, F_is_functor = False, False
    if is_show: # 圏の初期状態を表示
        show_graphs([est_A,est_B],["shell","shell"],True,False,["sim start A\C","sim start B\C"])

    #est_Aとest_Bの対応付け探索する
    fork_edges, A_rem_edges, B_rem_edges, BMF_node_dict, F_node_dict = categories_nt_search(g, A, B, est_A, est_B,nt_cutoff)

    #対応を取るときはAがBにfでつながった形にしなければいけない。それでは元に戻せないので現状の圏を保管しておく
    tmp_estB = est_B
    tmp_fork,tmp_A,tmp_B = fork_edges, A_rem_edges, B_rem_edges


    # 対応がつかなかった部分をanti-fork ruleで削除する
    if anti_type == "forced":
        est_A,est_B,A_rem_edges,B_rem_edges = forced_anti_fork_rule(est_A,est_B,A,B,fork_edges,A_rem_edges,B_rem_edges,F_node_dict)
    elif anti_type == "non_identity":
        est_A,est_B,A_rem_edges,B_rem_edges = non_indentity_forced_anti_fork_rule(est_A,est_B,A,B,fork_edges,A_rem_edges,B_rem_edges,F_node_dict)
    elif anti_type == "full":
        est_A,est_B = full_anti_fork_rule(est_A,est_B,A,fork_edges,A_rem_edges,B_rem_edges)

    if is_show: # anti-fork ruleで緩和された後の圏を表示
        show_nt_graph(est_A,est_B,A,B,fork_edges,A_rem_edges,B_rem_edges,["sim A\C metaphar","sim B\C"])

    #もしAnti-fork ruleですべてのノードがなくなってしまった場合飛ばす
    if len(list(est_A.nodes())) == 0 or len(list(est_B.nodes())) == 0:
        print("Anti-fork ruleで全てのノードがなくなってしまった")
        return

    # BMFが関手かどうかをチェック
    BMF_edge_dict = edge_correspondence_dict(est_B,est_A,BMF_node_dict)
    if BMF_edge_dict != None:
        BMF_is_functor = functor(est_B,est_A,BMF_node_dict,BMF_edge_dict)

    # Fが関手かどうかをチェック
    F_edge_dict = edge_correspondence_dict(est_B,est_A,F_node_dict)
    if F_edge_dict != None:
        F_is_functor = functor(est_B,est_A,F_node_dict,F_edge_dict)

    if BMF_is_functor == False or F_is_functor == False:
        print("Fが関手にならなかった")
        return

    # 自然変換をなしているかどうかを判定
    metaphar = natural_transfomation(est_B,est_A,BMF_node_dict,BMF_edge_dict,F_node_dict,F_edge_dict)
    if metaphar == None:
        print("自然変換にならなかった")
        return

    #ここでコスライス圏に戻す
    cos_A = coslice_category(est_A,A)
    identity_morphism(cos_A)

    cos_B = coslice_category(est_B,B)
    identity_morphism(cos_B)

    if is_show:
        # show_metaphar(F_edge_dict,A,B)
        pass

    # 関手Fでの対象と射の対応を辞書形で記憶する
    recoder.recode_functor_F(A,B,1,F_edge_dict)

    # 以下表示、記録用

    #コスライス圏Bからコスライス圏AにFでうつされる部分の圏を作る（表示用）
    F_est_A = nx.DiGraph()
    for dom,cod in F_edge_dict.values():
        F_est_A.add_edge(dom,cod)

    if is_show or is_save:
        # 関手で対応付けられた対象と射を分かりやすいように、名前の後に対応番号を付ける
        n_A,n_B = cor_label_graph(F_est_A, est_B, F_node_dict, F_edge_dict)

        # 通常の圏での関手をコスライス圏での関手に変換する
        cos_F_node_dict,cos_F_edge_dict = coslice_functor(A, B, F_edge_dict)

        # コスライス圏の対象の名前を(A,B)から(A→B)に変換する
        new_cos_A,node_name_dict_A,edge_name_dict_A = modify_coslice_node_name(cos_A)
        new_cos_B,node_name_dict_B,edge_name_dict_B = modify_coslice_node_name(cos_B)

        # コスライス圏での関手を（A->B）に変更したので、関手の辞書も変更する
        new_node_dict,new_edge_dict = modify_coslice_functor_name(cos_F_node_dict,cos_F_edge_dict,
                                        node_name_dict_A,edge_name_dict_A,
                                        node_name_dict_B,edge_name_dict_B)

        # コスライス圏も同様に対応番号を付けた形の新しい圏を作る
        cos_n_A,cos_n_B = cor_label_graph(new_cos_A, new_cos_B, new_node_dict, new_edge_dict)
        identity_morphism(cos_n_A)
        identity_morphism(cos_n_B)

    if is_show: #Fによって移された部分に対応番号を付けたものを表示
        show_graphs([n_A,n_B],["shell","shell"],True,True,["sim  A\C F","sim  B\C F"])

    if is_save:#圏の画像を保存する
        header = config.data_header if isinstance(config, Three_metaphor_TINT_config) else ""
        # まずそれぞれの圏を画像として保存する
        save_graphs([n_A,n_B],
            ["EST_"+header+"seed_"+str(seed)+"_"+A_name_dict[A]+"_F",
            "EST_"+header+"seed_"+str(seed)+"_"+B_name_dict[B]+"_F"],
            ["circo","circo"])
        # コスライス圏の状態にしたものも同様に保存する
        save_graphs([cos_n_A,cos_n_B],
            ["EST_"+header+"seed_"+str(seed)+"_"+A_name_dict[A]+"_F_cos",
            "EST_"+header+"seed_"+str(seed)+"_"+B_name_dict[B]+"_F_cos"],
            ["origin","origin"])

        # それぞれの圏の画像を横に結合する
        img1 = [IMAGE_DIR+"EST_"+header+"seed_"+str(seed)+"_"+A_name_dict[A]+"_F.png",1]
        img2 = [IMAGE_DIR+"EST_"+header+"seed_"+str(seed)+"_"+B_name_dict[B]+"_F.png",1]
        filepath = IMAGE_DIR+"EST_"+header+"seed_"+str(seed)+"_"+A_name_dict[A]+"_"+B_name_dict[B]+"_F.png"
        image_hcombine(img1,img2,filepath,False)
        recoder.recode_image_file_name_recode(A, B, filepath)

        # コスライス圏も同様に画像を横に結合する
        img1 = [IMAGE_DIR+"EST_"+header+"seed_"+str(seed)+"_"+A_name_dict[A]+"_F_cos.png",1]
        img2 = [IMAGE_DIR+"EST_"+header+"seed_"+str(seed)+"_"+B_name_dict[B]+"_F_cos.png",1]
        filepath = IMAGE_DIR+"EST_"+header+"seed_"+str(seed)+"_"+A_name_dict[A]+"_"+B_name_dict[B]+"_F_cos.png"
        image_hcombine(img1,img2,filepath,False)
        recoder.recode_cos_image_file_name_recode(A, B, filepath)

    if is_show: # そのステップでの最終的な結果を表示する
        show_graphs([est_A,est_B],["shell","shell"],True,False,["sim A\C result","sim B\C result"])

#3つの比喩についてTINTのシミュレーションを実行する関数
def established_three_metaphor_sim():
    #連想データが置いてあるディレクトリ
    DIR = "./../three_metaphor_data/"

    #全てのイメージのデータを取得する
    node_data = pd.read_csv(DIR+"three_metaphor_images.csv",header=None,encoding="SHIFT-JIS")

    A_targets = ["蝶"]         # 被喩辞
    B_targets = ["踊り子"]       # 喩辞
    A_fname = ["butterfly"]  # 被喩辞の英語
    B_fname = ["dancer"] # 喩辞の英名


    A_name_dict = {key:value for key,value in zip(A_targets,A_fname)} #喩辞のファイル名（記録用）
    B_name_dict = {key:value for key,value in zip(B_targets,B_fname)} #被喩辞のファイル名（記録用）
    sim_iter = 1000                     # シミュレーション回数
    sim_times = 1                       # 圏を発展させるステップ数
    anti_time = 1                       # anti-fork ruleを掛けるタイミング（何ステップごとに掛けるか）
    nt_step = 1                         # 自然変換を探索する際に何ステップで行ける系をまでを見るか
    is_show = False                     # シミュレーション中に圏を表示するか
    is_save = False                     # シミュレーションの結果を画像として保存するか
    anti_type = "full"                  # anti-fork ruleの種類(full,forced)
    seed = 6000                         # ランダムシード
    data_index = "all"                  # 4人のデータを使ったので、記録する際にどのデータを使ったのかのヘッダー(0~3,all)

    # 連想強度データから潜在圏を作る
    assoc_data = load_three_metaphor_data()# データの読み込みindexは4人のうちどのデータを使うか(Noneは全員の平均)
    # pandas の dataframe を入力として nexworkX のグラフにする
    assoc_net = nx.from_pandas_edgelist(df = assoc_data, source='source', target='target',edge_attr=["weight"], create_using=nx.DiGraph)
    identity_morphism(assoc_net) # 恒等射の追加

    #シード値でのループ
    for seed_inc in trange(0,1,desc="SEED_LOOP",leave=False):
        random.seed(seed+seed_inc)
        np.random.seed(seed=seed+seed_inc)

        recoder = Three_metaphor_TINT_recoder(A_targets,B_targets,sim_iter,sim_times,anti_time,anti_type,data_index,seed+seed_inc)
        #比喩でのループ
        for target_A,target_B in zip(tqdm(A_targets,desc="HIYU LOOP",leave=False),B_targets):
            A_nodes = None
            config = Three_metaphor_TINT_config(sim_times,anti_time,anti_type,seed+seed_inc,A_nodes,nt_step,is_show,is_save,A_name_dict,B_name_dict,data_index)

            #コスライス圏Aの対象だけを確立させる（A->?）の射だけ
            est_A = nx.DiGraph()
            A_node_data = node_data[node_data[0]==target_A]
            for cod in A_node_data[1]:
                if target_A == cod or not assoc_net.has_edge(target_A,cod):
                    continue
                est_A.add_edge(target_A,cod)
            identity_morphism(est_A)

            #コスライス圏Bの対象だけを確立させる(B->?)の射だけ
            est_B = nx.DiGraph()
            B_node_data = node_data[node_data[0]==target_B]
            for cod in B_node_data[1]:
                if target_B == cod or not assoc_net.has_edge(target_B,cod):
                        continue
                est_B.add_edge(target_B,cod)
            identity_morphism(est_B)

            #シミュレーション回数でのループ
            for i in trange(sim_iter,desc="SIM_LOOP",leave=False):
                #実際のシミュレーションを行う関数を呼び出す
                TINT_simu_est(assoc_net, target_A, target_B, est_A, est_B, config, recoder)

        # recodeに記録されている関手Fをすべてファイルに吐き出す
        recoder.all_dict_to_csv(True,True)


if __name__ == "__main__":
    established_three_metaphor_sim()
