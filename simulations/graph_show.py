from common import *
from TINT_rule import *
import networkx as nx
from networkx.algorithms import community as cm
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
import os


PLT_IMAGE_DIR = "./graphs/"

#コスライス圏にした場合のノード名を(A,B)タプルから(A->B)の文字列に変換する
def modify_coslice_node_name(cos):
    new_cat = nx.DiGraph()
    origin_nodes = list(cos.nodes())
    origin_edges = list(cos.edges())
    new_node_name_dict = dict()
    new_edge_name_dict = dict()
    #対象を新しい名前に変換する
    for node in origin_nodes:
        dom,cod = node#コスライス圏をcoslice_categoryで作るとnodeがタプルになる
        new_node_name = "("+dom+"->"+cod+")"#タプルから文字列に変換
        new_cat.add_node(new_node_name)
        new_node_name_dict[node] = new_node_name

    #名前を変更する新しい圏に射を引く
    for edge in origin_edges:
        dom,cod = edge
        n_dom,n_cod = new_node_name_dict[dom],new_node_name_dict[cod]
        edge_attr=cos[dom][cod]
        new_cat.add_edge(n_dom,n_cod)
        #元々の射の属性をコピーする
        for key,value in edge_attr.items():
            new_cat[n_dom][n_cod][key] = value
        new_edge_name_dict[edge] = (n_dom,n_cod)
    return new_cat,new_node_name_dict,new_edge_name_dict

#コスライス圏の関手をmodify_coslice_node_nameで変更した文字列に沿うように変更
def modify_coslice_functor_name(func_node_dict,func_edge_dict,node_name_dict_A,edge_name_dict_A,node_name_dict_B,edge_name_dict_B):
    new_func_node_dict = dict()
    new_func_edge_dict = dict()
    #コスライス圏の関手において対象の辞書の対象の名前を変える
    for key,value in func_node_dict.items():
        n_B,n_A = node_name_dict_B[key],node_name_dict_A[value]
        new_func_node_dict[n_B] = n_A
    #コスライス圏の関手において射の辞書の射を構成する対象の名前を変える
    for key,value in func_edge_dict.items():
        n_B,n_A = edge_name_dict_B[key],edge_name_dict_A[value]
        new_func_edge_dict[n_B] = n_A
    return new_func_node_dict,new_func_edge_dict

#通常の圏での関手をコスライス圏での関手に変換する
def coslice_functor(target_A,target_B,F_edge_dict):
    coslice_node_dict = dict()
    coslice_edge_dict = dict()
    for b_edge,t_edge in F_edge_dict.items():
        #print(b_edge,t_edge)
        if b_edge[0] == b_edge[1]:#通常の圏において恒等射であるものは省く
            continue
        if b_edge[0] == target_B:#ドメインがtargetBであればコスライス圏の対象になる
            coslice_node_dict[b_edge] = t_edge
            #またコスライス圏での恒等射の対応を作る
            coslice_edge_dict[(b_edge,b_edge)] = (t_edge,t_edge)
        else:
            dom,cod = b_edge
            cos_edge_B = ((target_B,dom),(target_B,cod))
            dom,cod = t_edge
            cos_edge_A = ((target_A,dom),(target_A,cod))
            coslice_edge_dict[cos_edge_B] = cos_edge_A
    return coslice_node_dict,coslice_edge_dict

#二つの圏の間で対象、射に関手Fの対応の番号を付与した圏を返す
def cor_label_graph(catA,catB,node_dict,edge_dict):
    new_graph_A = nx.DiGraph()
    new_graph_B = nx.DiGraph()

    nodes_idx_B = dict()
    nodes_idx_A = dict()
    new_node_dict_A = dict()
    new_node_dict_B = dict()

    node_pair = []
    node_len = len(set(node_dict.values()))
    idx_alphas = [chr(i) for i in range(97, 97+node_len)]
    node_idx_dict = {node:i for node,i in zip(set(node_dict.values()),idx_alphas)}
    # print(node_idx_dict)
    for pair in node_dict.items():
        node_pair.append(pair)
    for pair in node_pair:
        dom,cod = pair
        idx = str(node_idx_dict[cod])
        cor_dom_name = dom+":"+idx
        cor_cod_name = cod+":"+idx
        new_graph_B.add_node(cor_dom_name)
        new_node_dict_B[dom] = cor_dom_name
        if not new_graph_A.has_node(cor_cod_name):
            new_graph_A.add_node(cor_cod_name)
            new_node_dict_A[cod] = cor_cod_name

    edge_pair = []
    edge_idx_dict = {node:i+1 for i,node in enumerate(set(edge_dict.values()))}

    for pair in edge_dict.items():
        B_edge,A_edge = pair
        #恒等射に属性があるとノードとかぶってしまうので削除
        #ただし射の間の射を考えるときには埋め込みで普通の射が埋め込みに写る可能性があるので
        #表示を考えなければいけない
        # B_edge[0] != B_edge[1]:#恒等射省きたい場合はコメントアウトを外す
        edge_pair.append(pair)
    for pair in edge_pair:
        B_edge,A_edge = pair
        idx = edge_idx_dict[A_edge]
        B_dom,B_cod = new_node_dict_B[B_edge[0]],new_node_dict_B[B_edge[1]]
        new_graph_B.add_edge(B_dom,B_cod)
        new_graph_B[B_dom][B_cod]["Cor"] = idx  #対応番号を付ける

        A_dom,A_cod = new_node_dict_A[A_edge[0]],new_node_dict_A[A_edge[1]]
        if not new_graph_A.has_edge(A_dom,A_cod):
            new_graph_A.add_edge(A_dom,A_cod)
            new_graph_A[A_dom][A_cod]["Cor"] = idx #対応番号を付ける
    return new_graph_A,new_graph_B


def get_layout(layout):
    if layout == "random":
        func = nx.random_layout
    elif layout == "shell":
        func = nx.shell_layout
    elif layout == "spring":
        func = nx.spring_layout
    elif layout == "kamada_kawai":
        func = nx.kamada_kawai_layout
    elif layout =="circular":
        func = nx.circular_layout
    else:
        func = None
    return func

def show_graphs(glaphs, layouts, node_label=True,edge_label=True,titles=None):
    # print("show_graph")
    num_glaph = len(glaphs)
    fig = plt.figure(figsize=(16,8))
    for i in range(num_glaph):
        plt.subplot(1,num_glaph,i+1)
        func = get_layout(layouts[i])
        pos = func(glaphs[i])
        nx.draw_networkx_nodes(glaphs[i], pos,node_size = 16,node_color = "r")
        if node_label:
            nx.draw_networkx_labels(glaphs[i], pos, font_size=14, font_weight="bold")

        #edge_width = [ d['weight']*10 for (u,v,d) in G.edges(data=True)]
        #nx.draw_networkx_edges(G, pos, alpha=0.4, width=edge_width)
        nx.draw_networkx_edges(glaphs[i], pos)
        if edge_label:
            nx.draw_networkx_edge_labels(glaphs[i],pos)

        #nx.draw_networkx(glaphs[i], pos,with_labels=label)
        plt.xticks(color="None")
        plt.yticks(color="None")
        plt.tick_params(length=0)
        if titles != None:
            plt.title(titles[i])
    plt.show()
    return fig

def plt_save_graphs(glaphs, layouts, fname, node_label=True,edge_label=True,titles=None):
    num_glaph = len(glaphs)
    fig = plt.figure(figsize=(16,8))
    for i in range(num_glaph):
        plt.subplot(1,num_glaph,i+1)
        func = get_layout(layouts[i])
        pos = func(glaphs[i])
        nx.draw_networkx_nodes(glaphs[i], pos,node_color="r")
        if node_label:
            nx.draw_networkx_labels(glaphs[i], pos, fontsize=14, font_weight="bold")

        #edge_width = [ d['weight']*10 for (u,v,d) in G.edges(data=True)]
        #nx.draw_networkx_edges(G, pos, alpha=0.4, width=edge_width)
        nx.draw_networkx_edges(glaphs[i], pos)
        if edge_label:
            nx.draw_networkx_edge_labels(glaphs[i],pos)

        #nx.draw_networkx(glaphs[i], pos,with_labels=label)
        plt.xticks(color="None")
        plt.yticks(color="None")
        plt.tick_params(length=0)
        if titles != None:
            plt.title(titles[i])
    plt.savefig(PLT_IMAGE_DIR+fname+".png")
    return fig

def show_weight_color_graph(glaph, layout, node_label=True, edge_label=True, title=None):
    edges = list(glaph.edges())
    weights = [glaph[edge[0]][edge[1]]["weight"] for edge in edges]
    # weights = map(lambda x: x*100,weights)
    plt.figure(figsize=(15,15))
    func = get_layout(layout)
    pos = func(glaph)
    nx.draw_networkx_nodes(glaph,pos,nodelist=list(glaph.nodes()),node_size=20)
    nx.draw_networkx_labels(glaph, pos, font_size=16 ,font_weight="bold",font_color="r")

    color_edges = nx.draw_networkx_edges(glaph,pos,edgelist=edges,edge_color=weights,edge_cmap=plt.cm.Blues,edge_vmin=1,edge_vmax=5)
    # if title != None:
    #     plt.title(title)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(title+".png")
    # plt.show()

def show_nt_graph(est_A,est_B,target_A,target_B,fork_edges,A_rem_edges,B_rem_edges,titles):
    A_nodes = set()
    B_nodes = set()
    share_nodes = set()
    for edge in A_rem_edges:
        dom,cod = edge
        A_nodes.add(cod)
    for edge in B_rem_edges:
        dom,cod = edge
        B_nodes.add(cod)
    BMF_edges = [(target_A,edge[1]) for edge in B_rem_edges if edge[0] == target_B and edge[1] != target_B]
    #print(A_rem_edges)
    #print(B_rem_edges)
    share_nodes = A_nodes & B_nodes
    A_color = "#FF9999"
    B_color = "#9999FF"
    nt_color = "#99FF99"
    fig = plt.figure(figsize=(16,8))
    plt.subplot(1,2,1)
    plt.title(titles[0])
    func = nx.shell_layout
    pos = func(est_A)
    nx.draw_networkx_nodes(est_A, pos,nodelist=A_nodes,node_color=A_color)
    nx.draw_networkx_nodes(est_A, pos,nodelist=[node for node in  B_nodes if node != target_B],node_color=B_color)
    nx.draw_networkx_nodes(est_A, pos,nodelist=share_nodes,node_color="#FF99FF")
    nx.draw_networkx_labels(est_A, pos, fontsize=10, font_weight="bold")
    nx.draw_networkx_edges(est_A, pos)
    nx.draw_networkx_edges(est_A, pos, edgelist=A_rem_edges, edge_color=A_color)
#    nx.draw_networkx_edges(est_A, pos, edgelist=B_rem_edges, edge_color=B_color)
    nx.draw_networkx_edges(est_A, pos, edgelist=[edge for edge in list(B_rem_edges) if edge[0]!=target_B], edge_color=B_color)
#    nx.draw_networkx_edges(est_A, pos, edgelist=BMF_edges, edge_color=B_color)
    nx.draw_networkx_edges(est_A, pos, edgelist=[(target_A,edge[1]) for edge in BMF_edges if edge[1] != target_B], edge_color=B_color)
    nx.draw_networkx_edges(est_A, pos, edgelist=fork_edges, edge_color=nt_color)
    plt.xticks(color="None")
    plt.yticks(color="None")
    plt.tick_params(length=0)

    #colors = [(random.random(),random.random(),random.random()) for i in range(len(corrct_edge))]
    plt.subplot(1,2,2)
    plt.title(titles[1])
    func = nx.shell_layout
    pos = func(est_B)
    nx.draw_networkx_nodes(est_B, pos,node_color=B_color)
    nx.draw_networkx_labels(est_B, pos, fontsize=10, font_weight="bold")
    nx.draw_networkx_edges(est_B, pos)
    #nx.draw_networkx_edges(B_clas, pos, edgelist=corrct_edge, edge_color=colors)
    plt.xticks(color="None")
    plt.yticks(color="None")
    plt.tick_params(length=0)

    plt.show()

def coslice_show_nt_graph(coslice_A,coslice_B,target_A,target_B,node_dict,edge_dict,fork_edges,titles):
    A_nodes = node_dict.values()
    B_nodes = node_dict.keys()
    BMF_nodes = [(target_A,node[1]) for node in B_nodes if node[0] == target_B]
    #print(B_nodes)
    #print(BMF_nodes)

    A_edges = edge_dict.values()
    #print(A_edges)
    B_edges = edge_dict.keys()
    BMF_edges = [((target_A,dom[1]),(target_A,cod[1])) for dom,cod in B_edges if dom[0] == target_B and cod[0] == target_B]

    coslice_fork_edges = [((target_A,edge[0]),(target_A,edge[1])) for edge in fork_edges]

    share_nodes = []
    for B_node in B_nodes:
        dom,cod = B_node
        if dom == target_B and (target_A,cod) in A_nodes:
            share_nodes.append((target_A,cod))

    A_color = "#FF9999"
    B_color = "#9999FF"
    nt_color = "#99FF99"
    fig = plt.figure(figsize=(16,8))
    plt.subplot(1,2,1)
    plt.title(titles[0])
    func = nx.shell_layout
    pos = func(coslice_A)
    nx.draw_networkx_nodes(coslice_A, pos,nodelist=A_nodes,node_color=A_color)
    nx.draw_networkx_nodes(coslice_A, pos,nodelist=BMF_nodes,node_color=B_color)#BMFで写ったもの
    nx.draw_networkx_nodes(coslice_A, pos,nodelist=share_nodes,node_color="#FF99FF")#AとBで共有した概念
    nx.draw_networkx_labels(coslice_A, pos, fontsize=10, font_weight="bold") #フォント

    nx.draw_networkx_edges(coslice_A, pos)
    nx.draw_networkx_edges(coslice_A, pos, edgelist=A_edges, edge_color=A_color)
#    nx.draw_networkx_edges(est_A, pos, edgelist=B_rem_edges, edge_color=B_color)
    nx.draw_networkx_edges(coslice_A, pos, edgelist=BMF_edges, edge_color=B_color)
#    nx.draw_networkx_edges(est_A, pos, edgelist=BMF_edges, edge_color=B_color)
#    nx.draw_networkx_edges(est_A, pos, edgelist=[(target_A,edge[1]) for edge in BMF_edges if edge[1] != target_B], edge_color=B_color)
    nx.draw_networkx_edges(coslice_A, pos, edgelist=coslice_fork_edges, edge_color=nt_color)
    plt.xticks(color="None")
    plt.yticks(color="None")
    plt.tick_params(length=0)

    plt.subplot(1,2,2)
    plt.title(titles[1])
    func = nx.shell_layout
    pos = func(coslice_B)
    nx.draw_networkx_nodes(coslice_B,pos)
    nx.draw_networkx_labels(coslice_B, pos, fontsize=10, font_weight="bold")
    nx.draw_networkx_edges(coslice_B, pos)

    plt.xticks(color="None")
    plt.yticks(color="None")
    plt.tick_params(length=0)
    plt.show()

# 2つの画像を横に連結する関数
# pygraphvizでは複数のグラフをつなげて保存する方法がわからなかったのでopencvで後で結合
# https://watlab-blog.com/2019/09/29/image-combine/#i-4
def image_hcombine(im_info1, im_info2,filepath,is_blank=True,is_frame =False,is_remove=True):
    path1 = im_info1[0]                      # 1つ目の画像のパス
    path2 = im_info2[0]                      # 2つ目の画像のパス
    color_flag1 = im_info1[1]                # 1つ目の画像のカラー/グレー判別値
    color_flag2 = im_info2[1]                # 2つ目の画像のカラー/グレー判別値

    img1 = cv2.imread(path1, color_flag1)    # 1つ目の画像を読み込み
    img2 = cv2.imread(path2, color_flag2)    # 2つ目の画像を読み込み
    # print(img1.shape[:3], img2.shape[:3])    # リサイズ前の画像サイズを表示（検証用）
    # 1つ目の画像に対しカラーかグレースケールかによって読み込みを変える
    if color_flag1 == 1:
        h1, w1, ch1 = img1.shape[:3]         # 画像のサイズを取得（グレースケール画像は[:2]
    else:
        h1, w1 = img1.shape[:2]

    # 2つ目の画像に対しカラーかグレースケールかによって読み込みを変える
    if color_flag2 == 1:
        h2, w2, ch2 = img2.shape[:3]         # 画像のサイズを取得（グレースケール画像は[:2]
    else:
        h2, w2 = img2.shape[:2]
    # img1 = cv2.copyMakeBorder(img1,w1,w1,h1,h1,cv2.BORDER_CONSTANT,value=[0,0,0])
    # img2 = cv2.copyMakeBorder(img1,w2,w2,h2,h2,cv2.BORDER_CONSTANT,value=[0,0,0])

    # 2つの画像の縦サイズを比較して、大きい方に合わせて一方をリサイズする
    if h1 < h2:                              # 1つ目の画像の方が小さい場合
        h1 = h2                              # 小さい方を大きい方と同じ縦サイズにする
        w1 = int((h2 / h1) * w2)             # 縦サイズの変化倍率を計算して横サイズを決定する
        img1 = cv2.resize(img1, (w1, h1))    # 画像リサイズ
    else:                                    # 2つ目の画像の方が小さい場合
        h2 = h1                              # 小さい方を大きい方と同じ縦サイズにする
        w2 = int((h1 / h2) * w1)             # 縦サイズの変化倍率を計算して横サイズを決定する
        img2 = cv2.resize(img2, (w2, h2))    # 画像リサイズ

    # print(img1.shape[:3], img2.shape[:3])    # リサイズ後の画像サイズを表示（検証用）
    if is_blank:
        blank = np.zeros((img1.shape[0],70,3),np.uint8)
        blank += 255
        img = cv2.hconcat([img1,blank])
        img = cv2.hconcat([img, img2])          # 2つの画像を横方向に連結
    else:
        img = cv2.hconcat([img1,img2])

    if is_frame:
        w,h = img.shape[:2]
        img = cv2.copyMakeBorder(img,70,70,30,30,cv2.BORDER_CONSTANT,value=[255,255,255])
        img = cv2.copyMakeBorder(img,2,2,2,2,cv2.BORDER_CONSTANT,value=[0,0,0])

    cv2.imwrite(filepath,img)
    if is_remove:
        if os.path.exists(path1):
            os.remove(path1)
        if os.path.exists(path2):
            os.remove(path2)

def image_vcombine(im_info1, im_info2,filepath,is_blank=True):
    path1 = im_info1[0]                      # 1つ目の画像のパス
    path2 = im_info2[0]                      # 2つ目の画像のパス
    color_flag1 = im_info1[1]                # 1つ目の画像のカラー/グレー判別値
    color_flag2 = im_info2[1]                # 2つ目の画像のカラー/グレー判別値

    img1 = cv2.imread(path1, color_flag1)    # 1つ目の画像を読み込み
    img2 = cv2.imread(path2, color_flag2)    # 2つ目の画像を読み込み
    # print(img1.shape[:3], img2.shape[:3])    # リサイズ前の画像サイズを表示（検証用）

    # 1つ目の画像に対しカラーかグレースケールかによって読み込みを変える
    if color_flag1 == 1:
        h1, w1, ch1 = img1.shape[:3]         # 画像のサイズを取得（グレースケール画像は[:2]
    else:
        h1, w1 = img1.shape[:2]

    # 2つ目の画像に対しカラーかグレースケールかによって読み込みを変える
    if color_flag2 == 1:
        h2, w2, ch2 = img2.shape[:3]         # 画像のサイズを取得（グレースケール画像は[:2]
    else:
        h2, w2 = img2.shape[:2]
    # img1 = cv2.copyMakeBorder(img1,w1,w1,h1,h1,cv2.BORDER_CONSTANT,value=[0,0,0])
    # img2 = cv2.copyMakeBorder(img1,w2,w2,h2,h2,cv2.BORDER_CONSTANT,value=[0,0,0])

    # 2つの画像の縦サイズを比較して、大きい方に合わせて一方をリサイズする
    if w1 < w2:                              # 1つ目の画像の方が小さい場合
        w1 = w2                              # 小さい方を大きい方と同じ縦サイズにする
        h1 = int((w2 / w1) * h2)             # 縦サイズの変化倍率を計算して横サイズを決定する
        bw = h1
        bh = w1
        img1 = cv2.resize(img1, (w1, h1))    # 画像リサイズ
    else:                                    # 2つ目の画像の方が小さい場合
        w2 = w1                              # 小さい方を大きい方と同じ縦サイズにする
        h2 = int((w1 / w2) * h1)             # 縦サイズの変化倍率を計算して横サイズを決定する
        bw = h1
        bh = w1
        img2 = cv2.resize(img2, (w2, h2))    # 画像リサイズ

    if is_blank:
        blank = np.zeros((2,img1.shape[1],3),np.uint8)
        img = cv2.vconcat([img1,blank])
        img = cv2.vconcat([img, img2])          # 2つの画像を横方向に連結
    else:
        img = cv2.vconcat([img1,img2])

    cv2.imwrite(filepath,img)
    if os.path.exists(path1):
        os.remove(path1)
    if os.path.exists(path2):
        os.remove(path2)
