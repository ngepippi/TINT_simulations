import networkx as nx
import pygraphviz as pyg
import matplotlib.pyplot as plt
import itertools as iter
import csv
import sys
import os
import math
from cairosvg import svg2png

save_dir = "./graphs/"
SVG_DIR = "./graphs/SVG/"
IMAGE_DIR = "./graphs/SVG/IMAGE/"
node_font = "Meiryo"
"""
関手の自分のイメージ
    ある圏の対処と射を別の圏の対処と射に構造を保ったまま移す
    F(・)と書いてあるがやっていることは対応付け
圏Cの対処Xに対して
X->F(X),id_F(x)=F(id_x),F(g・f) = F(g)・F(f)であること
F(g・f) = F(g)・F(f)
合成射が等しいとは最終的なドメインとコドメインが同じであることではないか
合成射であるかどうかを判断はできた
合成射であることを定義できたとして、合成元をどのように判断するのか
合成射は属性として合成元の情報をもつ？
    合成の仕方は1通りではないよな
その射の構成要素を属性として持つ
    単独の射だけならば自分のみを持つ
    ただ合成射であるのに別の名前を付けられている場合もある
その場合は可換である複数の経路を追加する？
    ただこれは合成が深くなると多くなるよな
    どう結合法則であるので正しい順番であればかっこの位置はなくてもよいのでは
対応付けのネット―ワークを生成するとどうか

論文の例ではf\C'->B\C'->A\C'関手は
    b1->b1・f,b2->b2・f
写像であることの確認
    任意のAに対してただ一つBの値が決まる

"""
#複数ステップかかるパスの確率を計算する。全てのステップの確率の掛け算
def path_prob(g,paths):
    edges = convert_node_to_edge(paths)
    mul_prob = 1
    for edge in edges:
        dom,cod=edge
        mul_prob *= g[dom][cod]["weight"]
    return mul_prob

def optimum_path_search(G,dom,cod):
    #重み付きネットワークからコスト最小の経路を探索する
    paths = get_all_paths(G,dom,cod)
    paths = [convert_node_to_edge(nodes) for nodes in paths]
    min_value = sys.float_info.max
    idx = 0
    if len(paths) == 0:
        return None
    for i,path in enumerate(paths):
        weight_sum = 0
        #print(path,end=" ")
        for edge in path:
            dom, cod = edge
            weight_sum += 1.0/G.edges[dom,cod]["weight"]
        #print(weight_sum)
        #最小値をとりたいなら毎回sumを最小値と比較してもうすでに大きいなら切る
        if weight_sum < min_value:
            min_value = weight_sum
            idx = i
    return min_value,paths[idx]

def optimum_path(G,paths):
    min_value = sys.float_info.max
    idx = 0
    if len(paths) == 0:
        return None
    for i,path in enumerate(paths):
        weight_sum = 0
        #print(path,end=" ")
        for edge in path:
            dom, cod = edge
            weight_sum += 1.0/G.edges[dom,cod]["weight"]
        #print(weight_sum)
        #最小値をとりたいなら毎回sumを最小値と比較してもうすでに大きいなら切る
        if weight_sum < min_value:
            min_value = weight_sum
            idx = i
    return min_value,paths[idx]

def load_category(f_node, f_edge, header=True):
    add_nodes=[]
    with open(f_node,encoding="utf-8_sig") as f:
        nodes = csv.reader(f)
        if header:
            node_header = next(nodes)
            #print(node_header)
        for node in nodes:
            add_nodes.extend(node)
            #print(node)
    add_edges=[]
    with open(f_edge,encoding="utf-8_sig") as f:
        edges = csv.reader(f)
        if header:
            edge_header = next(edges)
            #print(edge_header)
        for edge in edges:
            edge[-1] = float(edge[-1])
            add_edges.append(tuple(edge))
            #print(edge)

    g = nx.DiGraph()
    g.add_nodes_from(add_nodes,fontname=node_font)
    g.add_weighted_edges_from(add_edges)
    return g

#すべてのノードの恒等射を作る
def identity_morphism(G,weighted=True):
    nodes = G.nodes()
    edges = G.edges()
    for node in nodes:
        if (node,node) not in edges:
            if weighted:
                G.add_edge(node,node,weight=1.0)
            else:
                G.add_edge(node,node)

def copy_edge(g,cg,edge):
    dom,cod = edge
    attr=g[dom][cod]
    cg.add_edge(dom,cod)
    for key,value in attr.items():
        cg[dom][cod][key] = value

#target圏(仮名)を作る
def target_category(G,target,weighted=False):
    succ = list(G.successors(target))#targetが始点である辺の終点の取得
    succ.remove(target)
    g = nx.DiGraph()
    g.add_node(target,fontname=node_font)
    #Gにあるtergetから終点までの辺を作製
    for cod in succ:#targetから出ていく射を追加する
        if weighted:
            prob = G[target][cod]["weight"]
        else:
            prob = 1.0
        g.add_edge(target,cod,weight=prob)
    for cod1,cod2 in list(iter.permutations(succ,2)):#射の間の射を
        if G.has_edge(cod1,cod2):
            if weighted:
                prob = G[cod1][cod2]["weight"]
            else:
                prob = 1.0
            g.add_edge(cod1,cod2,weight=prob)
    return g

#コスライス圏を作る
def coslice_category(G,target,weighted=False):
    succ = list(G.successors(target))#targetが始点である辺の終点の取得
    succ.remove(target)#コスライス圏での恒等射の扱い方がわからない
    g = nx.DiGraph()
    g.add_nodes_from([(target,cod) for cod in succ],fontname="Meiryo")
    for pair in list(iter.permutations(succ,2)):
        cod1, cod2 = pair
        if G.has_edge(cod1,cod2):
            dom, cod = (target,cod1),(target,cod2)
            if  weighted:
                prob = G[cod1][cod2]["weight"]#重みが必要かはわからないけども
            else:
                prob=1.0
            g.add_edge(dom,cod,weight=prob)
    return g

#与えられた圏とtargetからtargetのコスライス圏を作り返す
def make_coslice_category(G,target):
    cods = list(G.successors(target))#targetが始点である辺の終点の取得
    g1 = nx.DiGraph()
    g1.add_node(target,fontname=node_font)
    #Gにあるtergetから終点までの辺を作製
    for cod in cods:
        g1.add_node(cod,fontname=node_font)
        copy_edge(G,g1,(target,cod))
    #codsの要素を始点とする辺の終点のリストを作り
    #そのリストの中にcodの要素があればその間に辺を作る
    for cod in cods:
        cod_cods = list(G.successors(cod))
        for cc in cod_cods:
            if cc in cods:
                copy_edge(G,g1,(cod,cc))
    return g1

def edge_category(G,edge_name_dict = None):
    #ドメインがtargetでない射をとる
    #その(target,dom),(target,cod)があればそれらの間に射を引く
    edges = G.edges()
    g = nx.DiGraph()
    for i, edge in enumerate(edges):
        #add_edge_category(G,g,edge)
        add_edge_category(G,g,[edge])
    return g

def is_between_edges(G,edge):
    is_share_domain = False
    share_domain = None
    added_dom, added_cod = edge[0],edge[1]
    dom_doms = list(G.pred[added_dom])#追加された射のdomをcodに持つ射のdomを取得
    cod_doms = list(G.pred[added_cod])#追加された射のcodをcodに持つ射のdomを取得

    for idx, dom_dom in enumerate(dom_doms):
        #print(idx,dom_dom)
        if dom_dom in cod_doms :
            is_share_domain = True
            share_domain = dom_dom
    return is_share_domain, share_domain

def add_edge_category(G, g, op_edge):
    op_dom, op_cod = op_edge[0][0],op_edge[-1][-1]
    #print("add_edge_category",op_dom,op_cod)
    is_share_domain, share_domain = is_between_edges(G,(op_dom,op_cod))
    if is_share_domain:
        add_dom, add_cod = (share_domain,op_dom),(share_domain,op_cod)
        g.add_edge(add_dom,add_cod)
        print("edge",add_dom,add_cod)
        #ここから重み対応
#他の属性も追加したい場合は考えて
        s = 0
        for edge in op_edge:
            dom,cod = edge
            attr = G[dom][cod]
            print(attr)
            s += attr["weight"]
#        for key,value in attr.items():
#            g[dom][cod][key] = value
        g[add_dom][add_cod]["weight"] = s/len(op_edge)
    else:
        print("node",(op_dom,op_cod))
        g.add_node((op_dom, op_cod),fontname=node_font)

def remove_edge_category(G, g, op_edge):
    op_dom, op_cod = op_edge[0],op_edge[1]
    is_share_domain, share_domain = is_between_edges(G,op_edge)
    if is_share_domain:
        print(share_domain)
        dom, cod = (share_domain,op_dom),(share_domain,op_cod)
#もし恒等射に値する部分を消そうとしたら消さない、もっとスマートにやりたい
#できるなら呼び出しがわで処理してもらいたい
        if dom == cod:
            return
        g.remove_edge(dom,cod)
    else:
#[要改善]本来ならばないものを消そうとした際はエラーが出てほしい
#よって呼び出し側で重複しているものは処理する必要がある
        if ((op_dom, op_cod)) in g.nodes():
            g.remove_node((op_dom, op_cod))

def apply_edge_category(G, g, op_edge,apply_function):
    #元の圏で行われた操作を、射が対象となるコスライス圏に等価な操作となるように行う
    #同じ対象に射が複数あるような場合を考えるときは拡張
    #射の名前をどうするかの解決を後で考える解決
    apply_function(G,g,op_edge)

def name_dict(edges,names):
    name_dict = {}
    for edge,name in zip(edges,names):
        #すでに存在しているキーの場合は写像でないのでNone返す
        if edge in name_dict:
            return None
        name_dict[edge] = name
    return name_dict

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

#グラフの表示
def show_graph(G, layout="random", node_label=True,edge_label=True):
    fig = plt.figure(figsize=(15,15))
    func = get_layout(layout)
    pos = func(G)
    #node_size = [ d['count']*20 for (n,d) in G.nodes(data=True)]
    nx.draw_networkx_nodes(G, pos)
    if node_label:
        nx.draw_networkx_labels(G, pos, fontsize=6, font_weight="bold")

    #edge_width = [ d['weight']*10 for (u,v,d) in G.edges(data=True)]
    #nx.draw_networkx_edges(G, pos, alpha=0.4, width=edge_width)
    nx.draw_networkx_edges(G, pos,overlap=False, splines='true')
    if edge_label:
        nx.draw_networkx_edge_labels(G, pos, fontsize=6, font_weight="bold")

    plt.axis("off")
    plt.show()
    return fig

def show_graphs(glaphs, layouts, node_label=True,edge_label=True):
    num_glaph = len(glaphs)
    fig = plt.figure(figsize=(16,8))
    for i in range(num_glaph):
        plt.subplot(1,num_glaph,i+1)
        func = get_layout(layouts[i])
        pos = func(glaphs[i])
        nx.draw_networkx_nodes(glaphs[i], pos)
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

    plt.show()
    return fig

def save_graph(G,graph_name,style):
    save_svg = SVG_DIR+graph_name+".svg"
    save_png = IMAGE_DIR+graph_name+".png"
    if os.path.exists(save_svg):
        os.remove(save_svg)
    if os.path.exists(save_png):
        os.remove(save_png)
    #g = nx.nx_agraph.to_agraph(G)
    g = networkX_to_pygraphviz(G,style)
    #g.attr(overlap="false")
    #g.attr(splines='true')
    #file.pdfという名前で出力，レイアウトはcircoを使う
    #g.graph_attr.update(K=4, overlap =False, splines=True)
    g.graph_attr.update(K=6)
    if style == "origin":
        g.layout()
        g.draw(save_png)
    else:
        g.draw(save_svg, prog=style, format="svg")
    # g.draw(save_png,prog=style)
        svg2png(url=save_svg, write_to=save_png)

def save_graphs(graphs,graph_names,styles):
    for graph, graph_name,style in zip(graphs,graph_names,styles):
        save_graph(graph,graph_name,style)

def networkX_to_pygraphviz(G,style):
    r = 2.5
    n = len(list(G.nodes()))
    g = pyg.AGraph(directed=True)
    for i,node in enumerate(list(G.nodes())):
        g.add_node(node,fontname="Meiryo")
        if style != "origin":
            continue
        x, y = r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n)
        g.get_node(node).attr['pos']='{},{}!'.format(x,y) # ノードの位置を指定できる？

    for edge in list(G.edges()):
        dom,cod = edge
        attr = G[dom][cod]
        label = ""
        for key,value in attr.items():
            # label += str(key)+":"+str(value)
            label += str(value)
            pass
        g.add_edge(dom,cod,label=label)
    return g

#辞書型で隣接行列を作る
def adjacency_matrix_dict(G):
    dict = {}
    nodes = G.nodes()
    for node in nodes:
        cods = list(G.successors(node))
        adj_dict = {}
        print(cods)
        for n in nodes:
            if n in cods:
                adj_dict[n] = 1
            else:
                adj_dict[n] = 0
        dict[node] = adj_dict
    return dict

#辞書型の隣接行列からリスト型の隣接行列をつくる
def convert_dict_to_matrix(dict):
    keys = dict.keys()
    matrix = []
    for d in dict.values():
        vec = []
        for key in keys:
            vec.append(d[key])
        matrix.append(vec)
    return matrix

#domainからcodmainまでのループのない経路を返す
def get_all_paths(G, domain, codmain, cutoff=None):
    #all_simple_pathsはsourceからtragetまでのエッジ数cutoffまでのループのない経路を返す
    paths = list(nx.all_simple_paths(G, source=domain, target=codmain,cutoff=cutoff))
    return paths

#domain,codomainまでの合成射を返す
def get_composition_path(G, domain, codmain, cutoff=None):
    #all_simple_pathsはsourceからtragetまでのエッジ数cutoffまでのループのない経路を返す
    paths = get_all_paths(G, domain, codmain, cutoff)
    paths.remove([domain, codmain])
    return paths

#ノードで表された経路をエッジで表現する
def convert_node_to_edge(nodes):
    #print("convert_node_to_edge",nodes)
    edges = []
    edges.append((nodes[0],nodes[1]))
    dom = nodes[1]
    for node in nodes[2:]:
        cod = node
        edges.append((dom,cod))
        dom = cod
    #print("convert_node_to_edge",edges)
    return edges

#最終的な始点と終点を取得
def final_dom_cod(edges):
    #最初にルートとして正しいか
    if is_correct_root(edges):
        return (edges[0][0],edges[-1][1])
    return None

#合成射かどうかを返す
def is_composition_morphism(G,domain,codmain,cutoff=None):
    #注意:恒等射はall_simple_pathsでとれない！！！
    #cutoffが２でよいという保証がない。まあでも圏の合成射は必ず2つの辺の合成
    paths = get_all_paths(G,domain,codmain,cutoff=cutoff)

    #恒等射しかない場合はpathsは空になる
    if paths == []:
        return False

    #恒等射でなければ、その射を削除する
    paths.remove([domain,codmain])
    #まだリストに要素があれば、それは合成射が存在する
    if not paths:
        return False
    else:
        return True

#渡された辺の組が正しい経路を通るか判定
def is_correct_root(edges):
    cod = edges[0][1]
    for edge in edges[1:]:
        dom = edge[0]
        if cod != dom:
            return False
        cod = edge[1]
    return True

def edges_correspondence_table(edges1,edges2):
    len1,len2 = len(edges1),len(edges2)
    if len1 != len2:#対応なので同じ数ないとまずいよね
        return None
    #対応の対角行列を作る
    correspondence_table = [[0 if col != row else 1 for col in range(len1)] for row in range(len2)]
    return correspondence_table

#2つの圏の対象の対応関係を辞書型にする
def node_correspondence_dict(nodes1,nodes2):
    dict = {}
    for node1,node2 in zip(nodes1,nodes2):
        #すでに存在しているキーの場合は写像でないのでNone返す
        if node1 in dict:
            print(node1,"はもうすでに写されている")
            return None
        dict[node1] = node2
    return dict

#2つの圏の対象の対応関係辞書から射の対応関係を辞書型にする
def edge_correspondence_dict(g1,g2,node_dict):
    edges1 = g1.edges()
    edges2 = g2.edges()
    dict = {}
    for edge1 in edges1:
        dom,cod = edge1
        F_dom,F_cod = node_dict[dom],node_dict[cod]
        #print(edge1,"->",(F_dom,F_cod))
        #移り先に存在しない射が指定されている
        if (F_dom,F_cod) not in edges2:
            print((dom,cod),"に対応する射",(F_dom,F_cod),"がない") #対応がない射の表示
            return None
        #同じ射が2回指定されている(写像ではない)
        if (dom,cod) in dict:
            # print("同じ射が、2つの写り先に指定されている")         # 写像ではない場合
            return None
        dict[(dom,cod)] = (F_dom,F_cod)
    return dict

#関手かどうかを判定する
def functor(g1, g2, node_dict, edge_dict):
    edges1 = list(edge_dict.keys())
    edges2 = list(edge_dict.values())
    #print("FUNCTOR")
    #print("is identity")
    #1.恒等射の移りが、自分の対象の移りであるかF(id_A) = id_F(A)
    for edge1 in edges1:
        dom, cod = edge1
        if dom == cod:#恒等射かどうか
            F_dom = node_dict[dom]  #F(A)
            if edge_dict[(dom,dom)] != (F_dom,F_dom):#F(id_A) = id_F(A)か
                return False
    #print("is correct morphism")
    #2.F(A->B) = F(A)->F(B)の判定。移った辺が移った点2組と同じか判定
    for i, edge1 in enumerate(edges1):
        dom,cod = edge1
        F_dom,F_cod = node_dict[dom],node_dict[cod]
        if (F_dom,F_cod) != edges2[i]:
            return False
    # print("get comp morphisms")
    comp_morphisms_indexes = []
    #合成射の添え字を取得する(合成射は別の条件もクリアする必要があるため)
    for i,edge in enumerate(edges1):
        dom,cod = edge
        # print(dom,cod)
        if is_composition_morphism(g1,dom,cod,2):#カットオフを2に指定しているが正しいかはわからない
            comp_morphisms_indexes.append(i)

    #3.合成射がF(g・f) = F(g)・F(f)であるかどうかの判定
    #print("is correct comp morphisms")
    for idx in comp_morphisms_indexes:
        dom,cod = edges1[idx]
        paths = get_composition_path(g1,dom,cod,2)
        paths = list(map(convert_node_to_edge,paths))
        F_comp = edges2[idx]
        for path in paths:
            f = path[0]
            g = path[1]
            f_idx = edges1.index(f)
            g_idx = edges1.index(g)
            F_f = edges2[f_idx]
            F_g = edges2[g_idx]
            if final_dom_cod([F_f,F_g]) != F_comp:
                #print("関手の定義を満たしていない")
                return False
    # print("関手の定義を満たしている")
    return True

#関手である2つの圏から自然変換を可能にする射を返す
def natural_transfomation(g1,g2,node_dict1,edge_dict1,node_dict2,edge_dict2):
    edges1, edges2 = list(edge_dict1.keys()), list(edge_dict2.keys())
    origin_edges = g2.edges()
    origin_edges_pairs = list(iter.product(origin_edges,repeat=2))
    natural_transfomations = {}
    #関手によって移された射を可換にするような自然変換の射を探す
    for edge1,edge2 in zip(edges1, edges2):
        dom1,cod1 = edge1
        dom2,cod2 = edge2
        F_A, F_B = node_dict1[dom1], node_dict1[cod1]#関手Fでの対象の写り先
        G_A, G_B = node_dict2[dom2], node_dict2[cod2]#関手Gでの対象の写り先
        F_f, G_f = edge_dict1[edge1], edge_dict2[edge2]#関手F,Gでの射の写り先
        #print(edge1,edge2)
        #print(F_f,G_f)

        #自然変換の記録に存在しなければ、初期化
        if F_A not in natural_transfomations:
            natural_transfomations[edge1[0]] = set()
        #自然変換の記録に存在しなければ、初期化
        if F_B not in natural_transfomations:
            natural_transfomations[edge1[1]] = set()
        for origin_pair in origin_edges_pairs:
            #print(origin_pair)
            alpha1,alpha2 = origin_pair
            G_alpha = final_dom_cod([alpha1,G_f])
            alpha_F = final_dom_cod([F_f,alpha2])
            if G_alpha == alpha_F and G_alpha != None and alpha_F != None:
                natural_transfomations[edge1[0]].add(alpha1)
                natural_transfomations[edge1[1]].add(alpha2)
                #print("射",F_f,G_f,"に対して",end=" ")
                #print("自然変換",alpha1,alpha2,"がある")
        #可換にできるような自然変換がない
        if len(natural_transfomations[edge1[0]]) == 0 and len(natural_transfomations[edge1[1]]) == 0:
            #なかったらその射に関してはNoneを記録する、これは自然変換の比率を取りたいから
            natural_transfomations[edge1[0]].add(None)
            natural_transfomations[edge1[1]].add(None)
            #return None
    return natural_transfomations


#グラフを統合する
def integration_graph(g1, g2):
    G = nx.DiGraph()
    for edge in list(g1.edges):
        copy_edge(g1,G,edge)
    for edge in list(g2.edges):
        copy_edge(g2,G,edge)
    #G.add_edges_from(edge1)
    #G.add_edges_from(edge2)
    return G

#圏を圏の定義を満たすように合成射を作る
def regulation_graph(G):
    g = integration_graph(G, nx.DiGraph())
    #g = nx.MultiDiGraph()
    nodes = G.nodes()
    edges = G.edges()
    added_edges = []
    #合成射を探し元の圏になければ作成する
    for node in nodes:
        for cod in G.successors(node):
            for c in G.successors(cod):
                if not (node, c) in edges and node != c:
                    g.add_edge(node,c)
                    added_edges.append((node,c))
    print(g.number_of_edges())
    return g,added_edges

def test():
    #true case 1
    nodes1 = ["p","X1","X2","X"]
    nodes2 = [0,0,1,1]
    edges1 = [("p","p"),("X1","X1"),("X2","X2"),("X","X"),("p","X1"),("p","X2"),("X1","X"),("X2","X"),("p","X")]
    edges2 = [(0,0),(0,1),(1,1)]
    g1 = nx.MultiDiGraph()
    g2 = nx.MultiDiGraph()
    g1.add_nodes_from(nodes1)
    g2.add_nodes_from(nodes2)
    g1.add_edges_from(edges1)
    g2.add_edges_from(edges2)
    node_dict1 = node_correspondence_dict(nodes1,nodes2)
    edge_dict1 = edge_correspondence_dict(g1,g2,node_dict1)
    functor(g1,g2,node_dict1,edge_dict1)

    nodes1 = ["p","X1","X2","X"]
    nodes2 = [0,1,1,1]
    edges1 = [("p","p"),("X1","X1"),("X2","X2"),("X","X"),("p","X1"),("p","X2"),("X1","X"),("X2","X"),("p","X")]
    edges2 = [(0,0),(0,1),(1,1)]
    g1 = nx.MultiDiGraph()
    g2 = nx.MultiDiGraph()
    g1.add_nodes_from(nodes1)
    g2.add_nodes_from(nodes2)
    g1.add_edges_from(edges1)
    g2.add_edges_from(edges2)
    show_graph(g1,"shell",True)
    show_graph(g2,"shell",True)
    node_dict2 = node_correspondence_dict(nodes1,nodes2)
    edge_dict2 = edge_correspondence_dict(g1,g2,node_dict2)
    functor(g1,g2,node_dict2,edge_dict2)
    print(natural_transfomation(g1,g2,node_dict1,edge_dict1,node_dict2,edge_dict2))
    print(natural_transfomation(g1,g2,node_dict2,edge_dict2,node_dict1,edge_dict1))

    nodes1 = ["p","X1","X2","X"]
    nodes2 = [0,0,0,1]
    edges1 = [("p","X1"),("p","X2"),("X1","X"),("X2","X"),("p","X")]
    edges2 = [(0,0),(0,1),(1,1)]
    g1 = nx.MultiDiGraph()
    g2 = nx.MultiDiGraph()
    g1.add_nodes_from(nodes1)
    g2.add_nodes_from(nodes2)
    g1.add_edges_from(edges1)
    g2.add_edges_from(edges2)
    node_dict2 = node_correspondence_dict(nodes1,nodes2)
    edge_dict2 = edge_correspondence_dict(g1,g2,node_dict2)
    functor(g1,g2,node_dict2,edge_dict2)
    print(natural_transfomation(g1,g2,node_dict1,edge_dict1,node_dict2,edge_dict2))
    print(natural_transfomation(g1,g2,node_dict2,edge_dict2,node_dict1,edge_dict1))

def test2():
    #離散圏からの関手は自然変換をなすのかの確認
    #実際はなった
    nodeA = [1,2,3,4,5]

    nodeB = ["a","i","u","e","o","あ","い","う","え","お"]
    edge =[("a","あ"),("i","い"),("u","う"),("e","え"),("o","お")]

    C = nx.DiGraph()
    D = nx.DiGraph()

    C.add_nodes_from(nodeA)
    D.add_nodes_from(nodeB)
    D.add_edges_from(edge)
    identity_morphism(C)
    identity_morphism(D)
    show_graphs([C,D],["shell","shell"])

    node_dict_1 = node_correspondence_dict(nodeA,nodeB[0:5])
    edge_dict_1 = edge_correspondence_dict(C,D,node_dict_1)

    node_dict_2 = node_correspondence_dict(nodeA,nodeB[5:])
    edge_dict_2 = edge_correspondence_dict(C,D,node_dict_2)
    print(node_dict_1)
    print(edge_dict_1)
    print(node_dict_2)
    print(edge_dict_2)

    print(natural_transfomation(C,D,node_dict_1,edge_dict_1,node_dict_2,edge_dict_2))

def one_category_rule_apply_test():
    def full_basic_rule(g_exc):
        added_edges = []
        #合成射を探し元の圏になければ作成する
        for node in g_exc.nodes():
            for cod in list(g_exc.successors(node)):
                for c in list(g_exc.successors(cod)):
                    #DiGraohを仮定しているので、射がもうあれば作らない
                    if (node, c) not in g_exc.edges() and node != c:
    #合成射の想起確率を決めてないから、もとの圏に合成射がない場合がある
    #現状はBasic ruleで励起した射の想起確率は1.0とする
    #                   recall_prob = g_exc[node][c]["weight"]
                        g_exc.add_edge(node,c)
                        #g_exc[node][c]["weight"] = 1.0
                        added_edges.append((node,c))
        return added_edges
    def target_category(G,target):
        succ = list(G.successors(target))#targetが始点である辺の終点の取得
        succ.remove(target)
        g = nx.DiGraph()
        g.add_node(target,fontname=node_font)
        #Gにあるtergetから終点までの辺を作製
        for cod in succ:#targetから出ていく射を追加する
            #prob = G[target][cod]["weight"]
            g.add_edge(target,cod)
        for cod1,cod2 in list(iter.permutations(succ,2)):#射の間の射を
            if G.has_edge(cod1,cod2):
                #prob = G[cod1][cod2]["weight"]
                g.add_edge(cod1,cod2)
        return g

    node_A = ["A","B","C","D"]
    node_B = [1,2,3,4]
    edge_A = [("A","B"),("A","C"),("A","D"),("B","C")]
    edge_B = [(1,2),(1,3),(1,4),(2,3)]
    g = nx.DiGraph()

    g.add_edges_from(edge_A)
    g.add_edges_from(edge_B)
    identity_morphism(g)
    #show_graph(g,"shell",True,False)
    g.add_edge("A",1)
    g.add_edge(2,"B")
    #g.add_edge(3,"C")
    #g.add_edge(4,"D")

    full_basic_rule(g)
    full_basic_rule(g)
    full_basic_rule(g)

    show_graph(g,"shell",True,False)
    target_A = target_category(g,"A")
    identity_morphism(target_A)
    target_B = target_category(g,1)
    identity_morphism(target_B)
    show_graphs([target_A,target_B],["shell","shell"],True,False)

    node_pair_B = [ 1 , 2, 3, 4, "B","C","D"]
    node_pair_A = ["A", 2, 3, 4, "B","C","D"]

    node_dict_bmf = node_correspondence_dict(node_pair_B,node_pair_A)
    edge_dict_bmf = edge_correspondence_dict(target_B,target_A,node_dict_bmf)
    is_functor = functor(target_B,target_A,node_dict_bmf,edge_dict_bmf)
    print("Is BMF Functor? : ",is_functor)

    node_pair_B = [ 1 , 2 , 3 , 4 ,"B","C","D"]
    node_pair_A = ["A","B","C","D","B","C","D"]

    node_dict_F = node_correspondence_dict(node_pair_B,node_pair_A)
    edge_dict_F = edge_correspondence_dict(target_B,target_A,node_dict_F)
    is_functor = functor(target_B,target_A,node_dict_F,edge_dict_F)
    print("Is F Functor? : ",is_functor)

    is_natul_trans = natural_transfomation(target_B,target_A,node_dict_bmf,edge_dict_bmf,node_dict_F,edge_dict_F)
    print("IS BMF to F Natural Transfomation? : ",is_natul_trans)

