from common import *
import networkx as nx
import itertools as iter
import random

#もともと考えていたbasic_ruleで合成射があるような場合は圏全体で作る
#合成射のつくりかたの順番で作られない合成射があるかどうか
def full_basic_rule(g_exc):
    added_edges = []
    #合成射を探し元の圏になければ作成する
    for node in list(g_exc.nodes()):
        for cod in list(g_exc.successors(node)):
            for c in list(g_exc.successors(cod)):
                #DiGraohを仮定しているので、射がもうあれば作らない
                if not g_exc.has_edge(node,c) and node != c:
#合成射の想起確率を決めてないから、もとの圏に合成射がない場合がある
#現状はBasic ruleで励起した射の想起確率は1.0とする
#                   recall_prob = g_exc[node][c]["weight"]
                    g_exc.add_edge(node,c,weight=1.0)
                    added_edges.append((node,c))
    return added_edges

def local_neighboring_rule(g,g_exc,target,A_nodes):
    cont_edges = [(target,node) for node in list(g.successors(target)) if node in A_nodes]
    add_edges = []
    if cont_edges == []:
        return
    #励起した射から続く射が励起するかどうかを探索
    for cont_edge in cont_edges:
        dom,cod = cont_edge
        prob = g[dom][cod]["weight"]
        if random.random() < prob and not g_exc.has_edge(dom,cod):
            g_exc.add_edge(dom,cod,weight = prob)
            #g_exc.add_edge(dom,cod)
            add_edges.append(cont_edge)
    return add_edges

def neighboring_rule(g, g_exc, prev_add_edges = []):
    exc_nodes = list(g_exc.nodes())#現在顕在圏にある対象
    for exc_node in exc_nodes:
        #その対象から続く対象
        cont_nodes = list(g.successors(exc_node))
        now_add_edges = []
        if cont_nodes == []:
            return
        #励起した射から続く射が励起するかどうかを探索
        #print(cod,"：",cont_edges)
        for cont_node in cont_nodes:
            dom,cod = exc_node, cont_node
            prob = g[dom][cod]["weight"]
            #現在励起していない射か、乱数で励起した場合
            if not g_exc.has_edge(dom,cod) and random.random() < prob:
                # print(dom,cod,prob)
                g_exc.add_edge(dom,cod,weight = prob)
                prev_add_edges.append((dom,cod))
                now_add_edges.append((dom,cod))
    return prev_add_edges

def local_fork_rule(g, g_exc, target,cutoff = 1):
    if len(list(g_exc.nodes())) == 0:
        print("fork ruleでノード数０が確認された")
        return
    succ = list(g_exc.successors(target))
    succ.remove(target)
    add_edges=[]
    for cod1,cod2 in list(iter.permutations(succ,2)):
        if cod1 == cod2:#恒等射は飛ばす
            # print("fork identity")
            continue
        if g_exc.has_edge(cod1,cod2):#もうすでにあれば飛ばす
            continue
        #域を共有する射の余域の間の経路の取得(これは潜在圏)
        #cutoffが1ならば直接つなぐような射、これはあとあと拡張する all_shortest_paths()とかに
        paths = get_all_paths(g,cod1,cod2,1)
        #pathが見つからない場合は飛ばす
        if paths == [] or paths == None:
            continue
        else:
            sum_prob = 0
            #全ての最短パスの積の和を取る
            for path in paths:
                sum_prob += path_prob(g,path)
            if random.random() <= sum_prob:
                g_exc.add_edge(cod1,cod2,weight=sum_prob)
                add_edges.append((cod1,cod2,sum_prob))
    return add_edges

def forced_anti_fork_rule(est_A,est_B,target_A, target_B, fork_edges,A_rem_edges,B_rem_edges,node_dict):
    #自然変換の要素を含む三角構造以外を切る　これだとコスライス圏は対象しか残らない
    correct_edge_B = set()
    correct_edge_A = set()

    #コスライス圏の射となる射で正しく写っている射を残す
    for edge in [node for node in list(est_B.edges()) if node != target_B]:
        B_dom,B_cod = edge
        if B_dom not in node_dict or B_cod not in node_dict:
            continue
        A_dom,A_cod = node_dict[B_dom],node_dict[B_cod]
        if est_A.has_edge(A_dom,A_cod):
            correct_edge_B.add((B_dom,B_cod))
            correct_edge_A.add((A_dom,A_cod))

    antied_estA, antied_estB = nx.DiGraph(), nx.DiGraph()
    antied_estA.add_edges_from(list(A_rem_edges))   #自然変換のコドメインとAをつなぐ射を残す
    antied_estA.add_edges_from(list(fork_edges))    #自然変換の要素を残す
    antied_estA.add_edges_from(list(correct_edge_A))#Fで正しく写る射を残す
    #antied_estA.add_edges_from(list(correct_edge_B))#Fで正しく写る射を残す(target_BがA\Cに入っている)
    antied_estA.add_edges_from([edge for edge in list(correct_edge_B) if edge[0]!=target_B])#Fで正しく写る射を残す(target_BがA\Cに入っていない)

    antied_estB.add_edges_from(list(B_rem_edges))   #自然変換のドメインとBをつなぐ射を残す
    antied_estB.add_edges_from(list(correct_edge_B))#Fで正しく写る射を残す

    #BMFで写される射を残す
    for edge in fork_edges:
        B_node ,_ = edge
        antied_estA.add_edge(target_A,B_node)

    identity_morphism(antied_estA)
    identity_morphism(antied_estB)

    return antied_estA,antied_estB,(A_rem_edges | correct_edge_A),(B_rem_edges|correct_edge_B)

def non_indentity_forced_anti_fork_rule(est_A,est_B,target_A, target_B, fork_edges,A_rem_edges,B_rem_edges,node_dict):
    #自然変換の要素を含む三角構造以外を切る　これだとコスライス圏は対象しか残らない
    correct_edge_B = set()
    correct_edge_A = set()

    #コスライス圏の射となる射で正しく写っている射を残す
    for edge in [node for node in list(est_B.edges()) if node != target_B]:
        B_dom,B_cod = edge
        if B_dom not in node_dict or B_cod not in node_dict:
            continue
        A_dom,A_cod = node_dict[B_dom],node_dict[B_cod]

        #A_domからA_codへの射が存在しなおかつそれが恒等射ではない場合だけ、残す
        if est_A.has_edge(A_dom,A_cod) and A_dom != A_cod:
            correct_edge_B.add((B_dom,B_cod))
            correct_edge_A.add((A_dom,A_cod))

    antied_estA, antied_estB = nx.DiGraph(), nx.DiGraph()
    antied_estA.add_edges_from(list(A_rem_edges))   #自然変換のコドメインとAをつなぐ射を残す
    antied_estA.add_edges_from(list(fork_edges))    #自然変換の要素を残す
    antied_estA.add_edges_from(list(correct_edge_A))#Fで正しく写る射を残す
    #antied_estA.add_edges_from(list(correct_edge_B))#Fで正しく写る射を残す(target_BがA\Cに入っている)
    antied_estA.add_edges_from([edge for edge in list(correct_edge_B) if edge[0]!=target_B])#Fで正しく写る射を残す(target_BがA\Cに入っていない)

    antied_estB.add_edges_from(list(B_rem_edges))   #自然変換のドメインとBをつなぐ射を残す
    antied_estB.add_edges_from(list(correct_edge_B))#Fで正しく写る射を残す

    #BMFで写される射を残す
    for edge in fork_edges:
        B_node ,_ = edge
        antied_estA.add_edge(target_A,B_node)

    identity_morphism(antied_estA)
    identity_morphism(antied_estB)

    return antied_estA,antied_estB,(A_rem_edges | correct_edge_A),(B_rem_edges|correct_edge_B)


def full_anti_fork_rule(est_A,est_B,target_A, fork_edges,A_rem_edges,B_rem_edges):
    #自然変換の要素を含む三角構造以外を切る　これだとコスライス圏は対象しか残らない
    antied_estA, antied_estB = nx.DiGraph(), nx.DiGraph()
    antied_estA.add_edges_from(list(A_rem_edges))
    antied_estA.add_edges_from(list(fork_edges))
    antied_estB.add_edges_from(list(B_rem_edges))
    for edge in fork_edges:
        B_node ,_ = edge
        antied_estA.add_edge(target_A,B_node)
    identity_morphism(antied_estA)
    identity_morphism(antied_estB)
    return antied_estA,antied_estB

#TINTを実行することで得られた、対応の良さを点数化する
#点数は多次元で現状（ノードがいくつ残ったか、射がいくつ残ったか、）
#ただし対応の良さが比喩の解釈の良さであるとは思わないこと
def cal_score(base_cat_S,base_cat_T,base_cos_cat_S,base_cos_cat_T,F_node_dict,F_edge_dict,F_cos_node_dict,F_cos_edge_dict):
    score = [0 for i in range(6)]

    S_all_node_num = len(base_cat_S.nodes())#元々のターゲットの圏にある全てのノード数
    S_all_edge_num = len(base_cat_S.edges())#元々のターゲットの圏にある全てのエッジ数
    T_all_node_num = len(base_cat_T.nodes())#元々のソースの圏にある全てのノード数
    T_all_edge_num = len(base_cat_T.edges())#元々のソースの圏にある全てのエッジ数

    base_nodes_S = list(base_cat_S.nodes())
    base_edges_S = list(base_cat_S.edges())
    base_nodes_T = list(base_cat_T.nodes())
    base_edges_T = list(base_cat_T.edges())

    antied_nodes_S = list(F_node_dict.keys())
    antied_edges_S = list(F_edge_dict.keys())
    antied_nodes_T = list(F_node_dict.values())
    antied_edges_T = list(F_edge_dict.values())

    #ソースの圏の対象がどれくらい保たれたか
    S_node_score = 0
    for base_node in base_nodes_S:
        if base_node in antied_nodes_S:
            S_node_score += 1
    #ソースの圏の射がどれくらい保たれたか
    S_edge_score = 0
    for base_edge in base_edges_S:
        if base_edge in antied_edges_S:
            S_edge_score += 1

    #ターゲットの圏の対象がどれくらい保たれたか
    T_node_score = 0
    for base_node in base_nodes_T:
        if base_node in antied_nodes_T:
            T_node_score += 1
    #ターゲットの圏の射がどれくらい保たれたか
    T_edge_score = 0
    for base_edge in base_edges_T:
        if base_edge in antied_edges_T:
            T_edge_score += 1

    #三角構造がどのくらい保たれたか
    #コスライス圏の恒等射でない射がターゲットでも恒等射ではない射に写っているか
    all_cos_edges  = [(dom,cod) for dom,cod in list(base_cos_cat_S.edges()) if dom != cod]
    antied_cos_edges = [(dom,cod) for dom,cod in F_cos_edge_dict.keys() if dom != cod]
    tri_num = len(all_cos_edges) # 三角構造の数
    tri_score = 0

    for dom,cod in antied_cos_edges:
        F_dom,F_cod = F_cos_node_dict[dom],F_cos_node_dict[cod]
        if F_dom != F_cod:
            tri_score += 1

    print("ソースの対象の数：",S_all_node_num,"ソースの射の数：",S_all_edge_num)
    print("ターゲットの対象の数",T_all_node_num,"ターゲットの射の数",T_all_edge_num)
    print("ソースの三角構造の数",tri_num)
    print("ソースで残った対象：",S_node_score,"ソースで残った射：",S_edge_score)
    print("ターゲット残った対象",T_node_score,"ターゲット残った射",T_node_score)
    print("三角構造が写された数",tri_score)
    print("ソースで残った対象の割合：",S_node_score/S_all_node_num,"ソースで残った射の割合：",S_edge_score/S_all_edge_num)
    print("ターゲット残った対象の割合",T_node_score/T_all_node_num,"ターゲット残った射の割合",T_node_score/T_all_edge_num)
    print("三角構造が写された数の割合",tri_score/tri_num)

    return [S_node_score,S_edge_score,T_node_score,T_edge_score,tri_score]
