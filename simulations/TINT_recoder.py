# TINTの対応づけの記録用クラス-基本的に継承しているものはシミュレーション毎に記録用ファイル名の命名規則などを変更している

class TINT_recoder(object):
    def __init__(self,A_nodes,B_nodes,sim_iter,sim_times,anti_time,anti_type,seed):
        self.sim_pair_dicts = {}
        self.sim_iter = sim_iter
        self.sim_times=sim_times
        self.anti_time = anti_time
        self.anti_type = anti_type
        self.num_anti = sim_times // anti_time
        self.keys = []
        self.anti_times = []
        self.seed = seed
        self.D_PATH = "./edge_correspondence/"

        for i in range(1,self.num_anti+1):
            for A_node,B_node in zip(A_nodes,B_nodes):
                key = (A_node,B_node,i*self.anti_time)
                self.keys.append(key)
                self.sim_pair_dicts[key] = dict()
                self.anti_times.append(i*self.anti_time)

    def show_correspondence(self,target_A,target_B,num_anti,remove_identity=False):
        dict = self.sim_pair_dicts[(target_A,target_B,num_anti*self.anti_time)]

        #この取り方だと重複してしまう、1回出てきたkeyをはじくように変える
        apper_keys = set()
        print(len(set(dict.items())),len(set(dict.values())))
        for key in dict.keys():
            sum_count=0
            if remove_identity:
                keys = [k for k in dict.keys() if k[0] == key[0] and k[0][0]!=k[0][1] and key not in apper_keys]
            else:
                keys = [k for k in dict.keys() if k[0] == key[0] and key not in apper_keys]
            for k in keys:
                # if k in apper_keys:
                #     continue
                B_edge,A_edge = k
                B_dom,B_cod = B_edge
                A_dom,A_cod = A_edge
                count = dict[k]
                sum_count+=count
                print("{0:<7} にとっての {1:<7}\t {2:<5} にとっての {3:<5}\t回数 {4:<3}\t確率 {5}".format(B_dom,B_cod,A_dom,A_cod,count,count/self.sim_iter))
            if keys != []:
                print("{0:<7} にとっての {1:<7}\t は対応がつかなかった \t回数 {2:<3}\t確率 {3}\n".format(B_dom,B_cod,(self.sim_iter-sum_count),(self.sim_iter-sum_count)/self.sim_iter))
            apper_keys |= set(keys)

    def to_csv(self,target_A,target_B,num_anti,remove_identity=False,is_na_count=True):
        #より賢く順番を取るのであれば最初に顕在化させている方のノードをすべて取っておく
        #それをこの関数に渡してなんやかんやするのが一番こうりつがいいか？
        #(B.?->B.?) => (A.?,A.?)の形式になっているのでBのノードを順次回せばできるのでは
        dict = self.sim_pair_dicts[(target_A,target_B,num_anti*self.anti_time)]
        fname = create_recode_file_name(target_A,target_B,num_anti)
        with open(fname,'w') as f:
            apper_keys = set()
            f.write("B_dom,B_cod,A_dom,A_cod,count,probability\n")
            for key in dict.keys():
                sum_count=0
                if remove_identity:
                    keys = [k for k in dict.keys() if k[0] == key[0] and k[0][0]!=k[0][1] and key not in apper_keys]
                else:
                    keys = [k for k in dict.keys() if k[0] == key[0] and key not in apper_keys]
                for k in keys:
                    # if k in apper_keys:
                    #     continue
                    B_edge,A_edge = k
                    B_dom,B_cod = B_edge
                    A_dom,A_cod = A_edge
                    count = dict[k]
                    sum_count+=count
                    f.write("{0},{1},{2},{3},{4},{5}\n".format(B_dom,B_cod,A_dom,A_cod,count,count/self.sim_iter))
                apper_keys |= set(keys)
                if is_na_count and keys != []:
                    f.write("{0},{1},{2},{3},{4},{5}\n".format(B_dom,B_cod,"NA","NA",(self.sim_iter-sum_count),(self.sim_iter-sum_count)/self.sim_iter))

    def all_dict_to_csv(self,remove_identity=False,is_na_count=True):
        for key in self.keys:
            dict = self.sim_pair_dicts[key]
            target_A,target_B,num_anti = key
            fname = self.create_recode_file_name(target_A,target_B,num_anti)
            with open(fname,'w') as f:
                apper_keys = set()
                f.write("B_dom,B_cod,A_dom,A_cod,count,probability\n")
                for key in dict.keys():
                    sum_count=0
                    if remove_identity:
                        keys = [k for k in dict.keys() if k[0] == key[0] and k[0][0]!=k[0][1] and key not in apper_keys]
                    else:
                        keys = [k for k in dict.keys() if k[0] == key[0] and key not in apper_keys]
                    for k in keys:
                        # if k in apper_keys:
                        #     continue
                        B_edge,A_edge = k
                        B_dom,B_cod = B_edge
                        A_dom,A_cod = A_edge
                        count = dict[k]
                        sum_count+=count
                        f.write("{0},{1},{2},{3},{4},{5}\n".format(B_dom,B_cod,A_dom,A_cod,count,count/self.sim_iter))
                    apper_keys |= set(keys)
                    if is_na_count and keys != []:
                        f.write("{0},{1},{2},{3},{4},{5}\n".format(B_dom,B_cod,"NA","NA",(self.sim_iter-sum_count),(self.sim_iter-sum_count)/self.sim_iter))

    def create_recode_file_name(self, target_A, target_B, num_anti):
        return self.D_PATH+"seed_" + str(self.seed)+"_"+ target_A+"_"+target_B+"_"+self.anti_type+"_anti_"+str(num_anti)+"_iter_"+str(self.sim_iter)+"_correspondence.csv"

    # 対応づいた連想を記録する関数
    def recode_functor_F(self,A,B,anti_time,F_edge_dict):
        F_dict = self.sim_pair_dicts[(A,B,anti_time)]
        for B_edge,A_edge in F_edge_dict.items():
            if (B_edge,A_edge) in F_dict:
                F_dict[(B_edge,A_edge)] +=1
            else:
                F_dict[(B_edge,A_edge)] = 1

    # 通常の圏の画像のファイル名を記録しておく
    def recode_image_file_name_recode(self, A, B, filepath):
        if (A,B) in self.img_name_dict:
            self.img_name_dict[(A,B)].append(filepath)
        else:
            self.img_name_dict[(A,B)] = [filepath]

    # コスライス圏の画像のファイル名を記録しておく
    def recode_cos_image_file_name_recode(self, A, B, filepath):
        if (A,B) in self.img_name_cos_dict:
            self.img_name_cos_dict[(A,B)].append(filepath)
        else:
            self.img_name_cos_dict[(A,B)] = [filepath]

# 3つの比喩に対して対象同士のシミュレーションを行う際の、記録用クラス
class Three_metaphor_TINT_recoder(TINT_recoder):
    def __init__(self,A_nodes,B_nodes,sim_iter,sim_times,anti_time,anti_type,data_index,seed):
        super().__init__(A_nodes,B_nodes,sim_iter,sim_times,anti_time,anti_type,seed)
        self.data_index = data_index
        self.D_PATH = "./object_edge_correspondence/"
        self.data_header = "Date_{}_".format(self.data_index)
        self.img_name_dict = dict()
        self.img_name_cos_dict = dict()
    def create_recode_file_name(self, target_A, target_B, num_anti):
        return self.D_PATH+self.data_header+"seed_"+str(self.seed)+"_"+target_A+"_"+target_B+"_"+self.anti_type+"_anti_"+str(num_anti)+"_iter_"+str(self.sim_iter)+"_correspondence.csv"

# 3つの比喩に対して三角構造同士のシミュレーションを1つの三角構造に対して行う際の、記録用クラス
class Three_metaphor_TINT_recoder_for_edge(TINT_recoder):
    def __init__(self,A_nodes,B_nodes,sim_iter,sim_times,anti_time,anti_type,data_index,seed):
        super().__init__(A_nodes,B_nodes,sim_iter,sim_times,anti_time,anti_type,seed)
        self.data_index = data_index
        self.data_header = "FOREDGE_Date_{}_".format(self.data_index)
        self.img_name_dict = dict()
        self.img_name_cos_dict = dict()
    def create_recode_file_name(self, target_A, target_B, num_anti):
        return self.D_PATH+self.data_header+"seed_"+str(self.seed)+"_"+target_A+"_"+target_B+"_"+self.anti_type+"_anti_"+str(num_anti)+"_iter_"+str(self.sim_iter)+"_correspondence.csv"

# 3つの比喩に対して三角構造同士のシミュレーションを全ての三角構造に対して行う際の、記録用クラス（三角構造の情報をファイル名につける）
class Three_metaphor_TINT_recoder_for_edge_all(TINT_recoder):
    def __init__(self,A_nodes,B_nodes,B_dom,B_cod,sim_iter,sim_times,anti_time,anti_type,data_index,seed):
        super().__init__(A_nodes,B_nodes,sim_iter,sim_times,anti_time,anti_type,seed)
        self.data_index = data_index
        self.D_PATH = "./tri_edge_correspondence/"
        self.data_header = "FOREDGE_Date_{}_".format(self.data_index)
        self.img_name_dict = dict()
        self.img_name_cos_dict = dict()
        self.B_dom = B_dom
        self.B_cod = B_cod
    def create_recode_file_name(self, target_A, target_B, num_anti):
        return self.D_PATH+self.data_header+"seed_"+str(self.seed)+"_"+target_A+"_"+target_B+"_"+self.B_dom+"_"+self.B_cod+"_"+self.anti_type+"_anti_"+str(num_anti)+"_iter_"+str(self.sim_iter)+"_correspondence.csv"


class word2vec_TINT_recoder(TINT_recoder):
    def __init__(self,A_nodes,B_nodes,sim_iter,sim_times,anti_time,anti_type,seed):
        super().__init__(A_nodes,B_nodes,sim_iter,sim_times,anti_time,anti_type,seed)
        self.img_name_dict = dict()
        self.img_name_cos_dict = dict()

    def create_recode_file_name(self, target_A, target_B, num_anti):
        return self.D_PATH+"word2vec_seed_"+str(self.seed)+"_"+target_A+"_"+target_B+"_"+self.anti_type+"_anti_"+str(num_anti)+"_iter_"+str(self.sim_iter)+"_correspondence.csv"
