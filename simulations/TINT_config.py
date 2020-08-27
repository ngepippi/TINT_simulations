import json
import time

#シミュレーションに使う設定をまとめたクラス
class TINT_config(object):
    """docstring for TINT_config."""
    def __init__(self,sim_times,anti_time,anti_type,seed,A_nodes,nt_cutoff,is_show,is_save,A_name_dict,B_name_dict):
        self.sim_times = sim_times      #ステップ何回繰り返すか
        self.anti_time = anti_time      #Anti-fork ruleを起用するステップ幅
        self.anti_type = anti_type
        self.seed = seed                #使用したシード値
        self.neigh_limit = A_nodes      #Neighboring ruleの制限
        self.nt_cutoff = nt_cutoff      #自然変換を何ステップまで探索するか
        self.is_show = is_show          #シミュレーション中にグラフを表示するか
        self.is_save = is_save
        #ここ一緒にしておそらく問題はない
        self.A_name_dict = A_name_dict  #日本語を英語に変換する辞書(ファイル名に使う)
        self.B_name_dict = B_name_dict  #日本語を英語に変換する辞書(ファイル名に使う)
    
    def write_config_json(self):
        config_dict = self.make_config_dict()
        with open("./config.json",mode="w") as fw:
            json.dump(config_dict,fw,indent=4)

    def make_config_dict(self):
        config_dict = {} 
        config_dict["date"]      = time.today()
        config_dict["seed"]      = self.seed
        config_dict["sim_times"] = self.sim_times
        config_dict["anti_time"] = self.anti_time
        config_dict["anti_type"] = self.anti_type
        config_dict["neigh_limit"] = self.neigh_limit
        config_dict["nt_cutoff"] = self.nt_cutoff
        return config_dict

#3つの比喩でが混ざったデータ用の設定をまとめたクラス
class Three_metaphor_TINT_config(TINT_config):
    """docstring for TINT_config."""
    def __init__(self,sim_times,anti_time,anti_type,seed,A_nodes,nt_cutoff,is_show,is_save,A_name_dict,B_name_dict,data_index):
        super().__init__(sim_times,anti_time,anti_type,seed,A_nodes,nt_cutoff,is_show,is_save,A_name_dict,B_name_dict)
        self.data_index = data_index
        self.data_header = "Data_"+str(data_index)+"_"

    def make_config_dict(self):
        config_dict = {} 
        config_dict["date"]      = time.today()
        config_dict["seed"]      = self.seed
        config_dict["sim_times"] = self.sim_times
        config_dict["anti_time"] = self.anti_time
        config_dict["anti_type"] = self.anti_type
        config_dict["neigh_limit"] = self.neigh_limit
        config_dict["nt_cutoff"] = self.nt_cutoff
        config_dict["data"]      = self.data_index
        return config_dict
