import os
from preprocess import time_extract
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from config import config
import pandas as pd
from util import check_dir

class AitLog:
    # 针对AitLog数据集进行预处理的方法
    name_path_map = {'audit': 'audit/audit.log',
                     'auth': 'auth.log',
                     'mail_log': 'mail.log',
                     'suricata_fast': 'suricata/fast.log',
                     'daemon': 'daemon.log',
                     'messages': 'messages',
                     'syslog': 'syslog',
                     'user': 'user.log',
                     'apache_access': 'apache2/access.log',
                     'apache_error': 'apache2/error.log',
                     'exim4_main': 'exim4/mainlog',
                     'suricata_eve': 'suricata/eve.json'}
    name_label_map = {'apache_error': {'0', 'upload', 'nikto'},
                      'auth': {'0', 'hydra'},
                      'user': {'0', 'upload', 'nikto', 'hydra', 'webshell'},
                      'apache_access': {'0', 'nikto', 'hydra', 'webshell'}}
    labels = ['0', 'upload', 'nikto', 'hydra', 'webshell']
    name_list = ['apache_error', 'auth', 'user', 'apache_access']
    label_stat_file = check_dir('data/AitLog/stat/label_stat.csv')
    cluster_stat_file = check_dir('data/AitLog/stat/cluster_stat.csv')
    host_list = ['cup', 'insect', 'onion', 'spiral']

    def __init__(self, root):
        self.root = root


class AitLogDerived(AitLog):
    def __init__(self, name):
        super().__init__(root='D:/data/AIT-LDS-v1_1')
        self.name = name
        self.size = 0
        self.log_file_list = []
        self.label_file_list = []
        self.framed_file=check_dir('data/AitLog/framed/{}.csv'.format(self.name))
        self.cluster_seq_file = check_dir('data/AitLog/cluster_seq/{}.csv'.format(self.name))
        self.drain_config_file = 'preprocess/drain3.ini'
        self.cluster_rate = 0.0001
        self.label_set = self.name_label_map[self.name]
        #self.set_path()

    def set_path(self):
        for host in self.host_list:
            log_file = '{}/data/mail.{}.com/{}'.format(self.root, host, self.name_path_map[self.name])
            label_file = '{}/labels/mail.{}.com/{}'.format(self.root, host, self.name_path_map[self.name])
            self.log_file_list.append(log_file)
            self.label_file_list.append(label_file)
            self.size += os.path.getsize(log_file)
        result = {'name': self.name, 'host_list': self.host_list, 'size': self.size,
                  'log_file_list': self.log_file_list, 'label_file_list': self.label_file_list,
                  'framed_file': self.framed_file, 'cluster_seq_file': self.cluster_seq_file,
                  'label_stat_file': self.label_stat_file, 'cluster_stat_file': self.cluster_stat_file}
        return result

    def gen_framed(self):
        print("generating framed file for {}...".format(self.name))
        drain_config = TemplateMinerConfig()
        drain_config.load(self.drain_config_file)
        drain_config.profiling_enabled = True
        template_miner = TemplateMiner(config=drain_config)

        framed_lines = []
        for i in range(len(self.host_list)):
            with open(self.log_file_list[i]) as f1, open(self.label_file_list[i]) as f2:
                log_lines = f1.readlines()
                label_lines = f2.readlines()
            for j in range(len(log_lines)):
                origin_log = log_lines[j].rstrip()
                label = label_lines[j].rstrip().split(',')[0]
                timestamp, input_drain = time_extract.match.get(self.name)(origin_log)
                drain_result = template_miner.add_log_message(input_drain)
                row = [self.host_list[i], origin_log, 'template', drain_result['cluster_id'], timestamp, label]
                framed_lines.append(row)
        # 先用list存储数据，再生成df，比直接插入df快得多
        framed = pd.DataFrame(framed_lines,
                              columns=['host', 'origin_log', 'template', 'cluster_id', 'timestamp', 'label'])

        def func(cluster_id):
            return str(template_miner.drain.id_to_cluster[cluster_id].get_template())

        framed['template'] = framed['cluster_id'].apply(func)
        framed.to_csv(self.framed_file, index=False)
        print("generation finished")

    def gen_cluster_seq(self):
        print("generating cluster_seq file for {}...".format(self.name))
        with open(self.framed_file) as f:
            df = pd.read_csv(f, low_memory=False)
            seq = df[['host', 'timestamp', 'cluster_id', 'label']].copy()
            del df
        seq['label'] = seq['label'].apply(func=lambda x: 'webshell' if x.find('webshell') != -1 else x)
        seq.to_csv(self.cluster_seq_file, index=False)
        print("generation finished")

    def filter_cluster_seq(self):
        print("filtering cluster_seq for {}...".format(self.name))
        seq_df = pd.read_csv(self.cluster_seq_file, low_memory=False)
        cluster_stat = pd.read_csv(self.cluster_stat_file, low_memory=False)
        cluster_set = set(
            cluster_stat.query('name=="{}" & rate>{}'.format(self.name, self.cluster_rate))['cluster_id'].values)
        # filter by cluster
        seq_df = seq_df[seq_df['cluster_id'].isin(cluster_set)]
        # filter by label
        seq_df = seq_df[seq_df['label'].isin(self.label_set)]
        seq_df.to_csv(self.cluster_seq_file, index=False)

    def get_label_stat(self):
        df = pd.read_csv(self.cluster_seq_file, low_memory=False)
        result = {'name': self.name}
        for label in self.labels:
            result[label] = df[df['label'] == label].shape[0]
        return result

    def cluster_stat(self):
        df = pd.read_csv(self.cluster_seq_file, low_memory=False)
        ser = df['cluster_id'].value_counts()
        result = {'name': self.name, 'cluster_id': ser.index, 'count': ser.values,
                  'rate': ser.apply(lambda x: x / df.shape[0])}
        result = pd.DataFrame(result).reset_index(drop=True)
        return result


def process_AitLog(gen_cluster_seq=False,filter_cluster=False,name_list=config['name_list']):
    if gen_cluster_seq:
        # gen_framed->gen_cluster->cluster_stat,label_stat
        label_stat = []  # dict list
        cluster_stat = []  # df list
        for name in name_list:
            curLog = AitLogDerived(name)
            #curLog.gen_framed()
            curLog.gen_cluster_seq()
            label_stat.append(curLog.get_label_stat())
            cluster_stat.append(curLog.cluster_stat())
        label_stat = pd.DataFrame(label_stat)
        cluster_stat = pd.concat(cluster_stat)
        label_stat.to_csv(AitLog.label_stat_file, index=False)
        cluster_stat.to_csv(AitLog.cluster_stat_file, index=False)
    if filter_cluster:
    # filter_cluster->cluster_stat,label_stat
    # filter 依赖cluster_stat，故再遍历一次
        label_stat = []  # dict list
        cluster_stat = []  # df list
        for name in name_list:
            curLog = AitLogDerived(name)
            curLog.filter_cluster_seq()
            label_stat.append(curLog.get_label_stat())
            cluster_stat.append(curLog.cluster_stat())
        label_stat = pd.DataFrame(label_stat)
        cluster_stat = pd.concat(cluster_stat)
        label_stat.to_csv(AitLog.label_stat_file, index=False)
        cluster_stat.to_csv(AitLog.cluster_stat_file, index=False)
