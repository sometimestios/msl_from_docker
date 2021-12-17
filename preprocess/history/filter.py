# -*- coding: utf-8 -*-
import pandas as pd
from config import config


class Filter:
    def __init__(self, name, seq_file, cluster_stat_flie, cluster_rate, label_set, label=config['label']):
        self.name = name
        self.seq = pd.read_csv(seq_file, low_memory=False)
        self.cluster_stat = pd.read_csv(cluster_stat_flie, low_memory=False)
        self.cluster_rate = cluster_rate
        self.label_set = label_set
        self.label = label

    def clusterFilter(self):
        seq = self.seq
        # 注意在query的条件中，字符串要用双引号括起来，否则会被当做变量处理
        cluster_set = set(
            self.cluster_stat.query('name=="{}" & rate>{}'.format(self.name, self.cluster_rate))['cluster_id'].values)
        # seq = seq.drop(seq[seq['cluster_id'].isin(cluster_set)].index)  # .isin方法而不是in
        seq = seq[seq['cluster_id'].isin(cluster_set)]
        self.seq = seq
        return seq

    def labelFilter(self):
        seq = self.seq
        # seq = seq.drop(seq[seq[self.label].isin(self.label_set)].index)
        seq = seq[seq[self.label].isin(self.label_set)]
        self.seq = seq
        return seq


name_label_map = {
    'apache_error': {'0', 'upload', 'nikto'},
    'auth': {'0', 'hydra'},
    'user': {'0', 'upload', 'nikto', 'hydra', 'webshell'},
    'apache_access': {'0', 'nikto', 'hydra', 'webshell'}
}

cluster_stat_file = "../data/cluster_stat.csv"
filtered_label_stat_file = "../data/filtered_label_stat.csv"
cluster_rate = 0.0001
name_list = config['name_list']
# 先后基于cluster和label进行过滤
for name in name_list:
    label_set = name_label_map[name]
    cur_seq_file = "../data/cluster_seq/{}.csv".format(name)
    filtered_file = "../data/filtered_seq/{}.csv".format(name)
    cur_filter = Filter(name, cur_seq_file, cluster_stat_file, cluster_rate, label_set)
    cur_filter.clusterFilter().to_csv(filtered_file, index=False)
    cur_filter.labelFilter().to_csv(filtered_file, index=False)
from preprocess.LogType import AllLabelStat

AllLabelStat(filtered_label_stat_file, "../data/filtered_seq/")
