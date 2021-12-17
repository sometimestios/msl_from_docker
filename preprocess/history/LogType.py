# -*- coding: utf-8 -*-
import logging
import sys
import time
import pandas as pd
from preprocess import time_extract
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from config import config
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
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


class LogType:
    def __init__(self, name, host_list):
        self.name = name
        self.host_list = host_list
        self.log_list = list(
            map(lambda host: "D:/data/AIT-LDS-v1_1/data/mail.{}.com/{}".format(host, name_path_map[self.name]),
                self.host_list))
        self.label_list = list(
            map(lambda host: "D:/data/AIT-LDS-v1_1/labels/mail.{}.com/{}".format(host, name_path_map[self.name]),
                self.host_list))
        self.framed_file = "../data/framed/{}.csv".format(self.name)
        self.cluster_seq_file = "../data/cluster_seq/{}.csv".format(self.name)
        self.drain_config = "../preprocess/drain3.ini"
        self.host_line_count = [0 for i in range(len(self.host_list))]  # 该类型日志按照host划分的区间的右端点

    def genFramed(self):
        config = TemplateMinerConfig()
        config.load(self.drain_config)
        config.profiling_enabled = True
        template_miner = TemplateMiner(config=config)

        line_count = 0
        log_lines = []
        label_lines = []
        for i in range(len(self.log_list)):
            with open(self.log_list[i]) as f1, open(self.label_list[i]) as f2:
                log_lines.extend(f1.readlines())
                label_lines.extend(f2.readlines())
                self.host_line_count[i] = len(log_lines)
        if len(log_lines) != len(label_lines):
            print(
                "error:{}-->len(log_lines)={},len(label_lines)={}".format(self.name, len(log_lines), len(label_lines)))
            exit(-1)
        start_time = time.time()
        batch_start_time = start_time
        batch_size = 10000
        framed_lines = []

        end = 0
        for i in range(len(self.host_line_count)):
            cur_host = self.host_list[i]
            start = end
            end = self.host_line_count[i]
            for j in range(start, end):
                log_line = log_lines[j].rstrip()
                timestamp, log = time_extract.match.get(self.name)(log_line)
                label_line = label_lines[j].rstrip().split(",")
                result = template_miner.add_log_message(log)
                # include: origin log, host, cluster id, timestamp, template,  label1,label2
                framed_lines.append(
                    [log_line, cur_host, result["cluster_id"], timestamp, result["template_mined"], label_line[0],
                     label_line[1]])

                line_count += 1
                if line_count % batch_size == 0:
                    time_took = time.time() - batch_start_time
                    rate = batch_size / time_took
                    logger.info(f"Processing line: {line_count}, rate {rate:.1f} lines/sec, "
                                f"{len(template_miner.drain.clusters)} clusters so far.")
                    batch_start_time = time.time()
                # if result["change_type"] != "none":
                #     result_json = json.dumps(result)
                #     logger.info(f"Input ({line_count}): " + log_line)
                #     logger.info("Result: " + result_json)
        framed_df = pd.DataFrame(framed_lines,
                                 columns=["origin_log", "host", "cluster_id", "timestamp", "template", "label1",
                                          "label2"])
        # print(framed_df[framed_df['label1']==0])
        framed_df.to_csv(self.framed_file, index=False)

        time_took = time.time() - start_time
        if time_took == 0:
            rate = 99999
        else:
            rate = line_count / time_took
        logger.info(
            f"--- Done processing file in {time_took:.2f} sec. Total of {line_count} lines, rate {rate:.1f} lines/sec, "
            f"{len(template_miner.drain.clusters)} clusters")
        sorted_clusters = sorted(template_miner.drain.clusters, key=lambda it: it.size, reverse=True)
        for cluster in sorted_clusters:
            logger.info(cluster)

        print("Prefix Tree:")
        template_miner.drain.print_tree()

    def genClusterSeq(self):
        with open(self.framed_file) as f:
            data = pd.read_csv(f, low_memory=False)  # pandas读数据是分块读取，如果内存不足可能导致某些块的数据自动压缩，如str类型读成int类型
        seq = data[["host", "cluster_id", "label1", "label2"]].copy()
        # df['b'] = df['b'].apply(func=lambda x: 'webshell' if x.find('ben') else x)
        seq['label1'] = seq['label1'].apply(func=lambda x: 'webshell' if x.find('webshell') != -1 else x)
        seq['label2'] = seq['label2'].apply(func=lambda x: 'webshell' if x.find('webshell') != -1 else x)
        seq.to_csv(self.cluster_seq_file, index=False)

    def labelStat(self):
        stat1 = {'name': self.name + '_label1'}
        stat2 = {'name': self.name + '_label2'}
        df = pd.read_csv(self.cluster_seq_file, low_memory=False)
        cate1 = df['label1'].unique()
        cate2 = df['label2'].unique()
        for c in cate1:
            cnt = df[df['label1'] == c].shape[0]
            stat1[c] = cnt
        for c in cate2:
            cnt = df[df['label2'] == c].shape[0]
            stat2[c] = cnt
        return stat1, stat2

    def clusterStat(self, prt=True):
        df = pd.read_csv(self.cluster_seq_file, low_memory=False)
        pd.set_option('display.max_columns', 100)  # 设置最大显示列数的多少
        pd.set_option('display.width', 100)  # 设置宽度,就是说不换行,比较好看数据
        pd.set_option('display.max_rows', 200)  # 设置行数的多少

        ser = df['cluster_id'].value_counts()
        result = {'name': self.name, 'cluster_id': ser.index, 'count': ser.values,
                  'rate': ser.apply(lambda x: x / df.shape[0])}
        result = pd.DataFrame(result).reset_index(drop=True)
        if prt:
            print("{}日志的总数:{}\ncluster的数量及比例:".format(self.name, df.shape[0]))
            print(result)
        return result


def AllLabelStat(label_stat_file=None, seq_file_dir=None):
    label_stat = []  # list of dict, merge by DataFrame()
    for i in range(len(name_list)):
        curType = LogType(name=name_list[i], host_list=host_list)
        if seq_file_dir:
            curType.cluster_seq_file = seq_file_dir + "{}.csv".format(
                curType.name)  # 更新curType的seq文件路径，以便统计非原始seq文件的label分布
        stat1, stat2 = curType.labelStat()
        label_stat.extend([stat1, stat2])
    label_stat = pd.DataFrame(label_stat)
    if label_stat_file:
        print("AllLabelStat saved in {}".format(label_stat_file))
        label_stat.to_csv(label_stat_file, index=False)
    else:
        print(label_stat)
        return label_stat


def AllClusterStat(cluster_stat_file=None, seq_file_dir=None):
    cluster_stat = []  # list of df, merge by concat()
    for i in range(len(name_list)):
        curType = LogType(name=name_list[i], host_list=host_list)
        if seq_file_dir:
            curType.cluster_seq_file = seq_file_dir + "{}.csv".format(
                curType.name)  # 更新curType的seq文件路径，以便统计非原始seq文件的cluster分布
        stat3 = curType.clusterStat()
        cluster_stat.append(stat3)
    cluster_stat = pd.concat(cluster_stat)
    if cluster_stat_file:
        print("AllClusterStat saved in {}".format(cluster_stat_file))
        cluster_stat.to_csv(cluster_stat_file, index=False)
    else:
        print(cluster_stat)
        return cluster_stat


name_list = config['name_list']
host_list = config['host_list']
if __name__ == '__main__':
    label_stat_file = "../data/label_stat.csv"
    cluster_stat_file = "../data/cluster_stat.csv"
    for i in range(len(name_list)):
        curType = LogType(name=name_list[i], host_list=host_list)
        curType.genFramed()  # 提取出簇的日志文件
        curType.genClusterSeq()  # cluster序列文件

    label_stat = AllLabelStat(label_stat_file)
    cluster_stat = AllClusterStat(cluster_stat_file)
