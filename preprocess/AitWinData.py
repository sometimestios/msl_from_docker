from sklearn.preprocessing import MinMaxScaler
from scipy.special import expit
from config import config
from preprocess.AitLog import AitLogDerived
from datetime import datetime
import numpy as np
import pandas as pd
import os
from util import check_dir

label_num_map = {'0': 0, 'nikto': 1, 'hydra': 2, 'webshell': 3, 'upload': 4}
num_label_map = {0: '0', 1: 'nikto', 2: 'hydra', 3: 'webshell', 4: 'upload'}


def no_weight(his_time, cur_time):
    return [1 for _ in range(len(his_time))]


def get_weight1(his_time, cur_time, alpha=0.5, beta=0.8, gamma=0.1):
    # result=[alpha^2*beta,alpha*beta,beta]
    his_time.append(cur_time)
    gap = list(reversed(his_time))
    result = [beta for _ in range(len(gap))]
    for i in range(1, len(gap)):
        if gap[i] == gap[i - 1]:
            result[i] = result[i - 1]
        else:
            result[i] = result[i - 1] * alpha
        if result[i] < gamma:
            # 后续都为gamma
            for j in range(i, len(gap)):
                result[j] = gamma
            break
    result = list(reversed(result))
    return result[:-1]


def get_weight2(his_time, cur_time, alpha=0.5, beta=0.8, gamma=0.1):
    return

def get_window(x, y, timestamp, win_len, wei_func):
    x_win = []
    y_win = []
    for i in range(len(x) - win_len):
        cur_x = x[i:i + win_len]
        cur_w = wei_func(timestamp[i:i + win_len], timestamp[i + win_len])
        new_x = list(map(lambda a, b: a * b, cur_x, cur_w))
        x_win.append(new_x)
        y_win.append(y[i + win_len])
    return x_win, y_win


def str2time(x):
    ret = datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    ret = pd.Timestamp(ret)
    return ret


def df2win(X, Y, win_len, wei_func):
    x_ret = []
    y_ret = []
    for host in config['host_list']:
        print("host:{}".format(host))
        tmp = X[X['host'] == host]
        vec = tmp.drop(['host', 'timestamp'], axis=1).values
        timestamp = tmp['timestamp'].values
        del tmp
        timestamp = list(map(str2time, timestamp))
        y = np.array(Y[Y['host'] == host]['label'])
        x, y = get_window(vec, y, timestamp, win_len, wei_func)
        x_ret.extend(x)
        y_ret.extend(y)
    return x_ret, y_ret


name_func_map = {'no_wei': no_weight, 'wei1': get_weight1, 'wei2': get_weight2}


class AitWinData:
    def __init__(self, name, win_len, wei_func_name):
        self.name = name
        self.cluster_seq_file = 'data/AitLog/cluster_seq/{}.csv'.format(self.name)
        self.win_len = win_len
        self.wei_func = name_func_map[wei_func_name]
        self.total_file = 'data/AitLog/win_data/win{}/{}/total/{}.csv'.format(self.win_len, wei_func_name, self.name)
        self.rd_state = None
        self.select_file = None
        check_dir(self.total_file)

    def gen_total_data(self):
        # 函数占用内存较多，调用del释放一些内存
        print("generating win_data for {}...".format(self.name))
        df = pd.read_csv(self.cluster_seq_file, low_memory=False)

        config['dense'] = len(df['label'].unique())
        X = pd.get_dummies(df['cluster_id'])
        X = df[['host', 'timestamp']].join(X)
        Y = df['label'].apply(lambda x: label_num_map[x])
        Y = df[['host']].join(Y)
        del df
        X, Y = df2win(X, Y, self.win_len, self.wei_func)
        X = np.array(X)
        Y = np.array(Y)
        X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
        X = pd.DataFrame(X)
        Y = pd.DataFrame(Y)
        df = pd.concat([X, Y], axis=1)
        del X,Y
        new_columns = list(df.columns)
        new_columns[-1] = 'label'
        df.columns = new_columns
        print("saving in {}".format(self.total_file))
        df.to_csv(self.total_file, index=False)
        print("finished")

    def stat(self):
        df = pd.read_csv(self.total_file, low_memory=True)
        print("count of each label of {}:".format(self.name))
        ret = []
        for i in range(len(label_num_map)):
            cnt = df[df['label'] == i].shape[0]
            print("{}:{}".format(num_label_map[i], cnt))
            ret.append(cnt)
        return ret

    def select(self, win_data_need_file):
        print("reading {}".format(self.total_file))
        data = pd.read_csv(self.total_file, low_memory=False)
        df = pd.read_csv(win_data_need_file)
        df.set_index(['name'], inplace=True)
        result = []
        for label in label_num_map:
            num = df.loc[self.name, label]
            if num:
                tmp = data[data['label'] == label_num_map[label]]
                result.append(tmp.sample(num, random_state=self.rd_state))
        result = pd.concat(result, axis=0)
        print("saving selected data in ", self.select_file)
        result.to_csv(self.select_file, index=False)


def gen_group_total_data(win_len, wei_func_name):
    for name in config['name_list']:
        curWinVec = AitWinData(name, win_len, wei_func_name)
        curWinVec.gen_total_data()


def stat_total_data(win_len, wei_func_name):
    lines = []
    win_data_stat_file = 'data/AitLog/win_data/win{}/win_data_stat.csv'.format(win_len)
    print("----total win_data stat:-----")
    for name in config['name_list']:
        curWinData = AitWinData(name, win_len, wei_func_name)
        line = [name]
        line.extend(curWinData.stat())
        lines.append(line)
    df = pd.DataFrame(lines, columns=['name', '0', 'nikto', 'hydra', 'webshell', 'upload'])
    df.to_csv(win_data_stat_file, index=False)
    print("----saved in {}:-----".format(win_data_stat_file))


def gen_group_select_data(win_len, wei_func_name, rd_state):
    win_data_need_file = 'data/AitLog/win_data/win{}/win_data_need.csv'.format(win_len)  # customized in advance
    for name in config['name_list']:
        # for name in ['apache_error']:
        curWinData = AitWinData(name, win_len, wei_func_name)
        curWinData.rd_state = rd_state
        curWinData.select_file = 'data/AitLog/win_data/' \
                                 'win{}/{}/select/rd_state{}/{}.csv'.format(win_len, wei_func_name, rd_state, name)
        check_dir(curWinData.select_file)
        curWinData.select(win_data_need_file)


def gen_all_total_data(win_len_list, wei_func_name_list):
    for win_len in win_len_list:
        for wei_func_name in wei_func_name_list:
            gen_group_total_data(win_len, wei_func_name)
            stat_total_data(win_len, wei_func_name)


def gen_all_select_data(win_len_list, wei_func_name_list, rd_state_list):
    for win_len in win_len_list:
        for wei_func_name in wei_func_name_list:
            for rd_state in rd_state_list:
                gen_group_select_data(win_len, wei_func_name, rd_state)
