# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from scipy import stats
from sklearn.model_selection import train_test_split
from config import config


def check_dir(file):
    p,n=os.path.split(file)
    if not os.path.exists(p):
        os.makedirs(p)
    return file


def load_select_win_data(file):
    from sklearn.utils import shuffle
    with open(file) as f:
        data = pd.read_csv(f, low_memory=False)
    x = np.array(data.drop(['label'], axis=1))
    y = np.array(data['label'])
    config['input_shape'][1] = x.shape[1] // config['input_shape'][0]
    config['class'] = len(data['label'].unique())
    x, y = shuffle(x, y)
    return x, y


def get_window(X, Y, window_len):
    # 窗口下一条log的label作为窗口的label
    x_win = []
    y_win = []
    for i in range(len(X) - window_len):
        x_win.append(X[i:i + window_len])
        y_win.append(Y[i + window_len])
    return x_win, y_win


def split_data(x, y, test_rate):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_rate, random_state=config['random_state'])
    return x_train, x_test, y_train, y_test


def report(test_y, predict):
    from sklearn import metrics
    result = {'name': '',
              'model': '',
              'class': config['class'],
              'cluster': config['input_shape'][1],
              'f1_score_micro': metrics.f1_score(test_y, predict, average='micro'),
              'f1_score_macro': metrics.f1_score(test_y, predict, average='macro'),
              'precision': metrics.precision_score(test_y, predict, average='macro'),
              'recall': metrics.recall_score(test_y, predict, average='macro'),
              'accuracy': metrics.accuracy_score(test_y, predict)}
    return result


def ave_k_report(cfer_reports_dict):
    ave_data_list = []
    for cfer, reports in cfer_reports_dict.items():
        if not reports:
            continue
        df = pd.DataFrame(reports)
        ave_data = {'name': df['name'][0],
                    'model': df['model'][0],
                    'class': df['class'][0],
                    'cluster': df['cluster'][0]}
        new_df = pd.DataFrame(columns=df.columns)
        ave_metrix = df.drop(['name', 'model', 'class', 'cluster'], axis=1).mean().to_dict()
        ave_data.update(ave_metrix)
        ave_data_list.append(ave_data)
    return ave_data_list
