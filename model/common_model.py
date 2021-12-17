# -*- coding: utf-8 -*-
import time
import pickle as pickle
import pandas as pd
import util
from config import config


# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    model.fit(train_x, train_y)
    return model


# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(train_x, train_y)
    return model


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(train_x, train_y)
    return model


# SVM Classifier
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    model.fit(train_x, train_y)
    return model


def lstm_classifier(name, train_x, train_y, input_shape, dense):
    from model.lstm import MyModel
    root = '../'
    myModel = MyModel(name + '_lstm', input_shape, dense)
    model = myModel.train_lstm(train_x, train_y)
    return model


# def read_data(data_file, test_rate):
#     # x, y = util.common_load_data(data_file, window_len)
#     x, y = util.load_select_win_data(data_file)
#     x_train, x_test, y_train, y_test = util.split_data(x, y, test_rate)
#     return x_train, x_test, y_train, y_test


def get_k_fold(x, y, k=config['k_fold']):
    # https://zhuanlan.zhihu.com/p/150446294
    # StratifiedKFold只能按比例筛选出index，其shuffle参数只能影响index的随机组合，而index的顺序永远按照从小到大的顺序排列，因此在这之前数据必须shuffle
    # 发现这个问题是因为相同label 的数据扎堆导致模型效果有问题
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=k, random_state=config['random_state'], shuffle=True)
    return skf.split(x, y)


def train_model_on_data(data_name,win_len, data_file):
    model_save_file = "None"
    model_save = {}
    test_classifiers = ['LSTM', 'NB', 'KNN', 'LR', 'RF', 'DT', 'SVM', 'GBDT']
    # test_classifiers = ['NB', 'KNN']
    classifiers = {'NB': naive_bayes_classifier,
                   'KNN': knn_classifier,
                   'LR': logistic_regression_classifier,
                   'RF': random_forest_classifier,
                   'DT': decision_tree_classifier,
                   'SVM': svm_classifier,
                   'GBDT': gradient_boosting_classifier,
                   'LSTM': lstm_classifier,
                   }

    print('reading training and testing data...')
    # 七三分方法
    # train_x, test_x, train_y, test_y = read_data(data_file, config['test_rate'])

    # kfold方法
    cfer_reports_dict = {'NB': [],
                         'KNN': [],
                         'LR': [],
                         'RF': [],
                         'DT': [],
                         'SVM': [],
                         'GBDT': [],
                         'LSTM': [],
                         }
    x, y = util.load_select_win_data(data_file)
    for train_index, test_index in get_k_fold(x, y):
        train_x, test_x = x[train_index], x[test_index]
        train_y, test_y = y[train_index], y[test_index]
        for classifier in test_classifiers:
            print('******************* %s ********************' % classifier)
            start_time = time.time()
            if classifier == 'LSTM':
                # lstm的classifier要多一个name参数,且输入数据为sample,time,feature的三维数据，判断是否需要reshape
                if (len(train_x.shape) == 2):
                    train_x = train_x.reshape(
                        (train_x.shape[0], win_len, train_x.shape[1] // win_len))
                    test_x = test_x.reshape((test_x.shape[0], win_len, test_x.shape[1] // win_len))
                model = classifiers[classifier](data_name, train_x, train_y, test_x.shape[1:3], config['dense'])
            else:
                # 其他模型输入数据为二维，sample*feature
                if (len(train_x.shape) == 3):
                    train_x = train_x.reshape((train_x.shape[0], train_x.shape[1] * train_x.shape[2]))
                    test_x = test_x.reshape((test_x.shape[0], test_x.shape[1] * test_x.shape[2]))
                model = classifiers[classifier](train_x, train_y)
            print('training took %fs!' % (time.time() - start_time))
            predict = model.predict(test_x)
            if model_save_file != "None":
                model_save[classifier] = model
            # 基于predict结果导出report
            report = util.report(test_y, predict)
            report['name'] = data_name
            report['cluster']=config['name_cluster_map'][data_name]
            report['model'] = classifier
            for i, k in enumerate(report.keys()):
                if i < 4:
                    print("{} = {}".format(k, report[k]))
                else:
                    print("{} = {:.2f}%".format(k, report[k] * 100))
            cfer_reports_dict[classifier].append(report)
        if model_save_file != "None":
            pickle.dump(model_save, open(model_save_file, 'wb'))
    return cfer_reports_dict


def train_model_on_dataset(win_len, wei_func_name, rd_state):
    select_dir = 'data/AitLog/win_data/win{}/{}/select/rd_state{}/'.format(win_len, wei_func_name, rd_state)
    test_report_file = 'data/AitLog/test_report/rd_state{}/win{}_{}.csv'.format(rd_state, win_len, wei_func_name)
    util.check_dir(test_report_file)
    test_report = []  # 字典列表
    name_list = config['name_list']
    for name in name_list:
        print("++++++++++++training model for {}++++++++++++++".format(name))
        cfer_reports_dict = train_model_on_data(name,win_len, select_dir + '{}.csv'.format(name))
        aved_report = util.ave_k_report(cfer_reports_dict)
        test_report.extend(aved_report)
        print("+++++++++++training model for {} end.+++++++++++++".format(name))
    test_report = pd.DataFrame(test_report)
    test_report.to_csv(test_report_file, index=False)


def train_model_on_dataset_list(win_len_list, wei_func_name_list, rd_state_list):
    for win_len in win_len_list:
        for wei_func_name in wei_func_name_list:
            for rd_state in rd_state_list:
                train_model_on_dataset(win_len, wei_func_name, rd_state)
