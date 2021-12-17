# -*- coding: utf-8 -*-
from model.lstm import MyModel
import os,sys,time
class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
    # 自定义目录存放日志文件
def save_log():
    log_path = './running_log/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # 日志文件名按照程序运行时间设置
    log_file_name = log_path + 'log-' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log'
    # 记录正常的 print 信息
    sys.stdout = Logger(log_file_name)
    # 记录 traceback 异常信息
    sys.stderr = Logger(log_file_name)

name_list=["apache_error","auth","suricata_fast"]
save_log()
for name in name_list:
    data_file="../data/cluster_seq/{}.csv".format(name)
    model_file="../saved_model/{}-lstm.h5".format(name)
    pic_name="../figure/{}-lstm".format(name)
    m=MyModel(name+'-lstm',data_file,model_file)
    m.train_lstm()