# -*- coding: utf-8 -*-
import os, csv

root=os.getcwd()
class Log_info:
    match = {'audit.log': 'audit',
             'auth.log': 'auth',
             'mail.log': 'mail_log',
             'fast.log': 'suricata_fast',
             'daemon.log': 'daemon',
             'messages': 'messages',
             'syslog': 'syslog',
             'user.log': 'user',
             'access.log': 'apache_access',
             'error.log': 'apache_error',
             'mainlog': 'exim4_main',
             'eve.json': 'suricata_eve',
             }

    def __init__(self, file, num, count, rate):
        self.file = file
        self.num = num  # 总数
        self.count = count  # 两种abnormal标签的数量
        self.rate = rate  # 两种abnormal标签的比例
        post_fix = self.file.split('/')[-1]
        # 将后缀与match中进行匹配，以确定log的type
        if self.match.get(post_fix):
            self.type = self.match.get(post_fix)
        elif self.match.get(post_fix[-9:]):
            self.type = self.match.get(post_fix[-9:])
        elif self.match.get(post_fix[-10:]):
            self.type = self.match.get(post_fix[-10:])
        else:
            self.type = 'other'

    def show(self):
        print("file:{},type:{},num:{},count:{},rate:{}".format(self.file, self.type, self.num, self.count, self.rate))

def read_in_chunks(file_path, chunk_size=1024 * 1024):
    file_object = open(file_path, 'r', encoding='utf-8')
    while True:
        chunk_data = file_object.read(chunk_size)
        if not chunk_data:
            break
        yield chunk_data


def scan_files(directory, prefix=None, postfix=None):
    files_list = []

    for root, sub_dirs, files in os.walk(directory):
        for special_file in files:
            if postfix:
                if special_file.endswith(postfix):
                    files_list.append(os.path.join(root, special_file))
            elif prefix:
                if special_file.startswith(prefix):
                    files_list.append(os.path.join(root, special_file))
            else:
                files_list.append(os.path.join(root, special_file))
    for i in range(len(files_list)):
        files_list[i] = files_list[i].replace('\\', '/')
    return files_list


# 统计单个label文件的信息
def file_parse(file):
    count = [0, 0]
    if os.path.getsize(file) > 100 * 1024 * 1024:
        print(file, " is too big, size=", os.path.getsize(file))
        #return
        chunks = read_in_chunks(file, 20 * 1024 * 1024)
        N = 0
        for chunk in chunks:
            chunk = chunk.split("\n")
            N += len(chunk)
            for line in chunk:
                # print(line)
                line = line.rstrip().split(",")
                if len(line) < 2:
                    N -= 1
                    continue
                if line[0] != "0":
                    count[0] += 1
                if line[1] != "0":
                    count[1] += 1
    else:
        with open(file) as f:
            lines = f.readlines()
        N = len(lines)
        for line in lines:
            line = line.rstrip().split(",")
            if line[0] != "0":
                count[0] += 1
            if line[1] != "0":
                count[1] += 1
    rate = [count[0] / N, count[1] / N]
    return Log_info(file, N, count, rate)


# print(file, ", N=", N, ",count= ", count, ",rate: ", rate)


def log_stat():
    label_root = "D:/data/AIT-LDS-v1_1/labels/"
    log_info_file = "../data/log_info_file.csv"
    files = scan_files(label_root)
    log_info_list = []  # 全部log的信息

    # 统计全部文件信息
    for file in files:
        if file[-3:] == "zip":
            continue
        tmp_info = file_parse(file)
        if tmp_info:
            log_info_list.append(tmp_info)
    log_info_list.sort(key=lambda x: x.rate[1], reverse=True)  # 按照rate2从大到小排序
    with open(log_info_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["file", "type", "num", "label1_count", "label2_count", "label1_rate", "label2_rate"])
        for info in log_info_list:
            writer.writerow([info.file, info.type, info.num, info.count[0], info.count[1], info.rate[0], info.rate[1]])
    return log_info_list


# 按照type统计信息
def type_stat(info_list):
    type_info_file = "../data/type_info_file.csv"
    # match={info_type:[Num,label1 count,label2 count]}
    match = {}
    for info in info_list:
        if info.type in match:
            match[info.type][0] += info.num
            match[info.type][1] += info.count[0]
            match[info.type][2] += info.count[1]
        else:
            match[info.type] = [info.num, info.count[0], info.count[1]]
    type_info_list = []
    for m in match:
        # tmp=[type,num,label1_count, label2_count, label1_rate,label2_rate]
        tmp = [m]
        tmp.extend(match[m])
        tmp.extend([tmp[2] / tmp[1], tmp[3] / tmp[1]])
        type_info_list.append(tmp)
    type_info_list.sort(key=lambda x:x[5],reverse=True)  # 按照label2_rate从大到小排列
    with open(type_info_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["type", "num", "label1_count", "label2_count", "label1_rate", "label2_rate"])
        for info in type_info_list:
            writer.writerow(info)
    return type_info_list


log_info_list = log_stat()
type_info_list = type_stat(log_info_list)
for log_info in log_info_list:
    log_info.show()
print(type_info_list)
