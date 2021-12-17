from config import config
from preprocess.AitLog import process_AitLog
from model.common_model import train_model_on_dataset_list
from preprocess.AitWinData import gen_all_total_data, gen_all_select_data

# process_AitLog(gen_cluster_seq=True,filter_cluster=True)
gen_all_total_data(config['win_len_list'],config['wei_func_name_list'])
# gen_all_select_data(config['win_len_list'],config['wei_func_name_list'],config['rd_state_list'])


#train_model_on_dataset_list(config['win_len_list'], config['wei_func_name_list'], config['rd_state_list'])
