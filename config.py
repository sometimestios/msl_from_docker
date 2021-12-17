# -*- coding: utf-8 -*-
config = {'input_shape': [20, -1],
          'dense': 5,
          'class': -1,
          'host_list': ['cup', 'insect', 'onion', 'spiral'],
          'win_len_list': [10, 15, 20, 25, 30],
          'name_list': ['apache_error', 'auth', 'user', 'apache_access'],
          'wei_func_name_list': ['no_wei', 'wei1'],
          'rd_state_list': [1, 2],
          'random_state': 100,
          'k_fold': 4,
          'name_cluster_map': {'apache_error': 19,
                               'auth': 6,
                               'user': 28,
                               'apache_access': 55}
          }
