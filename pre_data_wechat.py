import numpy as np
ROOT_PATH = "/home/zyang/code-2021/decFair/data/"
# 比赛数据集路径
DATASET_PATH = ROOT_PATH + '/wechat_algo_data1/'
# 训练集
USER_ACTION = DATASET_PATH + "user_action.csv"
FEED_INFO = DATASET_PATH + "feed_info.csv"
FEED_EMBEDDINGS = DATASET_PATH + "feed_embeddings.csv"
# 测试集
TEST_FILE = DATASET_PATH + "test_a.csv"
# ACTION_LIST = ["like", "click", "follow", "finish"]
FEA_FEED_LIST = ['item_id', 'duration_time']
# 初赛待预测行为列表
ACTION_LIST = ["read_comment", "like", "click_avatar", "forward"]
ACTION_LIST.append('finish')
# FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]
# FEA_COLUMN_LIST.append('finish')
# FEA_FEED_LIST = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']
# 负样本下采样比例(负样本:正样本)
ACTION_SAMPLE_RATE = {"read_comment": 5, "like": 5, "click_avatar": 5, "forward": 10, "comment": 10, "follow": 10,
                      "favorite": 10}