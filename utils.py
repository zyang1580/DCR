import numpy as np
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models.deepfm import *
from deepctr_torch.models.basemodel import *
from deepctr_torch.callbacks import EarlyStopping

class metrics(object):
    def __init__(self,user,label):
        super().__init__()
        self.user = user
        self.label = label
        self.sort_users()
        
    def sort_users(self):
        self.arg_idx = np.argsort(self.user)
        self.user = self.user[self.arg_idx]
        self.label = self.label[self.arg_idx]
        u,inverse,counts = np.unique(self.user, return_inverse=True, return_counts=True)
        self.id_map_to_u = inverse
        self.counts = counts
        self.pos_counts = np.zeros_like(u)
        self.neg_counts = np.zeros_like(u)
        i = 0
        start_idx = 0
        for ui in u:
            ui_count = counts[i]
            ui_pos = self.label[start_idx:start_idx+ui_count].sum() # label for ui
            ui_neg = ui_count - ui_pos
            self.pos_counts[i] = ui_pos  # count of neg list
            self.neg_counts[i] = ui_neg  # count of neg list
            i += 1
            start_idx += ui_count

    def sort_according_user(self, x):
        return x[self.arg_idx]

    def test(self,predict,topK=10):
        predict = self.sort_according_user(predict)
        batch_size = 1024
        uauc = []
        if isinstance(topK,list):
            map_all = [[] for i in range(len(topK))]
            ndcg_all = [[] for i in range(len(topK))]
        else:    
            map_all = [[]]
            ndcg_all = [[]]   # just one K

        user_number = self.pos_counts.shape[0]
        for i in range(0,user_number,batch_size):
            start_u = i
            end_u = min(i+batch_size,user_number)
            real_size = end_u - start_u 
            start_id, end_id = self.batch_id(start_u,end_u)
            
            batch_pre = predict[start_id:end_id].squeeze()
            batch_label = self.label[start_id:end_id].squeeze()
            batch_idmap2u = self.id_map_to_u[start_id:end_id].copy()
            batch_idmap2u -= batch_idmap2u.min() # start from 0

            batch_u = self.user[start_u:end_u]
            batch_pos_count = self.pos_counts[start_u:end_u]
            batch_neg_count = self.neg_counts[start_u:end_u]
            batch_count = batch_pos_count + batch_neg_count
            
            batch_count_max = batch_count.max()
            init_matrix_pre = np.zeros([real_size,batch_count_max],dtype=np.float) - np.inf
            init_matrix_label = np.zeros([real_size,batch_count_max],dtype=np.float)
            # print(batch_idmap2u.shape,real_size)
            cumsum_idx = np.zeros_like(batch_count)
            cumsum_idx[1:] = np.cumsum(batch_count)[0:-1]
            cumsum_idx = np.repeat(cumsum_idx,batch_count)
            mapping_idx = (batch_idmap2u, np.arange(batch_idmap2u.shape[0])-cumsum_idx)
            #np.concatenate([batch_idmap2u.reshape(-1,1), np.arange(batch_idmap2u.shape[0]).reshape(-1,1)],axis=-1)
            # print(mapping_idx.shape)
            # print(init_matrix_label.shape)
            init_matrix_label[mapping_idx] = batch_label
            init_matrix_pre[mapping_idx] = batch_pre

            sort_idx  = np.argsort(-init_matrix_pre,axis=-1)
            # map_list = self.batch_map(sort_idx,init_matrix_label,batch_pos_count,topK=topK)
            ndcg_list = self.batch_NDCG(sort_idx,init_matrix_label,batch_pos_count,topK=topK)
            map_list = ndcg_list #[0 for x in ndcg_list]
            

            for i in range(len(topK)):
                map_all[i].extend(map_list[i])
                ndcg_all[i].extend(ndcg_list[i])  # auume
        print("test user num:",len(ndcg_all[0]))
        return 0, np.array(map_all).mean(axis=-1), np.array(ndcg_all).mean(axis=-1)

    def test2(self,predict,topK=10):
        predict = self.sort_according_user(predict)
        batch_size = 1024
        uauc = 0
        if isinstance(topK,list):
            map_all = [[] for i in range(len(topK))]
            ndcg_all = [[] for i in range(len(topK))]
            recall_all = [[] for i in range(len(topK))]
        else:    
            map_all = [[]]
            ndcg_all = [[]]   # just one K
            recall_all = [[]]

        user_number = self.pos_counts.shape[0]
        for i in range(0,user_number,batch_size):
            start_u = i
            end_u = min(i+batch_size,user_number)
            real_size = end_u - start_u 
            start_id, end_id = self.batch_id(start_u,end_u)
            
            batch_pre = predict[start_id:end_id].squeeze()
            batch_label = self.label[start_id:end_id].squeeze()
            batch_idmap2u = self.id_map_to_u[start_id:end_id].copy()
            batch_idmap2u -= batch_idmap2u.min() # start from 0

            batch_u = self.user[start_u:end_u]
            batch_pos_count = self.pos_counts[start_u:end_u]
            batch_neg_count = self.neg_counts[start_u:end_u]
            batch_count = batch_pos_count + batch_neg_count
            
            batch_count_max = batch_count.max()
            init_matrix_pre = np.zeros([real_size,batch_count_max],dtype=np.float) - np.inf
            init_matrix_label = np.zeros([real_size,batch_count_max],dtype=np.float)
            # print(batch_idmap2u.shape,real_size)
            cumsum_idx = np.zeros_like(batch_count)
            cumsum_idx[1:] = np.cumsum(batch_count)[0:-1]
            cumsum_idx = np.repeat(cumsum_idx,batch_count)
            mapping_idx = (batch_idmap2u, np.arange(batch_idmap2u.shape[0])-cumsum_idx)
            #np.concatenate([batch_idmap2u.reshape(-1,1), np.arange(batch_idmap2u.shape[0]).reshape(-1,1)],axis=-1)
            # print(mapping_idx.shape)
            # print(init_matrix_label.shape)
            init_matrix_label[mapping_idx] = batch_label
            init_matrix_pre[mapping_idx] = batch_pre

            sort_idx  = np.argsort(-init_matrix_pre,axis=-1)
            recall_list = self.batch_recall(sort_idx,init_matrix_label,batch_pos_count,topK=topK)
            map_list = self.batch_map(sort_idx,init_matrix_label,batch_pos_count,topK=topK)
            ndcg_list = self.batch_NDCG(sort_idx,init_matrix_label,batch_pos_count,topK=topK)
            
            for i in range(len(topK)):
                map_all[i].extend(map_list[i])
                ndcg_all[i].extend(ndcg_list[i])  # auume
                recall_all[i].extend(recall_list[i])
            # map_all.extend(map)
            # ndcg_all.extend(ndcg)
        print("test user num:",len(ndcg_all[0]))
        return np.array(recall_all).mean(axis=-1), np.array(map_all).mean(axis=-1), np.array(ndcg_all).mean(axis=-1)

    
    # def test_return_dict(self,predict,topK=10):
    #     results = self.test(predict,topK=topK)
    def batch_map(self, sort_idx, batch_label, pos_count,topK=10):
        if isinstance(topK,list):
            map_list = []
        else:
            topK = [topK]
            map_list = []
        start_id = batch_label.shape[1] * np.ones(batch_label.shape[0])
        start_id[0] = 0
        start_id = np.cumsum(start_id)
        sort_idx_ = sort_idx + start_id.reshape(-1,1)
        sort_idx_ = sort_idx_.reshape(-1).astype(np.int)
        batch_label_ = batch_label.reshape(-1)
        rank_hit = batch_label_[sort_idx_]
        rank_hit = rank_hit.reshape(-1,batch_label.shape[1])

        for K in topK:
            max_K = min(K,rank_hit.shape[1])
            cum_hit = np.cumsum(rank_hit[:,0:max_K],axis=-1)
            consider_rank = np.arange(max_K).reshape(1,-1) + 1
            cum_hit_ratio = cum_hit / consider_rank
            ap = rank_hit[:,:max_K] * cum_hit_ratio
            pos_count_ = pos_count.copy()
            pos_count_[np.where(pos_count>max_K)] = max_K
            valid_user  = np.where(pos_count>0)[0]
            map = ap.sum(axis=-1)[valid_user] / pos_count_[valid_user]
            map_list.append(map)
        return map_list
    
    def batch_recall(self, sort_idx, batch_label, pos_count,topK=10):
        if isinstance(topK,list):
            recall_list = []
        else:
            topK = [topK]
            recall_list = []
        start_id = batch_label.shape[1] * np.ones(batch_label.shape[0])
        start_id[0] = 0
        start_id = np.cumsum(start_id)
        sort_idx_ = sort_idx + start_id.reshape(-1,1)
        sort_idx_ = sort_idx_.reshape(-1).astype(np.int)
        batch_label_ = batch_label.reshape(-1)
        rank_hit = batch_label_[sort_idx_]
        rank_hit = rank_hit.reshape(-1,batch_label.shape[1])

        for K in topK:
            max_K = min(K,rank_hit.shape[1])
            rank_hit_K = rank_hit[:,:max_K]
            sum_hit = rank_hit_K.sum(axis=-1)
            pos_count_ = pos_count.copy()
            pos_count_[np.where(pos_count>max_K)] = max_K
            valid_user  = np.where(pos_count>0)[0]
            recall_ = sum_hit[valid_user] / pos_count_[valid_user]
            # map = ap.sum(axis=-1)[valid_user] / pos_count_[valid_user]
            recall_list.append(recall_)
        return recall_list
    

    def batch_NDCG(self, sort_idx, batch_label, pos_count, topK=10):
        '''
        sort_idx: matrix
        batch_label: matrix
        '''
        if isinstance(topK,list):
            ndcg_list = []
        else:
            topK = [topK]
            ndcg_list = []
        start_id = batch_label.shape[1] * np.ones(batch_label.shape[0])
        start_id[0] = 0
        start_id = np.cumsum(start_id)
        sort_idx_ = sort_idx + start_id.reshape(-1,1)
        sort_idx_ = sort_idx_.reshape(-1).astype(np.int)
        batch_label_ = batch_label.reshape(-1)
        rank_hit = batch_label_[sort_idx_]
        rank_hit = rank_hit.reshape(-1,batch_label.shape[1])
        for K in topK:
            max_K = min(K,rank_hit.shape[1])
            log_rank = 1.0 / np.log2(np.arange(2, 2 + max_K)) # batch 
            log_rank_cumsum = np.cumsum(log_rank)
            pos_count_ = pos_count.copy()
            valid_user = np.where(pos_count>0)[0]
            pos_count_[np.where(pos_count_>=max_K)] = max_K
            pos_count_ -= 1
            idcg = log_rank_cumsum[pos_count_][valid_user]
            dcg = (rank_hit[:,0:max_K] * log_rank.reshape(1,-1)).sum(axis=-1)[valid_user]
            ndcg = dcg / idcg
            ndcg_list.append(ndcg)
        return ndcg_list
        
    def batch_id(self,start_u,end_u):
        bacth_u = self.user[start_u: end_u]
        start_id = self.counts[0:start_u].sum()
        end_id = self.counts[0:end_u].sum()
        return start_id,end_id





class early_stoper(object):
    def __init__(self,refer_metric='uauc',stop_condition=10):
        super().__init__()
        self.best_epoch = 0
        self.best_eval_result = None
        self.not_change = 0
        self.stop_condition = stop_condition
        self.init_flag = True
        self.refer_metric = refer_metric

    def update_and_isbest(self,eval_metric,epoch):
        if self.init_flag:
            self.best_epoch = epoch
            self.init_flag = False
            self.best_eval_result = eval_metric
            return True
        else:
            if eval_metric[self.refer_metric] > self.best_eval_result[self.refer_metric]: # update the best results
                self.best_eval_result = eval_metric
                self.not_change = 0
                self.best_epoch = epoch
                return True              # best
            else:                        # add one to the maker for not_change information 
                self.not_change += 1     # not best
                return False

    def is_stop(self):
        if self.not_change > self.stop_condition:
            return True
        else:
            return False