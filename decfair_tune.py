# -*- coding: utf-8 -*-
"""
This is the version for search hyper-parameters by using the package Ray.
you need install it with pip
"""
from operator import pos
from re import A, S
import os
from typing import Dict
from deepctr_torch.layers.interaction import BiInteractionPooling
import numpy as np
from numpy.core.defchararray import index
from numpy.lib.type_check import real
import pandas as pd
from pandas.core.algorithms import isin
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.nn.modules import linear
from torch.nn.parameter import Parameter
from tqdm import tqdm
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models.deepfm import *
# from deepctr_torch.models.basemodel import *
from basemodel_ours import *
from deepctr_torch.callbacks import EarlyStopping
import time
import argparse
import math
from torch.utils.tensorboard import SummaryWriter

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from utils import metrics,early_stoper

# 存储数据的根目录
ROOT_PATH = "/home/zyang/Code/decFair/decFair/data/"
# 比赛数据集路径
DATASET_PATH = ROOT_PATH + '/wechat_algo_data1/'
# 训练集
USER_ACTION = DATASET_PATH + "user_action.csv"
FEED_INFO = DATASET_PATH + "feed_info.csv"
FEED_EMBEDDINGS = DATASET_PATH + "feed_embeddings.csv"
# 测试集
TEST_FILE = DATASET_PATH + "test_a.csv"
ACTION_LIST = ["like", "click", "follow", "finish"]
FEA_FEED_LIST = ['item_id', 'duration_time']
# 初赛待预测行为列表
# ACTION_LIST = ["read_comment", "like", "click_avatar", "forward"]
# ACTION_LIST.append('finish')
# FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]
# FEA_COLUMN_LIST.append('finish')
# FEA_FEED_LIST = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']
# 负样本下采样比例(负样本:正样本)
ACTION_SAMPLE_RATE = {"read_comment": 5, "like": 5, "click_avatar": 5, "forward": 10, "comment": 10, "follow": 10,
                      "favorite": 10}



class ConfounderStackLayers(nn.Module):
    def __init__(self,confounder_num,hiden_units=[256,128],input_dim=32,activation='relu'):
        super(ConfounderStackLayers,self).__init__()
        parameter_list = []
        self.confounder_num = confounder_num
        bias_list = []
        for layer_ouput in hiden_units:
            m =[nn.parameter.Parameter(torch.empty(layer_ouput, input_dim)) for i in range(confounder_num)] # dnn layer
            parameter_list.extend(m)
            bias_list.append(nn.parameter.Parameter(torch.empty(layer_ouput*confounder_num))) # bias
            input_dim = layer_ouput
        parameter_list.extend([nn.parameter.Parameter(torch.empty(1, hiden_units[-1])) for i in range(confounder_num)]) # ouput layer
        self.parameters_list = nn.ParameterList(parameter_list)               
        bias_list.append(nn.parameter.Parameter(torch.empty(confounder_num)))  # ouput layer bias
        self.bias_list = nn.ParameterList(bias_list)
        self.activation = F.relu
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(len(self.bias_list)):
            for weights in self.parameters_list[i* self.confounder_num:(i+1)*self.confounder_num]:
                nn.init.kaiming_uniform_(weights, a=math.sqrt(5))
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_list[i],-bound,bound)
    def forward(self,x):
        h = x
        for i in range(len(self.bias_list)):
            layer_i_parameters = self.parameters_list[i*self.confounder_num:(i+1)*self.confounder_num]
            layer_i_bias = self.bias_list[i]
            if i == 0:
                weights = torch.cat([mm for mm in layer_i_parameters],dim=0)
                h = F.linear(h,weights,bias=layer_i_bias) 
            else:
                h = self.activation(h)
                weights = torch.block_diag(*layer_i_parameters)
                h = F.linear(h,weights,bias=layer_i_bias)# + layer_i_bias
        return h
       

class ConfounderBaseModel2(BaseModel):
    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, initial_epoch=0, validation_split=0.,
            validation_data=None, shuffle=True, callbacks=None, split_by_u=True, args_=None,save_name=None,log_file=None):
        if isinstance(x, dict):
            temp = []
            for feature in self.feature_index:
                temp.append(x[feature])
            temp.append(x[self.confounder_name]) # add the confounder to the last
            x = temp
            # x = [x[feature] for feature in self.feature_index]
        
        aux_loss_coff = 0
        try:
            aux_loss_coff = args_.alpha
        except:
            aux_loss_coff = 0
        print("aux loss coff:",aux_loss_coff,file=log_file)
        report_K = 10 #args_.report_K
        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, '
                    'it must contain either 2 items (x_val, y_val), '
                    'or 3 items (x_val, y_val, val_sample_weights), '
                    'or alternatively it could be a dataset or a '
                    'dataset or a dataset iterator. '
                    'However we received `validation_data=%s`' % validation_data)
            if isinstance(val_x, dict):
                temp = [val_x[feature] for feature in self.feature_index]
                temp.append(val_x[self.confounder_name])
                val_x = temp
        else:
            val_x = []
            val_y = []
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)
        
        y = y[:,0:1]
        val_y_post = val_y[:,1:]    # post action as a testing target
        val_y = val_y[:,0:1]        # trained action  

        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(
                np.concatenate(x, axis=-1)),
            torch.from_numpy(y))
        if batch_size is None:
            batch_size = 256

        model = self.train()
        loss_func = self.loss_func
        optim = self.optim

        if self.gpus:
            print('parallel running on these gpus:', self.gpus,file=log_file)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)

        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size, num_workers=4, pin_memory=False)

        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        
        refer_metric = args_.stop_refer
        stoper = early_stoper(refer_metric=refer_metric, stop_condition=args_.patience)
        self.evaluator = None
        self.my_metrics = metrics(val_x[0].values, val_y)
        self.my_metrics_post = metrics(val_x[0].values, val_y_post)
        
        # Train
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch), file=log_file)
        for epoch in range(initial_epoch, epochs):
            # callbacks.on_epoch_begin(epoch)
            self.set_pre_mode(mode='condition') # in training, only train one branch base one the confounder
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            train_result = {}
            try:
                # with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                # with enumerate(train_loader) as t:
                for _, (x_train, y_train) in enumerate(train_loader):
                    x = x_train.to(self.device).float()
                    y = y_train.to(self.device).float()

                    y_pred = model(x,y_label=y,aux_loss=True).squeeze()

                    optim.zero_grad()
                    loss = loss_func(y_pred, y.squeeze(), reduction='sum')
                    reg_loss = self.get_regularization_loss()

                    total_loss = loss + reg_loss + aux_loss_coff * self.aux_loss  

                    loss_epoch += loss.item()
                    total_loss_epoch += total_loss.item()
                    total_loss.backward()
                    optim.step()

                    if verbose > 0:
                        for name, metric_fun in self.metrics.items():
                            if name not in train_result:
                                train_result[name] = []
                            try:
                                temp = metric_fun(
                                    y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64"))
                            except Exception:
                                temp = 0
                            finally:
                                train_result[name].append(temp)
            except KeyboardInterrupt:
                # t.close()
                raise
            # t.close()

            # Add epoch_logs
            epoch_logs["loss"] = total_loss_epoch / sample_num
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                eval_result, pred_ans = self.evaluate(val_x, val_y, batch_size, y_post=val_y_post, log_file=log_file)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
                # pre_modes = ['condition', 'do', 'do-even']
                # epoch_logs["val_uauc"] = epoch_logs["val_uauc_post_"+'do']   # used for stopping 
                need_saving = stoper.update_and_isbest(epoch_logs, epoch)
                if need_saving:
                    torch.save(self.state_dict(),"/data/zyang/decFair/logs/best-"+save_name+"-m.pth") # need to be the same to the one in the run_a_main()
                    best_pred_ans = pred_ans

            #add tensorboard
            if self.writer is not None:
                self.writer.add_scalar('trian/loss', epoch_logs["loss"], epoch)
                # self.writer.add_scalar('valid/uAuc', epoch_logs["val_uauc"], epoch)

            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch + 1, epochs), file=log_file)

                eval_str = "{0}s - ".format(
                    epoch_time)
                
                for name,name_result in epoch_logs.items():
                        eval_str += ' -' + name +':{0:.4f}'.format(name_result)
                print(eval_str, file=log_file)
                print(eval_str)
            if epoch % 10 == 0:
                with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                    pass
                        # path = os.path.join(checkpoint_dir, "checkpoint")
                        # torch.save((self.state_dict(), self.optim.state_dict()), path)
            if report_K==10:
                tune.report(val_condition_ndcg10=epoch_logs['val_condition_ndcg'+str(report_K)], val_do_ndcg10=epoch_logs['val_do_ndcg'+str(report_K)],val_condition_ndcg_post10=epoch_logs['val_condition_ndcg_post'+str(report_K)], val_do_ndcg_post10=epoch_logs['val_do_ndcg_post'+str(report_K)])
            if report_K==1:
                tune.report(val_condition_ndcg1=epoch_logs['val_condition_ndcg'+str(report_K)], val_do_ndcg1=epoch_logs['val_do_ndcg'+str(report_K)],val_condition_ndcg_post1=epoch_logs['val_condition_ndcg_post'+str(report_K)], val_do_ndcg_post1=epoch_logs['val_do_ndcg_post'+str(report_K)])
            # callbacks.on_epoch_end(epoch, epoch_logs)
            if stoper.is_stop():
                break
        print("best epoch:",stoper.best_epoch,"best-result:",str(stoper.best_eval_result),file=log_file)
        # tune.report(val_condition_ndcg10=stoper.best_eval_result['val_condition_ndcg'+str(report_K)], val_do_ndcg10=stoper.best_eval_result['val_do_ndcg'+str(report_K)],val_condition_ndcg_post10=stoper.best_eval_result['val_condition_ndcg_post'+str(report_K)], val_do_ndcg_post10=stoper.best_eval_result['val_do_ndcg_post'+str(report_K)])
        if report_K==10:
            tune.report(val_condition_ndcg10=epoch_logs['val_condition_ndcg'+str(report_K)], val_do_ndcg10=epoch_logs['val_do_ndcg'+str(report_K)],val_condition_ndcg_post10=epoch_logs['val_condition_ndcg_post'+str(report_K)], val_do_ndcg_post10=epoch_logs['val_do_ndcg_post'+str(report_K)])
        if report_K==1:
            tune.report(val_condition_ndcg1=epoch_logs['val_condition_ndcg'+str(report_K)], val_do_ndcg1=epoch_logs['val_do_ndcg'+str(report_K)],val_condition_ndcg_post1=epoch_logs['val_condition_ndcg_post'+str(report_K)], val_do_ndcg_post1=epoch_logs['val_do_ndcg_post'+str(report_K)])
        
        # if report_K==10:
        #     tune.report(val_condition_ndcg10=stoper.best_eval_result['val_condition_ndcg'+str(report_K)], val_do_ndcg10=stoper.best_eval_result['val_do_ndcg'+str(report_K)],val_condition_ndcg_post10=stoper.best_eval_result['val_condition_ndcg_post'+str(report_K)], val_do_ndcg_post10=stoper.best_eval_result['val_do_ndcg_post'+str(report_K)])
        # if report_K==1:
        #     tune.report(val_condition_ndcg1=stoper.best_eval_result['val_condition_ndcg'+str(report_K)], val_do_ndcg1=stoper.best_eval_result['val_do_ndcg'+str(report_K)],val_condition_ndcg_post1=stoper.best_eval_result['val_condition_ndcg_post'+str(report_K)], val_do_ndcg_post1=stoper.best_eval_result['val_do_ndcg_post'+str(report_K)])
        # return self.history

    def evaluate(self, x, y, batch_size=256, y_post=None,log_file=None):
        """

        :param x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per evaluation step. If unspecified, `batch_size` will default to 256.
        :return: Dict contains metric names and metric values.
        """
        self.set_pre_mode('all') #pre_mode = 
        pred_ans,pred_ans_2,pred_ans_3 = self.predict(x, batch_size)
        pred_list = [pred_ans,pred_ans_2,pred_ans_3]
        pred_ans = pred_ans.squeeze()

        eval_result = {}
        for name, metric_fun in self.metrics.items():
            try:
                temp = metric_fun(y, pred_ans)
            except Exception:
                temp = 0
            finally:
                eval_result[name] = metric_fun(y, pred_ans)
        
        user = x[0]
        # if self.evaluator is None:
        #     self.evaluator = candidates_test(user,y,label_post=y_post)
        
        k = 0
        for mode in ['condition','do','do-even']:
            pre_k = pred_list[k]
            s_time = time.time()


            topK=[10]
            recall, map_, ndcg = self.my_metrics.test(pre_k,topK=topK) # uauc refer to recall here 
            recall_post, map_post, ndcg_post = self.my_metrics_post.test(pre_k,topK=topK)
            auc_post = 0 
            print(mode,": recall (~~~~):", recall, 'map:', map_, 'ndcg:', ndcg,file=log_file)
            print(mode,": recall:(post) ", recall_post, 'map:', map_post, 'ndcg:', ndcg_post, file=log_file)
            i = 0
            for K in topK:
                eval_result[mode+'_map'+str(K)] = map_[i]
                eval_result[mode+'_ndcg'+str(K)] = ndcg[i]
                # eval_result[mode+'_uauc_post'+str(K)] = uauc_post[i]
                eval_result[mode+'_map_post'+str(K)] = map_post[i]
                eval_result[mode+'_ndcg_post'+str(K)] = ndcg_post[i]
                i += 1
            k += 1
        return eval_result, pred_list

    def predict(self, x, batch_size=256):
        """
        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param batch_size: Integer. If unspecified, it will default to 256.
        :return: Numpy array(s) of predictions.
        """
        model = self.eval()
        if isinstance(x, dict):
            temp = [x[feature] for feature in self.feature_index]
            temp.append(x[self.confounder_name])
            x = temp
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        tensor_data = Data.TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1)))
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=batch_size)

        pred_ans_1 = []
        pred_ans_2 = []
        pred_ans_3 = []
        pred_ans = []
        
        with torch.no_grad():
            for _, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()
                y_pred = model(x)
                if isinstance(y_pred,list):
                    pred_ans_1.append(y_pred[0].cpu().data.numpy())
                    pred_ans_2.append(y_pred[1].cpu().data.numpy())
                    pred_ans_3.append(y_pred[2].cpu().data.numpy())
                else:
                    pred_ans.append(y_pred.cpu().data.numpy())
        if len(pred_ans_1) > 0:
            pred_ans =  (np.concatenate(pred_ans_1).astype("float64"),np.concatenate(pred_ans_2).astype("float64"),np.concatenate(pred_ans_3).astype("float64"))
        else:
            pred_ans = np.concatenate(pred_ans).astype("float64")
        return pred_ans


class FastFairNFM(ConfounderBaseModel2): # --> DCR-MOE
    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, use_fm=True,
                 dnn_hidden_units=(256, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                 dnn_dropout=0,dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', gpus=None, 
                 writer=None, emb_dim = 0, confounder_num=1, confounder_prob=None, confounder_name=None, log_file=None):

        super(FastFairNFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                     device=device, gpus=gpus)
        self.writer = writer
        self.use_fm = use_fm
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        self.confounder_num = confounder_num    # number of confounder
        self.confounder_prob = torch.from_numpy(confounder_prob).to(device)  # probality of confounder
        self.confounder_name = confounder_name  # confoudner name
        self.set_pre_mode()

        self.fc_input_dim = 1
        if len(dnn_feature_columns) > 0:
            self.fc_input_dim += emb_dim
            print("read embedding dim:", self.fc_input_dim)
        
        if use_fm:
            self.bilinear_fm = BiInteractionPooling()
        
        self.confounder_layes = ConfounderStackLayers(confounder_num,hiden_units=dnn_hidden_units,input_dim=self.fc_input_dim,activation=dnn_activation)
        self.add_regularization_weight(filter(lambda x: 'weight' in x[0] and 'bn' not in x[0],self.confounder_layes.named_parameters()), l2=l2_reg_dnn)
        self.to(device)
    
    def set_pre_mode(self,mode='condition'):
        self.pre_mode=mode
    
    def read_confouder(self,X):
        # feat = self.confounder_name
        confounder = X[:, -1].long()
        return confounder.reshape(-1,1)

    def forward(self, X,y_label=None,aux_loss=False):
        self.aux_loss = 0
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        # linear
        linear_output = self.linear_model(X).reshape(-1,1)
        # bilinear pooling (FM)
        if self.use_fm and len(sparse_embedding_list) > 0:
            fm_input = torch.cat(sparse_embedding_list, dim=1)
            fm_ouput = self.bilinear_fm(fm_input)
        
        # total ouput
        tot_nfm_ouput = torch.cat([linear_output, fm_ouput.squeeze()],dim=-1) # concat the three branch as the out put
        # confounder tower
        confounder_ouput = self.confounder_layes(tot_nfm_ouput)
        # sigmoid
        y_pred = self.out(confounder_ouput) # batch_size * confounder_num
        if aux_loss:
            # add containts
            confounder = self.read_confouder(X).squeeze()
            confounder_onehot = F.one_hot(confounder, self.confounder_num)
            confounder_onehot_verse = 1 - confounder_onehot
            masked_all_pred = y_pred.mul(confounder_onehot_verse) # we slect the elements that not beloing to i-th confounder
            pos_ = masked_all_pred.mul(y_label.reshape(-1,1))
            pos_average = pos_.sum(dim=0)
            neg_ = masked_all_pred.mul(1-y_label.reshape(-1,1))
            neg_average = neg_.sum(dim=0)
            non_zero_info_pos = confounder_onehot_verse.mul(y_label.reshape(-1,1)).sum(dim=0)
            pos_average = pos_average / (non_zero_info_pos + 1e-6)
            non_zero_pos = torch.heaviside(non_zero_info_pos,torch.tensor([0.0],device=self.device))
            non_zero_info_neg = confounder_onehot_verse.mul(1-y_label.reshape(-1,1)).sum(dim=0)
            neg_average = neg_average / (non_zero_info_neg+1e-6)
            non_zero_neg = torch.heaviside(non_zero_info_neg,torch.tensor([0.0],device=self.device))
            valid_flag = non_zero_pos.mul(non_zero_neg)
            self.aux_loss = - torch.log(torch.sigmoid(pos_average - neg_average)).mul(valid_flag).sum() / valid_flag.sum()

            confounder_samples  = confounder_onehot.sum(dim=0)
            confoudner_samples_verse = confounder_onehot.shape[0] - confounder_samples
            confounder_average = y_pred.mul(confounder_onehot).sum(dim=0) / (confounder_samples + 1e-6)
            confounder_verse_average = y_pred.mul(confounder_onehot_verse).sum(dim=0) / (confoudner_samples_verse + 1e-6)
            constraints = torch.norm(confounder_average - confounder_verse_average,p=2).sum()
            self.aux_loss += 1e-0 * constraints
            

        # ouput 
        if self.pre_mode == 'condition':
            confounder = self.read_confouder(X)
            y_pred = torch.gather(y_pred,1,confounder)
        elif self.pre_mode == 'do':
            prob = self.confounder_prob.reshape(-1,1)
            y_pred = torch.matmul(y_pred,prob.float())
        elif self.pre_mode == 'do-even':
            y_pred = y_pred.sum(dim=-1)
        elif self.pre_mode == "all":
            confounder = self.read_confouder(X)
            y_pred_condition = torch.gather(y_pred,1,confounder)
            prob = self.confounder_prob.reshape(-1,1)
            y_pred_do = torch.matmul(y_pred,prob.float())
            y_pred_do_even = y_pred.sum(dim=-1)
            return [y_pred_condition,y_pred_do,y_pred_do_even]
        elif self.pre_mode == 'each':
            return y_pred
        else:
            raise "pre mode does not exist in FairNFM"
        return y_pred
