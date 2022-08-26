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


class CRBaseModel(BaseModel):
    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, initial_epoch=0, validation_split=0.,
            validation_data=None, shuffle=True, callbacks=None, split_by_u = True,save_name=None,args_=None,log_file=None):
        if isinstance(x, dict):
            temp = []
            for feature in self.feature_index:
                temp.append(x[feature])
            x = temp
            # x = [x[feature] for feature in self.feature_index]

        do_validation = False
        alpha = args_.alpha
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
                val_x = [val_x[feature] for feature in self.feature_index]
        
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)
        
        y = y[:,0:1]
        val_y_post = val_y[:,1:]
        val_y = val_y[:,0:1]

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
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size, num_workers=4, pin_memory=True)

        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        refer_metric = args_.stop_refer
        stoper = early_stoper(refer_metric=refer_metric,stop_condition=args_.patience)
        self.my_metrics = metrics(val_x[0],val_y)
        self.my_metrics_post = metrics(val_x[0],val_y_post)
        # Train
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch),file=log_file)
        for epoch in range(initial_epoch, epochs):
            # callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            train_result = {}
            try:
                for _, (x_train, y_train) in enumerate(train_loader):
                    x = x_train.to(self.device).float()
                    y = y_train.to(self.device).float()

                    _,y_pred, y_pred2 = model(x)
                    y_pred = y_pred.squeeze()
                    y_pred2 = y_pred2.squeeze()

                    optim.zero_grad()
                    loss = loss_func(y_pred, y.squeeze(), reduction='sum')
                    loss2 = loss_func(y_pred2, y.squeeze(), reduction='sum')
                    reg_loss = self.get_regularization_loss()

                    total_loss = loss + alpha * loss2 + reg_loss

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
            

            # Add epoch_logs
            epoch_logs["loss"] = total_loss_epoch / sample_num
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                eval_result, pred_ans = self.evaluate(val_x, val_y, batch_size, y_post=val_y_post,log_file=log_file)
                for name, result in eval_result.items():

                    epoch_logs["val_" + name] = result
                if epoch % 10 == 0:
                    with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                        pass
                        # path = os.path.join(checkpoint_dir, "checkpoint")
                        # torch.save((self.state_dict(), self.optim.state_dict()), path)
                tune.report(val_ndcg10=epoch_logs['val_ndcg10'], val_ndcg_post10=epoch_logs['val_ndcg_post10'],val_uauc=epoch_logs['val_uauc'],val_uauc_post=epoch_logs['val_uauc_post'])

                need_saving = stoper.update_and_isbest(epoch_logs, epoch)
                if need_saving:
                    torch.save(self.state_dict(),"/data/zyang/decFair/logs/best-"+save_name+"-m.pth")
                    best_pred_ans = pred_ans
                

            #add tensorboard
            if self.writer is not None:
                self.writer.add_scalar('trian/loss', epoch_logs["loss"], epoch)
                self.writer.add_scalar('valid/uAuc', epoch_logs["val_uauc"], epoch)

            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch + 1, epochs))

                eval_str = "{0}s - ".format(
                    epoch_time)
                print(eval_str,file=log_file)
                for name,name_result in epoch_logs.items():
                        eval_str += ' -' + name +':{0:.4f}'.format(name_result)
                # print(eval_str)
            # callbacks.on_epoch_end(epoch, epoch_logs)
            # if self.stop_training:
            #     break
            if stoper.is_stop():  # stoper by myself, to save the best model
                break
        # callbacks.on_train_end()
        print("--best_epoch:",stoper.best_epoch,"-- best_result:",stoper.best_eval_result,file=log_file)
        tune.report(val_ndcg10=epoch_logs['val_ndcg10'], val_ndcg_post10=epoch_logs['val_ndcg_post10'],val_uauc=epoch_logs['val_uauc'],val_uauc_post=epoch_logs['val_uauc_post'])

        # return self.history

    def evaluate(self, x, y, batch_size=256, y_post=None, log_file=None):
        """
        :param x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per evaluation step. If unspecified, `batch_size` will default to 256.
        :return: Dict contains metric names and metric values.
        """
        pred_ans = self.predict(x, batch_size)
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
        # uauc_,_,_ = uAUC(user,pred_ans,y)
        topK=[10]
        uauc, map, ndcg = self.my_metrics.test(pred_ans,topK=topK) # uauc==0
        uauc_post, map_post, ndcg_post = self.my_metrics_post.test(pred_ans,topK=topK)
        eval_result['uauc'] = uauc
        eval_result['uauc_post'] = uauc_post
        i = 0
        for K in topK:
            eval_result['map'+str(K)] = map[i]
            eval_result['ndcg'+str(K)] = ndcg[i]
            eval_result['map_post'+str(K)] = map_post[i]
            eval_result['ndcg_post'+str(K)] = ndcg_post[i]
            i += 1
        print("uauc, map, ndcg(finish):",uauc, 'map:', map, 'ndcg:', ndcg, file=log_file)
        print("uauc, map, ndcg(~~post):",uauc_post, 'map', map_post, 'ndcg', ndcg_post, file=log_file)
        return eval_result, pred_ans

    def predict(self, x, batch_size=256):
        """
        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param batch_size: Integer. If unspecified, it will default to 256.
        :return: Numpy array(s) of predictions.
        """
        model = self.eval()
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        tensor_data = Data.TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1)))
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=batch_size)

        pred_ans = []
        with torch.no_grad():
            for _, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()
                y_pred,_,_ = model(x)  # .squeeze()
                y_pred = y_pred.cpu().data.numpy()
                pred_ans.append(y_pred)
        return np.concatenate(pred_ans).astype("float64")

class CR_NFM(CRBaseModel):
    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, use_fm=True,
                 dnn_hidden_units=(256, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                 dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', gpus=None, writer=None,emb_dim = None,spurious_feat_name='position'):

        super(CR_NFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                     device=device, gpus=gpus)
        self.writer = writer
        self.use_fm = use_fm
        self.use_dnn = True #len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        self.spurious_feat_name = spurious_feat_name
        if use_fm:
            self.fm = BiInteractionPooling()

        if self.use_dnn:
            self.dnn = DNN(emb_dim, dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            self.dnn_linear = nn.Linear(
                dnn_hidden_units[-1], 1, bias=True).to(device)

            self.dnn_spurious_feat = DNN(emb_dim, dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            self.dnn_linear_spurious_feat = nn.Linear(
                dnn_hidden_units[-1], 1, bias=True).to(device)
            # reg for dnn
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)

            # reg for spurious feat dnn
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn_spurious_feat.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.dnn_linear_spurious_feat.weight, l2=l2_reg_dnn)
        self.to(device)

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        linear_logits = self.linear_model(X)
        if self.use_fm and len(sparse_embedding_list) > 0:
            fm_input = torch.cat(sparse_embedding_list, dim=1)
            fm_output = self.fm(fm_input).squeeze(dim=1)
        if self.use_dnn:
            dnn_output = self.dnn(fm_output)
            dnn_logit = self.dnn_linear(dnn_output)  
        logits_main_branch = dnn_logit + linear_logits # M-branch
        spurious_feats_emb = self.embedding_dict[self.spurious_feat_name](X[:,-1].long()) 
        spurious_feats_houtput = self.dnn_spurious_feat(spurious_feats_emb)
        spurious_feats_output = self.dnn_linear_spurious_feat(spurious_feats_houtput)
        spurious_feats_logits = torch.tanh(spurious_feats_output)
        logits = logits_main_branch + spurious_feats_logits
        y_pred = self.out(logits)
        y_pre2 = self.out(spurious_feats_output)
        return logits_main_branch, y_pred, y_pre2
