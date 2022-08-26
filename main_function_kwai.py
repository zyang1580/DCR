# -*- coding: utf-8 -*-
"""
This is the version for search hyper-parameters by using the package Ray.
you need install it with pip
"""
from operator import pos
from re import A, S
import os
from typing import Dict, List
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
from ipw_nfm import ipw_myNFM

from utils import metrics
from decfair_tune import FastFairNFM
from baseline_tune import DeepNFM, NFM, MyDeepFM, MyFM, DeepFM, FM_witout_FirstOrder
from counterfactual_reasoning import CR_NFM
from pre_data_wechat import *
from FairGo import myNFM,FairGo
from ipw_nfm import ipw_myNFM
# from decfair_tune_nfm import DCR_NFM,DCR_NFM_apr

# path
ROOT_PATH = "/home/zyang/code-2021/decFair/data/"
# # 

class parameters(object):
    def __init__(self) -> None:
        super().__init__()
        self.lr = 1e-3
        self.reg_emb = 1e-6
        self.reg_para = 1e-2
        self.dim = 16
        self.batch_size=1024
        self.epoch = 100
        self.use_videolen = 0
        self.patience = 10
        self.use_codelen = 1
        self.model= 'fastFairNFM' # 'NFM' 
        self.post_action='like' #comment
        self.stop_refer = 'val_do_ndcg_post10' 
        self.opt = 'adagrad'
        self.pretrain = 0
        self.alpha = 0
        self.cuda=0
        self.pretrained_model="/data/zyang/decFair/logs/best-kwai-adagrad-myNFM-vl-0-vc-1-like-myNFM-lr_0.01-reg_emb_0.001-reg_para_0-dim_16-stop-val_ndcg_post10-auxloss-1.0-stop-val_ndcg_post10-m.pth"
    def reset(self, config:Dict):
        for name,val in config.items():
            setattr(self,name,val)

def change_columns_name(columns_name,uname='user_id',iname='item_id'):
    index_u = columns_name.index(uname)
    columns_name[index_u] = 'user_id'
    index_i = columns_name.index(iname)
    columns_name[index_i] = 'item_id'

    return columns_name
def run_a_model(config,need_train=True):
    dataset='kwai'
    print("dataset:",dataset)
    args = parameters()
    args.reset(config)
    used_optimizer = args.opt
    lr = args.lr
    post_action = args.post_action     # the label for testing the deconfouded results
    save_name = dataset + "ipwmax_" + used_optimizer+"-"+args.model+"-vl-"+str(args.use_videolen)+'-vc-'+str(args.use_codelen)+'-' + post_action + '-' + args.model+"-lr_{}-reg_emb_{}-reg_para_{}-dim_{}-stop-{}-auxloss-{}".format(args.lr,args.reg_emb, args.reg_para, args.dim, args.stop_refer,args.alpha)+"-stop-"+args.stop_refer+'-train-'+str(need_train)
    log_file_path = "/home/zyang/code-2021/decFair/logs/"+dataset+'/'+ save_name + ".txt"
    log_file  = open(log_file_path,'wt')
    FEA_FEED_LIST = ['item_id','duration_time'] #['item_id','bgm_song_id','bgm_singer_id',"videoplayseconds"]
    print("FEA_FEED_LIST:", FEA_FEED_LIST, file=log_file)
    print("**** post_action:", post_action, file=log_file)
    # length_name = 'videoplayseconds'
    length_name =  'duration_time' # "videoplayseconds"#
    code_length_name = 'code_duration' # "code_videolen" #
    item_name = 'item_id'
    if args.use_videolen == 0: # not use any video length information
        FEA_FEED_LIST = list(filter(lambda x: x!=length_name, FEA_FEED_LIST))
        print("please make sure that the video length is not used, used features",FEA_FEED_LIST,file=log_file)
    if args.use_codelen > 0:
        FEA_FEED_LIST.append(code_length_name) # use the code lenght in FM 
    # submit = pd.read_csv(ROOT_PATH + '/coded_test_data.csv')[['userid', 'feedid']]
    post_action = args.post_action     # the label for testing the deconfouded results
    print("**** post_action:", post_action,file=log_file)
    for action in ['finish']:          # ACTION_LIST: for training
        train = pd.read_csv(ROOT_PATH + '/'+dataset+f'/final_train_data_for_{action}.csv')
        print("origin columns:",train.columns,file=log_file)
        valid = pd.read_csv(ROOT_PATH + '/'+dataset+f'/final_valid_data_for_{action}.csv')
        test = pd.read_csv(ROOT_PATH + '/'+dataset+f'/final_test_data_for_{action}.csv')
        # train = train.sample(frac=1, random_state=42).reset_index(drop=True)
        train.columns = change_columns_name(train.columns.tolist(), uname='user_id', iname='item_id')
        valid.columns = change_columns_name(valid.columns.tolist(), uname='user_id', iname='item_id')
        test.columns = change_columns_name(test.columns.tolist(), uname='user_id', iname='item_id')
        USE_FEAT = ['user_id', 'item_id', action, post_action] + FEA_FEED_LIST[1:]
        train= train[USE_FEAT]
        test = test[USE_FEAT]
        valid = valid[USE_FEAT]
        print("posi prop:",sum((train[action]==1)*1)/train.shape[0],file=log_file)
        # print()
        # test = pd.read_csv(ROOT_PATH + '/test_data.csv')[[i for i in USE_FEAT if i != action and i!=post_action]]
        target = [action, post_action] # here we have two task, one for direct target and one post target
        data = pd.concat((train, valid, test)).reset_index(drop=True)
        if args.use_videolen>0:
            dense_features = [length_name]
            data[dense_features] = data[dense_features].fillna(0)
        else:
            dense_features = []
        sparse_features = [i for i in USE_FEAT if i not in dense_features and i not in target]
        data[sparse_features] = data[sparse_features].fillna(0)
        
        # 2.count #unique features for each sparse field,and record dense feature field name
        if args.use_videolen > 0:
            print("use video length",file=log_file)
            fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(),embedding_dim=args.dim)
                                    for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                                    for feat in dense_features]
        else: # not use video lenght information
            print("not use video length",file=log_file)
            fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(),embedding_dim=args.dim)
                                    for feat in sparse_features]

        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns

        feature_names = get_feature_names(
            linear_feature_columns + dnn_feature_columns)

        # 3.generate input data for model
        train, valid, test = data.iloc[:train.shape[0]].reset_index(drop=True),  data.iloc[train.shape[0]:train.shape[0]+valid.shape[0]].reset_index(drop=True), data.iloc[train.shape[0]+valid.shape[0]:].reset_index(drop=True)
        
        train_model_input = {name: train[name] for name in feature_names}
        valid_model_input = {name: valid[name] for name in feature_names}
        test_model_input = {name: test[name] for name in feature_names}
        if args.model == 'FairNFM' or args.model =='fastFairNFM' or args.model=='CR_NFM' or args.model=='FairGo' or args.model=='ipw' or args.model=="DCR_NFM" or args.model=="ADCR_NFM":
            confounder = code_length_name
            assert confounder == feature_names[-1] # the confounder must be to place at the last column.
        print("input features:",train_model_input.keys(),file=log_file)

        # 4.Define Model,train,predict and evaluate
        device = 'cpu'
        use_cuda = True
        if use_cuda and torch.cuda.is_available():
            print('cuda ready...',file=log_file)
            device = 'cuda:'+ str(args.cuda)
       
        if args.model == 'deepFM':
            used_model = MyDeepFM
        elif args.model == 'deepNFM':
            used_model = DeepNFM
        elif args.model == 'NFM':
            used_model = NFM
        elif args.model == 'myNFM':
            used_model = myNFM
        elif args.model =='2FM':
            used_model = FM_witout_FirstOrder
        elif args.model =='FM':
            used_model = MyFM
        elif args.model =='CR_NFM':
            used_model = CR_NFM
        elif args.model == 'ipw':
            used_model = ipw_myNFM
            vlen_info = data.groupby(confounder).agg({'finish':'mean'})
            con = vlen_info.index.values
            proensity_pos = vlen_info.values
            proensity_neg = 1 - proensity_pos
            # proensity_pos = (proensity_pos/proensity_pos.sum())**0.5
            # proensity_neg = (proensity_neg/proensity_neg.sum())**0.5
            proensity_pos = (proensity_pos/proensity_pos.max())**0.5
            proensity_neg = (proensity_neg/proensity_neg.max())**0.5
            con_idx = np.argsort(con)
            propensity_pos = proensity_pos[con_idx]
            propensity_neg = proensity_neg[con_idx]
            linear_feature_columns = list(filter(lambda x: x.name != confounder, linear_feature_columns))
            dnn_feature_columns = list(filter(lambda x: x.name!= confounder, dnn_feature_columns))
            # the confusing feature is not inputted for ipw
        elif args.model == 'FairGo':
            used_model = FairGo
            confounder_num = data[confounder].unique().shape[0]
            if args.use_codelen==0:
                linear_feature_columns = list(filter(lambda x: x.name != confounder, linear_feature_columns))
                dnn_feature_columns = list(filter(lambda x: x.name!= confounder, dnn_feature_columns))
        elif args.model == 'FairNFM' or args.model =='fastFairNFM':
            if args.model == 'FairNFM':
                used_model = FairNFM
            else:
                used_model =FastFairNFM
            confounder_num = data[confounder].unique().shape[0]
            confounder_pd = train[[item_name,confounder]] #.drop_duplicates('feedid')
            # confounder_total = train[confounder].values.reshape(-1)
            _, confoudner_prob = np.unique(confounder_pd[confounder].values.reshape(-1),return_counts=True) # compute by the interaction
            confoudner_prob = confoudner_prob * 1.0 / confoudner_prob.sum()  # the probability
            linear_feature_columns = list(filter(lambda x: x.name != confounder, linear_feature_columns))
            dnn_feature_columns = list(filter(lambda x: x.name!= confounder, dnn_feature_columns))   # filter the confouder in the feature that  model will utiliz
        else:
            print("don't have this type model:", args.model,fiel=log_file)
            exit()
        
        dnn_hidden_units = (256,128)
        print("used model:",used_model,file=log_file)
        # creat model based on args
            
        if args.model == 'FairNFM' or args.model =='fastFairNFM':  # DCR-MOE
            model = used_model(linear_feature_columns=linear_feature_columns, dnn_hidden_units=dnn_hidden_units, dnn_feature_columns=dnn_feature_columns,
                        task='binary', l2_reg_embedding=args.reg_emb, device=device, l2_reg_dnn=args.reg_para, 
                        l2_reg_linear=args.reg_para,emb_dim=args.dim,confounder_num=confounder_num,confounder_name=confounder,confounder_prob=confoudner_prob)

        elif args.model == 'CR_NFM': # CR
            # print("load pretrained model...")
            model = used_model(linear_feature_columns=linear_feature_columns,dnn_hidden_units=dnn_hidden_units, dnn_feature_columns=dnn_feature_columns,
                        task='binary', l2_reg_embedding=args.reg_emb, device=device, l2_reg_dnn=args.reg_para, 
                        l2_reg_linear=args.reg_para,emb_dim=args.dim,spurious_feat_name=confounder)
        elif args.model == 'FairGo':
            # print("load pretrained model...")
            model = used_model(linear_feature_columns=linear_feature_columns,dnn_hidden_units=dnn_hidden_units, dnn_feature_columns=dnn_feature_columns,
                        task='binary', l2_reg_embedding=args.reg_emb, device=device, l2_reg_dnn=args.reg_para, 
                        l2_reg_linear=args.reg_para,emb_dim=args.dim,confounder_num=confounder_num,confounder=confounder)
            model.load_pretrained_recommender(args.pretrained_model)
        elif args.model == 'ipw':
            # print("load pretrained model...")
            model = used_model(linear_feature_columns=linear_feature_columns,dnn_hidden_units=dnn_hidden_units, dnn_feature_columns=dnn_feature_columns,
                        task='binary', l2_reg_embedding=args.reg_emb, device=device, l2_reg_dnn=args.reg_para, 
                        l2_reg_linear=args.reg_para,emb_dim=args.dim,propensity_pos=propensity_pos,propensity_neg=propensity_neg,confounder_name=confounder)
        else: # NFM
            # print("load pretrained model...")
            model = used_model(linear_feature_columns=linear_feature_columns,dnn_hidden_units=dnn_hidden_units, dnn_feature_columns=dnn_feature_columns,
                        task='binary', l2_reg_embedding=args.reg_emb, device=device, l2_reg_dnn=args.reg_para, 
                        l2_reg_linear=args.reg_para,emb_dim=args.dim)

        if lr == 0:
            lr=None # taking default leaning rate
        model.compile(used_optimizer, "binary_crossentropy", metrics=["binary_crossentropy", "auc"],lr=lr)
        
        print("the best model will be saved as:","best-"+save_name, file=log_file)
        if need_train: # training.....
            history = model.fit(train_model_input, train[target].values, batch_size=args.batch_size, epochs=args.epoch, verbose=1,\
                validation_data=(valid_model_input,valid[target].values), save_name=save_name,args_=args,log_file=log_file)
        else:
            print("test on pretrined model....",file=log_file)
        # given the prediction for test data at the best model
        print("test model on the best model")
        if not need_train:
            save_name = dataset + "ipwmax_" + used_optimizer+"-"+args.model+"-vl-"+str(args.use_videolen)+'-vc-'+str(args.use_codelen)+'-' + post_action + '-' + args.model+"-lr_{}-reg_emb_{}-reg_para_{}-dim_{}-stop-{}-auxloss-{}".format(args.lr,args.reg_emb, args.reg_para, args.dim, args.stop_refer,args.alpha)+"-stop-"+args.stop_refer+'-train-'+str(True)
        

        """
        Please keep the following pathe same to the one in the .fit()
        """
        if os.path.exists("/data/zyang/decFair/logs/best-"+save_name+"-m.pth"): # load the best model
            model.load_state_dict(torch.load("/data/zyang/decFair/logs/best-"+save_name+"-m.pth"))
        else:
            print("can not load model...",log_file)
            return 0
        test_y = test[target].values
        test_metrics  = metrics(test['user_id'].values.squeeze(),test_y[:,0])   
        test_metrics_post  = metrics(test['user_id'].values.squeeze(),test_y[:,1])  
        if args.model == 'FairNFM' or args.model =='fastFairNFM': # FairNFM
            for pre_mode in ['condition','do']: # condition: correlation-based prediction, do: causal prediction
                print("\n||| **********prediction mode:",pre_mode,file=log_file)
                model.set_pre_mode(mode=pre_mode)
                pred_ans = model.predict(test_model_input, 256)
                # uauc1,_,_ = uAUC(test['user_id'],pred_ans,test_y[:,0])
                topK = [10,20,50]
                uauc, map_list, ndcg_list = test_metrics.test(pred_ans,topK=topK)
                print('test with the best model, finish uauc:', uauc, 'map:', map_list,'ndcg:', ndcg_list, file=log_file)
                # uauc2,_,_ = uAUC(test['user_id'].values, pred_ans, test_y[:,1])
                uauc_post, map_post_list, ndcg_post_list = test_metrics_post.test(pred_ans,topK=topK)
                print('test with the best model, post action uauc', uauc_post, 'map:', map_post_list, 'ndcg', ndcg_post_list, file=log_file)
                torch.cuda.empty_cache()
        else:
            pred_ans = model.predict(test_model_input, 256)
            test_y = test[target].values
            topK = [10,20,50]
            uauc, map_list, ndcg_list = test_metrics.test(pred_ans,topK=topK)
            print('test with the best model, finish uauc:', uauc, 'map:', map_list,'ndcg:', ndcg_list,file=log_file)
            # uauc2,_,_ = uAUC(test['user_id'].values, pred_ans, test_y[:,1])
            uauc_post, map_post_list, ndcg_post_list = test_metrics_post.test(pred_ans,topK=topK)
            print('test with the best model, post action uauc', uauc_post, 'map:', map_post_list, 'ndcg', ndcg_post_list, file=log_file)
            # uauc1,_,_ = uAUC(test['user_id'],pred_ans,test_y[:,0])
            # print('test with the best model, finish uauc:', uauc1)
            # uauc2,_,_ = uAUC(test['user_id'].values, pred_ans, test_y[:,1])
            # print('test with the best model, like uauc:',uauc2)
            torch.cuda.empty_cache()

if __name__=='__main__':
    config={}
    config['model']='fastFairNFM'
    run_a_model(config)