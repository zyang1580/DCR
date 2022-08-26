# DCR
This is an implementation for our paper "Addressing Confounding Feature Issue for Causal Recommendation" (accepted by TOIS) based on PyTorch. 

We implement our model and baselines based on a famous package of deep-learning-based CTR models --- DeepCTR (torch version). And we take the renowned package ray[tune] to search hyper-parameters automatically.

Partial work was done when Yang Zhang was an intern at [WeChat, Tencent](https://weixin.qq.com/).


## 1. Requirements
+ pytorch == 1.8
+ deepctr-torch == 0.2.7
+ ray
+ ray[tune]
+ Numpy
+ python >= 3.7

We run codes on devices with NVIDIA GTX 3090 GPU or 2080Ti GPU.  


## 2. Parameters
Key parameters:
+ --lr: learning rate.
+ --reg_emb: L2 regularization cofficient for user/item embeddings.
+ --reg_para: L2 regularization cofficient for other model parameters.
+ --report_K: the referenced topk-N (N=report_K) recommendation for early stopping. Note than referenced metric is NDCG@N. (default:N=10)
+ --stop_refer: the referenced metric for early stopping. (need to set `val_ndcg_post10` for baselines and `val_do_ndcg_post10` for DCR-MoE)

Note that, compared with baselines, our model has not additional hyper-parameters.


## 3. Reproduce Results
We provide two methods:

### 3.1 Simple Methods:
We have saved the best model of all models, including DCR-MoE and baselines. We provide the following jupyter notebook file to reproduce the results:
```
# for kwai
best_kwai.ipynb
```
The best models and datsets can be downloaded at this [URL](https://rec.ustc.edu.cn/share/59a3e280-253c-11ed-aad3-51d42ffa3214). The instruction for downloading can be found at [data/README.md](data/README.md). 

***NOTE: the file name of the best models records the corresponding best hyper-parameters.

### 3.2 Start from Scratch
+ If you use a new dataset, you need to:
1. Preprocess your dataset, referring to the file "prepare.py" and the file "prepare_data2.py".
2. Update the main_function_kwai.py for the new dataset, focusing on several variables:
```
post_action: testing label
action: trainig: trainign label
FEA_FEED_LIST: feature list (potential to be utilized)
USE_FEAT: utilized features
length_name: confounding feature
code_length_name: coded confounding feature
```

+ Then, to search for hyper-parameters, execute the code "searching_hyper_DCR_MOE_kwai.py". The search space is controlled by the following code in the file (different models need to search different hyper-parameters):
```
    config={
        'lr':tune.grid_search([1e-3,1e-2]),
        'reg_emb':tune.grid_search([[1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,0]]), #
        'reg_para':tune.grid_search([0,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]),
        'model':tune.grid_search(['fastFairNFM']), # DCR-MoE
        'alpha':tune.grid_search([0]), # not used for DCR-MoE
        'stop_refer':tune.grid_search([refer_metrics])
    }
```
  Please note that other hyper-parameters are controlled by the [`parameters class`](https://github.com/zyang1580/DCR/blob/3c8bbbcd4508366efd5590289253669b2eba2eac/main_function_kwai.py#L49) in "main_function_kwai.py".


  Meanwhile, you can control the hyper-parameter `model` to select the DCE-MoE or baselines. 
  ```
  model = fastFairNFM: DCR-MoE
  model = MyNFM (or NFM) and used_codelen=0: NFM-WOA
  model = MyNFM (or NFM) and used_codelen=1 (defined in the above *parameters class*): NFM-WA
  model = ipw: IPW
  model = FairGo : FairGo
  model = CR_NFM: CR
  ```



