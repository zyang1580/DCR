# DCR
This is an implementation for our paper "Addressing Confounding Feature Issue for Causal Recommendation" based on PyTorch. 

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
+ --report_K: the referenced topk-N (N=report_K) recommendation for early stopping. Note than referenced metric is NDCG@N.
+ --stop_refer: the referenced metric for early stopping.

Note that, compared with baselines, our model has not additional hyper-parameters.


## 3. Reproduce Results
We provide two methods:

### 3.1 Simple Methods:
We have saved the best model of all models, including DCR-MoE and baselines. We provide the following jupyter notebook file to reproduce the results:
```
best_kwai.ipynb
```
The best models can be downloaded at here (https://rec.ustc.edu.cn)

### 3.2 Start from Scratch

