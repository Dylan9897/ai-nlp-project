# 异常图片识别
VIT，Image-classify，out of detection，K-means，PQ

### 数据集下载

1. [DAGM2007](https://pan.baidu.com/s/1wQMfgtjgjwtYxgO2NSf7aQ) 提取码：v7t1
2. [MVTEC](https://pan.baidu.com/s/1YTqJn5otJBKzN7woPA04eA) 提取码：2etq 
3. [其他](https://pan.baidu.com/s/1WOLpik9-MMQIJlrklgtFUA) 提取码：ph5c
4. [预训练模型](https://pan.baidu.com/s/1GFlDvWmNyzjaGfstihvzsA) 提取码：l4dd

### 主要方法                                                                                                                                                                                                                                        

​	主 要 参 考 论 文 [Out-of-Distribution Detection with Deep Nearest Neighbors](https://arxiv.org/abs/2204.06507)和 [Deep Nearest Neighbor Anomaly Detection](https://arxiv.org/abs/2002.10445)。                                                                                                                                                                                                                                                                                             

​	采 用 预 训 练 模 型 ， 对 已 知 正 常 图 片 库 中 的 所 有 图 片 推 理 得 到 它 们 的 特 征 向 量 ， 形 成 正 常 图 片 特 征 库 。                                                                                                                                                                                                                                                                                                                                                                                               

​	对 于 未 知 图 片 ， 采 用 相 同 模 型 计 算 其 特 征 向 量 ， 与 正 常 图 片 特 征 库 中 所 有 向 量 计 算 距 离 ， 得 到 前 k个 临 近 特 征 (knn)的 平 均 距 离 ， 该 距 离 如 果 大 于 阈 值 ， 则 判 断 为 异 常 。                                                                                                                                                                                                                                                                                                                                   

​	如 何 确 定 阈 值 ： 对 正 常 图 片 特 征 库 中 的 所 有 图 片 当 作 未 知 图 片 进 行 推 理 ， 90%-95%的 图 片 需 被 识 别 为 正 常 图 片 。    

### 环境依赖

1. [pytorch 1.12.1](https://pytorch.org/get-started/previous-versions/#v1121) 根 据 cuda版 本 对 应 安 装     

   ```python
   # CUDA 10.2
   conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
   # CUDA 11.3
   conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
   # CUDA 11.6
   conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
   # CPU Only
   conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cpuonly -c pytorch
   ```

2. [clip](https://github.com/openai/CLIP)（ 需 用 到 基 于 CLIP预 训 练 的 ViT模 型 ）  

   ```python
   pip install git+https://github.com/openai/CLIP.git
   ```

3. [faiss](https://github.com/facebookresearch/faiss)（ 加 速 相 似 性 搜 索 ） 

   ```python
   # GPU
   conda install -c pytorch faiss-gpu
   # CPU
   conda install -c pytorch faiss-cpu
   ```

4. [loguru](https://github.com/Delgan/loguru)

   ```python
   pip install loguru
   ```

5. [tqdm](https://github.com/tqdm/tqdm)  

   ```
   pip install tqdm
   ```

### 功能描述

接 口 函 数 为  `main.py` 中 的  `detect_abnormal`， 参 数 如 下 ： 

```python
test_dir: 要检测的图片所在目录
db_dir (None): 构建的正常图片特征库所保存的目录 (如果db_dir目录存在，则从该目录导入库；如果不存在，则重新构建库，并保存在此目录)
image_dir ('image/train'): 包含已知的正常图片
model_name ('vit'): 使用的模型名称，支持 'vit', 'resnet18', 'resnet50', 'resnet101'
k (100): knn中的k值
thres_ratio (0.92): 多少正常图片需被识别为正常，用于构建库时计算阈值，一般为0.9-0.95
gpu (True): 是否使用gpu
batch_size (128): mini-batch规模
num_workers (8)
```

该函数的功能

1. 构 建 正 常 图 片 特 征 库                                                                                                                                                                                                                           

\- 如 果 `db_dir`存 在 且 合 法 ， 会 从 `db_dir`直 接 加 载 库 ， 此 时 传 入 的 `image_dir, thres_ratio, k`都 不 会 被 程 序 使 用 ， 由 库 给 定 。                                                                                                                             

\- 如 果 `db_dir`不 存 在 ， 会 从 `image_dir`中 的 图 片 构 建 正 常 图 片 特 征 库 ， 并 保 存 在 `db_dir`中                                                                                                                                                                                                                                                                                                                                                                                                     

2. 检 测                                                                                                                                                                                                                                         

对 `test_dir`中 的 图 片 进 行 检 测 ， 在 `result`目 录 中 新 建 一 个 以 运 行 时 间 命 名 的 文 件 夹 ， 将 检 测 结 果 保 存 在 其 中 。                                                                                                                                            

检 测 结 果 为 `abnormal.result`， 保 存 了 所 有 模 型 检 测 到 的 异 常 图 片 ， 每 行 是 一 个 图 片 路 径                                                                                                                                                                 

也 会 保 存 运 行 配 置 为 `config.json`  

### 实验结果

#### DAGM
准确率：0.8763736263736264

|              | precision | recall | f1-score | support |
| ------------ | --------- | ------ | -------- | ------- |
| 0            | 0.86      | 0.97   | 0.91     | 1150    |
| 1            | 0.93      | 0.72   | 0.81     | 670     |
| accuracy     |           |        | 0.88     | 1820    |
| macro avg    | 0.89      | 0.54   | 0.86     | 1820    |
| weighted avg | 0.88      | 0.88   | 0.87     | 1820    |

####  OOD

准确率：0.9903100775193798

|              | precision | recall | f1-score | support |
| ------------ | --------- | ------ | -------- | ------- |
| 0            | 0.00      | 0.00   | 0.00     | 0       |
| 1            | 1.00      | 0.99   | 1.00     | 516     |
| accuracy     |           |        | 0.99     | 516     |
| macro avg    | 0.50      | 0.50   | 0.50     | 516     |
| weighted avg | 1.00      | 0.99   | 1.00     | 516     |

#### MVTEC

准确率：0.543768115942029

|              | precision | recall | f1-score | support |
| ------------ | --------- | ------ | -------- | ------- |
| 0            | 0.36      | 0.87   | 0.51     | 467     |
| 1            | 0.90      | 0.42   | 0.57     | 1258    |
| accuracy     |           |        | 0.54     | 1725    |
| macro avg    | 0.63      | 0.65   | 0.54     | 1725    |
| weighted avg | 0.75      | 0.54   | 0.56     | 1725    |





