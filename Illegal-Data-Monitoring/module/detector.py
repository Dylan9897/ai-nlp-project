import os
import sys

import faiss
import numpy as np
import torch
import tqdm
from loguru import logger
from torch import nn
import json


normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10

class KNNDetector(object):
    def __init__(self, device):
        self.device = device
        logger.info(f'Using device: {device}.')

    def setup_from_db(self, db_dir):
        
        db_path = os.path.join(db_dir, 'db.npy')
        config_path = os.path.join(db_dir, 'config.json')
        
        try:
            assert os.path.exists(db_dir) and os.path.exists(db_path) and os.path.exists(config_path)
        except:
            logger.error(f'invalid database {db_dir}!')
            sys.exit()

        logger.info(f'{db_dir} found, setup detector from {db_dir}.')
        
        with open(config_path, 'r') as f: config = json.load(f)
        self.threshold = config['threshold']
        self.K = config['K']
        self.thres_ratio = config['thres_ratio']
        self.model_name = config['model_name']
        
        train_features = np.load(db_path)
        # * store in faiss
        if self.device.type == 'cuda':
            res = faiss.StandardGpuResources() # use a single GPU
            index_flat = faiss.IndexFlatL2(train_features.shape[1])
            gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
            self.index = gpu_index_flat
        else:
            self.index = faiss.IndexFlatL2(train_features.shape[1])
        self.index.add(train_features)
        
        logger.info(f'Detector setup finished. Total samples: {self.index.ntotal}. K={self.K}. thres_ratio={self.thres_ratio}.')
        return self.threshold
    
    @torch.no_grad()
    def setup_from_dataset(self, net: nn.Module, train_loader, K=100, thres_ratio=0.92, save_db_dir=None):
        logger.info(f'Setup detector from dataset.')
        
        self.K = K
        self.thres_ratio = thres_ratio
        self.model_name = net.name
        
        train_features = []

        net = net.to(self.device)
        net.eval()

        ## inference
        for batch in tqdm.tqdm(train_loader, desc='Setting up database: '):
            # print(f"batch size shape is {batch.shape}")
            data = batch.to(self.device)
            feature = net(data)
            train_features.append(normalizer(feature.cpu().numpy()))

        train_features = np.concatenate(train_features, axis=0)
        # * store in faiss
        if self.device.type == 'cuda':
            res = faiss.StandardGpuResources() # use a single GPU
            index_flat = faiss.IndexFlatL2(train_features.shape[1])
            gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
            self.index = gpu_index_flat
        else:
            self.index = faiss.IndexFlatL2(train_features.shape[1])  # 构建索引
        self.index.add(train_features) # 添加向量

        # * determine threshold (`thres_ratio` should be predicted as normal)
        ## search K+1 neighbors, because the nearest neightbor is itself
        D, _ = self.index.search(train_features, self.K+1)  # 对 train_features 执行 self.k 的最近邻搜索
        D = D[:, 1:]
        try:
            assert D.shape == (self.index.ntotal, self.K)
        except:
            logger.error('D.shape should be (num_database, K)!')
            sys.exit()
            
        ## get k neighbors mean distance
        D = np.mean(D, axis=-1)
        ## sort
        D = np.sort(D)
        # print("start to write D")
        with open("a.txt",'w',encoding='utf-8') as fl:
            fl.write(str(D.tolist()))
        ## get threshold
        self.threshold = D[int(self.thres_ratio*self.index.ntotal)]  # 取前第 0.92*5335 的样本的阈值
        logger.info(f'Threshold: {self.threshold}.')
        ## save
        if save_db_dir is not None:
            os.makedirs(save_db_dir, exist_ok=True)
            db_path = os.path.join(save_db_dir, 'db.npy')
            with open(os.path.join(save_db_dir, 'config.json'), 'w') as f:
                json.dump({
                    'threshold': float(self.threshold),
                    'K': self.K,
                    'thres_ratio': self.thres_ratio,
                    'model_name': self.model_name
                }, f, indent=1)
                
            np.save(db_path, train_features)
        
        return self.threshold
        
    @torch.no_grad()
    def detect(self, net: nn.Module, test_loader):
        """detect abnormality using predefined threshold"""
        try:
            assert self.__check_model_name(net.name)
        except:
            logger.error('The detection network is different from the network used for initialization!')
            sys.exit()

        D = self.cal_distance(net, test_loader)    
        return self.detect_given_distance(D)


    @torch.no_grad()
    def predict(self,net:nn.Module,data):
        """calculate distance for the test_loader"""
        try:
            assert self.__check_model_name(net.name)
        except:
            logger.error('The detection network is different from the network used for initialization!')
            sys.exit()
        test_features = []
        net = net.to(self.device)
        net.eval()
        data = data.unsqueeze(0)
        feature = net(data)
        test_features.append(normalizer(feature.cpu().numpy()))
        test_features = np.concatenate(test_features, axis=0)
        D, _ = self.index.search(test_features, self.K)  # shape of D: (num, self.K)

        ## get k neighbors mean distance
        D = np.mean(D, axis=-1).tolist()[0]
        return D>self.threshold,D



    @torch.no_grad()
    def cal_distance(self, net: nn.Module, test_loader):
        """calculate distance for the test_loader"""
        try:
            assert self.__check_model_name(net.name)
        except:
            logger.error('The detection network is different from the network used for initialization!')
            sys.exit()
            
        test_features = []
        net = net.to(self.device)
        net.eval()

        # * inference
        for batch in tqdm.tqdm(test_loader, desc='Parsing testset: '):
            data = batch.to(self.device)
            feature = net(data)
            test_features.append(normalizer(feature.cpu().numpy()))

        test_features = np.concatenate(test_features, axis=0)

        # * search
        D, _ = self.index.search(test_features, self.K) # shape of D: (num, self.K)

        ## get k neighbors mean distance
        D = np.mean(D, axis=-1)

        return D
    
    def detect_given_distance(self, D):
        """detect abnormality using predefined threshold, given distances"""
        ab_idx = np.where(D > self.threshold)[0].tolist()
        return ab_idx
    
    def __check_model_name(self, model_name):
        return model_name == self.model_name

    
