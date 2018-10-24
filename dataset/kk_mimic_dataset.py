# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 10:40:35 2018

@author: hamed 
"""

try: 
    import torch
    from torch.utils import data
except:
    pass
    
import numpy as np
from sklearn import datasets

#%%
class kk_mimic_dataset(data.Dataset):
    
    def __init__(self, phase="train", seq_len=10, data_norm=True, test=False):
        
        super(kk_mimic_dataset, self).__init__()
        if phase == "train": 
            if test:
                data_path = "../../mimic-libsvm/" + "PATIENTS_SPLIT_XGB_TRAIN"
            else:    
                data_path = "../mimic-libsvm/" + "PATIENTS_SPLIT_XGB_TRAIN"                
            data = datasets.load_svmlight_file(data_path)           
        else:
            if test:
                data_path = "../../mimic-libsvm/" + "PATIENTS_SPLIT_XGB_VALID"
            else: 
                data_path = "../mimic-libsvm/" + "PATIENTS_SPLIT_XGB_VALID"                
            data = np.array(datasets.load_svmlight_file(data_path))
        
            percent = 20
            if  phase == "valid":#               
                data = [ data[0][:data[1].shape[0]//percent], data[1][:data[1].shape[0]//percent] ]
            else:            
                data = [ data[0][data[1].shape[0]//percent:], data[1][data[1].shape[0]//percent:] ]
                
        # TODO: ONLY for fast debugging
#        factor = 10
        # First factor ones
#        data = [ data[0][:data[1].shape[0]//factor], data[1][:data[1].shape[0]//factor] ]
        
        #Random selection
        factor = 20
        n_data = data[0].shape[0]
        ind_ = np.ones(factor)
        ind_ = np.concatenate((ind_, np.zeros(n_data-factor)))
        ind_ = np.random.permutation(ind_)
        ind_ = np.greater(ind_, 0)
        data = [ data[0][ind_], data[1][ind_] ]
        
        
#        data = np.nan_to_num(data)
        self.d_feat = 14400
        self.seq_len = seq_len
        self.features = np.array(data[0].todense())
        self.labels = np.array(data[1])
        
        # Removing last irrelevant features
        self.temporal_features = np.split(self.features[:,:self.d_feat], self.seq_len, axis=1)
        self.temporal_features = np.reshape(self.temporal_features, (-1,self.seq_len, self.d_feat//self.seq_len))
        self.fixed_features = self.features[:,self.d_feat:]                
        
        #Data normalization 
        if data_norm:
            if test:
                file_name = "stats.npy"
            else:
                file_name = "dataset/stats.npy"            
            stats = np.load(file_name)  #stats = ("mean_", "scale_", "min_", "max_", "var_") * 1440
            mean_ = stats[0,:]
            scale_= stats[1,:]
            self.temporal_features = np.nan_to_num( np.divide( np.subtract(self.temporal_features, mean_), scale_) )
        
    #%%   
    def __len__(self):
        return self.labels.shape[0]
       
    #%%
    def __getitem__(self, index):
        src_seq = self.temporal_features[index]
        src_fixed_feats = self.fixed_features[index]
        gold = self.labels[index]        
        src_pos = np.array([pos_i for pos_i, _ in enumerate(src_seq)])  #TODO pos_i <- pos_i + 1 ??!        
        src_seq = torch.FloatTensor(src_seq)
        src_pos = torch.LongTensor(src_pos)
        src_fixed_feats = torch.FloatTensor(src_fixed_feats)
        gold = torch.LongTensor([gold])
        return src_seq, src_pos, gold, src_fixed_feats

#%% Data loader     
def loader(dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True):
    torch.initial_seed()  #to change the seed
    params = {'batch_size': batch_size, 'shuffle': shuffle, 'num_workers':num_workers}
    return data.DataLoader(dataset, **params) #, collate_fn=collate_fn_temp)

#%% test data loader
#dataset_ = kk_mimic_dataset(test=True)
#loader_ = iter(loader(dataset_, batch_size=2))
#
#for j in range(5):
#    print('*'*50 )
#    for i in range(10):
#        x = next(loader_)
#        temp_features = x[0]
#        print("sum temp_features = ", np.sum(temp_features.numpy()))
#        print("labels = ", x[2])















