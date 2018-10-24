# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 15:49:04 2018

@author: hamed
"""

import numpy as np

#import matplotlib.pyplot as plt
from tqdm import tqdm

#import pdb
from kk_mimic_dataset import kk_mimic_dataset

#%% Test dataloader
                
train_set = kk_mimic_dataset(data_norm=False, test=True)
valid_set = kk_mimic_dataset(phase='valid',data_norm=False, test=True)
test_set = kk_mimic_dataset(phase='test',data_norm=False, test=True)

print("len(training_set)", len(train_set))
print("len(validation_set)", len(valid_set))
print("len(test_set)", len(test_set))

#%%

num_feats = 1440  #per seq token
seq_len = 10


max_ = [-np.inf]*num_feats
min_ = [np.inf]*num_feats
sum_ = [0.]*num_feats
sum2_ = [0.]*num_feats

training_counter = 0
for x in tqdm(train_set):
    x = x[0]
    training_counter += 1
    
    for j in range(num_feats):
        temp = x[:,j]
        max_[j] = max(max_[j], temp.max())
        min_[j] = min(min_[j], temp.min())
        sum_[j] += temp.sum()
        sum2_[j] += np.power(temp,2).sum()
        
validation_counter = 0    
for x in tqdm(valid_set):
    x = x[0]
    validation_counter += 1
    
    for j in range(num_feats):
        temp = x[:,j]
        max_[j] = max(max_[j], temp.max())
        min_[j] = min(min_[j], temp.min())
        sum_[j] += temp.sum()
        sum2_[j] += np.power(temp,2).sum()

test_counter = 0        
for x in tqdm(test_set):
    x = x[0]
    test_counter += 1
    
    for j in range(num_feats):
        temp = x[:,j]
        max_[j] = max(max_[j], temp.max())
        min_[j] = min(min_[j], temp.min())
        sum_[j] += temp.sum()
        sum2_[j] += np.power(temp,2).sum()
        
mean_ = np.divide (sum_,   seq_len * float(len(train_set) + len(valid_set) + len(test_set))) 
ex2_ =  np.divide (sum2_ , seq_len * float(len(train_set) + len(valid_set) + len(test_set)))

var_ = ex2_ - np.power(mean_, 2)

#%%
print("max(max_)", max(max_))
print("min(min_)", min(min_))

scale_ = np.maximum( np.abs(np.subtract(max_, mean_)), np.abs(np.subtract(min_, mean_)) ) #Because mean will be dedcuted later

#%%
file_name = 'stats.npy'
stats = (mean_, scale_, min_, max_, var_)
np.save(file_name, stats, allow_pickle=True, fix_imports=True)










