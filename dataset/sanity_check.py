# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 17:52:45 2018

@author: hamed
"""

from kk_mimic_dataset import kk_mimic_dataset
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import colored_traceback; colored_traceback.add_hook()

#%%
train_set = kk_mimic_dataset(test=True)
valid_set = kk_mimic_dataset(phase='valid', test=True)
test_set = kk_mimic_dataset(phase='test', test=True)

print("n_train = ", len(train_set))
print("n_valid = ", len(valid_set))
print("n_test = ", len(test_set))

#%%
num_feats = 1440
seq_len  = 10

max_train = [-np.inf]*num_feats
min_train = [np.inf]*num_feats
sum_train = [0.]*num_feats

max_valid = [-np.inf]*num_feats
min_valid = [np.inf]*num_feats
sum_valid = [0.]*num_feats

max_test = [-np.inf]*num_feats
min_test = [np.inf]*num_feats
sum_test = [0.]*num_feats

#%%
training_counter = 0
for x in tqdm(train_set):
    x = x[0]
    training_counter += 1
    
    for j in range(num_feats):
        temp = x[:,j]
        max_train[j] = max(max_train[j], temp.max() )
        min_train[j] = min(min_train[j], temp.min() )
        sum_train[j] += temp.sum()

#%%        
validation_counter = 0    
for x in tqdm(valid_set):
    x = x[0]
    validation_counter += 1
    
    for j in range(num_feats):
        temp = x[:,j]
        max_valid[j] = max(max_valid[j], temp.max() )
        min_valid[j] = min(min_valid[j], temp.min() )
        sum_valid[j] += temp.sum()
        
#%%
test_counter = 0        
for x in tqdm(test_set):
    x = x[0]
    test_counter += 1
    
    for j in range(num_feats):
        temp = x[:,j]
        max_test[j] = max(max_test[j], temp.max() )
        min_test[j] = min(min_test[j], temp.min() )
        sum_test[j] += temp.sum()

#%%       
mean_train = np.divide (sum_train, seq_len * len(train_set) )  
scale_train = np.maximum( np.abs(np.subtract(max_train, mean_train)), np.abs(np.subtract(min_train, mean_train) ))

mean_valid = np.divide (sum_valid, seq_len * len(valid_set) )  
scale_valid = np.maximum( np.abs(np.subtract(max_valid, mean_valid)), np.abs(np.subtract(min_valid, mean_valid) ))

mean_test = np.divide (sum_test, seq_len * len(test_set) )  
scale_test = np.maximum( np.abs( np.subtract(max_test,  mean_test)), np.abs( np.subtract(min_test, mean_test) ))


mean_ = np.divide( np.add(np.add(sum_train, sum_valid), sum_test) ,  seq_len * float( len(train_set) + len(valid_set) + len(test_set)))

scale_train_mean_  = np.maximum( np.abs( np.subtract( max_train, mean_) ),  np.abs(  np.subtract( min_train, mean_) ))
scale_valid_mean_  = np.maximum( np.abs( np.subtract( max_valid, mean_) ) , np.abs(  np.subtract( min_valid, mean_) ))
scale_test_mean_  =  np.maximum( np.abs( np.subtract( max_test , mean_) ),  np.abs(  np.subtract( min_test , mean_) ))
scale_ =   np.maximum(scale_train_mean_, np.maximum(scale_valid_mean_, scale_test_mean_) )


#%%
plt.figure(1)

plt.subplot(241)
plt.title("mean_{}".format("train"))
plt.plot(mean_train)
plt.subplot(242)
plt.title("scale_{}".format("train"))
plt.plot(scale_train)


plt.subplot(243)
plt.title("mean_{}".format("valid"))
plt.plot(mean_valid)
plt.subplot(244)
plt.title("scale_{}".format("valid"))
plt.plot(scale_valid)

plt.subplot(245)
plt.title("mean_{}".format("test"))
plt.plot(mean_test)
plt.subplot(246)
plt.title("scale_{}".format("test"))
plt.plot(scale_test)


plt.subplot(247)
plt.title("mean_{}".format("total"))
plt.plot(mean_)

plt.subplot(248)
plt.title("scale_{}".format("total"))
plt.plot(scale_)

plt.show()
