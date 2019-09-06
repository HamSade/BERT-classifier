# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 16:46:24 2018

@author: hamed
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.Models import Encoder
#from transformer.SubLayers import PositionwiseFeedForward


class ffn_compressed(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, d_out, dropout=0.1):
        super(ffn_compressed, self).__init__()
        
#        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
#        self.w_2 = nn.Conv1d(d_hid, d_out, 1) # position-wise
        self.w_1 = nn.Conv1d(d_in, d_out, 1)
        self.w_2 = nn.Conv1d(d_out, d_out, 1)
        
        self.layer_norm = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = x.transpose(1, 2)
        
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output)
        return output
        
        
class model(nn.Module):
        
    def __init__(self, args):
        
        super(model, self).__init__()
        self.d_src_vec = args.d_src_vec
        self.d_emb_vec = args.d_emb_vec
        self.seq_len = args.seq_len
        self.n_layers= args.n_layers
        self.n_head = args.n_head
        self.dropout = args.dropout
        self.d_inner = args.d_inner
        
        self.ffn = ffn_compressed(d_in=self.d_src_vec, d_hid=self.d_inner,
                                  d_out=self.d_emb_vec)

        #len_seq, n_layers, n_head, d_k, d_v, d_inner, dropout=0.1
        self.encoder = Encoder(len_seq=self.seq_len,
                               n_layers=self.n_layers,
                               n_head=self.n_head,
                               d_k=self.d_emb_vec//self.n_head,
                               d_v=self.d_emb_vec//self.n_head,
                               d_inner=self.d_inner,
                               dropout=self.dropout)
        #Fully connected. Seems to have a lot of params
#        self.FC1 = nn.Linear(self.d_emb_vec * self.len_seq , 64)   
#        self.FC2 = nn.Linear(64, 8)
#        self.FC3 = nn.Linear(8, 2)
        #Average pooling
        self.avg_pooling = nn.AvgPool1d(self.d_emb_vec, stride=1)  #d_emb_vec-1: so that all features are average and become a scalar
        self.FC = nn.Linear(self.seq_len, 2)  #2: binary classification
        self.softmax = nn.Softmax(dim=-1)
        
        
    def forward(self, x, x_pos):
        
        x = self.ffn(x)       
        x = self.encoder(x, x_pos, return_attns=False)
        
        #Fully connected
#        x =  x.view(x.shape[0], x.shape[1]*x.shape[2])
#        x = self.FC1(x); x = self.FC2(x); x = self.FC3(x)
        
        #
        print("size before avg pooling = ", x.shape)
        x = self.avg_pooling(x)
        print("size after avg pooling = ", x.shape)
        x = torch.squeeze(x)  #To get rid of 1-dimensional feature and results in [batch_size, len_seq] size
        print('shape after squeeze = ', x.shape)
        x = self.FC(x)
        
        return self.softmax(x)
    
       