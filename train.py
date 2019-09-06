'''
This script handling the training process.
'''
import argparse
import math
import time

from tqdm import tqdm
import colored_traceback; colored_traceback.add_hook()

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from transformer.Optim import ScheduledOptim

import sys
sys.path.insert(0, './dataset/')
from kk_mimic_dataset import kk_mimic_dataset, loader

from Transformer_classifier import model
from AUCMeter import AUCMeter
    
print_chunk_size = 600000  #TODO So that it does not clutter with printing! :D

#%%
def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
#    print("gold.shape", gold.shape)
#    print("pred.shape", pred.shape)
#    gold = gold.view(-1)
    
    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, reduction='sum')
    return loss
    
#%%
def train_epoch(model_, training_data, optimizer, device):
    ''' Epoch operation in training phase'''

    model_.train()

    total_loss = 0
    pred = []
    gold = []
    n_seq_total = 0             
    for batch in training_data:
#            tqdm(training_data, mininterval=2,
#            desc='  - (Training)   ', leave=False):

        # prepare data
        src_seq, src_pos, gold_, src_fixed_feats = map(lambda x: x.to(device), batch)
        gold_ =  torch.cuda.LongTensor(gold_) #TODO ...
        gold_ = gold_.view(-1) #TODO ... doing it here instead of cal_l0ss. good for AUC too!
              
        # forward
        optimizer.zero_grad()
        pred_ = model_(src_seq, src_pos)
#        print("pred_.shape = ", pred_.shape)            
        
        # backward
        loss = cal_loss(pred_, gold_)
        loss.backward()
        # update parameters
        optimizer.step_and_update_lr()        
        # note keeping
        total_loss += loss.item() 
             
        pred_ = pred_.max(1)[1]
#        print("pred_.shape", pred_.shape)
        pred.append(pred_.cpu().numpy())
        gold.append(gold_.cpu().numpy())
        n_seq_total += 1                
        # printing loss
        if n_seq_total%print_chunk_size == print_chunk_size-1:
            print("training loss = ", loss.item()) 
    
    total_loss = total_loss/n_seq_total
    
    auc_train = AUCMeter()
    for i in range(len(pred)):
        auc_train.add(pred[i], gold[i])    
    auc_ = auc_train.value()[0]
    return total_loss, auc_

#%%
def eval_epoch(model_, validation_data, device):
    ''' Epoch operation in evaluation phase '''

    model_.eval()

    total_loss = 0
    pred = []
    gold = []
    n_seq_total = 0
    
    with torch.no_grad():
        for batch in validation_data:
#                tqdm(validation_data, mininterval=2,
#                desc='  - (Validation) ', leave=False):

            # prepare data
            src_seq, src_pos, gold_, src_fixed_feats = map(lambda x: x.to(device), batch)
            gold_ =  torch.cuda.LongTensor(gold_) #TODO 123
            gold_ = gold_.view(-1)
            
            # forward
            pred_ = model_(src_seq, src_pos) #TODO self_attn_mat is unused now, should be used later
            loss = cal_loss(pred_, gold_, smoothing=False)  #no smoothing in evaluation

            # note keeping
            total_loss += loss.item()
                 
            pred_ = pred_.max(1)[1]
            pred.append(pred_.cpu().numpy())
            gold.append(gold_.cpu().numpy())
            n_seq_total += 1

            # Printing loss
            
            if n_seq_total%print_chunk_size == print_chunk_size-1:
                print("validation loss = ", loss.item())

    total_loss = total_loss/n_seq_total

    auc_valid = AUCMeter()
    for i in range(len(pred)):
        auc_valid.add(pred[i], gold[i])
        auc_ = auc_valid.value()[0] 
    return total_loss, auc_

#%%
def train(model_, training_data, validation_data, optimizer, device, opt):
    ''' Start training '''      
        
    log_train_file = "log/train.log"
    log_valid_file = "log/eval.log"

    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write(' epoch, loss, AUC\n')

    valid_auc = []
    for epoch_i in tqdm(range(opt.epoch)):
        print('\n[ Epoch = ', epoch_i, ']')

        start = time.time()
        train_loss_, train_auc_ = train_epoch(
            model_, training_data, optimizer, device)
        print('  - (Training)   loss: {loss: 8.5f}, AUC: {AUC:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  loss=train_loss_, AUC=100*train_auc_,
                  elapse=(time.time()-start)/60))

        start = time.time()
        valid_loss_, valid_auc_ = eval_epoch(model_, validation_data, device)
        print('  - (Validation) loss: {loss: 8.5f}, AUC: {AUC:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(
                    loss=valid_loss_, AUC=100*valid_auc_,
                    elapse=(time.time()-start)/60))

        valid_auc += [valid_auc_]

        model_state_dict = model_.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_AUC_{AUC:3.3f}.chkpt'.format(AUC=100*valid_auc_)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '.chkpt'
                if valid_auc_ >= max(valid_auc):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{AUC:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss_, AUC=100*train_auc_))
                log_vf.write('{epoch},{loss: 8.5f},{AUC:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss_, AUC=100*valid_auc_))

#%%                                
def main():
    
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', default='trained.chkpt', required=False)
    parser.add_argument('-epoch', type=int, default=1000)
    parser.add_argument('-batch_size', type=int, default=16)

    
    parser.add_argument('-d_emb_vec', type=int, default=1440)
    parser.add_argument('-d_k', type=int, default=1440//8)
    parser.add_argument('-d_v', type=int, default=1440//8)

    
    parser.add_argument('-len_seq', type=int, default=10)
    
    parser.add_argument('-d_src_vec', type=int, default=1440)    
    parser.add_argument('-d_inner', type=int, default=2048) #TODO 304/512.*2048=1216.0
    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=3)  #TODO n_layer=6?
    parser.add_argument('-n_warmup_steps', type=int, default=4000)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
#    parser.add_argument('-label_smoothing', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    
    #========= Loading Dataset =========#
#    data = torch.load(opt.data) #TODO only used for next line, why should we?
    training_data =   loader(kk_mimic_dataset(phase="train"), batch_size=opt.batch_size) #TODO :fix
    validation_data = loader(kk_mimic_dataset(phase="valid"), batch_size=opt.batch_size) #TODO :fix
        

    #%%========= Preparing Model =========#
#    if opt.embs_share_weight:
#        assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx, \
#            'The src/tgt word2idx table are different but asked to share word embedding.'

    print('opt = ', opt)
    device = torch.device('cuda' if opt.cuda else 'cpu')  #TODO: Check if gpu works   
    
    model_ = model(d_src_vec=opt.d_src_vec,            
                 len_seq=opt.len_seq,
                 d_emb_vec=opt.d_emb_vec,
                 n_layers = opt.n_layers,
                 n_head=opt.n_head, d_k=opt.d_emb_vec//opt.n_head,
                 d_v=opt.d_emb_vec//opt.n_head,
                 d_inner=opt.d_inner, dropout=opt.dropout).cuda(device=device)    #TODO

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, model_.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        opt.d_emb_vec * opt.n_head, opt.n_warmup_steps)  #TODO 0.0001 == 100 times more!!

    train(model_, training_data, validation_data, optimizer, device ,opt)

#%%
if __name__ == '__main__':
    main()
