
''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np

#import transformer.Constants as Constants
#from transformer.Layers import EncoderLayer#, DecoderLayer

import Constants as Constants
from Layers import EncoderLayer#, DecoderLayer

##########################################################
def get_sinusoid_encoding_table(n_position, d_emb_vec, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx): #position in the sequence and index in the hidden dimension (#512)
        return position / np.power(10000., 2. * (hid_idx // 2) / d_emb_vec) #in the paper the pos is modeled as 2i and 2i+1
                                                                        # hid_idx//2 = i

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_emb_vec)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])  #n_position should be equal to "max_seq_len" + 1 (EOS)
 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

########  TESTing table
#T = get_sinusoid_encoding_table(2, 4).numpy()
#print(T)
#print(np.sum(T, axis=0))

##########################################################

def get_non_pad_mask(seq):
    '''Just pads the parts that are equal to Constants.PAD'''    
#    print("\nseq.shape = ", seq.shape, '\n')
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)
  
#### TEST
#seq= np.random.normal(size=(5,7))
#seq[0,0]=0.
#seq=torch.FloatTensor(seq)
#mask = get_non_pad_mask(seq)
#print(mask)

##########################################################
def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention MATRIX.
    len_q = seq_q.size(1)  #TODO zdxasd
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    
#    print("padding_mask.shape", padding_mask.shape)
    return padding_mask
    
### TEST
#seq= np.random.normal(size=(10,20))
#seq[0,0]=0.
#seq=torch.FloatTensor(seq)
#padding_mask = get_non_pad_mask(seq)
#print(padding_mask)  

##########################################################
class Encoder(nn.Module):
    ''' An encoder model with self attention mechanism. '''

    def __init__(
            self,
            len_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_inner, dropout=0.1):

        super(Encoder, self).__init__()
                      
        n_position = len_seq #+ 1  #TODO Because of SOS. Not required for continuous inputs
        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_k*n_head, padding_idx=0), #padding index is for SOS;;;; Also d_wrd_vec was changed to d_k (true #features) 
            
            freeze=True)  #Loading the table as a pretrained embedding. freeze=True makes sure it will not be updated and the same
            #across encoder and decoder

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

#        enc_slf_attn_list = []
        # -- Prepare masks
#        src_seq_ = src_seq.reshape(-1, src_seq.shape[-1])
#        print("src_seq_.shape", src_seq_.shape)
        slf_attn_mask = None #TODO get_attn_key_pad_mask(seq_k=src_seq[0], seq_q=src_seq[0]) 
#        non_pad_mask = get_non_pad_mask(src_seq[0])

        # -- Forward
#        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos) 
        src_pos = self.position_enc(src_pos)
#        src_pos = src_pos.reshape(-1, src_pos.shape[-1])
#        print("src_pos.shape = ", src_pos.shape)
        
#        print("src_seq.shape = ", src_seq.shape)
#        print("src_pos.shape = ", src_pos.shape)
        
        enc_output = src_seq + src_pos
#        print("enc_output.shape = ", enc_output.shape)
        
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
#                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
#            if return_attns:
#                enc_slf_attn_list += [enc_slf_attn]        

#        if return_attns:
#            return enc_output, enc_slf_attn_list
        return enc_output

##########################################################
#def get_subsequent_mask(seq):
#    ''' For masking out the subsequent info. '''
#
#    sz_b, len_s = seq.size()
#    subsequent_mask = torch.triu(  torch.ones((len_s, len_s), #takes a 2D matrix
#                        device=seq.device, dtype=torch.uint8), diagonal=1)
#    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls  #.unsqueeze() embrace it another list. expand extends it to all batches
#
#    return subsequent_mask








