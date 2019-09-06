''' Define the Layers '''
import torch.nn as nn

try:
    from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward
except:
    from SubLayers import MultiHeadAttention, PositionwiseFeedForward

#%%
class EncoderLayer(nn.Module):
    ''' Composed of two layers '''

    def __init__(self, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(n_head*d_k, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        if slf_attn_mask is not None:
            enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        if slf_attn_mask is not None:
            enc_output *= non_pad_mask

        return enc_output, enc_slf_attn

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
