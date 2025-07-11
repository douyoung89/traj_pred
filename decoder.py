import torch
import torch.nn as nn
from traj_encoder_4hot import PositionalEncoding 
from typing import Optional
import math

class AISDecoder(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 num_layers: int,
                 dim_feedforward: int,
                 dropout: float,
                 max_len: int,
                 lat_size: int,
                 lon_size: int,
                 sog_size: int,
                 cog_size: int,
                 lat_embd: int, 
                 lon_embd: int,
                 sog_embd: int,
                 cog_embd: int,):
        super().__init__()
        # ──────────────────────────────────────────────────────
        # 1) Four-hot embedding (토큰 입력용)
        self.register_buffer(
            'att_sizes', torch.tensor([lat_size, lon_size, sog_size, cog_size])
        )
        self.lat_emb = nn.Embedding(lat_size, lat_embd)
        self.lon_emb = nn.Embedding(lon_size, lon_embd)
        self.sog_emb = nn.Embedding(sog_size, sog_embd)
        self.cog_emb = nn.Embedding(cog_size, cog_embd) 
        # 2) Positional encoding
        self.pos_emb = PositionalEncoding(d_model, max_len, dropout)
        # 3) TransformerDecoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model, n_head, dim_feedforward, dropout, activation='gelu'
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)
        # 4) Output heads: 각 분류(classifier) 또는 회귀(regressor)
        #    여기서는 분류 head 예시
        self.head_lat = nn.Linear(d_model, lat_size)
        self.head_lon = nn.Linear(d_model, lon_size)
        self.head_sog = nn.Linear(d_model, sog_size)
        self.head_cog = nn.Linear(d_model, cog_size)
        
        self.loss_fn = nn.CrossEntropyLoss() 

    def forward(self,
                tgt_seq: torch.FloatTensor,          # (B, L) 
                memory: torch.Tensor,               # (B, L, d_model) fusion_emb
                tgt_key_padding_mask: torch.BoolTensor, 
                memory_key_padding_mask: Optional[torch.BoolTensor] = None):
        # 1) embed tgt_seq (4hot) → (B, T, d_model)
        idxs = (tgt_seq * self.att_sizes).long() 
        lat_i, lon_i, sog_i, cog_i = idxs.unbind(-1)
        e = torch.cat([
            self.lat_emb(lat_i),
            self.lon_emb(lon_i),
            self.sog_emb(sog_i),
            self.cog_emb(cog_i),
        ], dim=-1)                                     # (B, T, d_model)

        # 2) positional → (T, B, d_model)
        e = e.transpose(0,1)
        e = self.pos_emb(e)
        # 3) causal mask 자동 생성
        T = e.size(0)
        causal_mask = torch.triu(torch.ones(T,T), diagonal=1).bool().to(e.device)

        # 4) decode
        dec_out = self.decoder(
            tgt=e,                           # (T, B, E)
            memory=memory.transpose(0,1),    # (L, B, E)
            tgt_mask=causal_mask,  
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )                                   # (T, B, E)
        dec_out = dec_out.transpose(0,1)    # (B, T, E)

        # 5) project to logits
        lat_logits = self.head_lat(dec_out)
        lon_logits = self.head_lon(dec_out)
        sog_logits = self.head_sog(dec_out)
        cog_logits = self.head_cog(dec_out)
        
        
        loss_lat = self.loss_fn(lat_logits.view(-1, self.att_sizes[0].item()), lat_i.view(-1))
        loss_lon = self.loss_fn(lon_logits.view(-1, self.att_sizes[1].item()), lon_i.view(-1))
        loss_sog = self.loss_fn(sog_logits.view(-1, self.att_sizes[2].item()), sog_i.view(-1))
        loss_cog = self.loss_fn(cog_logits.view(-1, self.att_sizes[3].item()), cog_i.view(-1))
        
        loss_tuple = (loss_lat, loss_lon, loss_sog, loss_cog)
        logits = (lat_logits, lon_logits, sog_logits, cog_logits)
        loss = sum(loss_tuple)
        
        return logits, loss 