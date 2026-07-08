import torch
import torch.nn as nn
from transformer import TransformerBlock

class GPTModel(nn.Module):
    def __init__(self,vocab_size,embed_dim=512,num_heads=8,num_layers=6,max_seq_len=512,dropout=0.1):
        super().__init__()
        self.tok=nn.Embedding(vocab_size,embed_dim)
        self.pos=nn.Embedding(max_seq_len,embed_dim)
        self.blocks=nn.ModuleList([TransformerBlock(embed_dim,num_heads,embed_dim*4,dropout) for _ in range(num_layers)])
        self.ln=nn.LayerNorm(embed_dim)
        self.head=nn.Linear(embed_dim,vocab_size)
    def forward(self,idx):
        B,T=idx.shape
        pos=torch.arange(T,device=idx.device).unsqueeze(0)
        x=self.tok(idx)+self.pos(pos)
        mask=torch.tril(torch.ones(T,T,device=idx.device)).unsqueeze(0).unsqueeze(0)
        for b in self.blocks:
            x=b(x,mask)
        x=self.ln(x)
        return self.head(x)
